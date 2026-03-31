import sys
import os
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from torchvision import transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from torch.utils.data import DataLoader
from torch import nn
from peft import get_peft_model, LoraConfig
from transformers import CLIPVisionModelWithProjection, ResNetForImageClassification

from models.RETFound_MAE import models_vit as retfound_vit
from models.RETFound_MAE.util import pos_embed as retfound_pos_embed
from models.UrFound.finetune import models_vit as urfound_vit
from models.UrFound.util import pos_embed as urfound_pos_embed
from timm.models.layers import trunc_normal_

from data_processing.glaucoma_dataset import CombinedGlaucomaDataset
from utilities.utils import identity_transform

NUM_CLASSES = 2
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR    = "../../../../datasets"
SRC_DIR     = "../../../"

CLASS_NAMES = ["Healthy", "Glaucoma"]

PUBLICATION_RC = {
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.titlesize': 11, 'figure.dpi': 150,
}


# -------------------------
# model wrappers
# -------------------------

class CLIPRetina(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.vision = CLIPVisionModelWithProjection.from_pretrained(model_name)
        self.classifier = nn.Linear(self.vision.config.projection_dim, num_classes)

    def forward(self, images):
        return self.classifier(self.vision(pixel_values=images).image_embeds)


# -------------------------
# model loaders
# -------------------------

def load_retfound(checkpoint_path):
    ckpt     = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    lora_cfg = ckpt.get("lora", {"r": 8, "alpha": 32, "dropout": 0.05})

    model = retfound_vit.__dict__["vit_large_patch16"](
        num_classes=NUM_CLASSES, drop_path_rate=0.2, global_pool=True
    )
    ckpt_model = ckpt["model_state_dict"]
    state_dict = model.state_dict()
    for k in ["head.weight", "head.bias"]:
        if k in ckpt_model and ckpt_model[k].shape != state_dict[k].shape:
            del ckpt_model[k]
    retfound_pos_embed.interpolate_pos_embed(model, ckpt_model)
    model.load_state_dict(ckpt_model, strict=False)
    trunc_normal_(model.head.weight, std=2e-5)

    peft_config = LoraConfig(
        r=lora_cfg["r"], lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=["qkv", "proj"], bias="none",
        modules_to_save=["head"]
    )
    model = get_peft_model(model, peft_config)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    return model.eval().to(DEVICE)


def load_urfound(checkpoint_path):
    ckpt     = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    lora_cfg = ckpt.get("lora", {"r": 8, "alpha": 32, "dropout": 0.05})

    model = urfound_vit.__dict__["vit_base_patch16"](
        num_classes=NUM_CLASSES, drop_path_rate=0.2, global_pool=True
    )
    ckpt_model = ckpt["model_state_dict"]
    state_dict = model.state_dict()
    for k in ["head.weight", "head.bias"]:
        if k in ckpt_model and ckpt_model[k].shape != state_dict[k].shape:
            del ckpt_model[k]
    urfound_pos_embed.interpolate_pos_embed(model, ckpt_model)
    model.load_state_dict(ckpt_model, strict=False)
    trunc_normal_(model.head.weight, std=2e-5)

    peft_config = LoraConfig(
        r=lora_cfg["r"], lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=["qkv", "proj"], bias="none",
        modules_to_save=["head"]
    )
    model = get_peft_model(model, peft_config)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    return model.eval().to(DEVICE)


def load_clip(checkpoint_path):
    ckpt     = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    lora_cfg = ckpt.get("lora", {"r": 8, "alpha": 32, "dropout": 0.05})

    model = CLIPRetina("openai/clip-vit-large-patch14", num_classes=NUM_CLASSES)
    peft_config = LoraConfig(
        r=lora_cfg["r"], lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=["q_proj", "k_proj", "v_proj"], bias="none",
        modules_to_save=["classifier"]
    )
    model = get_peft_model(model, peft_config)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    return model.eval().to(DEVICE)


def load_resnet50(checkpoint_path):
    ckpt  = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ResNetForImageClassification.from_pretrained(
        "microsoft/resnet-50", num_labels=NUM_CLASSES, ignore_mismatched_sizes=True
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    return model.eval().to(DEVICE)


# -------------------------
# attention rollout — ViTs
# -------------------------

def attention_rollout_timm(model, image_tensor):
    """
    Attention rollout for timm ViT (RETFound, UrFound).
    Hooks attn_drop output in each block — during eval, dropout is identity
    so this captures the raw softmaxed attention weights.
    Returns (H, H) numpy array normalised 0-1.
    """
    attention_maps = []
    hooks = []

    def make_hook(store):
        def hook(module, input, output):
            store.append(output.detach().cpu())
        return hook

    # unwrap PEFT to access base ViT blocks
    base = model.base_model.model
    for block in base.blocks:
        h = block.attn.attn_drop.register_forward_hook(make_hook(attention_maps))
        hooks.append(h)

    with torch.no_grad():
        _ = model(image_tensor.to(DEVICE))

    for h in hooks:
        h.remove()

    # rollout: A_hat = 0.5*A + 0.5*I, then multiply through layers
    seq_len = attention_maps[0].size(-1)
    result  = torch.eye(seq_len)
    for attn in attention_maps:
        attn_mean    = attn[0].mean(dim=0)                        # (seq, seq)
        attn_hat     = 0.5 * attn_mean + 0.5 * torch.eye(seq_len)
        attn_hat     = attn_hat / attn_hat.sum(dim=-1, keepdim=True)
        result       = attn_hat @ result

    # CLS token row, drop CLS itself → patch attentions
    mask = result[0, 1:].numpy()
    n    = int(len(mask) ** 0.5)
    mask = mask.reshape(n, n)
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    return mask


def attention_rollout_clip(model, image_tensor):
    """
    Attention rollout for CLIP ViT using output_attentions=True.
    Returns (H, H) numpy array normalised 0-1.
    """
    # unwrap PEFT → CLIPRetina → vision model
    vision = model.base_model.model.vision

    with torch.no_grad():
        outputs = vision(
            pixel_values=image_tensor.to(DEVICE),
            output_attentions=True
        )

    attentions = outputs.attentions   # tuple of (1, heads, seq, seq)

    seq_len = attentions[0].size(-1)
    result  = torch.eye(seq_len)
    for attn in attentions:
        attn_mean = attn[0].mean(dim=0).cpu()
        attn_hat  = 0.5 * attn_mean + 0.5 * torch.eye(seq_len)
        attn_hat  = attn_hat / attn_hat.sum(dim=-1, keepdim=True)
        result    = attn_hat @ result

    mask = result[0, 1:].numpy()
    n    = int(len(mask) ** 0.5)
    mask = mask.reshape(n, n)
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    return mask


# -------------------------
# Grad-CAM — ResNet50
# -------------------------

def gradcam_resnet(model, image_tensor, target_class):
    """
    Grad-CAM on the last encoder stage of HuggingFace ResNet50.
    Returns (H, W) numpy array normalised 0-1.
    """
    features   = []
    gradients  = []

    def fwd_hook(module, input, output):
        features.append(output.detach())

    def bwd_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    target_layer = model.resnet.encoder.stages[-1]
    fh = target_layer.register_forward_hook(fwd_hook)
    bh = target_layer.register_full_backward_hook(bwd_hook)

    model.zero_grad()
    output = model(pixel_values=image_tensor.to(DEVICE)).logits
    output[0, target_class].backward()

    fh.remove()
    bh.remove()

    # global average pool gradients over spatial dims → channel weights
    weights = gradients[0].mean(dim=(2, 3), keepdim=True)
    cam     = (weights * features[0]).sum(dim=1).squeeze()
    cam     = F.relu(cam)
    cam     = cam.cpu().numpy()
    cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam


# -------------------------
# image selection
# -------------------------

def find_representative_images(models_dict, dataset, threshold=0.6):
    """
    For each class, find the first image that ALL models correctly classify.
    Returns dict: {class_idx: (image_tensor, raw_pil_image, image_path)}
    """
    transform_tensor = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform_display = transforms.Compose([
        transforms.Resize((224, 224)),
    ])

    found    = {}
    needed   = set(range(NUM_CLASSES))

    for idx in range(len(dataset)):
        if not needed:
            break

        image_path = dataset.image_paths[idx]
        true_label = dataset.labels[idx]

        if true_label not in needed:
            continue

        pil_img    = Image.open(image_path).convert("RGB")
        img_tensor = transform_tensor(pil_img).unsqueeze(0)

        all_correct = True
        for name, model in models_dict.items():
            with torch.no_grad():
                if name == "ResNet50":
                    logits = model(pixel_values=img_tensor.to(DEVICE)).logits
                else:
                    logits = model(img_tensor.to(DEVICE))
                prob       = F.softmax(logits, dim=1)[0, 1].item()
                predicted  = 1 if prob >= threshold else 0
                if predicted != true_label:
                    all_correct = False
                    break

        if all_correct:
            found[true_label] = (img_tensor, transform_display(pil_img), image_path)
            needed.discard(true_label)
            print(f"Found representative image for class {CLASS_NAMES[true_label]}: {os.path.basename(image_path)}")

    if needed:
        print(f"Warning: could not find jointly-correct images for classes: {[CLASS_NAMES[c] for c in needed]}")

    return found


# -------------------------
# overlay helper
# -------------------------

def overlay_heatmap(pil_image, mask, patch_size_display=224):
    """Resize mask to image size and blend as heatmap overlay."""
    mask_resized = Image.fromarray((mask * 255).astype(np.uint8)).resize(
        (patch_size_display, patch_size_display), Image.BILINEAR
    )
    heatmap = np.array(cm.jet(np.array(mask_resized) / 255.0))[:, :, :3]
    original = np.array(pil_image).astype(float) / 255.0
    blended  = 0.55 * original + 0.45 * heatmap
    blended  = np.clip(blended, 0, 1)
    return blended


# -------------------------
# plotting
# -------------------------

def plot_attention_grid(found_images, models_dict, output_path):
    """
    Grid: rows = classes (Healthy, Glaucoma), columns = models.
    First column shows the original image.
    Remaining columns show attention/grad-cam overlays.
    """
    plt.rcParams.update(PUBLICATION_RC)

    class_indices = sorted(found_images.keys())
    model_names   = list(models_dict.keys())
    n_rows        = len(class_indices)
    n_cols        = 1 + len(model_names)   # original + one per model

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.5 * n_rows))

    # ensure axes is always 2D
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    col_headers = ["Original"] + model_names
    for col, header in enumerate(col_headers):
        axes[0, col].set_title(header, fontsize=11, fontweight='bold', pad=6)

    for row, cls_idx in enumerate(class_indices):
        img_tensor, pil_img, img_path = found_images[cls_idx]

        # row label
        axes[row, 0].set_ylabel(CLASS_NAMES[cls_idx], fontsize=11,
                                fontweight='bold', rotation=90, labelpad=8)

        # original image
        axes[row, 0].imshow(pil_img)
        axes[row, 0].axis("off")

        # attention map per model
        for col, (model_name, model) in enumerate(models_dict.items(), start=1):
            if model_name == "RETFound" or model_name == "UrFound":
                mask = attention_rollout_timm(model, img_tensor)
            elif model_name == "CLIP":
                mask = attention_rollout_clip(model, img_tensor)
            elif model_name == "ResNet50":
                mask = gradcam_resnet(model, img_tensor, target_class=cls_idx)
            else:
                raise ValueError(f"Unknown model: {model_name}")

            overlay = overlay_heatmap(pil_img, mask)
            axes[row, col].imshow(overlay)
            axes[row, col].axis("off")

    plt.suptitle(
        "Glaucoma Detection — Attention Maps per Class and Model\n"
        "ViT models: attention rollout   |   ResNet50: Grad-CAM",
        fontsize=12, y=1.01
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# -------------------------
# main
# -------------------------

def main():
    root_dirs = {
        "G1020":            f"{DATA_DIR}/G1020",
        "ORIGA":            f"{DATA_DIR}/ORIGA",
        "EYEPACS_GLAUCOMA": f"{DATA_DIR}/EYEPACS_GLAUCOMA",
    }

    test_transformations = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_dataset = CombinedGlaucomaDataset(
        root_directories=root_dirs, split="test",
        img_transform=test_transformations,
        label_transform=identity_transform
    )
    print(f"Test samples: {len(test_dataset)}")

    print("\nLoading models...")
    models_dict = {
        "RETFound": load_retfound(f"{SRC_DIR}/best_models/best_retfound_glaucoma_model.pth"),
        "UrFound":  load_urfound(f"{SRC_DIR}/best_models/best_urfound_glaucoma_lora_model.pth"),
        "CLIP":     load_clip(f"{SRC_DIR}/best_models/best_clip_glaucoma_lora_model.pth"),
        "ResNet50": load_resnet50(f"{SRC_DIR}/best_models/best_resnet50_glaucoma_model.pth"),
    }

    print("\nFinding representative images (correctly classified by all models)...")
    found_images = find_representative_images(models_dict, test_dataset)

    if len(found_images) < NUM_CLASSES:
        print("Could not find jointly-correct images for all classes. Exiting.")
        return

    output_dir = "../../../plots/attention-maps"
    os.makedirs(output_dir, exist_ok=True)

    plot_attention_grid(
        found_images, models_dict,
        os.path.join(output_dir, "glaucoma_attention_maps.png")
    )


if __name__ == "__main__":
    main()
