import sys
import os
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib import cm as mpl_cm
from scipy.stats import entropy as scipy_entropy
from PIL import Image
from torchvision import transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

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
DATA_DIR    = "../../../datasets"
SRC_DIR     = "../../"

CLASS_NAMES = ["Healthy", "Glaucoma"]

PUBLICATION_RC = {
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.titlesize': 11, 'figure.dpi': 150,
}


# model wrappers

class CLIPRetina(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.vision = CLIPVisionModelWithProjection.from_pretrained(model_name)
        self.classifier = nn.Linear(self.vision.config.projection_dim, num_classes)

    def forward(self, images):
        return self.classifier(self.vision(pixel_values=images).image_embeds)


# model loaders

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


# attention rollout — ViTs

def attention_rollout_timm(model, image_tensor):
    attention_maps = []
    hooks = []

    def make_hook(store):
        def hook(module, input, output):
            store.append(input[0].detach().cpu())
        return hook

    base = model.base_model.model
    for block in base.blocks:
        block.attn.fused_attn = False

    for name, module in model.named_modules():
        if name.endswith("attn.attn_drop"):
            hooks.append(module.register_forward_hook(make_hook(attention_maps)))

    model.train()
    with torch.no_grad():
        _ = model(image_tensor.to(DEVICE))
    model.eval()

    for h in hooks:
        h.remove()

    for block in base.blocks:
        block.attn.fused_attn = True

    if not attention_maps:
        raise RuntimeError("No attention maps captured.")

    seq_len = attention_maps[0].size(-1)
    result  = torch.eye(seq_len)
    for attn in attention_maps:
        attn_mean = attn[0].mean(dim=0)
        attn_hat  = 0.5 * attn_mean + 0.5 * torch.eye(seq_len)
        attn_hat  = attn_hat / attn_hat.sum(dim=-1, keepdim=True)
        result    = attn_hat @ result

    mask = result[0, 1:].numpy()
    n    = int(len(mask) ** 0.5)
    mask = mask.reshape(n, n)
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    return mask


def attention_rollout_clip(model, image_tensor):
    vision = model.base_model.model.vision
    vision.config._attn_implementation = "eager"

    with torch.no_grad():
        outputs = vision(
            pixel_values=image_tensor.to(DEVICE),
            output_attentions=True
        )

    attentions = outputs.attentions
    seq_len    = attentions[0].size(-1)
    result     = torch.eye(seq_len)

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


# Grad-CAM — ResNet50

def gradcam_resnet(model, image_tensor, target_class):
    """
    Grad-CAM on the last encoder stage of HuggingFace ResNet50.
    Returns (H, W) numpy array normalised 0-1.
    """
    features  = []
    gradients = []

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

    weights = gradients[0].mean(dim=(2, 3), keepdim=True)
    cam     = (weights * features[0]).sum(dim=1).squeeze()
    cam     = F.relu(cam)
    cam     = cam.cpu().numpy()
    cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam


# attention entropy

def attention_entropy(mask):
    """
    Shannon entropy of an attention map in bits.
    The map is flattened and normalised to a probability distribution
    before computing H = -sum(p * log2(p)).
    Higher entropy = more diffuse attention.
    """
    flat  = mask.flatten().astype(np.float64)
    flat  = np.clip(flat, 0, None)
    total = flat.sum()
    if total == 0:
        return 0.0
    prob = flat / total
    return float(scipy_entropy(prob, base=2))


def find_representative_images(models_dict, dataset, threshold=0.6):
    """
    For each class, find the first image correctly classified by ALL models.
    Also captures per-model softmax probabilities for prediction confidence badges.
    Returns dict: {class_idx: (image_tensor, pil_image, image_path, model_probs)}
    """
    transform_tensor = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform_display = transforms.Compose([
        transforms.Resize((224, 224)),
    ])

    found  = {}
    needed = set(range(NUM_CLASSES))

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
        model_probs = {}
        for name, model in models_dict.items():
            with torch.no_grad():
                if name == "ResNet50":
                    logits = model(pixel_values=img_tensor.to(DEVICE)).logits
                else:
                    logits = model(img_tensor.to(DEVICE))
                probs     = F.softmax(logits, dim=1)[0]
                predicted = probs.argmax().item()
                model_probs[name] = (predicted, float(probs[predicted]))
                if predicted != true_label:
                    all_correct = False
                    break

        if all_correct:
            found[true_label] = (
                img_tensor,
                transform_display(pil_img),
                image_path,
                model_probs,   # dict: model_name -> (pred_class, pred_prob)
            )
            needed.discard(true_label)
            print(f"Found representative image for class {CLASS_NAMES[true_label]}: "
                  f"{os.path.basename(image_path)}")

    if needed:
        print(f"Warning: could not find jointly-correct images for classes: "
              f"{[CLASS_NAMES[c] for c in needed]}")

    return found


# overlay helper

def overlay_heatmap(pil_image, mask, patch_size_display=224):
    """Resize mask to image size and blend as heatmap overlay."""
    mask_resized = Image.fromarray((mask * 255).astype(np.uint8)).resize(
        (patch_size_display, patch_size_display), Image.BILINEAR
    )
    heatmap  = np.array(cm.jet(np.array(mask_resized) / 255.0))[:, :, :3]
    original = np.array(pil_image).astype(float) / 255.0
    blended  = np.clip(0.55 * original + 0.45 * heatmap, 0, 1)
    return blended


# plotting

def plot_attention_grid(found_images, models_dict, output_path):
    """
    Grid: rows = classes (Healthy, Glaucoma), columns = original + one per model.

    Each attention map cell shows:
      - blended heatmap overlay
      - subtitle: predicted class, correct/incorrect label, confidence %
      - below-axes label: attention entropy in bits

    A shared horizontal colorbar is added at the bottom of the figure.
    """
    plt.rcParams.update(PUBLICATION_RC)

    class_indices = sorted(found_images.keys())
    model_names   = list(models_dict.keys())
    n_rows        = len(class_indices)
    n_cols        = 1 + len(model_names)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.5 * n_rows))

    if n_rows == 1:
        axes = axes[np.newaxis, :]

    # column headers (top row only)
    col_headers = ["Original"] + model_names
    for col, header in enumerate(col_headers):
        axes[0, col].set_title(header, fontsize=11, fontweight='bold', pad=6)

    for row, cls_idx in enumerate(class_indices):
        img_tensor, pil_img, img_path, model_probs = found_images[cls_idx]

        # --- original image column ---
        axes[row, 0].set_ylabel(
            CLASS_NAMES[cls_idx], fontsize=11,
            fontweight='bold', rotation=90, labelpad=8,
        )
        axes[row, 0].imshow(pil_img)
        # use individual spine/tick removal so ylabel is preserved
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])
        for spine in axes[row, 0].spines.values():
            spine.set_visible(False)

        # --- attention map columns ---
        for col, (model_name, model) in enumerate(models_dict.items(), start=1):
            if model_name in ("RETFound", "UrFound"):
                mask = attention_rollout_timm(model, img_tensor)
            elif model_name == "CLIP":
                mask = attention_rollout_clip(model, img_tensor)
            elif model_name == "ResNet50":
                mask = gradcam_resnet(model, img_tensor, target_class=cls_idx)
            else:
                raise ValueError(f"Unknown model: {model_name}")

            overlay = overlay_heatmap(pil_img, mask)
            axes[row, col].imshow(overlay)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            for spine in axes[row, col].spines.values():
                spine.set_visible(False)

            # prediction confidence badge
            pred_cls, pred_prob = model_probs[model_name]
            correct    = (pred_cls == cls_idx)
            badge_col  = "#2ecc71" if correct else "#e74c3c"
            badge_text = "correct" if correct else "incorrect"
            axes[row, col].set_title(
                f"Pred: {CLASS_NAMES[pred_cls]} ({badge_text}, {pred_prob:.0%})",
                fontsize=7.5, color=badge_col, pad=3,
            )

            # attention entropy below the subplot
            ent = attention_entropy(mask)
            axes[row, col].text(
                0.5, -0.04,
                f"H = {ent:.2f} bits",
                transform=axes[row, col].transAxes,
                ha="center", va="top",
                fontsize=7, color="#444444", fontstyle="italic",
            )

    plt.suptitle(
        "Glaucoma Detection — Attention Maps per Class and Model\n"
        "ViT models: attention rollout   |   ResNet50: Grad-CAM",
        fontsize=12, y=1.01,
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)   # make room for colorbar

    # shared colorbar
    norm       = Normalize(vmin=0, vmax=1)
    scalar_map = mpl_cm.ScalarMappable(cmap="jet", norm=norm)
    scalar_map.set_array([])
    cbar_ax = fig.add_axes([0.12, 0.02, 0.78, 0.018])
    cbar    = fig.colorbar(scalar_map, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Normalised attention weight  (0 = low,  1 = high)", fontsize=8)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.tick_params(labelsize=7)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# main

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
