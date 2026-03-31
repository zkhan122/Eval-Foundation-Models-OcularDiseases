import sys
import os
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from torchvision import transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from torch import nn
from peft import get_peft_model, LoraConfig
from transformers import CLIPVisionModelWithProjection

from models.RETFound_MAE import models_vit as retfound_vit
from models.RETFound_MAE.util import pos_embed as retfound_pos_embed
from models.UrFound.finetune import models_vit as urfound_vit
from models.UrFound.util import pos_embed as urfound_pos_embed
from timm.models.layers import trunc_normal_

from data_processing.dataset import CombinedDRDataSet
from utilities.utils import identity_transform

NUM_CLASSES = 5
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR    = "../../../datasets"
SRC_DIR     = "../../"

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

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
        embedding_dim = self.vision.config.hidden_size
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, images):
        outputs = self.vision(pixel_values=images)
        image_features = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(image_features)
        return logits

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

    # set eager on every CLIPAttention layer directly
    # the config-level flag is too late — layers are already instantiated
    from transformers.models.clip.modeling_clip import CLIPAttention
    for module in model.modules():
        if isinstance(module, CLIPAttention):
            module.config._attn_implementation = "eager"

    peft_config = LoraConfig(
        r=lora_cfg["r"], lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=["q_proj", "k_proj", "v_proj"], bias="none",
        modules_to_save=["classifier"]
    )
    model = get_peft_model(model, peft_config)
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

    # disable fused attention — forces timm to use explicit softmax + attn_drop
    # which is the path our hooks can intercept
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

    # restore fused attention after capture
    for block in base.blocks:
        block.attn.fused_attn = True

    if not attention_maps:
        raise RuntimeError("No attention maps captured even after disabling fused attention.")

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
    """
    Attention rollout for CLIP ViT using output_attentions=True.
    Returns (H, H) numpy array normalised 0-1.
    """
    vision = model.base_model.model.vision

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


# image selection - images that all the models got correct (sampling)

def find_representative_images(models_dict, dataset):
    """
    for each DR grade (0-4), we find the first image correctly classified by ALL THE MODELS
    """
    transform_tensor = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform_display = transforms.Resize((224, 224))

    found  = {}
    needed = set(range(NUM_CLASSES))

    for idx in range(len(dataset)):
        if not needed:
            break

        # DR dataset returns (image, label) or (image, label, source)
        item       = dataset[idx]
        true_label = int(item[1])

        if true_label not in needed:
            continue

        image_path = dataset.image_paths[idx]
        pil_img    = Image.open(image_path).convert("RGB")
        img_tensor = transform_tensor(pil_img).unsqueeze(0)

        all_correct = True
        for name, model in models_dict.items():
            with torch.no_grad():
                logits    = model(img_tensor.to(DEVICE))
                predicted = logits.argmax(dim=1).item()
                if predicted != true_label:
                    all_correct = False
                    break

        if all_correct:
            found[true_label] = (img_tensor, transform_display(pil_img), image_path)
            needed.discard(true_label)
            print(f"Found representative image for class '{CLASS_NAMES[true_label]}': "
                  f"{os.path.basename(image_path)}")

    if needed:
        print(f"Warning: could not find jointly-correct images for: "
              f"{[CLASS_NAMES[c] for c in sorted(needed)]}")

    return found


# overlay util for heatmap

def overlay_heatmap(pil_image, mask):
    mask_resized = Image.fromarray((mask * 255).astype(np.uint8)).resize(
        (224, 224), Image.BILINEAR
    )
    heatmap  = np.array(cm.jet(np.array(mask_resized) / 255.0))[:, :, :3]
    original = np.array(pil_image).astype(float) / 255.0
    blended  = np.clip(0.55 * original + 0.45 * heatmap, 0, 1)
    return blended


# plotting

def plot_attention_grid(found_images, models_dict, output_path):
    """
    Grid: rows = DR grades (0-4), columns = original image + one per model.
    """
    plt.rcParams.update(PUBLICATION_RC)

    class_indices = sorted(found_images.keys())
    model_names   = list(models_dict.keys())
    n_rows        = len(class_indices)
    n_cols        = 1 + len(model_names)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.5 * n_rows))

    if n_rows == 1:
        axes = axes[np.newaxis, :]

    col_headers = ["Original"] + model_names
    for col, header in enumerate(col_headers):
        axes[0, col].set_title(header, fontsize=11, fontweight='bold', pad=6)

    for row, cls_idx in enumerate(class_indices):
        img_tensor, pil_img, img_path = found_images[cls_idx]

        axes[row, 0].set_ylabel(CLASS_NAMES[cls_idx], fontsize=10,
                                fontweight='bold', rotation=90, labelpad=8)
        axes[row, 0].imshow(pil_img)
        axes[row, 0].axis("off")

        for col, (model_name, model) in enumerate(models_dict.items(), start=1):
            if model_name == "CLIP":
                mask = attention_rollout_clip(model, img_tensor)
            else:
                mask = attention_rollout_timm(model, img_tensor)

            overlay = overlay_heatmap(pil_img, mask)
            axes[row, col].imshow(overlay)
            axes[row, col].axis("off")

    plt.suptitle(
        "DR Severity Grading — Attention Rollout per Grade and Model",
        fontsize=12, y=1.01
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# main

def main():
    test_root_directories = {
        "DEEPDRID": f"{DATA_DIR}/DeepDRiD",
        "DDR":      f"{DATA_DIR}/DDR",
        "EYEPACS":  f"{DATA_DIR}/EYEPACS",
        "MFIDDR":   f"{DATA_DIR}/MFIDDR",
    }

    test_csv_paths = {
        "EYEPACS":  f"{test_root_directories['EYEPACS']}/all_labels.csv",
        "DEEPDRID": f"{test_root_directories['DEEPDRID']}/regular_fundus_images/Online-Challenge1&2-Evaluation/Challenge1_labels.csv",
        "DDR":      f"{test_root_directories['DDR']}/DR_grading.csv",
        "MFIDDR":   f"{test_root_directories['MFIDDR']}/sample/test_fourpic_label.csv",
    }

    test_transformations = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_dataset = CombinedDRDataSet(
        root_directories=test_root_directories,
        split="test",
        img_transform=test_transformations,
        label_transform=identity_transform
    )
    test_dataset.load_labels_from_csv_for_test(test_csv_paths)
    test_dataset.prune_unlabeled()
    print(f"Test samples: {len(test_dataset)}")

    print("\nLoading models...")
    models_dict = {
        "RETFound": load_retfound(f"{SRC_DIR}/best_models/best_retfound_lora.pth"),
        "UrFound":  load_urfound(f"{SRC_DIR}/best_models/best_urfound_model.pth"),
        "CLIP":     load_clip(f"{SRC_DIR}/best_models/best_clip_model.pth"),
    }

    print("\nInspecting RETFound attention module names:")
    for name, module in models_dict["RETFound"].named_modules():
        if "attn" in name.lower():
            print(f"  {name}  |  {type(module).__name__}")
    
    print("\nFinding representative images (correctly classified by all models)...")
    found_images = find_representative_images(models_dict, test_dataset)

    if len(found_images) < NUM_CLASSES:
        print("Could not find jointly-correct images for all DR grades. Exiting.")
        return

    output_dir = "../../../plots/attention-maps"
    os.makedirs(output_dir, exist_ok=True)

    plot_attention_grid(
        found_images, models_dict,
        os.path.join(output_dir, "dr_attention_maps.png")
    )


if __name__ == "__main__":
    main()
