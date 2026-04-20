import sys
import os
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from torchvision import transforms
from scipy.stats import entropy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from torch import nn
from peft import get_peft_model, LoraConfig
from transformers import ResNetForImageClassification
from timm.models.layers import trunc_normal_

from models.RETFound_MAE import models_vit as retfound_vit
from models.RETFound_MAE.util import pos_embed as retfound_pos_embed

from data_processing.mixed_dataset import ODIRDataset, ODIR_CLASS_NAMES, NUM_CLASSES
from utilities.utils import identity_transform

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "../../../datasets"
SRC_DIR  = "../../"

PUBLICATION_RC = {
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.titlesize': 10, 'figure.dpi': 150,
}

# prediction threshold for multi-label
THRESHOLD = 0.5


def attention_entropy(mask):
    flat  = mask.flatten().astype(np.float64)
    flat  = np.clip(flat, 0, None)
    total = flat.sum()
    if total == 0:
        return 0.0
    prob = flat / total
    return float(entropy(prob, base=2))


def load_retfound(checkpoint_path):
    ckpt     = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    lora_cfg = ckpt.get("lora", {"r": 8, "alpha": 32, "dropout": 0.05})

    model = retfound_vit.__dict__["vit_large_patch16"](
        num_classes=NUM_CLASSES, drop_path_rate=0.2, global_pool=True
    )
    pretrained_path = f"{SRC_DIR}/models/RETFound_MAE/weights/RETFound_cfp_weights.pth"
    pretrained  = torch.load(pretrained_path, map_location="cpu", weights_only=False)
    ckpt_model  = pretrained["model"]
    state_dict  = model.state_dict()
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


def load_resnet50(checkpoint_path):
    ckpt  = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ResNetForImageClassification.from_pretrained(
        "microsoft/resnet-50", num_labels=NUM_CLASSES, ignore_mismatched_sizes=True
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    return model.eval().to(DEVICE)



def gradcam_retfound(model, image_tensor, target_class):
    features  = []
    gradients = []
 
    def fwd_hook(module, input, output):
        features.append(output.detach())
 
    def bwd_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())
 
    target_layer = model.base_model.model.blocks[-1].norm1
    fh = target_layer.register_forward_hook(fwd_hook)
    bh = target_layer.register_full_backward_hook(bwd_hook)
 
    model.zero_grad()
    output = model(image_tensor.to(DEVICE))
    output[0, target_class].backward()
 
    fh.remove()
    bh.remove()
 
    feat = features[0][0, 1:, :]
    grad = gradients[0][0, 1:, :]
    
    weights = grad.mean(dim=0, keepdim=True)
    cam = (weights * feat).sum(dim=-1)
    cam = F.relu(cam).cpu().numpy()
    
    n = int(len(cam) ** 0.5)
    cam = cam.reshape(n, n)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam


def gradcam_resnet(model, image_tensor, target_class):
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
    # backprop through the target class logit
    output[0, target_class].backward()

    fh.remove()
    bh.remove()

    weights = gradients[0].mean(dim=(2, 3), keepdim=True)
    cam     = (weights * features[0]).sum(dim=1).squeeze()
    cam     = F.relu(cam).cpu().numpy()
    cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam


# image selection

def find_representative_images(models_dict, dataset):
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

        img_path   = dataset.image_paths[idx]
        true_label = dataset.labels[idx]

        positive_needed = [c for c in needed if true_label[c] == 1.0]
        if not positive_needed:
            continue

        pil_img    = Image.open(img_path).convert("RGB")
        img_tensor = transform_tensor(pil_img).unsqueeze(0)

        for cls_idx in positive_needed:
            all_correct  = True
            model_probs  = {}   # model_name -> list of (class_name, prob) for predicted positives

            for model_name, model in models_dict.items():
                with torch.no_grad():
                    if model_name == "ResNet50":
                        logits = model(pixel_values=img_tensor.to(DEVICE)).logits
                    else:
                        logits = model(img_tensor.to(DEVICE))
                    probs     = torch.sigmoid(logits)[0]   # (8,)
                    predicted = (probs >= THRESHOLD).int().cpu().numpy()

                    # capture all predicted positive classes and their probs
                    pos_preds = [
                        (ODIR_CLASS_NAMES[i], float(probs[i]))
                        for i in range(NUM_CLASSES) if predicted[i] == 1
                    ]
                    model_probs[model_name] = pos_preds

                    if predicted[cls_idx] != 1:
                        all_correct = False
                        break

            if all_correct:
                found[cls_idx] = (img_tensor, transform_display(pil_img),
                                  img_path, model_probs)
                needed.discard(cls_idx)
                print(f"Found image for class '{ODIR_CLASS_NAMES[cls_idx]}': "
                      f"{os.path.basename(img_path)}")

    if needed:
        print(f"Warning: no jointly-correct image found for: "
              f"{[ODIR_CLASS_NAMES[c] for c in sorted(needed)]}")

    return found

# overlay helper

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
    from matplotlib.colors import Normalize as MplNorm
    from matplotlib import cm as mpl_cm

    plt.rcParams.update(PUBLICATION_RC)

    class_indices = sorted(found_images.keys())
    model_names   = list(models_dict.keys())
    n_cols        = len(class_indices)       # one column per disease class
    n_rows        = 1 + len(model_names)     # original row + one row per model

    # landscape: 8 disease classes across, 3 rows deep
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.8 * n_cols, 6.5 * n_rows))

    if n_cols == 1:
        axes = axes[:, np.newaxis]

    # column headers — disease class names
    for col, cls_idx in enumerate(class_indices):
        axes[0, col].set_title(ODIR_CLASS_NAMES[cls_idx],
                               fontsize=30, fontweight='bold', pad=8)

    # row labels on the left
    row_labels = ["Original"] + model_names
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=20, fontweight='bold',
                                rotation=90, labelpad=10)

    for col, cls_idx in enumerate(class_indices):
        img_tensor, pil_img, img_path, model_probs = found_images[cls_idx]

        # --- original image row ---
        axes[0, col].imshow(pil_img)
        axes[0, col].set_xticks([])
        axes[0, col].set_yticks([])
        for spine in axes[0, col].spines.values():
            spine.set_visible(False)

        # --- one row per model ---
        for row, (model_name, model) in enumerate(models_dict.items(), start=1):
            if model_name == "RETFound":
                mask = gradcam_retfound(model, img_tensor, target_class=cls_idx)
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

            # predicted positive classes badge
            pos_preds = model_probs.get(model_name, [])
            target_name = ODIR_CLASS_NAMES[cls_idx]
            if pos_preds:
                label_lines = [f"{name}: {prob:.0%}" for name, prob in pos_preds]
                badge_text = "\n".join(label_lines)
                badge_col = "#2ecc71" if any(
                    name == target_name for name, _ in pos_preds
                ) else "#e74c3c"
            else:
                badge_text = "no positives predicted"
                badge_col  = "#e74c3c"

            axes[row, col].set_title(
                badge_text,
                fontsize=30, color=badge_col, pad=4,
                linespacing=1.4,
            )

            # entropy beneath subplot
            ent = attention_entropy(mask)
            axes[row, col].text(
                0.5, -0.08,
                f"H = {ent:.2f} bits",
                transform=axes[row, col].transAxes,
                ha='center', va='top',
                fontsize=30, color='#444444', fontstyle='italic',
            )

    plt.suptitle(
        "ODIR-5K Multi-Label — Attention Maps per Disease Class\n"
        "RETFound: attention rollout   |   ResNet50: Grad-CAM",
        fontsize=40, y=1.01,
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.subplots_adjust(hspace=0.5, wspace=0.08)

    # shared horizontal colorbar
    norm       = MplNorm(vmin=0, vmax=1)
    scalar_map = mpl_cm.ScalarMappable(cmap="jet", norm=norm)
    scalar_map.set_array([])
    cbar_ax = fig.add_axes([0.12, 0.02, 0.78, 0.012])
    cbar    = fig.colorbar(scalar_map, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Normalised attention weight  (0 = low,  1 = high)", fontsize=40)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.tick_params(labelsize=30)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

# main

def main():
    csv_path = f"{DATA_DIR}/ODIR-5K/full_df.csv"
    img_dir = f"{DATA_DIR}/ODIR-5K/training"

    test_transformations = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # use val split — same 20% holdout used for evaluation
    dataset = ODIRDataset(
        img_dir, csv_path, split="val",
        img_transform=test_transformations
    )
    print(f"Dataset samples: {len(dataset)}")

    print("\nLoading models...")
    models_dict = {
        "RETFound": load_retfound(f"{SRC_DIR}/best_models/best_retfound_mixed_lora_model.pth"),
        "ResNet50": load_resnet50(f"{SRC_DIR}/best_models/best_resnet50_mixed_disease_model.pth"),
    }

    print("\nFinding representative images (true positive for each class, "
          "correctly predicted by all models)...")
    found_images = find_representative_images(models_dict, dataset)

    if len(found_images) == 0:
        print("No representative images found. Exiting.")
        return

    print(f"\nFound images for {len(found_images)}/{NUM_CLASSES} classes.")

    output_dir = "../../plots/attention-maps"
    os.makedirs(output_dir, exist_ok=True)

    plot_attention_grid(
        found_images, models_dict,
        os.path.join(output_dir, "mixed_disease_attention_maps.png")
    )


if __name__ == "__main__":
    main()
