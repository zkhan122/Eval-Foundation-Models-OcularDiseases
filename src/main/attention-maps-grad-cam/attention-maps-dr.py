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


# -------------------------
# GradCAM
# -------------------------

def gradcam_retfound(model, image_tensor, target_class=None, block_idx=-6):
    """
    GradCAM for RETFound/UrFound (ViT-based models)
    
    Args:
        block_idx: which transformer block to hook 
                   -6 = 6th from end (good balance of localization and semantics)
    """
    features  = []
    gradients = []

    def fwd_hook(module, input, output):
        features.append(output.detach())

    def bwd_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    device = next(model.parameters()).device
    target_layer = model.base_model.model.blocks[block_idx].norm1
    fh = target_layer.register_forward_hook(fwd_hook)
    bh = target_layer.register_full_backward_hook(bwd_hook)

    model.zero_grad()
    output = model(image_tensor.to(device))
    
    if target_class is not None:
        output[0, target_class].backward()
    else:
        output[0].sum().backward()

    fh.remove()
    bh.remove()

    feat = features[0][0, 1:, :]  # exclude CLS token
    grad = gradients[0][0, 1:, :]
    
    weights = grad.mean(dim=0, keepdim=True)
    cam = (weights * feat).sum(dim=-1)
    cam = F.relu(cam).cpu().numpy()
    
    n = int(len(cam) ** 0.5)
    cam = cam.reshape(n, n)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam


def gradcam_clip(model, image_tensor, target_class=None, block_idx=-6):
    """
    GradCAM for CLIP vision encoder
    
    Args:
        block_idx: which transformer layer to hook 
                   -6 = 6th from end
    """
    features  = []
    gradients = []

    def fwd_hook(module, input, output):
        features.append(output.detach())  # LayerNorm outputs tensor directly

    def bwd_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    device = next(model.parameters()).device
    vision = model.base_model.model.vision
    target_layer = vision.vision_model.encoder.layers[block_idx].layer_norm1
    fh = target_layer.register_forward_hook(fwd_hook)
    bh = target_layer.register_full_backward_hook(bwd_hook)

    model.zero_grad()
    output = model(image_tensor.to(device))
    
    if target_class is not None:
        output[0, target_class].backward()
    else:
        output[0].sum().backward()

    fh.remove()
    bh.remove()

    feat = features[0][0, 1:, :]  # exclude CLS token
    grad = gradients[0][0, 1:, :]
    
    weights = grad.mean(dim=0, keepdim=True)
    cam = (weights * feat).sum(dim=-1)
    cam = F.relu(cam).cpu().numpy()
    
    n = int(len(cam) ** 0.5)
    cam = cam.reshape(n, n)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam


# -------------------------
# attention entropy
# -------------------------

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


# -------------------------
# image selection
# -------------------------

def find_representative_images(models_dict, dataset):
    """
    For each DR grade (0-4), find the first image correctly classified
    by ALL models.  Also captures per-model softmax probabilities for
    use in the prediction confidence badges.
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

        item       = dataset[idx]
        true_label = int(item[1])

        if true_label not in needed:
            continue

        image_path = dataset.image_paths[idx]
        pil_img    = Image.open(image_path).convert("RGB")
        img_tensor = transform_tensor(pil_img).unsqueeze(0)

        all_correct = True
        model_probs = {}
        for name, model in models_dict.items():
            with torch.no_grad():
                logits    = model(img_tensor.to(DEVICE))
                probs     = torch.softmax(logits, dim=1)[0]
                predicted = logits.argmax(dim=1).item()
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
            print(f"Found representative image for class '{CLASS_NAMES[true_label]}': "
                  f"{os.path.basename(image_path)}")

    if needed:
        print(f"Warning: could not find jointly-correct images for: "
              f"{[CLASS_NAMES[c] for c in sorted(needed)]}")

    return found


# -------------------------
# overlay util
# -------------------------

def overlay_heatmap(pil_image, mask):
    mask_resized = Image.fromarray((mask * 255).astype(np.uint8)).resize(
        (224, 224), Image.BILINEAR
    )
    heatmap  = np.array(cm.jet(np.array(mask_resized) / 255.0))[:, :, :3]
    original = np.array(pil_image).astype(float) / 255.0
    blended  = np.clip(0.55 * original + 0.45 * heatmap, 0, 1)
    return blended


# -------------------------
# plotting
# -------------------------

def plot_attention_grid(found_images, models_dict, output_path):
    plt.rcParams.update(PUBLICATION_RC)

    class_indices = sorted(found_images.keys())
    model_names   = list(models_dict.keys())
    n_cols        = len(class_indices)          # one column per DR grade
    n_rows        = 1 + len(model_names)        # original row + one row per model

    # landscape: wide and not too tall
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.2 * n_cols, 5.2 * n_rows))

    if n_cols == 1:
        axes = axes[:, np.newaxis]

    # column headers — DR grade names
    for col, cls_idx in enumerate(class_indices):
        axes[0, col].set_title(CLASS_NAMES[cls_idx],
                               fontsize=30, fontweight='bold', pad=8)

    # row labels on the left — Original then model names
    row_labels = ["Original"] + model_names
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=25, fontweight='bold',
                                rotation=90, labelpad=10)

    for col, cls_idx in enumerate(class_indices):
        img_tensor, pil_img, img_path, model_probs = found_images[cls_idx]

        # --- original image row ---
        axes[0, col].imshow(pil_img)
        axes[0, col].set_xticks([])
        axes[0, col].set_yticks([])
        for spine in axes[0, col].spines.values():
            spine.set_visible(False)

        # --- one row per model (using GradCAM) ---
        for row, (model_name, model) in enumerate(models_dict.items(), start=1):
            if model_name == "CLIP":
                mask = gradcam_clip(model, img_tensor, target_class=cls_idx)
            else:  # RETFound, UrFound
                mask = gradcam_retfound(model, img_tensor, target_class=cls_idx)

            overlay = overlay_heatmap(pil_img, mask)
            axes[row, col].imshow(overlay)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            for spine in axes[row, col].spines.values():
                spine.set_visible(False)

            # prediction badge as subplot title
            pred_cls, pred_prob = model_probs[model_name]
            correct    = (pred_cls == cls_idx)
            badge_col  = "#2ecc71" if correct else "#e74c3c"
            badge_text = "correct" if correct else "incorrect"
            axes[row, col].set_title(
                f"Pred: {CLASS_NAMES[pred_cls]}\n({badge_text}, {pred_prob:.0%})",
                fontsize=30, color=badge_col, pad=4,
            )

            # entropy label beneath subplot
            ent = attention_entropy(mask)
            axes[row, col].text(
                0.5, -0.08,
                f"H = {ent:.2f} bits",
                transform=axes[row, col].transAxes,
                ha="center", va="top",
                fontsize=30, color="#444444", fontstyle="italic",
            )

    plt.suptitle(
        "DR Severity Grading — Grad-CAM per Grade and Model",
        fontsize=40, y=1.01,
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.07, hspace=0.5, wspace=0.08)

    # shared colorbar
    norm       = Normalize(vmin=0, vmax=1)
    scalar_map = mpl_cm.ScalarMappable(cmap="jet", norm=norm)
    scalar_map.set_array([])
    cbar_ax = fig.add_axes([0.12, 0.02, 0.78, 0.015])
    cbar    = fig.colorbar(scalar_map, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Normalised attention weight  (0 = low,  1 = high)", fontsize=40)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.tick_params(labelsize=9)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

# -------------------------
# main
# -------------------------

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

    print("\nFinding representative images (correctly classified by all models)...")
    found_images = find_representative_images(models_dict, test_dataset)

    if len(found_images) < NUM_CLASSES:
        print("Could not find jointly-correct images for all DR grades. Exiting.")
        return

    output_dir = "../../plots/attention-maps"
    os.makedirs(output_dir, exist_ok=True)

    plot_attention_grid(
        found_images, models_dict,
        os.path.join(output_dir, "dr_attention_maps_gradcam.png")
    )


if __name__ == "__main__":
    main()
