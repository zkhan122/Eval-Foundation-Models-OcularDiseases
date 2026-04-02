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



def load_retfound(checkpoint_path):
    ckpt     = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    lora_cfg = ckpt.get("lora", {"r": 8, "alpha": 32, "dropout": 0.05})

    model = retfound_vit.__dict__["vit_large_patch16"](
        num_classes=NUM_CLASSES, drop_path_rate=0.2, global_pool=True
    )
    pretrained_path = f"{SRC_DIR}/models/RETFound_MAE/weights/RETFound_cfp_weights.pth"
    pretrained      = torch.load(pretrained_path, map_location="cpu", weights_only=False)
    ckpt_model      = pretrained["model"]
    state_dict      = model.state_dict()
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
    """
    For each of the 8 ODIR disease classes, find the first image where:
      - the ground truth label for that class is 1
      - ALL models correctly predict that class as positive
    Returns dict: {class_idx: (img_tensor, pil_img, img_path)}
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

        img_path  = dataset.image_paths[idx]
        true_label = dataset.labels[idx]   # (8,) float array

        # find which needed classes this image is positive for
        positive_needed = [c for c in needed if true_label[c] == 1.0]
        if not positive_needed:
            continue

        pil_img    = Image.open(img_path).convert("RGB")
        img_tensor = transform_tensor(pil_img).unsqueeze(0)

        for cls_idx in positive_needed:
            all_correct = True
            for model_name, model in models_dict.items():
                with torch.no_grad():
                    if model_name == "ResNet50":
                        logits = model(pixel_values=img_tensor.to(DEVICE)).logits
                    else:
                        logits = model(img_tensor.to(DEVICE))
                    prob      = torch.sigmoid(logits)[0, cls_idx].item()
                    predicted = 1 if prob >= THRESHOLD else 0
                    if predicted != 1:
                        all_correct = False
                        break

            if all_correct:
                found[cls_idx] = (img_tensor, transform_display(pil_img), img_path)
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
    plt.rcParams.update(PUBLICATION_RC)

    class_indices = sorted(found_images.keys())
    model_names   = list(models_dict.keys())
    n_rows        = len(class_indices)
    n_cols        = 1 + len(model_names)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.2 * n_cols, 4.0 * n_rows))

    if n_rows == 1:
        axes = axes[np.newaxis, :]

    col_headers = ["Original"] + model_names
    for col, header in enumerate(col_headers):
        axes[0, col].set_title(header, fontsize=10, fontweight='bold', pad=6)

    for row, cls_idx in enumerate(class_indices):
        img_tensor, pil_img, img_path = found_images[cls_idx]

        # row ylabel on the left
        axes[row, 0].set_ylabel(ODIR_CLASS_NAMES[cls_idx], fontsize=10,
                                fontweight='bold', rotation=90, labelpad=8)

        # original image
        axes[row, 0].imshow(pil_img)
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])
        for spine in axes[row, 0].spines.values():
            spine.set_visible(False)

        # class name text label beneath the original image
        axes[row, 0].text(
            0.5, -0.08,
            ODIR_CLASS_NAMES[cls_idx],
            transform=axes[row, 0].transAxes,
            ha='center', va='top',
            fontsize=9, fontweight='bold',
            color='#222222', fontfamily='serif',
        )

        for col, (model_name, model) in enumerate(models_dict.items(), start=1):
            if model_name == "RETFound":
                mask = attention_rollout_timm(model, img_tensor)
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

    plt.suptitle(
        "ODIR-5K Multi-Label — Attention Maps per Disease Class\n"
        "RETFound: attention rollout   |   ResNet50: Grad-CAM",
        fontsize=11, y=1.01
    )
    plt.tight_layout(rect=[0, 0.02, 1, 1])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

# main

def main():
    csv_path     = f"{DATA_DIR}/ODIR-5K/full_df.csv"
    img_dir      = f"{DATA_DIR}/ODIR-5K/training"

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

    output_dir = "../../../plots/attention-maps"
    os.makedirs(output_dir, exist_ok=True)

    plot_attention_grid(
        found_images, models_dict,
        os.path.join(output_dir, "mixed_disease_attention_maps.png")
    )


if __name__ == "__main__":
    main()
