import time
import sys
import os
import json
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from models.RETFound_MAE import models_vit
from models.RETFound_MAE.util import pos_embed
from timm.models.layers import trunc_normal_
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn
from sklearn.metrics import roc_auc_score

from data_processing.dataset import CombinedDRDataSet
from utilities.utils import (
    identity_transform,
    test_retfound,
    json_to_csv, plot_confusion_matrix_with_ci
)
from peft import get_peft_model, LoraConfig

# -------------------------
# Minimal, training-aligned hparams
# -------------------------
NUM_CLASSES = 5
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "../../../../datasets"
SRC_DIR = "../../../"

PUBLICATION_RC = {
    'font.family':       'serif',
    'font.serif':        ['Times New Roman', 'DejaVu Serif'],
    'axes.labelsize':    13,
    'axes.titlesize':    14,
    'xtick.labelsize':   11,
    'ytick.labelsize':   11,
    'legend.fontsize':   11,
    'figure.dpi':        150,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.linewidth':    0.8,
    'xtick.direction':   'out',
    'ytick.direction':   'out',
}

# colour used for all benchmark lines
BENCH_COLOR   = '#4c72b0'
MEAN_COLOR    = '#c0392b'   # red for mean reference line
BAND_COLOR    = '#4c72b0'   # +- std


def _apply_rc():
    plt.rcParams.update(PUBLICATION_RC)

# -------------------------
# Backbone loader (identical logic to training)
# -------------------------
def load_retfound_backbone(model):
    checkpoint_path = f"{SRC_DIR}/models/RETFound_MAE/weights/RETFound_cfp_weights.pth"
    pretrained = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_model = pretrained["model"]
    state_dict = model.state_dict()
    for k in ["head.weight", "head.bias"]:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            del checkpoint_model[k]
    pos_embed.interpolate_pos_embed(model, checkpoint_model)
    model.load_state_dict(checkpoint_model, strict=False)
    trunc_normal_(model.head.weight, std=2e-5)
    return model


def main():
    # ==================== DATA ====================
    test_root_directories = {
        "DEEPDRID": f"{DATA_DIR}/DeepDRiD",
        "DDR": f"{DATA_DIR}/DDR",
        "EYEPACS": f"{DATA_DIR}/EYEPACS",
        "MFIDDR": f"{DATA_DIR}/MFIDDR",
    }

    test_transformations = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    test_dataset = CombinedDRDataSet(
        root_directories=test_root_directories,
        split="test",
        img_transform=test_transformations,
        label_transform=identity_transform
    )

    test_csv_paths = {
        "EYEPACS": f"{test_root_directories['EYEPACS']}/all_labels.csv",
        "DEEPDRID": f"{test_root_directories['DEEPDRID']}/regular_fundus_images/Online-Challenge1&2-Evaluation/Challenge1_labels.csv",
        "DDR": f"{test_root_directories['DDR']}/DR_grading.csv",
        "MFIDDR": f"{test_root_directories['MFIDDR']}/sample/test_fourpic_label.csv",
    }

    test_dataset.load_labels_from_csv_for_test(test_csv_paths)
    test_dataset.prune_unlabeled()

    print(f"Test samples after pruning: {len(test_dataset)}")

    # ==================== LOAD CHECKPOINT ====================
    best_path = f"{SRC_DIR}/best_models/best_retfound_lora.pth"
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)

    # Unified checkpoint parsing (training-compatible)
    lora_cfg = checkpoint.get("lora", {"r": 8, "alpha": 32, "dropout": 0.05})
    train_cfg = checkpoint.get("train", {})
    batch_size = train_cfg.get("micro_batch", 8)

    print("\nLoaded checkpoint configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  LoRA r: {lora_cfg['r']}")
    print(f"  LoRA alpha: {lora_cfg['alpha']}")
    print(f"  LoRA dropout: {lora_cfg['dropout']}")

    # ==================== MODEL ====================
# ==================== MODEL ====================
    model = models_vit.__dict__["vit_large_patch16"](
        num_classes=NUM_CLASSES,
        drop_path_rate=0.2,
        global_pool=True
    )

    # Load backbone weights FROM .pth
    model = load_retfound_backbone(model)

    for name, _ in model.named_modules():
        if "qkv" in name or "proj" in name:
            print(name)


    # Apply LoRA AFTER loading
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=["qkv", "proj"],
        bias="none",
        modules_to_save=["head"],
    )

    model = get_peft_model(model, peft_config)
    
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)


    model = model.to(DEVICE)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    for name, param in model.named_parameters():
        if "lora" in name or "head" in name:
            param.requires_grad = True
        if "norm" in name.lower():
            param.requires_grad = True

    # ==================== EVALUATION ====================
    criterion = nn.CrossEntropyLoss()

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )

    test_loss, test_acc, precision, recall, f1, qwk, per_class_auc, macro_auc, weighted_auc, per_class_recall, per_class_precision, y_true, y_probs = test_retfound(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=DEVICE
    )

    print("\nFINAL TEST RESULTS")
    print(f"Accuracy: {test_acc:.2f}%")
    print(f"Loss: {test_loss:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"QWK: {qwk:.4f}")
    print("Per-class AUROC:")
    for c, auc in per_class_auc.items():
        if auc is None:
            print(f"{c}: N/A")
        else:
            print(f"{c}: {auc:.4f}")
    print(f"Macro AUC: {macro_auc:.4f}")
    print(f"Weighted AUC: {weighted_auc:.4f}")
    
    
    per_class_auc_list = [per_class_auc[f"DR{i}"] for i in range(len(per_class_auc)) if per_class_auc[f"DR{i}"] is not None]


    # ==================== SAVE RESULTS ====================
    results = {
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "quadratic_weighted_kappa": float(qwk),
        "Per-class AUC": per_class_auc_list,
        "macro_auc": float(macro_auc),
        "weighted_auc": float(weighted_auc),
        "per_class_recall": per_class_recall,
        "per_class_precision": per_class_precision,
        "checkpoint": os.path.basename(best_path),
        "lora": lora_cfg,
        "train": train_cfg,
    }

    os.makedirs("results/retfound", exist_ok=True)
    results_path = "results/retfound/retfound_test_results.json"
    
    if os.path.exists(results_path):
        os.remove(results_path)
        print("Existing JSON removed")
        time.sleep(3)

    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    json_to_csv(results_path, "results/retfound", "retfound_results")
    

    np.save("../probs_numpy/retfound_dr_lora_true.npy", y_true)
    np.save("../probs_numpy/retfound_dr_lora_probs.npy", y_probs)

    print(f"\nResults saved to: {results_path}")
    print("=" * 70)

    y_pred = np.argmax(y_probs, axis=1)
    class_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"] 
    plot_confusion_matrix_with_ci(
        y_true      = y_true,
        y_pred      = y_pred,
        class_names = class_names,
        title       = "RETFound-LoRA DR Grading",
        save_path   = "../../../plots/confusion_matrices/dr/lora/retfound_cf.png",
    )

if __name__ == "__main__":
    main()

