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
    json_to_csv,
)


# -------------------------
# Minimal, training-aligned hparams
# -------------------------
NUM_CLASSES = 5
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "../../../../datasets"
SRC_DIR = "../../../"


# -------------------------
# Backbone loader (identical logic to training)
# -------------------------
def load_retfound_backbone(model):
    checkpoint_path = f"{SRC_DIR}/best_models/best_retfound_nonlora_model.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    checkpoint_model = checkpoint["model_state_dict"]
    state_dict = model.state_dict()

    for k in ["head.weight", "head.bias"]:
        if k in checkpoint_model and k in state_dict:
            if checkpoint_model[k].shape != state_dict[k].shape:
                del checkpoint_model[k]

    pos_embed.interpolate_pos_embed(model, checkpoint_model)
    model.load_state_dict(checkpoint_model, strict=False)
    trunc_normal_(model.head.weight, std=2e-5)

    return model


def strip_prefix(state_dict, prefix="base_model.model."):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_state_dict[k[len(prefix):]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

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
    best_path = f"{SRC_DIR}/best_models/best_retfound_nonlora_model.pth"
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)

    # Unified checkpoint parsing (training-compatible)
    train_cfg = checkpoint.get("train", {})
    batch_size = train_cfg.get("micro_batch", 8)

    print("\nLoaded checkpoint configuration:")
    print(f"  Batch size: {batch_size}")
    # ==================== MODEL ====================
    model = models_vit.__dict__["vit_large_patch16"](
        num_classes=NUM_CLASSES,
        drop_path_rate=0.2,
        global_pool=True
    )

    model = models_vit.__dict__["vit_large_patch16"](
    num_classes=NUM_CLASSES,
    drop_path_rate=0.2,
    global_pool=True
)

    checkpoint_model = strip_prefix(checkpoint["model_state_dict"])
    checkpoint_model = {k: v for k, v in checkpoint_model.items() if "lora_" not in k}

    pos_embed.interpolate_pos_embed(model, checkpoint_model)

    missing, unexpected = model.load_state_dict(checkpoint_model, strict=False)

    print("Missing:", missing)
    print("Unexpected:", unexpected)

    checkpoint_model = model.to(DEVICE)
    checkpoint_:model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

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

    test_loss, test_acc, precision, recall, f1, qwk, per_class_auc, macro_auc, weighted_auc, y_probs = test_retfound(
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
        "checkpoint": os.path.basename(best_path),
        "train": train_cfg,
    }

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(BASE_DIR, "results", "retfound")

    os.makedirs(results_dir, exist_ok=True)

    results_path = os.path.join(results_dir, "retfound_nonlora_test_results.json")
    
    if os.path.exists(results_path):
        os.remove(results_path)
        print("Existing JSON removed")
        time.sleep(3)

    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    json_to_csv(results_path, "results/retfound", "retfound_nonlora_test_results")
    
    np.save("../probs_numpy/retfound_dr_nonlora_probs.npy", y_probs)

    print(f"\nResults saved to: {results_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

