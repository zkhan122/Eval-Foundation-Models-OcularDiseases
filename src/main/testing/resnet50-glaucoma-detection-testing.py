import time
import sys
import os
import json
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn
from transformers import ResNetForImageClassification
from sklearn.metrics import (
    roc_auc_score, balanced_accuracy_score, f1_score,
    classification_report, confusion_matrix
)

from data_processing.glaucoma_dataset import CombinedGlaucomaDataset
from utilities.utils import identity_transform, json_to_csv

NUM_CLASSES = 2
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "../../../datasets"
SRC_DIR = "../../"


def evaluate(model, dataloader, criterion, device, threshold):
    model.eval()
    total_loss = 0.0
    y_true, y_pred, y_probs = [], [], []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Test inference")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(pixel_values=images)
            loss = criterion(outputs.logits, labels)
            total_loss += loss.item()

            probs = F.softmax(outputs.logits, dim=1)
            preds = (probs[:, 1] >= threshold).long()

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            y_probs.extend(probs.cpu().numpy())

            pbar.set_postfix({
                "loss": total_loss / (pbar.n or 1),
                "acc":  100.0 * sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
            })

    y_true  = np.array(y_true)
    y_pred  = np.array(y_pred)
    y_probs = np.array(y_probs)

    avg_loss = total_loss / len(dataloader)
    acc      = 100.0 * (y_true == y_pred).mean()
    bal_acc  = 100.0 * balanced_accuracy_score(y_true, y_pred)
    macro_f1 = 100.0 * f1_score(y_true, y_pred, average="macro")

    class_names = ["Healthy", "Glaucoma"]
    per_class_auc = {}
    for k, name in enumerate(class_names):
        binary_true = (y_true == k).astype(int)
        if binary_true.sum() > 0:
            per_class_auc[name] = 100.0 * roc_auc_score(binary_true, y_probs[:, k])
        else:
            per_class_auc[name] = None

    try:
        macro_auc    = 100.0 * roc_auc_score(y_true, y_probs[:, 1])
        weighted_auc = 100.0 * roc_auc_score(
            y_true,
            y_probs[:, 1],
            sample_weight=np.array(
                [1.0 / np.sum(y_true == y_true[i]) for i in range(len(y_true))]
            )
        )
    except ValueError:
        macro_auc = weighted_auc = 0.0

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sensitivity = 100.0 * tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = 100.0 * tn / (tn + fp) if (tn + fp) > 0 else 0.0

    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0
    )

    return (avg_loss, acc, bal_acc, macro_f1,
            per_class_auc, macro_auc, weighted_auc,
            sensitivity, specificity, report)


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
        root_directories=root_dirs,
        split="test",
        img_transform=test_transformations,
        label_transform=identity_transform
    )

    print(f"Test samples: {len(test_dataset)}")
    print(test_dataset.get_dataset_statistics())

    best_path = f"{SRC_DIR}/best_models/best_resnet50_glaucoma_model.pth"
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)

    train_cfg  = checkpoint.get("train", {})
    batch_size = train_cfg.get("micro_batch", 8)

    print(f"\nCheckpoint: bal_acc={checkpoint.get('val_bal_acc', 'N/A'):.2f}%")

    # reconstruct with same num_labels and ignore_mismatched_sizes as training,
    # then immediately overwrite all weights from checkpoint — no imagenet download needed
    model = ResNetForImageClassification.from_pretrained(
        "microsoft/resnet-50",
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(DEVICE)
    model.eval()

    criterion = nn.CrossEntropyLoss()

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )

    THRESHOLD = 0.6
    (test_loss, acc, bal_acc, macro_f1,
     per_class_auc, macro_auc, weighted_auc,
     sensitivity, specificity, report) = evaluate(
        model, test_loader, criterion, DEVICE, THRESHOLD
    )

    print("\nFINAL TEST RESULTS")
    print("=" * 50)
    print(f"accuracy         : {acc:.2f}%")
    print(f"balanced accuracy: {bal_acc:.2f}%")
    print(f"macro f1         : {macro_f1:.2f}%")
    print(f"sensitivity      : {sensitivity:.2f}%")
    print(f"specificity      : {specificity:.2f}%")
    print(f"macro auc        : {macro_auc:.2f}%")
    print(f"weighted auc     : {weighted_auc:.2f}%")
    print("per-class auc:")
    for name, auc_val in per_class_auc.items():
        print(f"  {name}: {auc_val:.2f}%" if auc_val is not None else f"  {name}: N/A")
    print(f"loss             : {test_loss:.4f}")
    print("=" * 50)
    print(report)

    results = {
        "accuracy":          float(acc),
        "balanced_accuracy": float(bal_acc),
        "macro_f1":          float(macro_f1),
        "sensitivity":       float(sensitivity),
        "specificity":       float(specificity),
        "macro_auc":         float(macro_auc),
        "weighted_auc":      float(weighted_auc),
        "per_class_auc":     {k: float(v) if v is not None else None
                              for k, v in per_class_auc.items()},
        "test_loss":         float(test_loss),
        "checkpoint":        os.path.basename(best_path),
        "train":             train_cfg,
    }

    os.makedirs("results/resnet50-glaucoma", exist_ok=True)
    results_path = "results/resnet50-glaucoma/resnet50_glaucoma_test_results.json"

    if os.path.exists(results_path):
        os.remove(results_path)
        time.sleep(1)

    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    json_to_csv(results_path, "results/resnet50-glaucoma", "resnet50_glaucoma_results")

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
