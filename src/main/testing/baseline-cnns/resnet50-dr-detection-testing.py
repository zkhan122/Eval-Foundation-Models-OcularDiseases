import time
import sys
import os
import json
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn
from transformers import ResNetForImageClassification
from sklearn.metrics import (
    roc_auc_score, balanced_accuracy_score, f1_score,
    classification_report, cohen_kappa_score
)

from data_processing.dataset import CombinedDRDataSet
from utilities.utils import identity_transform, json_to_csv

NUM_CLASSES  = 5
NUM_WORKERS  = 4
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "../../../../datasets"
SRC_DIR  = "../../../"

DR_CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]


def evaluate(model, dataloader, criterion, device):
    # no threshold parameter — multiclass uses argmax not binary threshold
    model.eval()
    total_loss = 0.0
    y_true, y_pred, y_probs = [], [], []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Test inference")
        for batch in pbar:
            images = batch[0].to(device)
            labels = batch[1].to(device)

            outputs = model(pixel_values=images)
            loss    = criterion(outputs.logits, labels)
            total_loss += loss.item()

            probs = F.softmax(outputs.logits, dim=1)
            preds = outputs.logits.argmax(dim=1)

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
    qwk      = cohen_kappa_score(y_true, y_pred, weights="quadratic")

    # per-class auc using one-vs-rest
    per_class_auc = {}
    for k, name in enumerate(DR_CLASS_NAMES):
        binary_true = (y_true == k).astype(int)
        if binary_true.sum() > 0:
            per_class_auc[name] = 100.0 * roc_auc_score(binary_true, y_probs[:, k])
        else:
            per_class_auc[name] = None

    try:
        macro_auc = 100.0 * roc_auc_score(
            y_true, y_probs, multi_class="ovr", average="macro"
        )
        weighted_auc = 100.0 * roc_auc_score(
            y_true, y_probs, multi_class="ovr", average="weighted"
        )
    except ValueError:
        macro_auc = weighted_auc = 0.0

    report = classification_report(
        y_true, y_pred,
        target_names=DR_CLASS_NAMES,
        digits=4,
        zero_division=0
    )

    return (avg_loss, acc, bal_acc, macro_f1, qwk,
            per_class_auc, macro_auc, weighted_auc,
            report, y_probs)


def main():
    root_dirs = {
        "DEEPDRID": f"{DATA_DIR}/DeepDRiD",
        "DDR":      f"{DATA_DIR}/DDR",
        "EYEPACS":  f"{DATA_DIR}/EYEPACS",
        "MFIDDR":   f"{DATA_DIR}/MFIDDR",
    }

    test_csv_paths = {
        "EYEPACS":  f"{root_dirs['EYEPACS']}/all_labels.csv",
        "DEEPDRID": f"{root_dirs['DEEPDRID']}/regular_fundus_images/Online-Challenge1&2-Evaluation/Challenge1_labels.csv",
        "DDR":      f"{root_dirs['DDR']}/DR_grading.csv",
        "MFIDDR":   f"{root_dirs['MFIDDR']}/sample/test_fourpic_label.csv",
    }

    test_transformations = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_dataset = CombinedDRDataSet(
        root_directories=root_dirs,
        split="test",
        img_transform=test_transformations,
        label_transform=identity_transform
    )
    test_dataset.load_labels_from_csv_for_test(test_csv_paths)
    test_dataset.prune_unlabeled()

    print(f"Test samples: {len(test_dataset)}")

    best_path  = f"{SRC_DIR}/best_models/best_resnet50_dr_model.pth"
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)

    train_cfg  = checkpoint.get("train", {})
    batch_size = train_cfg.get("micro_batch", 8)

    print(f"\nCheckpoint: bal_acc={checkpoint.get('val_bal_acc', 'N/A'):.2f}%")

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

    (test_loss, acc, bal_acc, macro_f1, qwk,
     per_class_auc, macro_auc, weighted_auc,
     report, y_probs) = evaluate(model, test_loader, criterion, DEVICE)

    print("\nFINAL TEST RESULTS")
    print("=" * 50)
    print(f"accuracy         : {acc:.2f}%")
    print(f"balanced accuracy: {bal_acc:.2f}%")
    print(f"macro f1         : {macro_f1:.2f}%")
    print(f"qwk              : {qwk:.4f}")
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
        "qwk":               float(qwk),
        "macro_auc":         float(macro_auc),
        "weighted_auc":      float(weighted_auc),
        "per_class_auc":     {k: float(v) if v is not None else None
                              for k, v in per_class_auc.items()},
        "test_loss":         float(test_loss),
        "checkpoint":        os.path.basename(best_path),
        "train":             train_cfg,
    }

    os.makedirs("results/resnet50-dr", exist_ok=True)
    results_path = "results/resnet50-dr/resnet50_dr_test_results.json"

    if os.path.exists(results_path):
        os.remove(results_path)
        time.sleep(1)

    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    json_to_csv(results_path, "results/resnet50-dr", "resnet50_dr_results")

    os.makedirs("../probs_numpy", exist_ok=True)
    np.save("../probs_numpy/resnet50-dr-testing.npy", y_probs)

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
