import time
import sys
import os
import json
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from models.UrFound.finetune import models_vit
from models.UrFound.util import pos_embed
from timm.models.layers import trunc_normal_
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn
from sklearn.metrics import (
    roc_auc_score, balanced_accuracy_score, f1_score,
    classification_report, confusion_matrix, 
)
from peft import get_peft_model, LoraConfig

from data_processing.glaucoma_dataset import CombinedGlaucomaDataset
from utilities.utils import identity_transform, json_to_csv, plot_confusion_matrix_with_ci

NUM_CLASSES = 2
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "../../../../datasets"
SRC_DIR = "../../../"


def load_urfound_backbone(model, checkpoint_path):
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


def evaluate(model, dataloader, criterion, device, threshold):
    model.eval()
    total_loss = 0.0
    y_true, y_pred, y_probs = [], [], []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Test inference")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            probs = F.softmax(outputs, dim=1)
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
        output_dict=True,
    )

    per_class_recall = {
        name: report[name]["recall"]
        for name in ["Healthy", "Glaucoma"]
    }

    per_class_precision = {
        name: report[name]["precision"]
        for name in ["Healthy", "Glaucoma"]
    }

    return (avg_loss, acc, bal_acc, macro_f1,
            per_class_auc, macro_auc, weighted_auc,
            sensitivity, specificity, report, per_class_precision, per_class_recall, y_true, y_probs)


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

    best_path = f"{SRC_DIR}/best_models/best_urfound_glaucoma_lora_model.pth"
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)

    lora_cfg   = checkpoint.get("lora",  {"r": 8, "alpha": 32, "dropout": 0.05})
    train_cfg  = checkpoint.get("train", {})
    batch_size = train_cfg.get("micro_batch", 8)

    print(f"\nCheckpoint: bal_acc={checkpoint.get('val_bal_acc', 'N/A'):.2f}%")
    print(f"LoRA r={lora_cfg['r']} alpha={lora_cfg['alpha']}")

    # urfound uses vit_base not vit_large
    model = models_vit.__dict__["vit_base_patch16"](
        num_classes=NUM_CLASSES,
        drop_path_rate=0.2,
        global_pool=True
    )

    model = load_urfound_backbone(model, best_path)

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
     sensitivity, specificity, report, per_class_precision, per_class_recall, y_true, y_probs) = evaluate(
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
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "test_loss":         float(test_loss),
        "checkpoint":        os.path.basename(best_path),
        "lora":              lora_cfg,
        "train":             train_cfg,
    }

    os.makedirs("results/urfound-glaucoma", exist_ok=True)
    results_path = "results/urfound-glaucoma/urfound_glaucoma_test_results.json"

    if os.path.exists(results_path):
        os.remove(results_path)
        time.sleep(1)

    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    json_to_csv(results_path, "results/urfound-glaucoma", "urfound_glaucoma_results")
    
    np.save("../probs_numpy/urfound_loca_glaucoma_true.npy", y_true)
    np.save("../probs_numpy/urfound_lora_glaucoma_probs.npy", y_probs)
    print(f"\nResults saved to: {results_path}")

    y_pred = np.argmax(y_probs, axis=1)
    class_names = ["Healthy", "Glaucoma"]

    plot_confusion_matrix_with_ci(
        y_true      = y_true,
        y_pred      = y_pred,
        class_names = class_names,
        title       = "UrFound-LoRA Glacuoma",
        save_path   = "../../../plots/confusion_matrices/glaucoma/lora/urfound_cf.png",
    )

if __name__ == "__main__":
    main()
