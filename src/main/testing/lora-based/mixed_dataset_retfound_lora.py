import time
import sys
import os
import json
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from models.RETFound_MAE import models_vit
from models.RETFound_MAE.util import pos_embed
from timm.models.layers import trunc_normal_
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from peft import get_peft_model, LoraConfig

from data_processing.mixed_dataset import ODIRDataset, ODIR_CLASS_NAMES, NUM_CLASSES
from utilities.utils import json_to_csv

NUM_WORKERS = 4
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "../../../../datasets"
SRC_DIR  = "../../../"


def load_retfound_backbone(model, checkpoint_path):
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


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_logits, all_labels = [], []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Test inference")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss   = criterion(logits, labels)
            total_loss += loss.item()

            all_logits.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            pbar.set_postfix({"loss": f"{total_loss / (pbar.n or 1):.4f}"})

    probs  = np.concatenate(all_logits, axis=0)   # (N, 8)
    labels = np.concatenate(all_labels, axis=0)   # (N, 8)
    preds  = (probs >= 0.3).astype(int)

    avg_loss = total_loss / len(dataloader)

    # per-class AUC
    per_class_auc = {}
    for i, name in enumerate(ODIR_CLASS_NAMES):
        try:
            per_class_auc[name] = float(roc_auc_score(labels[:, i], probs[:, i]))
        except ValueError:
            per_class_auc[name] = None

    try:
        macro_auc    = float(roc_auc_score(labels, probs, average="macro"))
        weighted_auc = float(roc_auc_score(labels, probs, average="weighted"))
    except ValueError:
        macro_auc = weighted_auc = 0.0

    # per-class F1
    per_class_f1 = {}
    for i, name in enumerate(ODIR_CLASS_NAMES):
        per_class_f1[name] = float(f1_score(labels[:, i], preds[:, i], zero_division=0))

    macro_f1    = float(f1_score(labels, preds, average="macro",    zero_division=0))
    weighted_f1 = float(f1_score(labels, preds, average="weighted", zero_division=0))

    # exact match accuracy (all 8 labels correct)
    exact_match = float((preds == labels.astype(int)).all(axis=1).mean())

    report = classification_report(
        labels, preds,
        target_names=ODIR_CLASS_NAMES,
        digits=4,
        zero_division=0
    )

    return (avg_loss, exact_match, macro_auc, weighted_auc,
            per_class_auc, macro_f1, weighted_f1, per_class_f1,
            report, labels, probs)


def main():
    csv_path     = f"{DATA_DIR}/ODIR-5K/full_df.csv"
    img_dir_test = f"{DATA_DIR}/ODIR-5K/training"

    test_transformations = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_dataset = ODIRDataset(
        img_dir_test, csv_path, split="test",
        img_transform=test_transformations
    )
    print(f"Test samples: {len(test_dataset)}")

    # loading checkpoint
    best_path  = f"{SRC_DIR}/best_models/best_retfound_mixed_lora_model.pth"
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)

    lora_cfg   = checkpoint.get("lora",  {"r": 8, "alpha": 32, "dropout": 0.05})
    train_cfg  = checkpoint.get("train", {})
    batch_size = train_cfg.get("micro_batch", 8)

    print(f"\nCheckpoint: val_auc={checkpoint.get('val_auc', 'N/A'):.4f}")
    print(f"LoRA r={lora_cfg['r']}  alpha={lora_cfg['alpha']}")

    # loading the model
    pretrained_path = f"{SRC_DIR}/models/RETFound_MAE/weights/RETFound_cfp_weights.pth"
    model = models_vit.__dict__["vit_large_patch16"](
        num_classes=NUM_CLASSES, drop_path_rate=0.2, global_pool=True
    )
    model = load_retfound_backbone(model, pretrained_path)

    peft_config = LoraConfig(
        r=lora_cfg["r"], lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=["qkv", "proj"], bias="none",
        modules_to_save=["head"],
    )
    model = get_peft_model(model, peft_config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(DEVICE)
    model.eval()

    criterion = nn.BCEWithLogitsLoss()

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
    )

    # eval
    (test_loss, exact_match, macro_auc, weighted_auc,
     per_class_auc, macro_f1, weighted_f1, per_class_f1,
     report, y_true, y_probs) = evaluate(model, test_loader, criterion, DEVICE)

    print("\nFINAL TEST RESULTS")
    print("=" * 55)
    print(f"exact match accuracy : {exact_match*100:.2f}%")
    print(f"macro AUC            : {macro_auc:.4f}")
    print(f"weighted AUC         : {weighted_auc:.4f}")
    print(f"macro F1             : {macro_f1:.4f}")
    print(f"weighted F1          : {weighted_f1:.4f}")
    print(f"loss                 : {test_loss:.4f}")
    print("\nPer-class AUC:")
    for name, auc in per_class_auc.items():
        print(f"  {name:<14}: {auc:.4f}" if auc is not None else f"  {name:<14}: N/A")
    print("\nPer-class F1:")
    for name, f1 in per_class_f1.items():
        print(f"  {name:<14}: {f1:.4f}")
    print("=" * 55)
    print(report)

    results = {
        "exact_match_accuracy": float(exact_match),
        "macro_auc":            float(macro_auc),
        "weighted_auc":         float(weighted_auc),
        "macro_f1":             float(macro_f1),
        "weighted_f1":          float(weighted_f1),
        "per_class_auc":        {k: float(v) if v is not None else None
                                 for k, v in per_class_auc.items()},
        "per_class_f1":         per_class_f1,
        "test_loss":            float(test_loss),
        "checkpoint":           os.path.basename(best_path),
        "lora":                 lora_cfg,
        "train":                train_cfg,
    }

    os.makedirs("results/retfound-mixed-disease", exist_ok=True)
    results_path = "results/retfound-mixed-disease/retfound_mixed_30percentCI-test_results.json"

    if os.path.exists(results_path):
        os.remove(results_path)
        time.sleep(1)

    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    json_to_csv(results_path, "results/retfound-mixed-disease", "retfound_mixed_disease_results")

    os.makedirs("../probs_numpy", exist_ok=True)
    # np.save("../probs_numpy/retfound_mixed_disease_true.npy",  y_true)
    # np.save("../probs_numpy/retfound_mixed_disease_probs.npy", y_probs)

    print(f"\nResults saved to: {results_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
