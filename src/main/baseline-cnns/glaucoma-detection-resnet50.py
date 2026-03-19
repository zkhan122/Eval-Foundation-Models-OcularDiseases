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
from sklearn.metrics import (
    roc_auc_score, balanced_accuracy_score, f1_score,
    classification_report, confusion_matrix
)
from peft import get_peft_model, LoraConfig

from data_processing.glaucoma_dataset import CombinedGlaucomaDataset
from utilities.utils import identity_transform, json_to_csv

NUM_CLASSES = 2
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "../../../../datasets"
SRC_DIR = "../../../"


def load_retfound_backbone(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_model = checkpoint["model_state_dict"]
    state_dict = model.state_dict()

    # dropping head if shape mismatches beacuse 2 classes
    for k in ["head.weight", "head.bias"]:
        if k in checkpoint_model and k in state_dict:
            if checkpoint_model[k].shape != state_dict[k].shape:
                del checkpoint_model[k]

    pos_embed.interpolate_pos_embed(model, checkpoint_model)
    model.load_state_dict(checkpoint_model, strict=False)
    trunc_normal_(model.head.weight, std=2e-5)
    return model

import sys
import os
import time
import math
import random
import numpy as np
import torch
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from torchvision import transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch import nn
from torch.cuda.amp import GradScaler
from transformers import ResNetForImageClassification

from data_processing.glaucoma_dataset import CombinedGlaucomaDataset
from utilities.utils import (
    identity_transform,
    save_metric_plot,
    plot_all_benchmark
)

NUM_CLASSES = 2

NUM_EPOCHS = 50
WARMUP_EPOCHS = 5
COOLDOWN_EPOCHS = 10

LR_MIN = 1e-6
LR_MAX = 1e-4

BETAS = (0.9, 0.99)
WEIGHT_DECAY = 5e-4

MICRO_BATCH_SIZE = 8
EFFECTIVE_BATCH_SIZE = 128
GRAD_ACCUM_STEPS = max(1, EFFECTIVE_BATCH_SIZE // MICRO_BATCH_SIZE)

NUM_WORKERS = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_GRAD_NORM = 1.0

print(f"Using device: {DEVICE}")
print(f"Micro batch: {MICRO_BATCH_SIZE} | Effective batch: {MICRO_BATCH_SIZE * GRAD_ACCUM_STEPS}")


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_everything(42)


def lr_at_epoch(epoch: int) -> float:
    if epoch < WARMUP_EPOCHS:
        t = (epoch + 1) / WARMUP_EPOCHS
        return LR_MIN + t * (LR_MAX - LR_MIN)
    if epoch >= NUM_EPOCHS - COOLDOWN_EPOCHS:
        return LR_MIN
    mid_total = NUM_EPOCHS - WARMUP_EPOCHS - COOLDOWN_EPOCHS
    t = (epoch - WARMUP_EPOCHS) / max(1, mid_total)
    return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * t))


def make_param_groups(model, weight_decay):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith(".bias") or "bn" in name.lower() or "norm" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay,    "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def create_balanced_sampler(dataset):
    labels = np.array(dataset.get_labels(), dtype=np.int64)
    class_counts = np.bincount(labels, minlength=NUM_CLASSES).astype(float)
    class_weights = 1.0 / class_counts
    sample_weights = torch.tensor(class_weights[labels], dtype=torch.float)

    print(f"class counts: {class_counts.astype(int).tolist()}")
    print(f"class weights: {[f'{w:.4f}' for w in class_weights]}")

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )


def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    optimizer.zero_grad()

    for step, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        with torch.amp.autocast(device_type=device.type):
            # HuggingFace ResNet returns a ImageClassifierOutputWithNoAttention
            # logits are at outputs.logits
            outputs = model(pixel_values=images)
            loss = criterion(outputs.logits, labels) / GRAD_ACCUM_STEPS

        scaler.scale(loss).backward()

        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item() * GRAD_ACCUM_STEPS
        preds = outputs.logits.argmax(dim=1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / len(dataloader), 100.0 * correct / total


def validate(model, dataloader, criterion, device):
    import torch.nn.functional as F
    from sklearn.metrics import (
        balanced_accuracy_score, f1_score, roc_auc_score, classification_report
    )

    model.eval()
    running_loss = 0.0
    y_true, y_pred, y_probs = [], [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(pixel_values=images)
            loss = criterion(outputs.logits, labels)
            running_loss += loss.item()

            probs = F.softmax(outputs.logits, dim=1)
            preds = outputs.logits.argmax(dim=1)

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            y_probs.extend(probs.cpu().numpy())

    y_true  = np.array(y_true)
    y_pred  = np.array(y_pred)
    y_probs = np.array(y_probs)

    val_loss = running_loss / len(dataloader)
    val_acc  = 100.0 * (y_true == y_pred).mean()
    bal_acc  = 100.0 * balanced_accuracy_score(y_true, y_pred)
    macro_f1 = 100.0 * f1_score(y_true, y_pred, average="macro")

    try:
        macro_auc = 100.0 * roc_auc_score(y_true, y_probs[:, 1])
    except ValueError:
        macro_auc = 0.0

    report = classification_report(
        y_true, y_pred,
        target_names=["Healthy", "Glaucoma"],
        digits=4,
        zero_division=0
    )

    return val_loss, val_acc, bal_acc, macro_f1, macro_auc, report


def main():
    DATA_DIR = "../../../datasets"
    SRC_DIR = "../../"

    root_dirs = {
        "G1020":  f"{DATA_DIR}/G1020",
        "ORIGA":  f"{DATA_DIR}/ORIGA",
        "REFUGE": f"{DATA_DIR}/REFUGE",
    }

    train_transformations = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    eval_transformations = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = CombinedGlaucomaDataset(
        root_directories=root_dirs,
        split="train",
        img_transform=train_transformations,
        label_transform=identity_transform
    )
    val_dataset = CombinedGlaucomaDataset(
        root_directories=root_dirs,
        split="val",
        img_transform=eval_transformations,
        label_transform=identity_transform
    )

    print("\n" + "=" * 60)
    print(train_dataset.get_dataset_statistics())
    print(val_dataset.get_dataset_statistics())
    print("=" * 60 + "\n")

    sampler = create_balanced_sampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=MICRO_BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=MICRO_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS // 2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    # num_labels=2 tells HuggingFace to replace the 1000-class ImageNet head
    # with a 2-class head automatically — no manual fc replacement needed
    model = ResNetForImageClassification.from_pretrained(
        "microsoft/resnet-50",
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True  # required because we're changing head size
    )
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(make_param_groups(model, WEIGHT_DECAY), lr=LR_MAX, betas=BETAS)
    scaler = GradScaler()

    best_val_bal_acc = 0.0
    best_val_acc = 0.0
    best_state = None

    print(f"\n{'=' * 60}")
    print(f"STARTING TRAINING — ResNet50 Glaucoma")
    print(f"{'=' * 60}")
    print(f"train samples : {len(train_dataset)}")
    print(f"val samples   : {len(val_dataset)}")
    print(f"{'=' * 60}\n")

    history_epochs    = []
    history_acc       = []
    history_bal_acc   = []
    history_macro_f1  = []
    history_macro_auc = []

    benchmark = {
        "epoch_times_s":    [],
        "peak_gpu_mb":      [],
        "train_throughput": [],
    }

    for epoch in range(NUM_EPOCHS):
        lr = lr_at_epoch(epoch)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(DEVICE)
        t0 = time.perf_counter()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE, scaler
        )

        epoch_time = time.perf_counter() - t0
        benchmark["epoch_times_s"].append(epoch_time)
        benchmark["train_throughput"].append(len(train_dataset) / epoch_time)
        benchmark["peak_gpu_mb"].append(
            torch.cuda.max_memory_allocated(DEVICE) / 1024 ** 2
            if torch.cuda.is_available() else float("nan")
        )

        peak_gpu   = benchmark["peak_gpu_mb"][-1]
        throughput = benchmark["train_throughput"][-1]
        print(f"Epoch {epoch + 1:03d}/{NUM_EPOCHS} | lr={lr:.2e} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% | "
              f"time={epoch_time:.1f}s | gpu={peak_gpu:.0f}MiB | "
              f"throughput={throughput:.1f}sps")

        if (epoch + 1) % 10 != 0:
            continue

        val_loss, val_acc, val_bal_acc, val_macro_f1, val_macro_auc, report = validate(
            model, val_loader, criterion, DEVICE
        )

        history_epochs.append(epoch + 1)
        history_acc.append(val_acc)
        history_bal_acc.append(val_bal_acc)
        history_macro_f1.append(val_macro_f1)
        history_macro_auc.append(val_macro_auc)

        print(f"val_acc={val_acc:.2f}% | bal_acc={val_bal_acc:.2f}% | "
              f"macro_f1={val_macro_f1:.2f}% | macro_auc={val_macro_auc:.2f}%")
        print(report)

        if val_bal_acc > best_val_bal_acc:
            best_val_bal_acc = val_bal_acc
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"new best balanced accuracy: {best_val_bal_acc:.2f}%")

    plot_save_dir = "./plots/resnet50-glaucoma"
    save_metric_plot(history_epochs, history_acc,       "Validation Accuracy", plot_save_dir)
    save_metric_plot(history_epochs, history_bal_acc,   "Balanced Accuracy",   plot_save_dir)
    save_metric_plot(history_epochs, history_macro_f1,  "Macro F1",            plot_save_dir)
    save_metric_plot(history_epochs, history_macro_auc, "Macro AUROC",         plot_save_dir)

    c_times = np.array(benchmark["epoch_times_s"][1:])
    c_gpu   = np.array([v for v in benchmark["peak_gpu_mb"][1:] if not math.isnan(v)])
    c_thru  = np.array(benchmark["train_throughput"][1:])

    benchmark["summary"] = {
        "epochs_measured":    len(c_times),
        "avg_epoch_time_s":   float(np.mean(c_times)),
        "std_epoch_time_s":   float(np.std(c_times)),
        "min_epoch_time_s":   float(np.min(c_times)),
        "max_epoch_time_s":   float(np.max(c_times)),
        "total_train_time_s": float(sum(benchmark["epoch_times_s"])),
        "avg_peak_gpu_mb":    float(np.mean(c_gpu)) if len(c_gpu) else None,
        "max_peak_gpu_mb":    float(np.max(c_gpu))  if len(c_gpu) else None,
        "avg_throughput_sps": float(np.mean(c_thru)),
    }

    s = benchmark["summary"]
    print(f"\n{'=' * 60}")
    print(f"BENCHMARK SUMMARY (epochs 2-{NUM_EPOCHS})")
    print(f"{'=' * 60}")
    print(f"  avg epoch time  : {s['avg_epoch_time_s']:.1f}s +/- {s['std_epoch_time_s']:.1f}s")
    print(f"  total train time: {s['total_train_time_s'] / 60:.1f} min")
    print(f"  avg peak gpu mem: {s['avg_peak_gpu_mb']:.0f} MiB")
    print(f"  max peak gpu mem: {s['max_peak_gpu_mb']:.0f} MiB")
    print(f"  avg throughput  : {s['avg_throughput_sps']:.1f} samples/s")
    print(f"{'=' * 60}\n")

    os.makedirs(f"{SRC_DIR}/best_models", exist_ok=True)
    with open(f"{SRC_DIR}/best_models/resnet50-glaucoma-benchmark.json", "w") as f:
        json.dump(benchmark, f, indent=4)

    plot_all_benchmark(
        source=benchmark,
        output_dir=f"{SRC_DIR}/plots/baseline-plots/resnet50-glaucoma-benchmark-plots",
        skip=1,
        model_name="ResNet50 Glaucoma",
    )

    save_path = f"{SRC_DIR}/best_models/best_resnet50_glaucoma_model.pth"
    torch.save({
        "val_acc":          best_val_acc,
        "val_bal_acc":      best_val_bal_acc,
        "model_state_dict": best_state,
        "num_classes":      NUM_CLASSES,
        "train": {
            "epochs":           NUM_EPOCHS,
            "lr_max":           LR_MAX,
            "lr_min":           LR_MIN,
            "warmup_epochs":    WARMUP_EPOCHS,
            "cooldown_epochs":  COOLDOWN_EPOCHS,
            "weight_decay":     WEIGHT_DECAY,
            "betas":            BETAS,
            "micro_batch":      MICRO_BATCH_SIZE,
            "grad_accum_steps": GRAD_ACCUM_STEPS,
            "effective_batch":  MICRO_BATCH_SIZE * GRAD_ACCUM_STEPS,
        }
    }, save_path)

    print(f"\n{'=' * 60}")
    print(f"TRAINING COMPLETE")
    print(f"{'=' * 60}")
    print(f"best val accuracy      : {best_val_acc:.2f}%")
    print(f"best balanced accuracy : {best_val_bal_acc:.2f}%")
    print(f"saved model to         : {save_path}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
