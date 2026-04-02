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
from torch.utils.data import DataLoader
from torch import nn
from torch.cuda.amp import GradScaler
from transformers import ResNetForImageClassification
from sklearn.metrics import roc_auc_score, f1_score

from data_processing.mixed_dataset import ODIRDataset, ODIR_CLASS_NAMES, NUM_CLASSES
from utilities.utils import identity_transform, save_metric_plot, plot_all_benchmark
from tqdm import tqdm

# -------------------------
# hyperparameters
# -------------------------
NUM_EPOCHS       = 50
WARMUP_EPOCHS    = 5
COOLDOWN_EPOCHS  = 10
LR_MIN           = 1e-6
LR_MAX           = 1e-4
BETAS            = (0.9, 0.99)
WEIGHT_DECAY     = 5e-4
MICRO_BATCH_SIZE     = 8
EFFECTIVE_BATCH_SIZE = 128
GRAD_ACCUM_STEPS     = max(1, EFFECTIVE_BATCH_SIZE // MICRO_BATCH_SIZE)
NUM_WORKERS      = 4
MAX_GRAD_NORM    = 1.0

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "../../../datasets"
SRC_DIR  = "../../"

print(f"Using device: {DEVICE}")
print(f"Micro batch: {MICRO_BATCH_SIZE} | Effective batch: {MICRO_BATCH_SIZE * GRAD_ACCUM_STEPS}")


def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_everything()


def lr_at_epoch(epoch):
    if epoch < WARMUP_EPOCHS:
        return LR_MIN + (epoch + 1) / WARMUP_EPOCHS * (LR_MAX - LR_MIN)
    if epoch >= NUM_EPOCHS - COOLDOWN_EPOCHS:
        return LR_MIN
    t = (epoch - WARMUP_EPOCHS) / max(1, NUM_EPOCHS - WARMUP_EPOCHS - COOLDOWN_EPOCHS)
    return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * t))


def make_param_groups(model, weight_decay):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        if name.endswith(".bias") or "bn" in name.lower() or "norm" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay,    "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()
    pbar = tqdm(dataloader, desc="Training")
    
    for step, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)

        with torch.amp.autocast(device_type=device.type):
            # HuggingFace ResNet returns ImageClassifierOutputWithNoAttention
            logits = model(pixel_values=images).logits
            loss   = criterion(logits, labels) / GRAD_ACCUM_STEPS

        scaler.scale(loss).backward()

        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item() * GRAD_ACCUM_STEPS
        pbar.set_postfix({"loss": running_loss / (step + 1)})

    return running_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_logits, all_labels = [], []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(pixel_values=images).logits
            loss   = criterion(logits, labels)
            running_loss += loss.item()
            all_logits.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            pbar.set_postfix({"loss": running_loss / (pbar.n or 1)})    

    probs  = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    try:
        macro_auc = roc_auc_score(labels, probs, average="macro")
    except ValueError:
        macro_auc = 0.0

    preds    = (probs >= 0.5).astype(int)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)

    return running_loss / len(dataloader), float(macro_auc), float(macro_f1)


def main():
    csv_path      = f"{DATA_DIR}/ODIR-5K/full_df.csv"
    img_dir_train = f"{DATA_DIR}/ODIR-5K/training"

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

    train_dataset = ODIRDataset(img_dir_train, csv_path, split="train",
                                img_transform=train_transformations)
    val_dataset   = ODIRDataset(img_dir_train, csv_path, split="val",
                                img_transform=eval_transformations)

    print("\n" + "="*60)
    print(train_dataset.get_dataset_statistics())
    print(val_dataset.get_dataset_statistics())
    print("="*60 + "\n")

    train_loader = DataLoader(train_dataset, batch_size=MICRO_BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS,
                              pin_memory=True, persistent_workers=True,
                              prefetch_factor=4)
    val_loader   = DataLoader(val_dataset, batch_size=MICRO_BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS // 2,
                              pin_memory=True, persistent_workers=True,
                              prefetch_factor=2)

    # num_labels=8 — HuggingFace replaces the 1000-class head automatically
    model = ResNetForImageClassification.from_pretrained(
        "microsoft/resnet-50",
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True
    )
    model = model.to(DEVICE)

    # BCEWithLogitsLoss for multi-label — NOT CrossEntropyLoss
    label_matrix = np.stack(train_dataset.labels)     # (N, 8)
    pos_counts   = label_matrix.sum(axis=0)           # positives per class
    neg_counts   = len(train_dataset) - pos_counts    # negatives per class
    pos_weight   = torch.tensor(neg_counts / np.clip(pos_counts, 1, None),
                             dtype=torch.float32).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(make_param_groups(model, WEIGHT_DECAY),
                                  lr=LR_MAX, betas=BETAS)
    scaler = GradScaler()

    best_auc    = 0.0
    best_state  = None
    history_epochs, history_auc, history_f1 = [], [], []

    benchmark = {"epoch_times_s": [], "peak_gpu_mb": [], "train_throughput": []}

    print(f"\n{'='*60}")
    print(f"STARTING TRAINING — ResNet50 ODIR Multi-Label")
    print(f"{'='*60}")
    print(f"train samples: {len(train_dataset)} | val samples: {len(val_dataset)}")
    print(f"{'='*60}\n")

    for epoch in range(NUM_EPOCHS):
        lr = lr_at_epoch(epoch)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(DEVICE)
        t0 = time.perf_counter()

        train_loss = train_one_epoch(model, train_loader, criterion,
                                     optimizer, DEVICE, scaler)

        epoch_time = time.perf_counter() - t0
        benchmark["epoch_times_s"].append(epoch_time)
        benchmark["train_throughput"].append(len(train_dataset) / epoch_time)
        benchmark["peak_gpu_mb"].append(
            torch.cuda.max_memory_allocated(DEVICE) / 1024**2
            if torch.cuda.is_available() else float("nan")
        )

        print(f"Epoch {epoch+1:03d}/{NUM_EPOCHS} | lr={lr:.2e} | "
              f"train_loss={train_loss:.4f} | time={epoch_time:.1f}s")

        if (epoch + 1) % 10 != 0:
            continue

        val_loss, val_auc, val_f1 = validate(model, val_loader, criterion, DEVICE)
        history_epochs.append(epoch + 1)
        history_auc.append(val_auc)
        history_f1.append(val_f1)

        print(f"  val_loss={val_loss:.4f} | macro_auc={val_auc:.4f} | macro_f1={val_f1:.4f}")

        if val_auc > best_auc:
            best_auc   = val_auc
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            print(f"  ★ New best macro AUC: {best_auc:.4f}")

    # plots
    plot_save_dir = "./plots/resnet50-mixed-disease"
    save_metric_plot(history_epochs, history_auc, "Macro AUC", plot_save_dir)
    save_metric_plot(history_epochs, history_f1,  "Macro F1",  plot_save_dir)

    # benchmark summary
    c_times = np.array(benchmark["epoch_times_s"][1:])
    c_gpu   = np.array([v for v in benchmark["peak_gpu_mb"][1:] if not math.isnan(v)])
    benchmark["summary"] = {
        "avg_epoch_time_s":   float(np.mean(c_times)),
        "total_train_time_s": float(sum(benchmark["epoch_times_s"])),
        "avg_peak_gpu_mb":    float(np.mean(c_gpu)) if len(c_gpu) else None,
    }

    os.makedirs(f"{SRC_DIR}/best_models", exist_ok=True)
    with open(f"{SRC_DIR}/best_models/resnet50-mixed-disease-benchmark.json", "w") as f:
        json.dump(benchmark, f, indent=4)

    save_path = f"{SRC_DIR}/best_models/best_resnet50_mixed_disease_model.pth"
    torch.save({
        "val_auc":          best_auc,
        "model_state_dict": best_state,
        "num_classes":      NUM_CLASSES,
        "train": {
            "epochs": NUM_EPOCHS, "lr_max": LR_MAX, "lr_min": LR_MIN,
            "warmup_epochs": WARMUP_EPOCHS, "cooldown_epochs": COOLDOWN_EPOCHS,
            "weight_decay": WEIGHT_DECAY, "betas": BETAS,
            "micro_batch": MICRO_BATCH_SIZE, "grad_accum_steps": GRAD_ACCUM_STEPS,
            "effective_batch": MICRO_BATCH_SIZE * GRAD_ACCUM_STEPS,
        }
    }, save_path)

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE — best macro AUC: {best_auc:.4f}")
    print(f"Saved to: {save_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
