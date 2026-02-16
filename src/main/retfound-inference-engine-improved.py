import sys
import os
import math
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.RETFound_MAE import models_vit
from models.RETFound_MAE.util import pos_embed
from timm.models.layers import trunc_normal_
from torchvision import transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch import nn

from data_processing.dataset import CombinedDRDataSet
from utilities.utils import (
    identity_transform,
    class_balanced_weights,
    train_one_epoch_retfound,
    validate_retfound,
    validate_retfound_with_metrics,
    subsample_dataset,
    save_metric_plot
)
from peft import get_peft_model, LoraConfig
from torch.cuda.amp import GradScaler
from sklearn.metrics import balanced_accuracy_score, f1_score, classification_report
from collections import Counter
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # class weights
        self.gamma = gamma  # focusing parameter
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


# -------------------------
# Paper-aligned training hparams
# -------------------------
NUM_CLASSES = 5

NUM_EPOCHS = 50
WARMUP_EPOCHS = 5
COOLDOWN_EPOCHS = 10

LR_MIN = 1e-6
LR_MAX = 5e-4

BETAS = (0.9, 0.99)
WEIGHT_DECAY = 5e-4

# Batch + accumulation (effective batch = 128)
MICRO_BATCH_SIZE = 8   # set to what fits on your GPU (e.g., 4, 8, 16)
EFFECTIVE_BATCH_SIZE = 128
GRAD_ACCUM_STEPS = max(1, EFFECTIVE_BATCH_SIZE // MICRO_BATCH_SIZE)

NUM_WORKERS = 8  # INCREASED from 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LoRA (fixed baseline; tune later once stable)
LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# Optional stability knobs
MAX_GRAD_NORM = 1.0

print(f"Using device: {DEVICE}")
print(f"Micro batch: {MICRO_BATCH_SIZE} | Effective batch: {MICRO_BATCH_SIZE * GRAD_ACCUM_STEPS} "
      f"| Grad accum steps: {GRAD_ACCUM_STEPS}")


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_everything(42)


# RETFOUND LR schedule: warmup -> cosine -> cooldown according to paper
def lr_at_epoch(epoch: int) -> float:
    # warmup: linear LR_MIN -> LR_MAX over WARMUP_EPOCHS
    if epoch < WARMUP_EPOCHS:
        t = (epoch + 1) / WARMUP_EPOCHS
        return LR_MIN + t * (LR_MAX - LR_MIN)

    # cooldown: keep LR at LR_MIN for the last COOLDOWN_EPOCHS
    if epoch >= NUM_EPOCHS - COOLDOWN_EPOCHS:
        return LR_MIN

    # cosine decay in the middle
    mid_total = NUM_EPOCHS - WARMUP_EPOCHS - COOLDOWN_EPOCHS
    t = (epoch - WARMUP_EPOCHS) / max(1, mid_total)
    return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * t))


# AdamW param groups: no weight decay on bias/norm
def make_param_groups(model: torch.nn.Module, weight_decay: float):
    decay, no_decay, lora = [], [], []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        name_l = name.lower()
        if "lora_" in name_l:
            lora.append(p)
        elif name.endswith(".bias") or "norm" in name_l or "bn" in name_l:
            no_decay.append(p)
        else:
            decay.append(p)

    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    if lora:
        groups.append({"params": lora, "weight_decay": 0.0})  # key: no WD on LoRA

    return groups


def create_balanced_sampler(dataset, num_classes=NUM_CLASSES):
    """
    Create a WeightedRandomSampler that balances classes during training.
    Each sample gets weighted inversely to its class frequency.
    """
    labels = np.array(dataset.labels, dtype=np.int64)
    
    class_counts = np.bincount(labels, minlength=num_classes)
    
    class_weights = class_balanced_weights(class_counts, beta=0.9999, device=DEVICE) 
    sample_weights = class_weights[labels]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True  # Allow oversampling minority classes
    )
    
    print(f"Created balanced sampler with {len(sample_weights)} samples")
    print(f"Sample weights range: [{sample_weights.min():.4f}, {sample_weights.max():.4f}]")
    
    return sampler


def main():
    DATA_DIR = "../../datasets"
    SRC_DIR = "../"

    train_root_directories = {
        "DEEPDRID": f"{DATA_DIR}/DeepDRiD",
        "EYEPACS": f"{DATA_DIR}/EYEPACS",
        "DDR": f"{DATA_DIR}/DDR",
    }
    val_root_directories = dict(train_root_directories)

    train_transformations = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    validation_transformations = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = CombinedDRDataSet(
        root_directories=train_root_directories,
        split="train",
        img_transform=train_transformations,
        label_transform=identity_transform
    )
    validation_dataset = CombinedDRDataSet(
        root_directories=val_root_directories,
        split="val",
        img_transform=validation_transformations,
        label_transform=identity_transform
    )

    train_csv_paths = {
        "DEEPDRID": f"{train_root_directories['DEEPDRID']}/regular_fundus_images/regular-fundus-training/regular-fundus-training.csv",
        "EYEPACS": f"{train_root_directories['EYEPACS']}/all_labels.csv",
        "DDR": f"{train_root_directories['DDR']}/DR_grading.csv"
    }

    val_csv_paths = {
        "DEEPDRID": f"{train_root_directories['DEEPDRID']}/regular_fundus_images/regular-fundus-validation/regular-fundus-validation.csv",
        "EYEPACS": f"{train_root_directories['EYEPACS']}/all_labels.csv",
        "DDR": f"{train_root_directories['DDR']}/DR_grading.csv"
     }

    train_dataset.load_labels_from_csv(train_csv_paths)
    validation_dataset.load_labels_from_csv(val_csv_paths)
    
    train_dataset.prune_unlabeled()
    validation_dataset.prune_unlabeled()
    
    print("\n" + "="*60)
    print("SUBSAMPLING LARGE DATASETS")
    print("="*60)
    print(f"Before subsampling: {len(train_dataset)} samples")

    train_dataset = subsample_dataset(train_dataset, max_samples_per_class=3000)

    print(f"After subsampling: {len(train_dataset)} samples")
    print("="*60 + "\n")


    # Get class distribution
    labels = np.array(train_dataset.labels, dtype=np.int64)
    class_counts = np.bincount(labels, minlength=NUM_CLASSES)

    print(f"\n{'='*60}")
    print(f"CLASS DISTRIBUTION:")
    print(f"{'='*60}")
    for i, count in enumerate(class_counts):
        print(f"Class {i}: {count:5d} samples ({count/len(labels)*100:.1f}%)")
    print(f"{'='*60}\n")

    class_weights_np = len(labels) / (NUM_CLASSES * class_counts.astype(float))
    
    # note: need to cap weights to prevent extreme values that cause model collapse
    max_weight = 12.0
    class_weights_np = np.clip(class_weights_np, None, max_weight)
    class_weights_np = class_weights_np / class_weights_np.sum() * NUM_CLASSES  # Re-normalize

    class_weights_tensor = torch.FloatTensor(class_weights_np).to(DEVICE)

    print(f"CLASS WEIGHTS (Balanced, capped at {max_weight}x):")
    for i, weight in enumerate(class_weights_np):
        print(f"Class {i}: {weight:.4f}")
    print()

    classes = list(range(NUM_CLASSES))
    counts = class_counts.tolist()

    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.bar(classes, counts, color='steelblue')
    plt.xlabel("Class Label")
    plt.ylabel("Number of Samples")
    plt.title("Original Training Distribution")
    plt.xticks(classes)

    plt.subplot(1, 2, 2)
    plt.bar(classes, class_weights_np, color='coral')
    plt.xlabel("Class Label")
    plt.ylabel("Loss Weight")
    plt.title("Class Weights Applied")
    plt.xticks(classes)
    plt.tight_layout()
    plt.savefig("../train_class_distribution_and_weights.jpg")
    plt.close()

    print("\n" + "="*60)
    print("CREATING BALANCED SAMPLER FOR TRAINING")
    print("="*60)
    
    print("="*60 + "\n")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=MICRO_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4  # Prefetch 4 batches per worker
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=MICRO_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS // 2,  # Fewer workers for validation
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2  # Less prefetching for validation
    )

    # Model + RETFound weights
    model = models_vit.__dict__["vit_large_patch16"](
        num_classes=NUM_CLASSES,
        drop_path_rate=0.2,
        global_pool=True
    )

    checkpoint_path = f"{SRC_DIR}/models/RETFound_MAE/weights/RETFound_cfp_weights.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    checkpoint_model = checkpoint["model"]
    state_dict = model.state_dict()

    for k in ["head.weight", "head.bias"]:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            del checkpoint_model[k]

    pos_embed.interpolate_pos_embed(model, checkpoint_model)
    model.load_state_dict(checkpoint_model, strict=False)
    trunc_normal_(model.head.weight, std=2e-5)

    # LoRA
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["qkv", "proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        modules_to_save=["head"]
    )
    model = get_peft_model(model, peft_config)
    model = model.to(DEVICE)

    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)
    
    print(f"\n{'='*60}")
    print(f"Using Focal Loss with gamma=2.0 and class weights")
    print(f"{'='*60}\n")

    # Optimizer (paper)
    optimizer = torch.optim.AdamW(
        make_param_groups(model, WEIGHT_DECAY),
        lr=LR_MAX,
        betas=BETAS
    )

    scaler = GradScaler()  

    best_val_acc = 0.0
    best_val_bal_acc = 0.0  # need to track val balanced accuracy too
    best_state = None

    print(f"\n{'='*60}")
    print(f"STARTING TRAINING")
    print(f"{'='*60}")
    print(f"Total epochs: {NUM_EPOCHS}")
    print(f"Warmup epochs: {WARMUP_EPOCHS}")
    print(f"Cooldown epochs: {COOLDOWN_EPOCHS}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(validation_dataset)}")
    print(f"{'='*60}\n")

    history_epochs = []
    history_acc = []
    history_bal_acc = []
    history_macro_f1 = []
    history_macro_auc = []


    for epoch in range(NUM_EPOCHS):
        lr = lr_at_epoch(epoch)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        train_loss, train_acc = train_one_epoch_retfound(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=DEVICE,
            epoch=epoch,
            scaler=scaler,
            grad_accum_steps=GRAD_ACCUM_STEPS,
            max_grad_norm=MAX_GRAD_NORM,
        )

        if (epoch + 1) % 10 != 0:
            print(f"Epoch {epoch+1:03d}/{NUM_EPOCHS} | "
                  f"lr={lr:.2e} | "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% | "
                  f"[Skipping validation]")
            continue

        val_loss, val_acc, val_bal_acc, val_macro_f1, val_macro_auc, report = validate_retfound_with_metrics(
            model, validation_loader, criterion, DEVICE, NUM_CLASSES
        )
        
        history_epochs.append(epoch + 1)
        history_acc.append(val_acc)
        history_bal_acc.append(val_bal_acc)
        history_macro_f1.append(val_macro_f1)
        history_macro_auc.append(val_macro_auc)


        print(f"Epoch {epoch+1:03d}/{NUM_EPOCHS} | "
              f"lr={lr:.2e} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}%")

        print(f"val_acc={val_acc:.2f}% | bal_acc={val_bal_acc:.2f}% | macro_f1={val_macro_f1:.2f}% | macro_auc={val_macro_auc:.2f}%")
        print(report)
        
        if val_bal_acc > best_val_bal_acc:
            best_val_bal_acc = val_bal_acc
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"★ New best balanced accuracy: {best_val_bal_acc:.2f}% (acc: {best_val_acc:.2f}%)")
    
    plot_save_dir = "./plots/retfound"
    save_metric_plot(history_epochs, history_acc, "Validation Accuracy", plot_save_dir)
    save_metric_plot(history_epochs, history_bal_acc, "Balanced Accuracy", plot_save_dir)
    save_metric_plot(history_epochs, history_macro_f1, "Macro F1", plot_save_dir)
    save_metric_plot(history_epochs, history_macro_auc, "Macro AUROC", plot_save_dir)



    os.makedirs("../best_models", exist_ok=True)
    save_path = "../best_models/best_retfound_model.pth"
    torch.save({
        "val_acc": best_val_acc,
        "val_bal_acc": best_val_bal_acc,
        "model_state_dict": best_state,
        "lora": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT},
        "train": {
            "epochs": NUM_EPOCHS,
            "lr_max": LR_MAX,
            "lr_min": LR_MIN,
            "warmup_epochs": WARMUP_EPOCHS,
            "cooldown_epochs": COOLDOWN_EPOCHS,
            "weight_decay": WEIGHT_DECAY,
            "betas": BETAS,
            "micro_batch": MICRO_BATCH_SIZE,
            "grad_accum_steps": GRAD_ACCUM_STEPS,
            "effective_batch": MICRO_BATCH_SIZE * GRAD_ACCUM_STEPS
        }
    }, save_path)

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"Best Balanced Accuracy: {best_val_bal_acc:.2f}%")
    print(f"Saved best model to: {save_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
