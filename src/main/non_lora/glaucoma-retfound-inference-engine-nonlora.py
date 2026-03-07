import sys
import os
import time
import math
import random
import sys
import os
import time
import math
import random
import sys
import os
import time
import math
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from models.RETFound_MAE import models_vit
from models.RETFound_MAE.util import pos_embed
from timm.models.layers import trunc_normal_
from torchvision import transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch import nn

# glaucoma dataset class
from data_processing.glaucoma_dataset import CombinedGlaucomaDataset
from utilities.utils import (
    identity_transform,
    train_one_epoch_retfound,
    validate_retfound_with_metrics,
    save_metric_plot,
    plot_all_benchmark
)
from peft import get_peft_model, LoraConfig
from torch.cuda.amp import GradScaler


from peft import get_peft_model, LoraConfig
from torch.cuda.amp import GradScaler


# -------------------------
# hyperparameters
# -------------------------

# binary: healthy vs glaucoma
NUM_CLASSES = 2

NUM_EPOCHS = 50
WARMUP_EPOCHS = 5
COOLDOWN_EPOCHS = 10

LR_MIN = 1e-6
LR_MAX = 5e-4

BETAS = (0.9, 0.99)
WEIGHT_DECAY = 5e-4

MICRO_BATCH_SIZE = 8
EFFECTIVE_BATCH_SIZE = 128
GRAD_ACCUM_STEPS = max(1, EFFECTIVE_BATCH_SIZE // MICRO_BATCH_SIZE)

NUM_WORKERS = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
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


def lr_at_epoch(epoch: int) -> float:
    # linear warmup phase
    if epoch < WARMUP_EPOCHS:
        t = (epoch + 1) / WARMUP_EPOCHS
        return LR_MIN + t * (LR_MAX - LR_MIN)
    # flat cooldown phase
    if epoch >= NUM_EPOCHS - COOLDOWN_EPOCHS:
        return LR_MIN
    # cosine decay middle
    mid_total = NUM_EPOCHS - WARMUP_EPOCHS - COOLDOWN_EPOCHS
    t = (epoch - WARMUP_EPOCHS) / max(1, mid_total)
    return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * t))


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
        # no weight decay on lora
        groups.append({"params": lora, "weight_decay": 0.0})
    return groups


def create_balanced_sampler(dataset):
    """oversample minority class (glaucoma)"""
    labels = np.array(dataset.get_labels(), dtype=np.int64)
    # per-class inverse frequency weights
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


def main():
    DATA_DIR = "../../../datasets"
    SRC_DIR = "../../"

    root_dirs = {
        "G1020":  f"{DATA_DIR}/G1020",
        "ORIGA":  f"{DATA_DIR}/ORIGA",
        "REFUGE": f"{DATA_DIR}/REFUGE",
    }

    # augmentation for training
    train_transformations = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # no augmentation for val/test
    eval_transformations = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # glaucoma dataset handles labels internally — no load_labels_from_csv needed
    train_dataset = CombinedGlaucomaDataset(
        root_directories=root_dirs,
        split="train",
        img_transform=train_transformations,
        label_transform=identity_transform
    )
    # refuge val = our test set (no refuge test labels exist)
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

    # sampler handles 3.3:1 healthy/glaucoma imbalance
    sampler = create_balanced_sampler(train_dataset)

    # shuffle=False because sampler controls ordering
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

    # load retfound backbone
    model = models_vit.__dict__["vit_large_patch16"](
        num_classes=NUM_CLASSES,
        drop_path_rate=0.2,
        global_pool=True
    )

    checkpoint_path = f"{SRC_DIR}/models/RETFound_MAE/weights/RETFound_cfp_weights.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_model = checkpoint["model"]
    state_dict = model.state_dict()

    # drop mismatched head weights — our head is 2-class not 1000-class
    for k in ["head.weight", "head.bias"]:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            del checkpoint_model[k]

    pos_embed.interpolate_pos_embed(model, checkpoint_model)
    model.load_state_dict(checkpoint_model, strict=False)
    # small init for new head
    trunc_normal_(model.head.weight, std=2e-5)

    # apply lora to attention layers
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

    # label smoothing reduces overconfidence
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = torch.optim.AdamW(
        make_param_groups(model, WEIGHT_DECAY),
        lr=LR_MAX,
        betas=BETAS
    )
    scaler = GradScaler()

    best_val_bal_acc = 0.0
    best_val_acc = 0.0
    best_state = None

    print(f"\n{'=' * 60}")
    print(f"STARTING TRAINING")
    print(f"{'=' * 60}")
    print(f"train samples : {len(train_dataset)}")
    print(f"val samples   : {len(val_dataset)}")
    print(f"{'=' * 60}\n")

    history_epochs = []
    history_acc = []
    history_bal_acc = []
    history_macro_f1 = []
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
            torch.cuda.reset_peak_memory_
