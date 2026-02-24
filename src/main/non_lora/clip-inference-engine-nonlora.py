import sys
import os
import torch
import optuna
import numpy as np
import math
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from timm.models.layers import trunc_normal_
import torchvision
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import DataLoader
from data_processing.dataset import CombinedDRDataSet
from utilities.utils import identity_transform, class_balanced_weights, get_specific_layer_names, train_one_epoch_clip, validate_clip_with_metrics, subsample_dataset, save_metric_plot
from torch import nn
from torch import optim
from torch.amp import GradScaler
from transformers import CLIPVisionModelWithProjection, CLIPProcessor


class CLIPRetina(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.vision = CLIPVisionModelWithProjection.from_pretrained(model_name)
        embedding_dim = self.vision.config.projection_dim
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, images):
        outputs = self.vision(pixel_values=images)
        image_embeds = outputs.image_embeds
        logits = self.classifier(image_embeds)
        return logits


NUM_CLASSES = 5

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

MAX_GRAD_NORM = 1.0


print(f"Using device: {DEVICE}")
print(f"Micro batch: {MICRO_BATCH_SIZE} | Effective batch: {MICRO_BATCH_SIZE * GRAD_ACCUM_STEPS} "
      f"| Grad accum steps: {GRAD_ACCUM_STEPS}")


def lr_at_epoch(epoch: int) -> float:
    if epoch < WARMUP_EPOCHS:
        t = (epoch + 1) / WARMUP_EPOCHS
        return LR_MIN + t * (LR_MAX - LR_MIN)

    if epoch >= NUM_EPOCHS - COOLDOWN_EPOCHS:
        return LR_MIN

    mid_total = NUM_EPOCHS - WARMUP_EPOCHS - COOLDOWN_EPOCHS
    t = (epoch - WARMUP_EPOCHS) / max(1, mid_total)
    return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * t))


def make_param_groups(model: torch.nn.Module, weight_decay: float):
    decay, no_decay = [], []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        name_l = name.lower()
        if name.endswith(".bias") or "norm" in name_l or "bn" in name_l:
            no_decay.append(p)
        else:
            decay.append(p)

    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})

    return groups


def main():
    DATA_DIR = "../../../datasets"
    SRC_DIR = "../../"

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

    labels = np.array(train_dataset.labels, dtype=np.int64)
    class_counts = np.bincount(labels, minlength=NUM_CLASSES)

    print(f"\n{'='*60}")
    print(f"CLASS DISTRIBUTION:")
    print(f"{'='*60}")
    for i, count in enumerate(class_counts):
        print(f"Class {i}: {count:5d} samples ({count/len(labels)*100:.1f}%)")
    print(f"{'='*60}\n")

    class_weights_np = len(labels) / (NUM_CLASSES * class_counts.astype(float))
    max_weight = 12.0
    class_weights_np = np.clip(class_weights_np, None, max_weight)
    class_weights_np = class_weights_np / class_weights_np.sum() * NUM_CLASSES
    class_weights_tensor = torch.FloatTensor(class_weights_np).to(DEVICE)

    print(f"CLASS WEIGHTS (Balanced, capped at {max_weight}x):")
    for i, weight in enumerate(class_weights_np):
        print(f"Class {i}: {weight:.4f}")
    print()

    classes = list(range(NUM_CLASSES))
    counts = class_counts.tolist()

    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.bar(classes, counts)
    plt.xlabel("Class Label")
    plt.ylabel("Number of Samples")
    plt.title("Original Training Distribution")
    plt.xticks(classes)

    plt.subplot(1, 2, 2)
    plt.bar(classes, class_weights_np)
    plt.xlabel("Class Label")
    plt.ylabel("Loss Weight")
    plt.title("Class Weights Applied")
    plt.xticks(classes)
    plt.tight_layout()
    plt.savefig("../train_class_distribution_and_weights.jpg")
    plt.close()

    train_loader = DataLoader(
        train_dataset,
        batch_size=MICRO_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=MICRO_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS // 2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    print("\n" + "="*50)
    print("Initializaing CLIP Model")
    print("="*50)

    model = CLIPRetina("openai/clip-vit-large-patch14", num_classes=NUM_CLASSES)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    print(model)
    print(processor)
    print("CLIP model loaded successfully")

    model = model.to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = torch.optim.AdamW(
        make_param_groups(model, WEIGHT_DECAY),
        lr=LR_MAX,
        betas=BETAS
    )

    scaler = GradScaler()

    best_val_acc = 0.0
    best_val_bal_acc = 0.0
    best_state = None

    history_epochs = []
    history_acc = []
    history_bal_acc = []
    history_macro_f1 = []
    history_macro_auc = []

    for epoch in range(NUM_EPOCHS):
        lr = lr_at_epoch(epoch)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        train_loss, train_acc = train_one_epoch_clip(
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

        val_loss, val_acc, val_bal_acc, val_macro_f1, val_macro_auc, report = validate_clip_with_metrics(
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

    plot_save_dir = "./plots/clip/nonlora"
    save_metric_plot(history_epochs, history_acc, "Validation Accuracy", plot_save_dir)
    save_metric_plot(history_epochs, history_bal_acc, "Balanced Accuracy", plot_save_dir)
    save_metric_plot(history_epochs, history_macro_f1, "Macro F1", plot_save_dir)
    save_metric_plot(history_epochs, history_macro_auc, "Macro AUROC", plot_save_dir)

    os.makedirs("../../best_models", exist_ok=True)
    save_path = "../../best_models/best_clip_nonlora_model.pth"

    torch.save({
        "val_acc": best_val_acc,
        "val_bal_acc": best_val_bal_acc,
        "model_state_dict": best_state,
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

    print(f"\nBest Val Accuracy: {best_val_acc:.2f}%")
    print(f"Best Balanced Accuracy: {best_val_bal_acc:.2f}%")
    print(f"Saved best model to: {save_path}")


if __name__ == "__main__":
    main()
