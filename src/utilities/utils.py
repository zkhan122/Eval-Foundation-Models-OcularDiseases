
from pathlib import Path
import json
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import torch
import torch.nn.functional as F
from transformers import Conv1D
from tqdm import tqdm
import numpy as np
from torch.amp import autocast
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score, balanced_accuracy_score, classification_report, roc_auc_score, roc_curve
import seaborn as sn
import pandas as pd
from .plots import generate_confusion_matrix
from PIL import Image, ImageFile

# need to allow truncated images such as those in EYEPACS
ImageFile.LOAD_TRUNCATED_IMAGES = True




def save_metric_plot(epochs, values, metric_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    plt.figure()
    plt.plot(epochs, values, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} vs Epoch")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f"{metric_name.lower().replace(' ', '_')}.png"))
    plt.close()





def _is_image_valid(path: str) -> bool:
    try:
        # quick empty-file guard
        if os.path.getsize(path) < 1024:
            return False

        with Image.open(path) as img:
            img.convert("RGB")   # force decode
        return True

    except Exception:
        return False


def subsample_dataset(dataset, max_samples_per_class):
    """Keep only max_samples_per_class from each class"""
    from collections import defaultdict

    class_indices = defaultdict(list)
    for idx, label in enumerate(dataset.labels):
        class_indices[label].append(idx)

    # Sample from each class
    selected_indices = []
    for class_label, indices in class_indices.items():
        if len(indices) <= max_samples_per_class:
            selected_indices.extend(indices)
        else:
            # Randomly sample
            import random
            selected_indices.extend(random.sample(indices, max_samples_per_class))

    # Update dataset
    dataset.image_paths = [dataset.image_paths[i] for i in selected_indices]
    dataset.labels = [dataset.labels[i] for i in selected_indices]
    dataset.sources = [dataset.sources[i] for i in selected_indices]

    print(f"Subsampled dataset: {len(selected_indices)} samples")
    return dataset

def identity_transform(x):
    return x


def normalize_stem(x) -> str:
    return Path(str(x).strip()).stem.lower()


def normalize_deepdrid(x) -> str:
    """
    Converts:
      343_l2.jpg -> 343
      343_r1.png -> 343
      343         -> 343
    """
    s = Path(str(x).strip()).stem.lower()
    return s.split("_")[0]



def class_balanced_weights(class_counts, beta, device):
    class_counts = np.asarray(class_counts, dtype=np.float32)

    effective_num = 1.0 - np.power(beta, class_counts)
    weights = (1.0-beta) / np.maximum(effective_num, 1e-8)
    weights = weights / weights.mean()
    weights = torch.FloatTensor(weights)
    if device is not None: 
        weights = weights.to(device)
    return weights


def wilson_ci(k: int, n: int, z: float = 1.96):
    """
    Wilson score interval for k successes in n trials.
    Returns (lower, upper) clipped to [0, 1].
    """
    if n == 0:
        return 0.0, 0.0
    p_hat = k / n
    centre = (p_hat + z**2 / (2 * n)) / (1 + z**2 / n)
    margin = (z / (1 + z**2 / n)) * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))
    return float(np.clip(centre - margin, 0, 1)), float(np.clip(centre + margin, 0, 1))
 
def plot_confusion_matrix_with_ci(
    y_true,
    y_pred,
    class_names,
    title: str = "Confusion Matrix",
    save_path: str = None,
    figsize: tuple = (8, 7),
    cmap_name: str = "viridis",
    show: bool = True,
):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_classes = len(class_names)

    cm_counts = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

    row_sums = cm_counts.sum(axis=1, keepdims=True)
    row_sums_safe = np.where(row_sums == 0, 1, row_sums)
    cm_norm = cm_counts / row_sums_safe

    ci_low  = np.zeros((n_classes, n_classes))
    ci_high = np.zeros((n_classes, n_classes))

    for i in range(n_classes):
        n_row = int(row_sums[i])
        for j in range(n_classes):
            lo, hi = wilson_ci(int(cm_counts[i, j]), n_row)
            ci_low[i, j]  = lo
            ci_high[i, j] = hi

    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.get_cmap(cmap_name)
    im = ax.imshow(cm_norm, vmin=0.0, vmax=1.0, cmap=cmap, aspect="auto")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalised Scale", fontsize=10)

    for i in range(n_classes):
        for j in range(n_classes):
            val  = cm_norm[i, j]
            lo   = ci_low[i, j]
            hi   = ci_high[i, j]

            brightness = (
                0.299 * cmap(val)[0]
                + 0.587 * cmap(val)[1]
                + 0.114 * cmap(val)[2]
            )
            text_color = "white" if brightness < 0.55 else "black"

            ax.text(
                j, i - 0.10,
                f"{val:.2f}",
                ha="center", va="center",
                fontsize=11, fontweight="bold",
                color=text_color,
            )
            ax.text(
                j, i + 0.22,
                f"({lo:.2f}, {hi:.2f})",
                ha="center", va="center",
                fontsize=7.5,
                color=text_color,
            )

    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Predicted Label", fontsize=11, labelpad=10)
    ax.set_ylabel("True Label",      fontsize=11, labelpad=10)
    ax.set_title(title, fontsize=13, pad=12)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved → {save_path}")

    if show:
        plt.show()

    plt.close(fig)
    return cm_norm, ci_low, ci_high


# calculating metrics
def calculate_metrics(labels, predictions):
    precision = precision_score(labels, predictions, average="macro", zero_division=0)
    recall = recall_score(labels, predictions, average="macro", zero_division=0)
    f1 = f1_score(labels, predictions, average="macro", zero_division=0)
    quadratic_weighted_kappa = cohen_kappa_score(labels, predictions, weights="quadratic")

    return precision, recall, f1, quadratic_weighted_kappa


def save_roc_curve_data(all_labels, all_probs, class_names, save_path):
    if class_names is None:
        class_names = [f"Class {i}" for i in range(all_probs.shape[1])]

    roc_data = {}

    for i, class_name in enumerate(class_names):
        y_true_binary = (all_labels == i).astype(int)
        y_score = all_probs[:, i]
        
        # calculating ROC curve
        fpr, tpr, thresholds = roc_curve(y_true_binary, y_score)
        
        # sample points at specific FPR values 
        target_fprs = [0.05, 0.10, 0.15, 0.20, 0.25]  
        sampled_tprs = []
        
        for target_fpr in target_fprs:
            idx = np.argmin(np.abs(fpr - target_fpr))
            sampled_tprs.append(float(tpr[idx]))
        
        roc_data[class_name] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
            "sampled_points": {
                "fpr_values": target_fprs,
                "tpr_values": sampled_tprs
            }
        }
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(roc_data, f, indent=2)
    
    print(f"ROC curve data saved to: {save_path}")


def train_one_epoch_retfound(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    epoch,
    scaler=None,
    grad_accum_steps: int = 1,
    max_grad_norm: float | None = None,   # gradient clipping
):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    grad_accum_steps = max(1, int(grad_accum_steps))
    optimizer.zero_grad(set_to_none=True)

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1} - Training")

    for batch_idx, batch  in enumerate(progress_bar):
        if len(batch) == 3:
            images, labels, sources = batch 
        else:
            images, labels = batch

        images = images.to(device)
        labels = labels.to(device)
                
        # forward + loss
        if scaler is not None:
            with autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            # scale loss for accumulation
            loss_to_backprop = loss / grad_accum_steps
            scaler.scale(loss_to_backprop).backward()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            (loss / grad_accum_steps).backward()
            
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        do_step = ((batch_idx + 1) % grad_accum_steps == 0) or ((batch_idx + 1) == len(dataloader))
        if do_step:
            if max_grad_norm is not None:
                # unscale before clipping when using AMP
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

        progress_bar.set_postfix({
            "loss": running_loss / (batch_idx + 1),
            "acc": 100.0 * correct / total,
            "accum": grad_accum_steps
        })

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc



def train_one_epoch_urfound(model, dataloader, criterion, optimizer, device, epoch, scaler, grad_accum_steps, max_grad_norm):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1} - Training")


    grad_accum_steps = max(1, int(grad_accum_steps))

    for batch_idx, batch  in enumerate(progress_bar):
        if len(batch) == 3:
            images, labels, sources = batch
        else:
            images, labels = batch

        images = images.to(device)
        labels = labels.to(device).long()

        optimizer.zero_grad()

        if scaler is not None:
            with autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            loss_to_backprop = loss / grad_accum_steps
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        progress_bar.set_postfix({
            "loss": running_loss / (batch_idx + 1),
            "acc": 100 * correct / total
        })


        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100 * correct / total

    return epoch_loss, epoch_acc



def train_one_epoch_urfound(model, dataloader, criterion, optimizer, device, epoch, scaler):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1} - Training")

    for batch_idx, batch in enumerate(progress_bar):
        if len(batch) == 3:
            images, labels, sources = batch
        else:
            images, labels = batch
        images = images.to(device)
        labels = labels.to(device).long()

        # feed forward pass
        optimizer.zero_grad()

        if scaler is not None:
            with autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        progress_bar.set_postfix({
            "loss": running_loss / (batch_idx + 1),
            "acc": 100 * correct / total
        })


        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100 * correct / total

    return epoch_loss, epoch_acc



def train_one_epoch_clip(model, dataloader, criterion, optimizer, device,
                         epoch, scaler, grad_accum_steps, max_grad_norm):

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1} - Training")
    grad_accum_steps = max(1, int(grad_accum_steps))

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(progress_bar):

        if len(batch) == 3:
            images, labels, sources = batch
        else:
            images, labels = batch

        images = images.to(device)
        labels = labels.to(device).long()

        # forward
        if scaler is not None:
            with autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            loss_to_backprop = loss / grad_accum_steps
            scaler.scale(loss_to_backprop).backward()

        else:
            outputs = model(images)
            loss = criterion(logits, labels)

            loss_to_backprop = loss / grad_accum_steps
            loss_to_backprop.backward()

        if (batch_idx + 1) % grad_accum_steps == 0:

            if scaler is not None:
                scaler.unscale_(optimizer)

                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                scaler.step(optimizer)
                scaler.update()
            else:
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                optimizer.step()

            optimizer.zero_grad()

        # ===== METRICS =====
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        progress_bar.set_postfix({
            "loss": running_loss / (batch_idx + 1),
            "acc": 100 * correct / total
        })

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total

    return epoch_loss, epoch_acc



def validate_clip(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for batch_idx, (images, labels, sources) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)

            outputs_obj = model(images)
            image_features = outputs_obj.image_embeds
            outputs = model.classifier(image_features)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })

    val_loss = running_loss / len(dataloader)
    val_acc = 100. * correct / total

    return val_loss, val_acc



def validate_clip_with_metrics(model, dataloader, criterion, device, num_classes: int):
    model.eval()
    total_loss = 0.0
    y_true, y_pred, y_probs = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                images, labels, _ = batch
            else:
                images, labels = batch

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            probs = F.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(preds.detach().cpu().tolist())
            y_probs.extend(probs.detach().cpu().numpy())

    val_loss = total_loss / len(dataloader)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    acc = 100.0 * (y_true == y_pred).mean()
    bal_acc = 100.0 * balanced_accuracy_score(y_true, y_pred)
    macro_f1 = 100.0 * f1_score(y_true, y_pred, average="macro")

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        digits=4,
        zero_division=0
    )

    try:
        if num_classes == 2:
            macro_auc = roc_auc_score(y_true, y_probs[:, 1])
        else:
            macro_auc = roc_auc_score(
                y_true,
                y_probs,
                multi_class="ovr",
                average="macro"
            )
        macro_auc *= 100.0
    except ValueError:
        macro_auc = float("nan")

    return val_loss, acc, bal_acc, macro_f1, macro_auc, report


def test_clip(model, dataloader, criterion, device):

    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_predictions = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")

        for batch_idx, batch in enumerate(pbar):

            if len(batch) == 3:
                images, labels, _ = batch
            else:
                images, labels = batch

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            pbar.set_postfix({
                "loss": running_loss / (batch_idx + 1),
                "acc": 100.0 * correct / total
            })

    val_loss = running_loss / len(dataloader)
    val_acc = 100.0 * correct / total

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    precision, recall, f1, qwk = calculate_metrics(
        all_labels,
        all_predictions
    )

    generate_confusion_matrix(
        all_labels,
        all_predictions,
        "results/clip",
        "clip_cf"
    )

    # AUROC
    num_classes = all_probs.shape[1]
    per_class_auc = {}

    for i in range(num_classes):
        try:
            auc = roc_auc_score(
                (all_labels == i).astype(int),
                all_probs[:, i]
            )
            per_class_auc[f"DR{i}"] = auc
        except ValueError:
            per_class_auc[f"DR{i}"] = None

    try:
        macro_auc = roc_auc_score(
            all_labels,
            all_probs,
            multi_class="ovr",
            average="macro"
        )
    except ValueError:
        macro_auc = float("nan")

    try:
        weighted_auc = roc_auc_score(
            all_labels,
            all_probs,
            multi_class="ovr",
            average="weighted"
        )
    except ValueError:
        weighted_auc = float("nan")
    
    from sklearn.metrics import classification_report
    report_dict = classification_report(
        all_labels, all_predictions,
        target_names=["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"],
        output_dict=True,
        zero_division=0
    )
    per_class_recall = {
        name: report_dict[name]["recall"]
        for name in ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
    }
    per_class_precision = {
        name: report_dict[name]["precision"]
        for name in ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
    }

    return (
        val_loss,
        val_acc,
        precision,
        recall,
        f1,
        qwk,
        per_class_auc,
        macro_auc,
        weighted_auc,
        per_class_recall,
        per_class_precision,
        all_labels,
        all_probs
    )


def validate_retfound(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for batch_idx, batch in enumerate(pbar):
            if len(batch) == 3:
                images, labels, _ = batch
            else:
                images, labels = batch

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })

    val_loss = running_loss / len(dataloader)
    val_acc = 100. * correct / total

    return val_loss, val_acc



def validate_retfound_with_metrics(model, dataloader, criterion, device, num_classes: int):
    model.eval()
    total_loss = 0.0
    y_true, y_pred = [], []
    y_probs = []   

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                images, labels, _ = batch
            else:
                images, labels = batch

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            probs = F.softmax(outputs, dim=1)  # convert logits → probabilities
            preds = outputs.argmax(dim=1)

            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(preds.detach().cpu().tolist())
            y_probs.extend(probs.detach().cpu().numpy())

    val_loss = total_loss / len(dataloader)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    acc = 100.0 * (y_true == y_pred).mean()
    bal_acc = 100.0 * balanced_accuracy_score(y_true, y_pred)
    macro_f1 = 100.0 * f1_score(y_true, y_pred, average="macro")

    try:
        if num_classes == 2:
            macro_auc = roc_auc_score(y_true, y_probs[:, 1])
        else:
            macro_auc = roc_auc_score(
                y_true,
                y_probs,
                multi_class="ovr",   # one-vs-rest (standard for medical papers)
                average="macro"
            )
        macro_auc *= 100.0
    except ValueError:
        macro_auc = 0.0  # if class is missing in batch

    report = classification_report(
        y_true, y_pred,
        labels=list(range(num_classes)),
        digits=4,
        zero_division=0
    )

    return val_loss, acc, bal_acc, macro_f1, macro_auc, report


def test_retfound(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_predictions = []
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Test Run Progress')
        for batch_idx, batch in enumerate(pbar):
            if len(batch) == 3:
                images, labels, sources = batch
            else:
                images, labels = batch

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            
            probabilities = torch.softmax(outputs, dim=1)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # storing predictions and labels to calculate metrics
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.append(probabilities.cpu().numpy())

            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })

    val_loss = running_loss / len(dataloader)
    val_acc = 100. * correct / total

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.concatenate(all_probabilities, axis=0)

    precision, recall, f1, quadratic_weighted_kappa = calculate_metrics(all_labels, all_predictions)
    
    metrics = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "quadratic_weighted_kappa": quadratic_weighted_kappa
    }

    per_class_auc = {}
    num_classes = all_probabilities.shape[1]

    for k in range(num_classes):
        binary_true = (all_labels == k).astype(int)
        if binary_true.sum() > 0:
            per_class_auc[f"DR{k}"] = roc_auc_score(binary_true, all_probabilities[:, k])
        else:
            per_class_auc[f"DR{k}"] = None

    macro_auc = roc_auc_score(
        all_labels,
        all_probabilities,
        multi_class="ovr",
        average="macro"
    )

    weighted_auc = roc_auc_score(
        all_labels,
        all_probabilities,
        multi_class="ovr",
        average="weighted"
    )

    
    generate_confusion_matrix(all_labels, all_predictions, "results/retfound", "retfound_cf")
    
    from sklearn.metrics import classification_report
    report_dict = classification_report(
        all_labels, all_predictions,
        target_names=["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"],
        output_dict=True,
        zero_division=0
    )
    per_class_recall = {
        name: report_dict[name]["recall"]
        for name in ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
    }
    per_class_precision ={
        name: report_dict[name]["precision"]
        for name in ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
    }

    return val_loss, val_acc, precision, recall, f1, quadratic_weighted_kappa, per_class_auc, macro_auc, weighted_auc,per_class_recall, per_class_precision, all_labels, all_probabilities




def validate_urfound(model, dataloader, criterion, device):

    # calculating metrics

    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for batch_idx, (images, labels, sources) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })

    val_loss = running_loss / len(dataloader)
    val_acc = 100. * correct / total

    return val_loss, val_acc


def validate_urfound_with_metrics(model, dataloader, criterion, device, num_classes: int):
    model.eval()
    total_loss = 0.0
    y_true, y_pred, y_probs = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                images, labels, _ = batch
            else:
                images, labels = batch

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            probs = F.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(preds.detach().cpu().tolist())
            y_probs.extend(probs.detach().cpu().numpy())

    val_loss = total_loss / len(dataloader)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    acc = 100.0 * (y_true == y_pred).mean()
    bal_acc = 100.0 * balanced_accuracy_score(y_true, y_pred)
    macro_f1 = 100.0 * f1_score(y_true, y_pred, average="macro")

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        digits=4,
        zero_division=0
    )

    try:
        if num_classes == 2:
            macro_auc = roc_auc_score(y_true, y_probs[:, 1])
        else:
            macro_auc = roc_auc_score(
                y_true,
                y_probs,
                multi_class="ovr",
                average="macro"
            )
        macro_auc *= 100.0
    except ValueError:
        macro_auc = float("nan")

    return val_loss, acc, bal_acc, macro_f1, macro_auc, report


def test_urfound(model, dataloader, criterion, device):

    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_predictions = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")

        for batch_idx, batch in enumerate(pbar):

            if len(batch) == 3:
                images, labels, _ = batch
            else:
                images, labels = batch

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            pbar.set_postfix({
                "loss": running_loss / (batch_idx + 1),
                "acc": 100.0 * correct / total
            })

    val_loss = running_loss / len(dataloader)
    val_acc = 100.0 * correct / total

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    precision, recall, f1, qwk = calculate_metrics(
        all_labels,
        all_predictions
    )

    generate_confusion_matrix(
        all_labels,
        all_predictions,
        "results/urfound",
        "urfound_cf"
    )

    # --------- AUROC ---------
    num_classes = all_probs.shape[1]
    per_class_auc = {}

    for i in range(num_classes):
        try:
            auc = roc_auc_score(
                (all_labels == i).astype(int),
                all_probs[:, i]
            )
            per_class_auc[f"DR{i}"] = auc
        except ValueError:
            per_class_auc[f"DR{i}"] = None

    try:
        macro_auc = roc_auc_score(
            all_labels,
            all_probs,
            multi_class="ovr",
            average="macro"
        )
    except ValueError:
        macro_auc = float("nan")

    try:
        weighted_auc = roc_auc_score(
            all_labels,
            all_probs,
            multi_class="ovr",
            average="weighted"
        )
    except ValueError:
        weighted_auc = float("nan")


    from sklearn.metrics import classification_report
    report_dict = classification_report(
        all_labels, all_predictions,
        target_names=["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"],
        output_dict=True,
        zero_division=0
    )
    per_class_recall = {
        name: report_dict[name]["recall"]
        for name in ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
    }
    per_class_precision = {
        name: report_dict[name]["precision"]
        for name in ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
    }

    return (
        val_loss,
        val_acc,
        precision,
        recall,
        f1,
        qwk,
        per_class_auc,
        macro_auc,
        weighted_auc,
        per_class_recall,
        per_class_precision,
        all_labels,
        all_probs
    )



def show_images(dataset, train_labels, num_images, start_idx=0):

    cols = 4
    rows = (num_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(20, 10))
    axes = axes.flatten()

    for plot_idx, data_idx in enumerate(range(start_idx, start_idx + num_images)):
        if data_idx >= len(dataset):  # Safety check
            break

        image, label, dataset_name = dataset[data_idx]

        img = image.clone()
        img[0] = img[0] * 0.229 + 0.485
        img[1] = img[1] * 0.224 + 0.456
        img[2] = img[2] * 0.225 + 0.406
        img = torch.clamp(img, 0, 1)

        axes[plot_idx].imshow(img.permute(1, 2, 0))
        axes[plot_idx].set_title(f"Dataset: {dataset_name} \n Label: {label}", fontsize=12, pad=5)
        axes[plot_idx].axis('off')

    for idx in range(num_images, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()


def get_specific_layer_names(model):
    # Create a list to store the layer names
    layer_names = []

    # Recursively visit all modules and submodules
    for name, module in model.named_modules():
        # Check if the module is an instance of the specified layers
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding, torch.nn.Conv2d, Conv1D)):
            # model name parsing

            layer_names.append('.'.join(name.split('.')[4:]).split('.')[0])

    return layer_names


def json_to_csv(json_path, save_path, filename):
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as file:
        data = json.loads(file.read())

    df = pd.json_normalize(data)
    df.to_csv(f"{save_path}/{filename}.csv", index=False, encoding="utf-8")

PUBLICATION_RC = {
    'font.family':       'serif',
    'font.serif':        ['Times New Roman', 'DejaVu Serif'],
    'axes.labelsize':    13,
    'axes.titlesize':    14,
    'xtick.labelsize':   11,
    'ytick.labelsize':   11,
    'legend.fontsize':   11,
    'figure.dpi':        150,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.linewidth':    0.8,
    'xtick.direction':   'out',
    'ytick.direction':   'out',
}

# Colour used for all benchmark lines (neutral slate-blue)
BENCH_COLOR   = '#4c72b0'
MEAN_COLOR    = '#c0392b'   # red for mean reference line
BAND_COLOR    = '#4c72b0'   # same hue, low alpha for ±std band


def _apply_rc():
    plt.rcParams.update(PUBLICATION_RC)


def _load_benchmark(source) -> dict:
    """Accept either a dict or a path to benchmark.json."""
    if isinstance(source, dict):
        return source
    with open(source) as f:
        return json.load(f)


def _clean(values: list, skip: int = 1):
    """Drop the first `skip` epochs (CUDA warmup) and NaNs."""
    arr = np.array(values[skip:], dtype=float)
    return arr[~np.isnan(arr)]


def _add_mean_band(ax, values, color=MEAN_COLOR, band_color=BAND_COLOR,
                   label_mean=True):
    """Overlay a dashed mean line and a ±1 std shaded band."""
    mu  = np.mean(values)
    std = np.std(values)
    ax.axhline(mu, color=color, linewidth=1.4, linestyle='--',
               label=f'Mean = {mu:.2f}' if label_mean else None, zorder=4)
    ax.axhspan(mu - std, mu + std, color=band_color, alpha=0.12,
               label=f'±1 std ({std:.2f})', zorder=3)


def _epoch_xaxis(ax, epochs):
    ax.set_xlabel('Epoch', labelpad=6)
    ax.set_xlim(epochs[0] - 0.5, epochs[-1] + 0.5)
    # Auto-thin x ticks if many epochs
    if len(epochs) > 30:
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=10))
    else:
        ax.set_xticks(epochs)
    ax.grid(axis='y', alpha=0.3, linewidth=0.6)
    ax.grid(axis='x', alpha=0.15, linewidth=0.4)


# epoch time
def plot_epoch_time(source, output_dir: str, skip: int = 1,
                    model_name: str = 'Model'):
    """
    Line plot of wall-clock seconds per training epoch.

    Args:
        source      : benchmark dict or path to benchmark.json
        output_dir  : directory to save the figure
        skip        : number of warmup epochs to exclude from statistics
                      (they are still drawn, but greyed out)
        model_name  : label used in the title
    """
    _apply_rc()
    bm     = _load_benchmark(source)
    times  = np.array(bm['epoch_times_s'], dtype=float)
    epochs = np.arange(1, len(times) + 1)
    clean  = _clean(bm['epoch_times_s'], skip)

    fig, ax = plt.subplots(figsize=(8, 4))

    # Warmup epochs (greyed out)
    if skip > 0:
        ax.plot(epochs[:skip], times[:skip], 'o--',
                color='#aaaaaa', linewidth=1.2, markersize=4,
                label='Warmup (excluded from stats)', zorder=2)

    ax.plot(epochs[skip:], times[skip:], 'o-',
            color=BENCH_COLOR, linewidth=1.6, markersize=4,
            label='Epoch time', zorder=3)

    _add_mean_band(ax, clean)
    _epoch_xaxis(ax, epochs)

    ax.set_ylabel('Time (s)')
    ax.set_title(f'{model_name} — Training Time per Epoch', pad=10)
    ax.legend(frameon=True, framealpha=0.9, edgecolor='#cccccc')

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, 'epoch_time.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


# peak GPU memory
def plot_gpu_memory(source, output_dir: str, skip: int = 1,
                    model_name: str = 'Model'):
    """
    Line plot of peak GPU memory (MiB) per training epoch.
    """
    _apply_rc()
    bm    = _load_benchmark(source)
    raw   = np.array(bm['peak_gpu_mb'], dtype=float)
    valid = ~np.isnan(raw)
    if not valid.any():
        print('No GPU memory data found — skipping gpu_memory.png')
        return

    epochs = np.arange(1, len(raw) + 1)
    clean  = _clean(bm['peak_gpu_mb'], skip)

    fig, ax = plt.subplots(figsize=(8, 4))

    if skip > 0:
        ax.plot(epochs[:skip], raw[:skip], 'o--',
                color='#aaaaaa', linewidth=1.2, markersize=4,
                label='Warmup (excluded from stats)', zorder=2)

    ax.plot(epochs[skip:], raw[skip:], 's-',
            color=BENCH_COLOR, linewidth=1.6, markersize=4,
            label='Peak GPU memory', zorder=3)

    _add_mean_band(ax, clean)
    _epoch_xaxis(ax, epochs)

    # Secondary axis in GiB
    ax2 = ax.twinx()
    ax2.set_ylim(np.array(ax.get_ylim()) / 1024)
    ax2.set_ylabel('Memory (GiB)', labelpad=6)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    ax2.spines['top'].set_visible(False)

    ax.set_ylabel('Peak GPU Memory (MiB)')
    ax.set_title(f'{model_name} — Peak GPU Memory per Epoch', pad=10)
    ax.legend(frameon=True, framealpha=0.9, edgecolor='#cccccc')

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, 'gpu_memory.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


# training throughput
def plot_throughput(source, output_dir: str, skip: int = 1,
                    model_name: str = 'Model'):
    """
    Line plot of training throughput (samples / second) per epoch.
    """
    _apply_rc()
    bm     = _load_benchmark(source)
    thru   = np.array(bm['train_throughput'], dtype=float)
    epochs = np.arange(1, len(thru) + 1)
    clean  = _clean(bm['train_throughput'], skip)

    fig, ax = plt.subplots(figsize=(8, 4))

    if skip > 0:
        ax.plot(epochs[:skip], thru[:skip], 'o--',
                color='#aaaaaa', linewidth=1.2, markersize=4,
                label='Warmup (excluded from stats)', zorder=2)

    ax.plot(epochs[skip:], thru[skip:], '^-',
            color=BENCH_COLOR, linewidth=1.6, markersize=4,
            label='Throughput', zorder=3)

    _add_mean_band(ax, clean)
    _epoch_xaxis(ax, epochs)

    ax.set_ylabel('Samples / second')
    ax.set_title(f'{model_name} — Training Throughput per Epoch', pad=10)
    ax.legend(frameon=True, framealpha=0.9, edgecolor='#cccccc')

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, 'throughput.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


# summary panel (all three + text table) 
def plot_benchmark_summary(source, output_dir: str, skip: int = 1,
                           model_name: str = 'Model'):
    """
    A 2×2 figure: epoch time | GPU memory | throughput | summary stats table.
    All panels share the same publication style.
    """
    _apply_rc()
    bm     = _load_benchmark(source)
    summary = bm.get('summary', {})

    times  = np.array(bm['epoch_times_s'],    dtype=float)
    gpu    = np.array(bm['peak_gpu_mb'],       dtype=float)
    thru   = np.array(bm['train_throughput'],  dtype=float)
    epochs = np.arange(1, len(times) + 1)

    c_times = _clean(bm['epoch_times_s'],    skip)
    c_gpu   = _clean(bm['peak_gpu_mb'],      skip)
    c_thru  = _clean(bm['train_throughput'], skip)

    fig = plt.figure(figsize=(14, 9))
    gs  = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

    axes_cfg = [
        (gs[0, 0], times, gpu,   'Time (s)',              'Epoch Time',          'o-', BENCH_COLOR),
        (gs[0, 1], times, gpu,   'Peak GPU Mem (MiB)',    'Peak GPU Memory',     's-', BENCH_COLOR),
        (gs[1, 0], times, thru,  'Samples / second',      'Training Throughput', '^-', BENCH_COLOR),
    ]

    datasets    = [times,  gpu,   thru]
    clean_sets  = [c_times, c_gpu, c_thru]
    ylabels     = ['Time (s)', 'Peak GPU Memory (MiB)', 'Samples / second']
    subtitles   = ['Epoch Time', 'Peak GPU Memory', 'Training Throughput']
    markers     = ['o-', 's-', '^-']
    subplot_pos = [gs[0, 0], gs[0, 1], gs[1, 0]]

    for pos, data, clean, ylabel, subtitle, marker in zip(
            subplot_pos, datasets, clean_sets, ylabels, subtitles, markers):

        ax = fig.add_subplot(pos)

        if skip > 0:
            ax.plot(epochs[:skip], data[:skip], 'o--',
                    color='#aaaaaa', linewidth=1.0, markersize=3, zorder=2)

        ax.plot(epochs[skip:], data[skip:], marker,
                color=BENCH_COLOR, linewidth=1.4, markersize=3,
                label=subtitle, zorder=3)

        if len(clean) > 0:
            _add_mean_band(ax, clean, label_mean=True)

        _epoch_xaxis(ax, epochs)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(subtitle, fontsize=12, pad=7)
        ax.legend(fontsize=9, frameon=True, framealpha=0.9, edgecolor='#cccccc')

    #Stats table panel
    ax_t = fig.add_subplot(gs[1, 1])
    ax_t.axis('off')

    total_min = summary.get('total_train_time_s', np.sum(times)) / 60

    rows = [
        ['Metric',                    'Value'],
        ['Epochs measured',           str(summary.get('epochs_measured',
                                           len(c_times)))],
        ['Mean epoch time',           f"{summary.get('avg_epoch_time_s',  np.mean(c_times)):.1f} s"],
        ['Std epoch time',            f"{summary.get('std_epoch_time_s',  np.std(c_times)):.1f} s"],
        ['Min / Max epoch time',      f"{summary.get('min_epoch_time_s',  np.min(c_times)):.1f} / "
                                      f"{summary.get('max_epoch_time_s',  np.max(c_times)):.1f} s"],
        ['Total training time',       f"{total_min:.1f} min"],
        ['Mean peak GPU memory',      f"{summary.get('avg_peak_gpu_mb',   np.mean(c_gpu) if len(c_gpu) else float('nan')):.0f} MiB"],
        ['Max peak GPU memory',       f"{summary.get('max_peak_gpu_mb',   np.max(c_gpu)  if len(c_gpu) else float('nan')):.0f} MiB"],
        ['Mean throughput',           f"{summary.get('avg_throughput_sps',np.mean(c_thru)):.1f} samp/s"],
    ]

    col_widths = [0.62, 0.38]
    row_h      = 0.09
    y_start    = 0.95

    # Header row
    for col_idx, header in enumerate(rows[0]):
        x = sum(col_widths[:col_idx]) + 0.02
        ax_t.text(x, y_start, header, transform=ax_t.transAxes,
                  fontsize=10.5, fontweight='bold', fontfamily='serif',
                  va='top')

    ax_t.plot([0, 1], [y_start - 0.01, y_start - 0.01], color="black", linewidth=0.8,
              transform=ax_t.transAxes, clip_on=False)

    for r_idx, row in enumerate(rows[1:]):
        y = y_start - (r_idx + 1) * row_h - 0.04
        bg = '#f2f4f8' if r_idx % 2 == 0 else 'white'
        ax_t.add_patch(mpatches.FancyBboxPatch(
            (0, y - 0.005), 1.0, row_h - 0.005,
            boxstyle='square,pad=0', transform=ax_t.transAxes,
            facecolor=bg, edgecolor='none', zorder=0))
        for col_idx, cell in enumerate(row):
            x = sum(col_widths[:col_idx]) + 0.02
            ax_t.text(x, y + row_h * 0.55, cell, transform=ax_t.transAxes,
                      fontsize=9.5, fontfamily='serif', va='center')

    ax_t.set_title('Summary Statistics', fontsize=12, pad=7)

    fig.suptitle(f'{model_name} — Training Benchmark', fontsize=15,
                 fontweight='bold', y=0.99)

    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, 'benchmark_summary.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


# master function for benchmarks
def plot_all_benchmark(source, output_dir: str, skip: int = 1,
                       model_name: str = 'Model'):
    """
    Generate all benchmark plots:
      • epoch_time.png        — wall-clock time per epoch
      • gpu_memory.png        — peak GPU memory per epoch
      • throughput.png        — training throughput per epoch
      • benchmark_summary.png — combined summary panel + stats table

    Args:
        source      : benchmark dict (in-memory) or path to benchmark.json
        output_dir  : directory where plots are saved (created if absent)
        skip        : warmup epochs to exclude from mean/std (default: 1)
        model_name  : model label used in plot titles (e.g. 'RETFound LoRA')
    """
    plot_epoch_time(source, output_dir, skip, model_name)
    plot_gpu_memory(source, output_dir, skip, model_name)
    plot_throughput(source, output_dir, skip, model_name)
    plot_benchmark_summary(source, output_dir, skip, model_name)
    print(f'\nAll benchmark plots saved to: {output_dir}')
