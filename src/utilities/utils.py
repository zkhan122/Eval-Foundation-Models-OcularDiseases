
from pathlib import Path
import json
import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformers import Conv1D
from tqdm import tqdm
import numpy as np
from torch.amp import autocast
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score, balanced_accuracy_score, classification_report, roc_auc_score
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



# calculating metrics
def calculate_metrics(labels, predictions):
    precision = precision_score(labels, predictions, average="macro", zero_division=0)
    recall = recall_score(labels, predictions, average="macro", zero_division=0)
    f1 = f1_score(labels, predictions, average="macro", zero_division=0)
    quadratic_weighted_kappa = cohen_kappa_score(labels, predictions, weights="quadratic")

    return precision, recall, f1, quadratic_weighted_kappa


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
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_predictions = []
    all_labels = []

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

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })

    val_loss = running_loss / len(dataloader)
    val_acc = 100. * correct / total

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    precision, recall, f1, quadratic_weighted_kappa = calculate_metrics(all_labels, all_predictions)
    
    metrics = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "quadratic_weighted_kappa": quadratic_weighted_kappa
    }

    generate_confusion_matrix(all_labels, all_predictions, "results/clip", "clip_cf")

    return val_loss, val_acc, precision, recall, f1, quadratic_weighted_kappa


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

    return val_loss, val_acc, precision, recall, f1, quadratic_weighted_kappa, per_class_auc, macro_auc, weighted_auc




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

    return (
        val_loss,
        val_acc,
        precision,
        recall,
        f1,
        qwk,
        per_class_auc,
        macro_auc,
        weighted_auc
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
