import sys
import os 
import json
import torch
import optuna
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from models.UrFound.finetune import models_vit
from models.UrFound.util import pos_embed
from timm.models.layers import trunc_normal_
import torchvision
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import DataLoader
from data_processing.dataset import CombinedDRDataSet
from utilities.utils import identity_transform, show_images, test_urfound,  calculate_metrics
from torch import nn
from torch import optim
from torch.cuda.amp import autocast, GradScaler
from peft import get_peft_model, LoraConfig, TaskType
import numpy as np

NUM_CLASSES = 5
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "../../../../datasets"
SRC_DIR = "../../../"

test_root_directories = {
    "DEEPDRID": f"{DATA_DIR}/DeepDRiD",
    "DDR": f"{DATA_DIR}/DDR",
    "EYEPACS": f"{DATA_DIR}/EYEPACS",
    "MFIDDR": f"{DATA_DIR}/MFIDDR",
}


test_transformations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


test_dataset = CombinedDRDataSet(root_directories=test_root_directories, split="test", img_transform=test_transformations, label_transform=identity_transform)



test_csv_paths = {
    "EYEPACS": f"{test_root_directories['EYEPACS']}/all_labels.csv",
    "DEEPDRID": f"{test_root_directories['DEEPDRID']}/regular_fundus_images/Online-Challenge1&2-Evaluation/Challenge1_labels.csv",
    "DDR": f"{test_root_directories['DDR']}/DR_grading.csv",
    "MFIDDR": f"{test_root_directories['MFIDDR']}/sample/test_fourpic_label.csv",
}


test_dataset.load_labels_from_csv_for_test(test_csv_paths)
test_dataset.prune_unlabeled()

# note to self: THIS IS THE MODEL (during testing phase we only load weights)

checkpoint_path = f"{SRC_DIR}/best_models/best_urfound_model.pth"
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
print("CHECKPOINT KEYS:", checkpoint.keys())


print("Checkpoint keys:", checkpoint.keys())
print("\nCheckpoint structure:")
for key in checkpoint.keys():
    print(f"  {key}: {type(checkpoint[key])}")
    if isinstance(checkpoint[key], dict):
        print(f"    Subkeys: {list(checkpoint[key].keys())[:5]}...")  


lora_cfg = checkpoint.get("lora", {"r": 8, "alpha": 32, "dropout": 0.05})
train_cfg = checkpoint.get("train", {})
batch_size = train_cfg.get("micro_batch", 8)  

print("\nLoaded checkpoint configuration:")
print(f"  LoRA r: {lora_cfg['r']}")
print(f"  LoRA alpha: {lora_cfg['alpha']}")
print(f"  LoRA dropout: {lora_cfg['dropout']}")


model = models_vit.__dict__["vit_base_patch16"](
    num_classes=NUM_CLASSES,
    drop_path_rate=0.2,
    global_pool=True
)

peft_config = LoraConfig(
    r=lora_cfg["r"],
    lora_alpha=lora_cfg["alpha"],
    lora_dropout=lora_cfg["dropout"],
    target_modules=["qkv", "proj"],
    bias="none",
    modules_to_save=["head"]
)



model = get_peft_model(model, peft_config)
model = model.to(DEVICE)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()



# param count
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")


criterion = nn.CrossEntropyLoss()

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)


test_loss, test_acc, precision, recall, f1, qwk, per_class_auc, macro_auc, weighted_auc, y_probs  = test_urfound(
    model=model,
    dataloader=test_loader,
    criterion=criterion,
    device=DEVICE
)


print("\nFINAL TEST RESULTS")
print(f"Accuracy: {test_acc:.2f}%")
print(f"Loss: {test_loss:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1: {f1:.4f}")
print(f"QWK: {qwk:.4f}")

print("Per-class AUROC:")
for c, auc in per_class_auc.items():
    if auc is None:
        print(f"{c}: N/A")
    else:
        print(f"{c}: {auc:.4f}")

print(f"Macro AUC: {macro_auc:.4f}")
print(f"Weighted AUC: {weighted_auc:.4f}")



print(f"\nCOMPARISON WITH TRAINING:")
print(f"BALANCED Validation accuracy during training: {checkpoint['val_bal_acc']:.2f}%")
print(f"Test accuracy on unseen data: {test_acc:.2f}%")
print(f"Difference: {test_acc - checkpoint['val_bal_acc']:+.2f}%")


per_class_auc_list = [
    per_class_auc[f"DR{i}"]
    for i in range(len(per_class_auc))
    if per_class_auc[f"DR{i}"] is not None
]

results = {
    "test_accuracy": float(test_acc),
    "test_loss": float(test_loss),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "quadratic_weighted_kappa": float(qwk),
    "Per-class AUC": per_class_auc_list,
    "macro_auc": float(macro_auc),
    "weighted_auc": float(weighted_auc),
    "training_validation_accuracy": float(checkpoint["val_bal_acc"]),
    "checkpoint": os.path.basename(checkpoint_path),
    "lora": lora_cfg,
    "train": train_cfg
}



results_path = "results/urfound/urfound_test_results.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=4)

np.save("../probs_numpy/urfound_lora_dr_probs.npy", y_probs)

print(f"\nResults saved to: {results_path}")
print("="*70)
