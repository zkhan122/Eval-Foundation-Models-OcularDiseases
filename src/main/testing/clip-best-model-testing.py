import sys
import os 
import json
import torch
import optuna
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from transformers import CLIPVisionModelWithProjection, CLIPProcessor
from timm.models.layers import trunc_normal_
import torchvision
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import DataLoader
from data_processing.dataset import CombinedDRDataSet
from utilities.utils import identity_transform, show_images, test_clip, calculate_metrics
from torch import nn
from torch import optim
from torch.cuda.amp import autocast, GradScaler
from peft import get_peft_model, LoraConfig, TaskType



class CLIPRetina(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()

        # Must be called "vision" — not vision_model
        self.vision = CLIPVisionModelWithProjection.from_pretrained(model_name)

        embedding_dim = self.vision.config.projection_dim
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, images):
        outputs = self.vision(pixel_values=images)
        image_embeds = outputs.image_embeds
        logits = self.classifier(image_embeds)
        return logits




NUM_CLASSES = 5
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.05


MICRO_BATCH_SIZE = 8


DATA_DIR = "../../../datasets"
SRC_DIR = "../../"


test_root_directories = {
    "DEEPDRID": f"{DATA_DIR}/DeepDRiD",
    "DDR": f"{DATA_DIR}/DDR",
    "EYEPACS": f"{DATA_DIR}/EYEPACS",
    "MFIDDR": f"{DATA_DIR}/MFIDDR",
}

test_transformations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

test_dataset = CombinedDRDataSet(
    root_directories=test_root_directories,
    split="test",
    img_transform=test_transformations,
    label_transform=identity_transform
)

test_csv_paths = {
    "EYEPACS": f"{test_root_directories['EYEPACS']}/all_labels.csv",
    "DEEPDRID": f"{test_root_directories['DEEPDRID']}/regular_fundus_images/Online-Challenge1&2-Evaluation/Challenge1_labels.csv",
    "DDR": f"{test_root_directories['DDR']}/DR_grading.csv",
    "MFIDDR": f"{test_root_directories['MFIDDR']}/sample/test_fourpic_label.csv",
}




test_dataset.load_labels_from_csv_for_test(test_csv_paths)
test_dataset.prune_unlabeled()

# note to self: THIS IS THE MODEL (during testing phase we only load weights)
checkpoint_path = f"{SRC_DIR}/best_models/best_clip_model.pth"
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)



print("Checkpoint keys:", checkpoint.keys())
print("\nCheckpoint structure:")
for key in checkpoint.keys():
    print(f"  {key}: {type(checkpoint[key])}")
    if isinstance(checkpoint[key], dict):
        print(f"    Subkeys: {list(checkpoint[key].keys())[:]}")  



model = CLIPRetina("openai/clip-vit-large-patch14", NUM_CLASSES)


peft_config = LoraConfig(
    r=checkpoint["params"]["lora_r"],
    lora_alpha=checkpoint["params"]["lora_alpha"],
    target_modules=["q_proj", "k_proj", "v_proj"],
    lora_dropout=0.05,  
    bias="none",
    modules_to_save=["classifier"],
    # task_type=TaskType.FEATURE_EXTRACTION
)


model = get_peft_model(model, peft_config)

model = model.to(DEVICE)

# Transform the checkpoint state dict keys
# Transform the checkpoint state dict keys
checkpoint_state_dict = checkpoint["model_state_dict"]
adapted_dict = {}

for key, value in checkpoint_state_dict.items():
    new_key = key
    
    # Handle vision_model transformation (insert "vision" before "vision_model")
    if "vision_model" in key:
        parts = key.split('.')
        for i, part in enumerate(parts):
            if part == "vision_model":
                # Insert "vision" before "vision_model"
                new_parts = parts[:i] + ["vision", "vision_model"] + parts[i+1:]
                new_key = '.'.join(new_parts)
                break
    
    # Handle visual_projection transformation (add "vision." prefix)
    elif "visual_projection" in key:
        parts = key.split('.')
        for i, part in enumerate(parts):
            if part == "visual_projection":
                # Add "vision." before "visual_projection"
                new_parts = parts[:i] + ["vision", "visual_projection"] + parts[i+1:]
                new_key = '.'.join(new_parts)
                break
    
    adapted_dict[new_key] = value

# Debug: Print some transformations to verify (fixed indexing)
print("Sample transformations:")
for i, (old_key, _) in enumerate(checkpoint_state_dict.items()):
    if i >= 5:
        break
    # Recompute new_key for display (simple but works)
    new_key = old_key
    if "vision_model" in old_key:
        parts = old_key.split('.')
        for j, part in enumerate(parts):
            if part == "vision_model":
                new_parts = parts[:j] + ["vision", "vision_model"] + parts[j+1:]
                new_key = '.'.join(new_parts)
                break
    elif "visual_projection" in old_key:
        parts = old_key.split('.')
        for j, part in enumerate(parts):
            if part == "visual_projection":
                new_parts = parts[:j] + ["vision", "visual_projection"] + parts[j+1:]
                new_key = '.'.join(new_parts)
                break
    print(f"  {old_key} -> {new_key}")



# Load the adapted state dict
model.load_state_dict(adapted_dict)
model.eval()


# param count
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")


criterion = nn.CrossEntropyLoss()


test_loader = DataLoader(test_dataset, batch_size=MICRO_BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)


test_loss, test_acc, precision, recall, f1, qwk, per_class_auc, macro_auc, weighted_auc = test_clip(
    model, test_loader, criterion, DEVICE
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


val_bal_acc = checkpoint.get('val_bal_acc', checkpoint.get('val_acc', None))
if val_bal_acc is None:
    raise KeyError("No validation accuracy found in checkpoint")

lora_cfg = checkpoint.get('lora', {})
train_cfg = checkpoint.get('train', {})

print(f"\nCOMPARISON WITH TRAINING:")
print(f"BALANCED Validation accuracy during training: {val_bal_acc:.2f}%")
print(f"Test accuracy on unseen data: {test_acc:.2f}%")
print(f"Difference: {test_acc - val_bal_acc}")


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
    "training_validation_accuracy": float(val_bal_acc),
    "checkpoint": os.path.basename(checkpoint_path),
    "lora": lora_cfg,
    "train": train_cfg
}


results_path = "results/clip/clip_test_results.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=4)

print(f"\nResults saved to: {results_path}")
print("="*70)
