import os
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split


ODIR_CLASS_NAMES = ["Normal", "Diabetes", "Glaucoma", "Cataract",
                    "AMD", "Hypertension", "Myopia", "Other"]
NUM_CLASSES = 8


class ODIRDataset(Dataset):
    """
    Multi-label dataset for ODIR-5K ocular disease classification.

    The CSV must contain:
      - 'filename' : image filename (e.g. '0_right.jpg')
      - 'target'   : one-hot label vector as a string '[1, 0, 0, 0, 0, 0, 0, 0]'

    Labels are returned as float32 tensors of shape (8,) for BCEWithLogitsLoss.
    """

    def __init__(self, img_dir, csv_path, split="train",
                 img_transform=None, label_transform=None,
                 val_size=0.2, random_seed=42):
        """
        Args:
            img_dir      : path to the folder containing the images
            csv_path     : path to full_df.csv
            split        : "train", "val", or "test"
                           train/val are split from the same CSV;
                           test uses the CSV as-is (no splitting)
            img_transform: torchvision transform applied to PIL images
            label_transform: optional transform applied to label tensor
            val_size     : fraction of training data used for validation
            random_seed  : seed for reproducible train/val split
        """
        self.img_dir         = img_dir
        self.img_transform   = img_transform
        self.label_transform = label_transform
        self.split           = split

        df = pd.read_csv(csv_path)

        # keep only rows where image actually exists
        df = df[df["filename"].notna()].reset_index(drop=True)

        if split in ("train", "val"):
            train_df, val_df = train_test_split(
                df, test_size=val_size,
                random_state=random_seed, shuffle=True
            )
            self.df = train_df.reset_index(drop=True) if split == "train" \
                      else val_df.reset_index(drop=True)
        else:
            # test split — use the CSV as provided
            self.df = df.reset_index(drop=True)

        # pre-parse filenames and labels
        self.image_paths = [
            os.path.join(img_dir, fn) for fn in self.df["filename"]
        ]
        self.labels = self._parse_labels(self.df["target"])

        # prune rows where the image file does not exist on disk
        self.prune_missing()

        print(f"ODIRDataset [{split}]: {len(self)} samples")

    def _parse_labels(self, target_series):
        """Parse target column from string '[1, 0, ...]' to float32 arrays."""
        labels = []
        for t in target_series:
            vec = ast.literal_eval(t)          # '[1, 0, ...]' → Python list
            labels.append(np.array(vec, dtype=np.float32))
        return labels

    def prune_missing(self):
        """Remove samples whose image file does not exist on disk."""
        valid_indices = [
            i for i, p in enumerate(self.image_paths) if os.path.exists(p)
        ]
        missing = len(self.image_paths) - len(valid_indices)
        if missing > 0:
            print(f"  Pruned {missing} missing images")
        self.image_paths = [self.image_paths[i] for i in valid_indices]
        self.labels      = [self.labels[i]      for i in valid_indices]
        self.df          = self.df.iloc[valid_indices].reset_index(drop=True)

    def get_dataset_statistics(self):
        """Print per-class counts — useful for checking class balance."""
        label_matrix = np.stack(self.labels)           # (N, 8)
        counts = label_matrix.sum(axis=0).astype(int)
        lines  = [f"ODIRDataset [{self.split}] — {len(self)} samples"]
        for name, count in zip(ODIR_CLASS_NAMES, counts):
            lines.append(f"  {name:<12}: {count:5d}  ({100*count/len(self):.1f}%)")
        return "\n".join(lines)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label    = torch.tensor(self.labels[idx], dtype=torch.float32)

        image = Image.open(img_path).convert("RGB")

        if self.img_transform:
            image = self.img_transform(image)
        if self.label_transform:
            label = self.label_transform(label)

        return image, label
