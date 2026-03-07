import os
import pandas as pd
from pathlib import Path
from typing import Optional, Dict
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

# We need train_test_split because glaucoma datasets have flat structures
# with no predefined splits — unlike your DR datasets which had split subfolders.
from sklearn.model_selection import train_test_split

# Reuse the same utilities you already have
from utilities.utils import normalize_stem, _is_image_valid

valid_file_extensions = ["jpg", "jpeg", "png"]

# Splits are derived from flat image pools using these fixed ratios.
# The random seed ensures the same images always end up in the same split,
# which is critical for reproducibility across experiments.
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
RANDOM_SEED = 42


def _split_paths(image_paths: list, labels: list, split: str):
    """
    Given a flat list of image paths and their labels, return only the
    portion corresponding to the requested split.

    We do a two-stage stratified split:
      1. Split into train vs (val+test)
      2. Split (val+test) into val vs test

    Stratification ensures class balance is preserved in every split —
    important here because glaucoma datasets can be imbalanced.
    """
    # Stage 1: carve off the training portion
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels,
        test_size=(VAL_RATIO + TEST_RATIO),
        random_state=RANDOM_SEED,
        stratify=labels          # preserves healthy/glaucoma ratio
    )

    # Stage 2: split the remainder into val and test equally
    relative_val_size = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=(1.0 - relative_val_size),
        random_state=RANDOM_SEED,
        stratify=temp_labels
    )

    if split == "train":
        return train_paths, train_labels
    elif split == "val":
        return val_paths, val_labels
    elif split == "test":
        return test_paths, test_labels
    else:
        raise ValueError(f"Invalid split '{split}'. Expected 'train', 'val', or 'test'.")


class CombinedGlaucomaDataset(Dataset):
    """
    Combined dataset class for binary glaucoma classification.
    
    Supports G1020 and ORIGA datasets. Both have flat image directories
    (no predefined splits), so splits are derived deterministically using
    stratified random splitting with a fixed seed.

    Labels:
        0 = Healthy (no glaucoma)
        1 = Glaucoma

    Usage example:
        datasets = {
            "G1020": "/path/to/g1020",
            "ORIGA": "/path/to/origa"
        }
        train_dataset = CombinedGlaucomaDataset(datasets, split="train", img_transform=my_transforms)
    """

    def __init__(self,
                 root_directories: Dict[str, str],
                 split: str,
                 img_transform: Optional[transforms.Compose] = None,
                 label_transform: Optional[transforms.Compose] = None):

        self.root_directories = root_directories
        self.split = split
        self.img_transform = img_transform
        self.label_transform = label_transform

        # These three lists are parallel — index i refers to the same sample
        # across all three, exactly as in your DR class.
        self.image_paths = []
        self.labels = []
        self.sources = []

        # Load whichever datasets were provided in root_directories
        if "G1020" in self.root_directories:
            self.load_G1020()

        if "ORIGA" in self.root_directories:
            self.load_ORIGA()

        if "REFUGE" in self.root_directories:
            self.load_REFUGE()


    # ------------------------------------------------------------------
    # G1020 loader
    # ------------------------------------------------------------------
    def load_G1020(self):
        """
        G1020 structure:
            G1020_ROOT/
                Images/
                    image_0.jpg
                    image_0.json   (segmentation annotations — ignored here)
                    image_1.jpg
                    ...
                G1020.csv          (imageID, binaryLabels)

        The CSV is the ground truth. We read it first into a lookup dict,
        then collect all valid images in the Images folder, and finally
        apply the train/val/test split deterministically.
        """
        G1020_ROOT = Path(self.root_directories["G1020"])
        image_dir  = G1020_ROOT / "Images"
        csv_path   = G1020_ROOT / "G1020.csv"

        print(f"\nG1020_ROOT:  {G1020_ROOT}")
        print(f"G1020 exists: {G1020_ROOT.exists()}")

        if not image_dir.exists():
            print(f"ERROR: G1020 Images folder not found at {image_dir}")
            return

        if not csv_path.exists():
            print(f"ERROR: G1020.csv not found at {csv_path}")
            return

        # Build a filename -> label lookup from the CSV.
        # The imageID column contains values like "image_0.jpg", so we can
        # match directly without stripping extensions (unlike some DR datasets).
        labels_df  = pd.read_csv(csv_path)
        label_dict = {
            str(row["imageID"]).strip(): int(row["binaryLabels"])
            for _, row in labels_df.iterrows()
        }
        print(f"G1020 CSV loaded: {len(label_dict)} labeled entries")

        # Collect all image paths and their labels from the flat Images folder
        all_paths  = []
        all_labels = []

        for filename in os.listdir(image_dir):
            name, ext = os.path.splitext(filename)
            if ext.lstrip('.').lower() not in valid_file_extensions:
                continue  # skip JSONs and anything else

            if filename not in label_dict:
                # Image exists but has no CSV entry — skip with a warning
                print(f"Warning: {filename} has no label in G1020.csv, skipping.")
                continue

            all_paths.append(str(image_dir / filename))
            all_labels.append(label_dict[filename])

        print(f"G1020: {len(all_paths)} labeled images found before splitting.")

        # Apply the deterministic split — this is the key difference from
        # your DR loaders which simply read from a split-specific subfolder.
        split_paths, split_labels = _split_paths(all_paths, all_labels, self.split)

        self.image_paths.extend(split_paths)
        self.labels.extend(split_labels)
        self.sources.extend(["G1020"] * len(split_paths))

        print(f"G1020 '{self.split}' split: {len(split_paths)} images loaded.")


    # ------------------------------------------------------------------
    # ORIGA loader
    # ------------------------------------------------------------------
    def load_ORIGA(self):
        """
        ORIGA structure:
            ORIGA_ROOT/
                Images/
                    001.jpg
                    002.jpg
                    ...
                OrigaList.csv   (Eye, Filename, ExpCDR, Set, Glaucoma)

        The 'Filename' column matches image filenames exactly (e.g. "001.jpg").
        The 'Glaucoma' column is the binary label: 0 = healthy, 1 = glaucoma.

        The original dataset includes a 'Set' column (A/B) denoting a
        predefined split, but we deliberately ignore it here and apply the
        same deterministic stratified split used for G1020, so that split
        logic is consistent across all glaucoma datasets.
        """
        ORIGA_ROOT = Path(self.root_directories["ORIGA"])
        image_dir  = ORIGA_ROOT / "Images"
        csv_path   = ORIGA_ROOT / "OrigaList.csv"

        print(f"\nORIGA_ROOT:  {ORIGA_ROOT}")
        print(f"ORIGA exists: {ORIGA_ROOT.exists()}")

        if not image_dir.exists():
            print(f"ERROR: ORIGA Images folder not found at {image_dir}")
            return

        if not csv_path.exists():
            print(f"ERROR: OrigaList.csv not found at {csv_path}")
            return

        labels_df  = pd.read_csv(csv_path)
        # 'Filename' matches the image file directly (e.g. "001.jpg")
        # 'Glaucoma' is already binary 0/1 — no conversion needed
        label_dict = {
            str(row["Filename"]).strip(): int(row["Glaucoma"])
            for _, row in labels_df.iterrows()
        }
        print(f"ORIGA CSV loaded: {len(label_dict)} labeled entries")

        all_paths  = []
        all_labels = []

        for filename in os.listdir(image_dir):
            name, ext = os.path.splitext(filename)
            if ext.lstrip('.').lower() not in valid_file_extensions:
                continue

            if filename not in label_dict:
                print(f"Warning: {filename} has no label in ORIGA CSV, skipping.")
                continue

            all_paths.append(str(image_dir / filename))
            all_labels.append(label_dict[filename])

        print(f"ORIGA: {len(all_paths)} labeled images found before splitting.")

        split_paths, split_labels = _split_paths(all_paths, all_labels, self.split)

        self.image_paths.extend(split_paths)
        self.labels.extend(split_labels)
        self.sources.extend(["ORIGA"] * len(split_paths))

        print(f"ORIGA '{self.split}' split: {len(split_paths)} images loaded.")


    # ------------------------------------------------------------------
    # REFUGE loader
    # ------------------------------------------------------------------
    def load_REFUGE(self):
        """
        REFUGE structure:
            REFUGE_ROOT/
                train/
                    Images/         <- actual image files (g0001.jpg, n0001.jpg, ...)
                    index.json      <- label file for this split
                val/
                    Images/
                    index.json
                test/
                    Images/
                    index.json

        Unlike G1020 and ORIGA, REFUGE ships with predefined train/val/test
        splits, so we use them directly rather than deriving our own.
        This is intentional — respecting the original split avoids any risk
        of test images leaking into training data.

        index.json format:
            {
              "0": {"ImgName": "g0001.jpg", "Label": 1, ...},
              "1": {"ImgName": "n0001.jpg", "Label": 0, ...},
              ...
            }

        Label values:
            1 = Glaucoma  (images typically prefixed with 'g')
            0 = Healthy   (images typically prefixed with 'n')
        """
        import json

        REFUGE_ROOT = Path(self.root_directories["REFUGE"])

        print(f"\nREFUGE_ROOT:  {REFUGE_ROOT}")
        print(f"REFUGE exists: {REFUGE_ROOT.exists()}")

        # Map the requested split name to the correct subfolder.
        # REFUGE uses the same names we do (train/val/test) so no remapping needed.
        split_dir = REFUGE_ROOT / self.split

        if not split_dir.exists():
            print(f"ERROR: REFUGE '{self.split}' folder not found at {split_dir}")
            return

        image_dir  = split_dir / "Images"
        json_path  = split_dir / "index.json"

        if not image_dir.exists():
            print(f"ERROR: REFUGE Images folder not found at {image_dir}")
            return

        if not json_path.exists():
            print(f"ERROR: REFUGE index.json not found at {json_path}")
            return

        # Parse the JSON into a filename -> label lookup dict.
        # The outer keys ("0", "1", ...) are just numeric indices we don't need —
        # we only care about ImgName and Label inside each entry.
        with open(json_path, "r") as f:
            index_data = json.load(f)

        # The REFUGE challenge withheld labels from the test split — those JSON
        # entries only contain ImgName and image dimensions, with no Label key.
        # We filter those out rather than crashing, and warn clearly.
        # In practice this means REFUGE test loads zero images — use val split
        # as your held-out evaluation set for REFUGE instead.
        label_dict = {}
        skipped = 0
        for entry in index_data.values():
            if "Label" not in entry:
                skipped += 1
                continue
            label_dict[entry["ImgName"]] = int(entry["Label"])

        if skipped > 0:
            print(f"WARNING: REFUGE '{self.split}' index.json has {skipped} entries "
                  f"with no 'Label' key — expected for the test split, whose labels "
                  f"were withheld by the challenge organisers. "
                  f"Use the val split for REFUGE evaluation instead.")

        if len(label_dict) == 0:
            print(f"REFUGE '{self.split}': no labeled entries found, skipping.")
            return

        print(f"REFUGE '{self.split}' index.json loaded: {len(label_dict)} labeled entries")

        loaded_count = 0
        for filename in os.listdir(image_dir):
            name, ext = os.path.splitext(filename)
            if ext.lstrip('.').lower() not in valid_file_extensions:
                continue

            if filename not in label_dict:
                # Shouldn't normally happen if index.json is complete,
                # but worth warning about rather than silently skipping.
                print(f"Warning: {filename} has no entry in index.json, skipping.")
                continue

            self.image_paths.append(str(image_dir / filename))
            self.labels.append(label_dict[filename])
            self.sources.append("REFUGE")
            loaded_count += 1

        print(f"REFUGE '{self.split}' split: {loaded_count} images loaded.")


    # ------------------------------------------------------------------
    # Core Dataset interface — identical pattern to your DR class
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        # Use cached image if available (populated by cache_images_to_memory),
        # otherwise load from disk on the fly.
        if hasattr(self, 'cached_images') and self.cached_images[idx] is not None:
            image = self.cached_images[idx].copy()
        else:
            image = Image.open(self.image_paths[idx]).convert("RGB")

        if self.img_transform:
            image = self.img_transform(image)

        label = self.labels[idx]
        if self.label_transform:
            label = self.label_transform(label)

        return image, label


    # ------------------------------------------------------------------
    # Utility methods — kept identical to DR class for consistency
    # ------------------------------------------------------------------
    def prune_unlabeled(self):
        """Remove any samples that failed label matching (label is None)."""
        filtered = [
            (p, l, s)
            for p, l, s in zip(self.image_paths, self.labels, self.sources)
            if l is not None
        ]
        if not filtered:
            print("WARNING: prune_unlabeled removed ALL samples!")
            return
        self.image_paths, self.labels, self.sources = map(list, zip(*filtered))
        print(f"Pruned unlabeled samples — {len(self.labels)} samples remain.")

    def prune_corrupted_images(self, num_workers: int = 16):
        """Validate every image file and remove corrupted ones using threads."""
        print("Pruning corrupted images (threaded)...")
        valid_indices = []

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(_is_image_valid, path): idx
                for idx, path in enumerate(self.image_paths)
            }
            for future in as_completed(futures):
                idx = futures[future]
                if future.result():
                    valid_indices.append(idx)
                else:
                    print(f"Removing corrupted image: {self.image_paths[idx]}")

        self.image_paths = [self.image_paths[i] for i in valid_indices]
        self.labels      = [self.labels[i]      for i in valid_indices]
        self.sources     = [self.sources[i]     for i in valid_indices]
        print(f"Dataset size after corruption prune: {len(self.labels)}")

    def get_dataset_statistics(self) -> Dict:
        """Return a summary of class and source distribution."""
        label_names  = {0: "Healthy", 1: "Glaucoma"}
        raw_counts   = Counter(self.labels)
        named_counts = {label_names.get(k, k): v for k, v in raw_counts.items()}

        return {
            "total_images":       len(self.image_paths),
            "images_per_source":  dict(Counter(self.sources)),
            "label_distribution": named_counts,
            "split":              self.split,
        }

    def get_labels(self) -> list:
        return self.labels
