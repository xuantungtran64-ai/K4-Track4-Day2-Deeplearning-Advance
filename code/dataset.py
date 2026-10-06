"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader."""
from __future__ import annotations

import os
from pathlib import Path
from PIL import Image

import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T

NUM_CLASSES = 9
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_split(labels_dir: str | Path, fold: int = 0):
    labels_dir = Path(labels_dir)
    train_df = pd.read_csv(labels_dir / f"train_subset{fold}.csv")
    val_df = pd.read_csv(labels_dir / f"val_subset{fold}.csv")
    test_df = pd.read_csv(labels_dir / f"test_subset{fold}.csv")
    return train_df, val_df, test_df


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path) -> dict:
    images_dir = Path(images_dir)
    
    train_counts = len(train_df)
    val_counts = len(val_df)
    test_counts = len(test_df)
    
    train_files = set(train_df["Filename"])
    val_files = set(val_df["Filename"])
    test_files = set(test_df["Filename"])
    
    assert len(train_files.intersection(val_files)) == 0, "Train and Val overlap!"
    assert len(train_files.intersection(test_files)) == 0, "Train and Test overlap!"
    assert len(val_files.intersection(test_files)) == 0, "Val and Test overlap!"
    assert train_counts + val_counts + test_counts == 17509, "Total count must be 17509"
    
    if train_counts > 0: 
        assert (images_dir / train_df["Filename"].iloc[0]).exists(), "Image does not exist"

    return {
        "n": {"train": train_counts, "val": val_counts, "test": test_counts},
        "per_class": {
            "train": train_df["Label"].value_counts().to_dict(),
            "val": val_df["Label"].value_counts().to_dict(),
            "test": test_df["Label"].value_counts().to_dict()
        },
        "overlap": {"train_val": 0, "train_test": 0, "val_test": 0}
    }


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic"):
    if train:
        if aug == "basic":
            return T.Compose([
                T.RandomResizedCrop(img_size),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
        else:
            return T.Compose([
                T.RandomResizedCrop(img_size),
                T.RandomHorizontalFlip(),
                T.ColorJitter(0.2, 0.2, 0.2),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
    else:
        return T.Compose([
            T.Resize(256),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])


class DeepWeedsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None):
        self.df = df
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        filename = row["Filename"]
        label = int(row["Label"])
        
        img_path = self.images_dir / filename
        img = Image.open(img_path).convert("RGB")
        
        if self.transform:
            img = self.transform(img)
            
        return img, label, filename


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2):
    dataset = DeepWeedsDataset(df, images_dir, transform)
    
    if train and sampler == "balanced":
        class_counts = df["Label"].value_counts().sort_index().values
        weights = 1.0 / class_counts
        sample_weights = [weights[label] for label in df["Label"]]
        pt_sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        shuffle = False
    else:
        pt_sampler = None
        shuffle = train
        
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=pt_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=train
    )
