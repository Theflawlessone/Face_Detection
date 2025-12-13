from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split

from src.paths import BASE_DIR, DATA_DIR, MODEL_DIR


# Config
@dataclass
class Config:
    seed: int = 42
    batch_size: int = 32
    epochs: int = 10
    lr: float = 1e-4
    num_workers: int = 2
    image_size: int = 224
    best_name: str = "best_age_resnet18.pt"
    split_pickle: str = "splits_age_seed42.pkl"


# Dataset
class UTKFaceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, base_dir: Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.base_dir = Path(base_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rel_path = str(row["path"]).replace("\\", "/")
        img_path = self.base_dir / rel_path

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        age = torch.tensor(row["age"], dtype=torch.float32)
        return img, age


# Helpers
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_model(device: torch.device):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)  # regression head
    return model.to(device)


def train_one_epoch(model, loader, optimizer, device, loss_fn):
    model.train()
    running_loss = 0.0
    running_mae = 0.0

    for x, y in tqdm(loader, desc="Train", leave=False):
        x = x.to(device)
        y = y.to(device).unsqueeze(1)  # (B,) -> (B,1)

        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        running_mae += torch.abs(pred - y).sum().item()

    n = len(loader.dataset)
    return running_loss / n, running_mae / n


@torch.inference_mode()
def eval_one_epoch(model, loader, device, loss_fn):
    model.eval()
    running_loss = 0.0
    running_mae = 0.0

    for x, y in tqdm(loader, desc="Val", leave=False):
        x = x.to(device)
        y = y.to(device).unsqueeze(1)

        pred = model(x)
        loss = loss_fn(pred, y)

        running_loss += loss.item() * x.size(0)
        running_mae += torch.abs(pred - y).sum().item()

    n = len(loader.dataset)
    return running_loss / n, running_mae / n


def main():
    cfg = Config()
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    metadata_path = DATA_DIR / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.csv not found at {metadata_path}")

    df = pd.read_csv(metadata_path)

    # Normalize paths
    df["path"] = df["path"].astype(str).str.replace("\\", "/", regex=False)

    # Stratify
    if "age_group_str" in df.columns:
        strat = df["age_group_str"]
    else:
        strat = pd.cut(df["age"], bins=[0, 12, 19, 39, 59, 120])

    train_df, temp_df = train_test_split(
        df, test_size=0.3, random_state=cfg.seed, stratify=strat
    )

    if "age_group_str" in temp_df.columns:
        strat2 = temp_df["age_group_str"]
    else:
        strat2 = pd.cut(temp_df["age"], bins=[0, 12, 19, 39, 59, 120])

    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=cfg.seed, stratify=strat2
    )

    print(f"Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    train_tf = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_ds = UTKFaceDataset(train_df, base_dir=BASE_DIR, transform=train_tf)
    val_ds = UTKFaceDataset(val_df, base_dir=BASE_DIR, transform=eval_tf)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True
    )

    model = make_model(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    best_val_mae = float("inf")
    best_path = MODEL_DIR / cfg.best_name
    history = []

    for epoch in range(1, cfg.epochs + 1):
        print(f"\nEpoch {epoch}/{cfg.epochs}")

        tr_loss, tr_mae = train_one_epoch(model, train_loader, optimizer, device, loss_fn)
        va_loss, va_mae = eval_one_epoch(model, val_loader, device, loss_fn)

        print(f"Train  loss={tr_loss:.4f}  MAE={tr_mae:.4f}")
        print(f"Val    loss={va_loss:.4f}  MAE={va_mae:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_mae": tr_mae,
            "val_loss": va_loss,
            "val_mae": va_mae,
        })

        if va_mae < best_val_mae:
            best_val_mae = va_mae
            torch.save(model.state_dict(), best_path)
            print(f"Saved new best model -> {best_path} (val MAE={best_val_mae:.4f})")

    # Save training log
    log_path = MODEL_DIR / "age_training_log.csv"
    pd.DataFrame(history).to_csv(log_path, index=False)
    print("\nSaved training log to:", log_path)
    print("Best val MAE:", best_val_mae)
    print("Best model:", best_path)


if __name__ == "__main__":
    main()
