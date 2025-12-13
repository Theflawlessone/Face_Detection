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

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)


class UTKFaceDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        img = Image.open(BASE_DIR / row["path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(row["age_group"], dtype=torch.long)


class AgeGroupResNet18(nn.Module):
    def __init__(self, num_age_groups=5):
        super().__init__()
        self.base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_feats = self.base.fc.in_features
        self.base.fc = nn.Linear(in_feats, num_age_groups)

    def forward(self, x):
        return self.base(x)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    df = pd.read_csv(DATA_DIR / "metadata.csv")
    df["path"] = df["path"].str.replace("\\", "/", regex=False)

    train_df, temp_df = train_test_split(
        df,
        test_size=0.3,
        random_state=SEED,
        stratify=df["age_group"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=SEED,
        stratify=temp_df["age_group"],
    )

    IMG_SIZE = 224
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_loader = DataLoader(
        UTKFaceDataset(train_df, train_tf),
        batch_size=64, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        UTKFaceDataset(val_df, val_tf),
        batch_size=64, shuffle=False, num_workers=2
    )

    model = AgeGroupResNet18().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-4, weight_decay=1e-4
    )

    best_val_acc = 0.0

    for epoch in range(1, 11):
        model.train()
        correct = total = 0
        running_loss = 0.0

        for imgs, y in tqdm(train_loader, desc=f"Epoch {epoch}"):
            imgs, y = imgs.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += imgs.size(0)

        train_acc = correct / total

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, y in val_loader:
                imgs, y = imgs.to(device), y.to(device)
                logits = model(imgs)
                correct += (logits.argmax(1) == y).sum().item()
                total += imgs.size(0)

        val_acc = correct / total
        print(f"Epoch {epoch}: train acc={train_acc:.4f}, val acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                model.state_dict(),
                MODEL_DIR / "best_classification_resnet18.pt",
            )
            print("Saved new best model")

    print("Best val acc:", best_val_acc)


if __name__ == "__main__":
    main()