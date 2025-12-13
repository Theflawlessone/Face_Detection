import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from facenet_pytorch import MTCNN

from src.paths import BASE_DIR, RAW_DIR, CROPPED_DIR, DATA_DIR


def parse_label(path: Path):
    """
    Parse UTKFace filename: age_gender_race_*.jpg
    gender: 0 = male, 1 = female
    race: 0 = White, 1 = Black, 2 = Asian, 3 = Indian, 4 = Others
    """
    name = path.name
    match = re.match(r"(\d+)_(\d+)_(\d+)_", name)
    if not match:
        return None, None, None
    age = int(match.group(1))
    gender = int(match.group(2))
    race = int(match.group(3))
    return age, gender, race


def build_df_raw():
    """
    Scan RAW_DIR for .jpg files and build df_raw with:
        raw_path (relative to BASE_DIR), age, gender, race
    """
    image_paths = sorted(RAW_DIR.rglob("*.jpg"))
    print(f"Found {len(image_paths)} raw images under {RAW_DIR}")

    records = []
    for p in image_paths:
        age, gender, race = parse_label(p)
        if age is None:
            continue
        rel_path = p.relative_to(BASE_DIR)
        records.append(
            [
                str(rel_path).replace("\\", "/"),
                age,
                gender,
                race,
            ]
        )

    df_raw = pd.DataFrame(records, columns=["raw_path", "age", "gender", "race"])
    print("df_raw rows:", len(df_raw))
    return df_raw


def build_cropped_dataframe(df_raw: pd.DataFrame, mtcnn: MTCNN):
    """
    Given df_raw with 'raw_path', crop faces into CROPPED_DIR if missing,
    and build df_cropped with 'path' (relative to BASE_DIR) + labels.

    Reuses existing crops if they already exist.
    """
    CROPPED_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    failed = []

    for _, row in tqdm(df_raw.iterrows(), total=len(df_raw), desc="Cropping / Reusing"):
        raw_rel = Path(row["raw_path"])
        raw_abs = BASE_DIR / raw_rel

        filename = raw_abs.name
        cropped_abs = CROPPED_DIR / filename

        # 1) If cropped already exists -> reuse
        if cropped_abs.is_file():
            rec = row.to_dict()
            rec["path"] = str(cropped_abs.relative_to(BASE_DIR)).replace("\\", "/")
            records.append(rec)
            continue

        # 2) Crop from raw image
        if not raw_abs.is_file():
            failed.append(str(raw_abs))
            continue

        try:
            img = Image.open(raw_abs).convert("RGB")
        except Exception:
            failed.append(str(raw_abs))
            continue

        try:
            with torch.no_grad():
                face_tensor = mtcnn(img)
        except Exception:
            failed.append(str(raw_abs))
            continue

        if face_tensor is None:
            failed.append(str(raw_abs))
            continue

        # tensor -> uint8 PIL
        face_tensor = face_tensor.clamp(0, 255)
        np_img = face_tensor.permute(1, 2, 0).cpu().numpy().astype("uint8")
        cropped_img = Image.fromarray(np_img)
        cropped_img.save(cropped_abs)

        rec = row.to_dict()
        rec["path"] = str(cropped_abs.relative_to(BASE_DIR)).replace("\\", "/")
        records.append(rec)

    df_cropped = pd.DataFrame(records)
    print(f"\nTotal raw rows: {len(df_raw)}")
    print(f"Total cropped rows: {len(df_cropped)}")
    print(f"Failed crops: {len(failed)}")
    return df_cropped, failed


def main():
    print("BASE_DIR:", BASE_DIR)
    print("RAW_DIR:", RAW_DIR)
    print("CROPPED_DIR:", CROPPED_DIR)

    # 0) Make sure there are images
    if not RAW_DIR.exists():
        print("RAW_DIR does not exist. Expected raw images under:", RAW_DIR)
        return

    # 1) Build df_raw
    df_raw = build_df_raw()
    if df_raw.empty:
        print("No valid UTKFace images found under", RAW_DIR)
        return

    # 2) Create MTCNN
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device for MTCNN:", device)
    mtcnn = MTCNN(image_size=224, margin=20, device=device)

    # 3) Build df_cropped via cropping / reuse
    df_cropped, failed = build_cropped_dataframe(df_raw, mtcnn=mtcnn)

    # 4) Drop raw_path; normalize path column
    if "raw_path" in df_cropped.columns:
        df_cropped = df_cropped.drop(columns=["raw_path"])

    df_cropped["path"] = df_cropped["path"].astype(str).str.replace("\\", "/", regex=False)

    # 5) Add convenience label columns
    gender_map = {0: "Male", 1: "Female"}
    race_map = {0: "White", 1: "Black", 2: "Asian", 3: "Indian", 4: "Other"}

    df_cropped["gender_str"] = df_cropped["gender"].map(gender_map)
    df_cropped["race_str"] = df_cropped["race"].map(race_map)

    bins = [0, 12, 19, 39, 59, 120]
    labels = [
        "Child (0–12)",
        "Teen (13–19)",
        "Young Adult (20–39)",
        "Middle-Aged (40–59)",
        "Senior (60+)",
    ]
    df_cropped["age_group_str"] = pd.cut(
        df_cropped["age"],
        bins=bins,
        labels=labels,
        right=True,
        include_lowest=True,
    )

    # 6) Save metadata.csv
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    metadata_path = DATA_DIR / "metadata.csv"
    df_cropped.to_csv(metadata_path, index=False)
    print("\nSaved metadata to:", metadata_path)

    # 7) Write failed crops list
    if failed:
        failed_path = DATA_DIR / "failed_crops.txt"
        with open(failed_path, "w") as f:
            for p in failed:
                f.write(p + "\n")
        print(f"Wrote {len(failed)} failed raw paths to:", failed_path)

    # 8) Quick summary
    print("\nSummary:")
    print(df_cropped[["gender_str"]].value_counts().rename("count"))
    print()
    print(df_cropped[["race_str"]].value_counts().rename("count"))
    print()
    print(df_cropped[["age_group_str"]].value_counts().rename("count"))


if __name__ == "__main__":
    main()