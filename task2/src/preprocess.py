"""
preprocess.py — Task 2: Fake News Detection in Malayalam
Handles data loading, label encoding, and balanced sampling.
"""

import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# ── Config 
TRAIN_CSV      = "data/train_data_mal_fake_detect.csv"
OUTPUT_DIR     = "data/processed"
LABEL_ENCODER  = "data/processed/label_encoder.pkl"
TEST_SIZE      = 0.2
RANDOM_STATE   = 42
NUM_CLASSES    = 5
SAMPLE_SIZE    = 200  # samples per class; minority classes are oversampled with replacement



def load_data(path: str) -> pd.DataFrame:
    """Load CSV and return DataFrame."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")
    df = pd.read_csv(path)
    print(f"[preprocess] Loaded {len(df)} rows from {path}")
    return df


def encode_labels(train_df: pd.DataFrame, val_df: pd.DataFrame):
    """Fit LabelEncoder on training labels and transform both splits."""
    le = LabelEncoder()
    train_df = train_df.copy()
    val_df   = val_df.copy()

    train_df["Label"] = le.fit_transform(train_df["Label"])
    val_df["Label"]   = le.transform(val_df["Label"])

    print(f"[preprocess] Classes: {list(le.classes_)}")
    return train_df, val_df, le


def balance_sample(df: pd.DataFrame, num_classes: int, sample_size: int) -> pd.DataFrame:
    """
    Draw exactly `sample_size` examples per class to create a balanced training set.
    Classes with fewer than `sample_size` rows are oversampled with replacement so
    every class contributes equally regardless of the original imbalance
    (e.g., 1,246 False vs 1 Mostly True).
    """
    frames = []
    for label in range(num_classes):
        subset = df[df["Label"] == label]
        if len(subset) == 0:
            print(f"[preprocess] Warning: no samples for class {label}")
            continue
        replace = len(subset) < sample_size   # oversample minority classes
        frames.append(subset.sample(sample_size, replace=replace, random_state=RANDOM_STATE))
    balanced = pd.concat(frames).reset_index(drop=True)
    print(f"[preprocess] Balanced training set: {len(balanced)} rows")
    return balanced


def save_splits(train_df, val_df, le, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"),   index=False)
    with open(os.path.join(output_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)
    print(f"[preprocess] Saved processed splits and label encoder to {output_dir}/")


def main():
    # 1. Load
    df = load_data(TRAIN_CSV)

    # 2. Clean — ensure text column is string
    df["News"] = df["News"].astype(str)

    # 3. Train / val split — stratify to preserve class proportions in both sets.
    # Falls back to a plain random split if any class is too small to stratify
    # (sklearn requires at least 2 samples per class).
    try:
        train_df, val_df = train_test_split(
            df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df["Label"]
        )
    except ValueError as e:
        print(f"[preprocess] Warning: stratified split failed ({e}), falling back to random split")
        train_df, val_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    print(f"[preprocess] Train: {len(train_df)} | Val: {len(val_df)}")

    # 4. Label encode
    train_df, val_df, le = encode_labels(train_df, val_df)

    # 5. Balanced sampling for training
    train_df = balance_sample(train_df, NUM_CLASSES, SAMPLE_SIZE)

    # 6. Save
    save_splits(train_df, val_df, le, OUTPUT_DIR)
    print("[preprocess] Done.")


if __name__ == "__main__":
    main()
