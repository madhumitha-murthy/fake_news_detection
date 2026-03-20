"""
evaluate.py — Task 2: Fake News Detection in Malayalam
Generates classification report, confusion matrix heatmap, and
F1/precision/recall heatmap from validation predictions.

Usage:
    python src/evaluate.py --model mbert
    python src/evaluate.py --model xlmr
"""

import os
import argparse
import pickle

import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for servers and scripts
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from config import MODEL_MAP


# ── Config ────────────────────────────────────────────────────────────────────
PROCESSED_DIR = "data/processed"
MODEL_DIR     = "models"
RESULTS_DIR   = "results"
MAX_LENGTH    = 256
BATCH_SIZE    = 32


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=MODEL_MAP.keys(), default="mbert",
        help="Which checkpoint to evaluate (mbert | xlmr)"
    )
    return parser.parse_args()


def get_predictions(model, tokenizer, texts, device):
    model.eval()
    all_preds = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        enc   = tokenizer(
            batch, truncation=True, padding=True,
            max_length=MAX_LENGTH, return_tensors="pt"
        )
        input_ids      = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
    return all_preds


def plot_classification_heatmap(true_labels, pred_labels, model_key, save_dir):
    report    = classification_report(true_labels, pred_labels,
                                      zero_division=1, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    plt.figure(figsize=(10, 6))
    sns.heatmap(report_df.iloc[:-1, :-1], annot=True, fmt=".2f",
                cmap="YlOrBr", cbar=True)
    plt.title(f"Validation Classification Report — {model_key.upper()}")
    plt.xlabel("Metrics")
    plt.ylabel("Class Labels")
    plt.tight_layout()
    out = os.path.join(save_dir, f"task2_{model_key}_classification_report.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[evaluate] Saved classification heatmap → {out}")


def plot_confusion_matrix(true_labels, pred_labels, class_names, model_key, save_dir):
    cm   = confusion_matrix(true_labels, pred_labels, labels=class_names)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=True)
    ax.set_title(f"Confusion Matrix — {model_key.upper()}")
    plt.tight_layout()
    out = os.path.join(save_dir, f"task2_{model_key}_confusion_matrix.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[evaluate] Saved confusion matrix → {out}")


def main():
    args       = parse_args()
    checkpoint = os.path.join(MODEL_DIR, f"task2_{args.model}")
    config     = MODEL_MAP[args.model]

    if not os.path.isdir(checkpoint):
        raise FileNotFoundError(
            f"No checkpoint found at '{checkpoint}'. "
            f"Run train.py --model {args.model} first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[evaluate] Device    : {device}")
    print(f"[evaluate] Checkpoint: {checkpoint}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(os.path.join(PROCESSED_DIR, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)

    tokenizer = config["tokenizer"].from_pretrained(checkpoint)
    model     = config["model"].from_pretrained(checkpoint)
    model.to(device)

    val_df         = pd.read_csv(os.path.join(PROCESSED_DIR, "val.csv"))
    val_df["News"] = val_df["News"].astype(str)
    texts          = val_df["News"].tolist()
    true_numeric   = val_df["Label"].tolist()

    pred_numeric = get_predictions(model, tokenizer, texts, device)
    true_labels  = le.inverse_transform(true_numeric)
    pred_labels  = le.inverse_transform(pred_numeric)

    print(f"\n[evaluate] Classification Report — {args.model.upper()}")
    print(classification_report(true_labels, pred_labels, zero_division=1))

    plot_classification_heatmap(true_labels, pred_labels, args.model, RESULTS_DIR)
    plot_confusion_matrix(true_labels, pred_labels, list(le.classes_),
                          args.model, RESULTS_DIR)


if __name__ == "__main__":
    main()
