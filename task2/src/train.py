"""
train.py — Task 2: Fake News Detection in Malayalam
Fine-tunes a transformer model (mBERT / XLM-RoBERTa) for 5-class
fake news classification and saves the best checkpoint.

Usage:
    python src/train.py --model mbert         # Multilingual BERT (default)
    python src/train.py --model xlmr          # XLM-RoBERTa
"""

import os
import argparse
import pickle

import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report

from config import MODEL_MAP


# ── Config ────────────────────────────────────────────────────────────────────
PROCESSED_DIR = "data/processed"
MODEL_DIR     = "models"
MAX_LENGTH    = 256
BATCH_SIZE    = 16
EPOCHS        = 3
LEARNING_RATE = 1e-5
NUM_LABELS    = 5


def parse_args():
    """Parse and return command-line arguments for model selection and hyperparameters."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=MODEL_MAP.keys(), default="mbert",
        help="Which transformer to fine-tune (mbert | xlmr)"
    )
    parser.add_argument("--epochs",     type=int,   default=EPOCHS)
    parser.add_argument("--batch_size", type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=LEARNING_RATE)
    return parser.parse_args()


def load_processed_data():
    """Load pre-processed train/val splits and the fitted LabelEncoder from disk."""
    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    val_df   = pd.read_csv(os.path.join(PROCESSED_DIR, "val.csv"))
    with open(os.path.join(PROCESSED_DIR, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)
    return train_df, val_df, le


def tokenize(tokenizer, texts, labels):
    """Tokenize texts and return a TensorDataset."""
    texts = [str(t) for t in texts]
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    label_tensor = torch.tensor(labels, dtype=torch.long)
    return TensorDataset(
        encodings["input_ids"],
        encodings["attention_mask"],
        label_tensor,
    )


def train_epoch(model, loader, optimizer, device):
    """Run one full training epoch and return the average cross-entropy loss.

    Args:
        model: HuggingFace sequence classification model in train mode.
        loader: DataLoader over the training TensorDataset.
        optimizer: AdamW optimiser.
        device: ``torch.device`` the model and tensors are on.

    Returns:
        Average loss (float) across all batches in the epoch.
    """
    model.train()
    total_loss = 0.0
    for batch in loader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        outputs.loss.backward()
        optimizer.step()
        total_loss += outputs.loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, le, device):
    """Run inference on a DataLoader and print a per-class classification report.

    Args:
        model: HuggingFace sequence classification model.
        loader: DataLoader over the validation TensorDataset.
        le: Fitted ``sklearn.LabelEncoder`` used to recover original class names.
        device: ``torch.device`` the model and tensors are on.

    Returns:
        Tuple of ``(pred_labels, true_labels)`` as numpy arrays of string class names.
    """
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    pred_labels = le.inverse_transform(all_preds)
    true_labels = le.inverse_transform(all_labels)
    print(classification_report(true_labels, pred_labels, zero_division=1))
    return pred_labels, true_labels


def main():
    """Orchestrate data loading, model training, validation, and checkpoint saving."""
    args   = parse_args()
    config = MODEL_MAP[args.model]

    print(f"[train] Model : {config['name']}")
    print(f"[train] Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")

    train_df, val_df, le = load_processed_data()

    # do_lower_case is a BERT-specific parameter and is not passed here;
    # XLM-RoBERTa uses SentencePiece and ignores it anyway.
    tokenizer = config["tokenizer"].from_pretrained(config["name"])
    model     = config["model"].from_pretrained(config["name"], num_labels=NUM_LABELS)
    model.to(device)

    train_ds = tokenize(tokenizer, train_df["News"].tolist(), train_df["Label"].tolist())
    val_ds   = tokenize(tokenizer, val_df["News"].tolist(),   val_df["Label"].tolist())

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size)

    optimizer = AdamW(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        avg_loss = train_epoch(model, train_loader, optimizer, device)
        print(f"\n[train] Epoch {epoch}/{args.epochs} — Loss: {avg_loss:.4f}")
        print(f"[train] Validation results (epoch {epoch}):")
        evaluate(model, val_loader, le, device)

    os.makedirs(MODEL_DIR, exist_ok=True)
    save_path = os.path.join(MODEL_DIR, f"task2_{args.model}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"\n[train] Model saved to {save_path}/")


if __name__ == "__main__":
    main()
