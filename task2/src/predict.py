"""
predict.py — Task 2: Fake News Detection in Malayalam
Loads a saved checkpoint and runs inference on the test set.
Saves predictions to outputs/task2_{model}_predictions.csv.

Usage:
    python src/predict.py --model mbert
    python src/predict.py --model xlmr
"""

import os
import argparse
import pickle

import pandas as pd
import torch

from config import MODEL_MAP


# ── Config ────────────────────────────────────────────────────────────────────
TEST_CSV      = "data/test-data.csv"
PROCESSED_DIR = "data/processed"
MODEL_DIR     = "models"
OUTPUT_DIR    = "outputs"
MAX_LENGTH    = 256
BATCH_SIZE    = 32


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=MODEL_MAP.keys(), default="mbert",
        help="Which saved checkpoint to load (mbert | xlmr)"
    )
    parser.add_argument("--test_csv",   default=TEST_CSV)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    return parser.parse_args()


def load_in_batches(tokenizer, texts, batch_size):
    """Generator: yields tokenized batches so large test sets fit in memory."""
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        yield tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )


def predict(model, tokenizer, texts, device, batch_size):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for enc in load_in_batches(tokenizer, texts, batch_size):
            input_ids      = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds   = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
    return all_preds


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
    print(f"[predict] Device    : {device}")
    print(f"[predict] Checkpoint: {checkpoint}")

    le_path = os.path.join(PROCESSED_DIR, "label_encoder.pkl")
    with open(le_path, "rb") as f:
        le = pickle.load(f)
    print(f"[predict] Classes: {list(le.classes_)}")

    tokenizer = config["tokenizer"].from_pretrained(checkpoint)
    model     = config["model"].from_pretrained(checkpoint)
    model.to(device)

    if not os.path.exists(args.test_csv):
        raise FileNotFoundError(f"Test CSV not found: {args.test_csv}")

    test_df         = pd.read_csv(args.test_csv)
    test_df["News"] = test_df["News"].astype(str)
    texts           = test_df["News"].tolist()
    print(f"[predict] Running inference on {len(texts)} samples …")

    numeric_preds        = predict(model, tokenizer, texts, device, args.batch_size)
    test_df["predicted"] = le.inverse_transform(numeric_preds)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"task2_{args.model}_predictions.csv")
    test_df.to_csv(out_path, index=False)
    print(f"[predict] Predictions saved to {out_path}")

    print("\n[predict] Prediction distribution:")
    print(test_df["predicted"].value_counts().to_string())


if __name__ == "__main__":
    main()
