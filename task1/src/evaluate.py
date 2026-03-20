"""
evaluate.py — Task 1: Fake News Detection in Malayalam
Loads a saved XLM-RoBERTa checkpoint and prints a classification report
(precision, recall, macro F1) on the development set.

Usage:
    cd task1/src
    python evaluate.py
"""

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report
from preprocess import clean_text


MODEL_PATH = "models/xlm-roberta-base"
DATA_PATH  = "data/dev.csv"
BATCH_SIZE = 32


def main():
    """Run batched inference on the dev set and print the classification report."""
    print(f"Evaluating model: {MODEL_PATH}")

    # DEVICE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # LOAD MODEL
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()

    # LOAD DATA
    df = pd.read_csv(DATA_PATH)
    label_map = {"fake": 1, "original": 0}
    df["label"] = df["label"].astype(str).str.strip().str.lower().map(label_map)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df["text"]  = df["text"].apply(clean_text)

    # BATCHED PREDICTION
    # Processing one sample at a time is ~30x slower; batching amortises
    # tokeniser overhead and fills GPU memory efficiently.
    predictions = []
    texts = df["text"].tolist()

    for i in range(0, len(texts), BATCH_SIZE):
        batch  = texts[i : i + BATCH_SIZE]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True,
                           padding=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=1).tolist()
        predictions.extend(preds)

    # RESULTS
    print(classification_report(df["label"], predictions,
                                target_names=["original", "fake"]))


if __name__ == "__main__":
    main()
