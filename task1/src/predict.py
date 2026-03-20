"""
predict.py — Task 1: Fake News Detection in Malayalam
Loads a saved XLM-RoBERTa checkpoint and generates predictions on the test set.
Also exposes ``predict_single()`` for use by app.py.

Usage:
    cd task1/src
    python predict.py
"""

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from preprocess import clean_text


MODEL_PATH = "models/xlm-roberta-base"
DATA_PATH  = "data/test.csv"
BATCH_SIZE = 32


def _load_model():
    """Load the checkpoint from MODEL_PATH, move it to the available device, and set eval mode."""
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()
    return model, tokenizer, device


def _batch_predict(model, tokenizer, device, texts):
    """Run batched inference and return a list of integer label predictions.

    Args:
        model: Loaded ``AutoModelForSequenceClassification`` in eval mode.
        tokenizer: Matching tokenizer.
        device: ``torch.device`` the model is on.
        texts: List of preprocessed strings.

    Returns:
        List of ints (0 = Original, 1 = Fake), one per input string.
    """
    predictions = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch  = texts[i : i + BATCH_SIZE]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True,
                           padding=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=1).tolist()
        predictions.extend(preds)
    return predictions


def predict_single(text: str) -> str:
    """Return 'Fake' or 'Original' for a single input string. Used by app.py."""
    model, tokenizer, device = _load_model()
    label = _batch_predict(model, tokenizer, device, [clean_text(text)])[0]
    return "Fake" if label == 1 else "Original"


def main():
    """Generate predictions for every row in DATA_PATH and write a submission CSV."""
    MODEL_NAME  = MODEL_PATH.split("/")[-1]
    output_file = f"outputs/{MODEL_NAME}_submission.csv"

    print(f"Generating predictions using: {MODEL_PATH}")

    model, tokenizer, device = _load_model()

    test_df = pd.read_csv(DATA_PATH)
    test_df["text"] = test_df["text"].apply(clean_text)
    texts = test_df["text"].tolist()

    predictions = _batch_predict(model, tokenizer, device, texts)

    submission = pd.DataFrame({"label": predictions})
    submission.to_csv(output_file, index=False)
    print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()
