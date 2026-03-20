"""
train.py — Task 1: Fake News Detection in Malayalam
Fine-tunes XLM-RoBERTa for binary fake news classification (Fake / Original)
using the HuggingFace Trainer API with automatic best-checkpoint selection.

Usage:
    cd task1/src
    python train.py
"""

import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from preprocess import clean_text


MODEL_NAME = "xlm-roberta-base"


def main():
    """Load data, fine-tune XLM-RoBERTa, and save the best checkpoint."""
    print(f"Training model: {MODEL_NAME}")

    # LOAD DATA
    train_df = pd.read_csv("data/train.csv")

    # LABEL MAPPING
    label_map = {"fake": 1, "original": 0}
    train_df["label"] = train_df["label"].astype(str).str.strip().str.lower().map(label_map)
    train_df = train_df.dropna(subset=["label"])
    train_df["label"] = train_df["label"].astype(int)

    print("Label distribution:")
    print(train_df["label"].value_counts())

    # CLEAN TEXT
    train_df["text"] = train_df["text"].apply(clean_text)

    # TRAIN / VALIDATION SPLIT
    # Held-out val set lets the Trainer track eval loss and pick the best checkpoint.
    train_split, val_split = train_test_split(
        train_df, test_size=0.1, random_state=42, stratify=train_df["label"]
    )
    print(f"Train: {len(train_split)} | Val: {len(val_split)}")

    train_dataset = Dataset.from_pandas(train_split.reset_index(drop=True))
    val_dataset   = Dataset.from_pandas(val_split.reset_index(drop=True))

    # TOKENIZER
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(example):
        return tokenizer(example["text"], padding=True, truncation=True, max_length=256)

    train_dataset = train_dataset.map(tokenize, batched=True)
    val_dataset   = val_dataset.map(tokenize, batched=True)

    cols = ["input_ids", "attention_mask", "label"]
    train_dataset.set_format(type="torch", columns=cols)
    val_dataset.set_format(type="torch", columns=cols)

    # MODEL
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    # TRAINING ARGS
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=50,
    )

    # TRAINER
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # TRAIN
    trainer.train()

    # SAVE BEST MODEL
    save_path = f"models/{MODEL_NAME.split('/')[-1]}"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Training complete! Model saved to {save_path}")


if __name__ == "__main__":
    main()
