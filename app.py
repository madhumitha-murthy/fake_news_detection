import os
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "task1/models/xlm-roberta-base"


@st.cache_resource
def load_model():
    if not os.path.isdir(MODEL_PATH):
        return None, None, None
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()
    return model, tokenizer, device


def predict(text: str, model, tokenizer, device) -> str:
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       padding=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    label_id = torch.argmax(logits, dim=1).item()
    return "Fake" if label_id == 1 else "Original"


st.title("Fake News Detection")

model, tokenizer, device = load_model()

if model is None:
    st.error(
        f"Model not found at '{MODEL_PATH}'. "
        "Train it first by running `python task1/src/train.py` from the project root."
    )
    st.stop()

text = st.text_area("Enter news article")

if st.button("Analyze"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        result = predict(text, model, tokenizer, device)
        st.success(f"Prediction: {result}")
