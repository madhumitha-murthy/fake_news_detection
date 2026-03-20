"""
config.py — Shared model registry for Task 2.
Centralises the MODEL_MAP so train.py, evaluate.py, and predict.py
all stay in sync without copy-pasting the same dict.
"""

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    XLMRobertaTokenizer,
    XLMRobertaForSequenceClassification,
)

# "bert-base-uncased" was removed: it is English-only and produces near-random
# embeddings for Malayalam text.  Use "mbert" (multilingual BERT) or "xlmr".
MODEL_MAP = {
    "mbert": {
        "tokenizer": BertTokenizer,
        "model":     BertForSequenceClassification,
        "name":      "bert-base-multilingual-uncased",
    },
    "xlmr": {
        "tokenizer": XLMRobertaTokenizer,
        "model":     XLMRobertaForSequenceClassification,
        "name":      "xlm-roberta-base",
    },
}
