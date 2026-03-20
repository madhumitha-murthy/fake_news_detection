"""
preprocess.py — Task 1: Fake News Detection in Malayalam
Text cleaning utilities applied to both training and inference inputs.
"""

import re


def clean_text(text: str) -> str:
    """Clean a raw Malayalam social media string for model input.

    Removes URLs, @mentions, #hashtags, and non-word characters, then
    lowercases the result. The regex ``\\w`` matches Unicode word characters,
    so Malayalam script is preserved while punctuation is stripped.

    Args:
        text: Raw input string (may be non-string; cast to str internally).

    Returns:
        Cleaned, lowercased string.
    """
    text = str(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[@#]\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.lower()
