# somali_phonemizer.py
import re
import unicodedata
from typing import Optional

# Somali → IPA mapping
_SOMALI_MAP = {
    # punctuation (kept)
    ",": ",", ".": ".", "?": "?", "!": "!", ";": ";", ":": ":", "-": "-",

    # vowels
    "aa": "aː",
    "ee": "eː",
    "ii": "iː",
    "oo": "oː",
    "uu": "uː",
    "a": "a",
    "e": "e",
    "i": "i",
    "o": "o",
    "u": "u",

    # consonants
    "'": "ʔ",
    "’": "ʔ",
    "b": "b",
    "t": "t",
    "j": "d͡ʒ",
    "x": "ħ",
    "kh": "χ",
    "d": "d",
    "r": "r",
    "s": "s",
    "sh": "ʃ",
    "dh": "ɖ",
    "c": "ʕ",
    "g": "ɡ",
    "p": "p",
    "v": "v",
    "f": "f",
    "q": "q",
    "k": "k",
    "l": "l",
    "m": "m",
    "n": "n",
    "w": "w",
    "h": "h",
    "y": "j",
}

# Regex helpers
_SPLIT_TOKENS = re.compile(r"[\s]+")

# Optional number handling
_NUM_RX = re.compile(r"[0-9]+(\.[0-9]+)?")

def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

def _expand_numbers(text: str) -> str:
    # Placeholder: numbers kept as-is
    return text

def _g2p_segment(word: str) -> str:
    """Very simple grapheme to phoneme for a single word with IPA mapping."""
    phonemes = []
    i = 0
    while i < len(word):
        # check digraphs (2 letters like sh, dh, kh, aa, ee...)
        if i + 1 < len(word) and word[i:i+2] in _SOMALI_MAP:
            phonemes.append(_SOMALI_MAP[word[i:i+2]])
            i += 2
            continue
        # single char
        if word[i] in _SOMALI_MAP:
            phonemes.append(_SOMALI_MAP[word[i]])
        else:
            phonemes.append(word[i])
        i += 1
    return " ".join(phonemes)

def somali_text_to_phonemes(text: str) -> str:
    """Convert Somali text (Latin orthography) to IPA phonemes."""
    text = _normalize_text(text)
    text = _expand_numbers(text)
    tokens = []
    for token in re.split(r"(\s+|[,.!?;:-])", text):
        if token is None or token == "":
            continue
        if token.isspace():
            continue
        if token in ",.!?;:-":
            tokens.append(token)
            continue
        phon = _g2p_segment(token)
        tokens.append(phon)
    return " ".join(tokens).strip()
