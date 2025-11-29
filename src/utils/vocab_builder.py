import json
from collections import Counter
from typing import Dict

from nltk.tokenize import word_tokenize

from src.config import CAPTIONS_JSON, VOCAB_JSON

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


def ensure_nltk():
    import nltk
    for resource in ["punkt", "punkt_tab"]:
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource)



def build_vocab(min_freq: int = 1) -> Dict[str, int]:
    if not CAPTIONS_JSON.exists():
        raise FileNotFoundError(f"{CAPTIONS_JSON} not found. Run caption_generator first.")

    with CAPTIONS_JSON.open("r") as f:
        captions = json.load(f)

    counter = Counter()
    for cap in captions.values():
        tokens = word_tokenize(cap.lower())
        counter.update(tokens)

    word2idx: Dict[str, int] = {}
    for i, tok in enumerate(SPECIAL_TOKENS):
        word2idx[tok] = i

    idx = len(SPECIAL_TOKENS)
    for word, freq in counter.items():
        if freq >= min_freq and word not in word2idx:
            word2idx[word] = idx
            idx += 1

    with VOCAB_JSON.open("w") as f:
        json.dump(word2idx, f, indent=2)

    print(f"Saved vocab of size {len(word2idx)} to {VOCAB_JSON}")
    return word2idx


if __name__ == "__main__":
    ensure_nltk()
    build_vocab(min_freq=1)
