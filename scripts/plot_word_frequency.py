import json
from collections import Counter
from pathlib import Path
import re

import matplotlib.pyplot as plt

CAPTIONS_JSON = Path("data/processed/captions.json")

# words you probably want to skip because they come from templates
STOPWORDS = {
    "<bos>", "<eos>", "<pad>",
    "in", "the", "a", "an", "scene", "image", "are", "is"
}

def tokenize(text: str):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)   # remove punctuation
    tokens = text.split()
    return tokens

def collect_word_counts():
    with CAPTIONS_JSON.open("r") as f:
        captions = json.load(f)

    counts = Counter()
    for cap in captions.values():
        for w in tokenize(cap):
            if w not in STOPWORDS:
                counts[w] += 1
    return counts

def plot_top_words(counts: Counter, k: int, out_path: Path):
    most_common = counts.most_common(k)
    labels = [w for w, _ in most_common]
    values = [c for _, c in most_common]

    plt.figure(figsize=(10, 4))
    plt.bar(labels, values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Frequency")
    plt.title(f"Top {k} words in synthetic captions")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved word-frequency figure to {out_path}")


if __name__ == "__main__":
    counts = collect_word_counts()
    print("Top 20 words:", counts.most_common(20))

    out_path = Path("figures/word_frequency_top20.png")
    out_path.parent.mkdir(exist_ok=True)
    plot_top_words(counts, k=20, out_path=out_path)
