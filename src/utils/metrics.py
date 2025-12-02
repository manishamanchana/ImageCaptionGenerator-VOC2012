import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import re

# Smoothing function for BLEU
_smooth = SmoothingFunction().method4

def normalize_caption(text: str) -> str:    #text normalization
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)   # remove punctuation
    text = re.sub(r"\s+", " ", text)      # remove multiple spaces
    return text


def tokenize_caption(text: str):
    return normalize_caption(text).split()

# BLEU

def compute_bleu1(pred: str, ref: str) -> float:
    pred_tokens = tokenize_caption(pred)
    ref_tokens = tokenize_caption(ref)

    if not pred_tokens or not ref_tokens:
        return 0.0

    return sentence_bleu(
        [ref_tokens],
        pred_tokens,
        weights=(1.0, 0, 0, 0),
        smoothing_function=_smooth
    )


def compute_bleu4(pred: str, ref: str) -> float:
    pred_tokens = tokenize_caption(pred)
    ref_tokens = tokenize_caption(ref)

    if not pred_tokens or not ref_tokens:
        return 0.0

    return sentence_bleu(
        [ref_tokens],
        pred_tokens,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=_smooth
    )

# METEOR
def compute_meteor(pred: str, ref: str) -> float:
    return meteor_score([normalize_caption(ref)], normalize_caption(pred))

# ROUGE-L
def lcs(a, b):
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m):
        for j in range(n):
            if a[i] == b[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])

    return dp[m][n]


def compute_rouge_l(pred: str, ref: str) -> float:
    pred_tokens = tokenize_caption(pred)
    ref_tokens = tokenize_caption(ref)

    if not pred_tokens or not ref_tokens:
        return 0.0

    lcs_len = lcs(pred_tokens, ref_tokens)
    recall = lcs_len / len(ref_tokens)
    precision = lcs_len / len(pred_tokens)

    if recall + precision == 0:
        return 0.0

    f_score = (2 * recall * precision) / (recall + precision)
    return f_score


# ALL METRICS (FOR ONE PAIR)
def compute_all_metrics(pred: str, ref: str):
    return {
        "BLEU-1": compute_bleu1(pred, ref),
        "BLEU-4": compute_bleu4(pred, ref),
        "METEOR": compute_meteor(pred, ref),
        "ROUGE-L": compute_rouge_l(pred, ref),
    }
