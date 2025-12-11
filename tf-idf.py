import re
import math
from collections import Counter

def split_sentences(text):
    # keeps end punctuation with each sentence
    return [s.strip() for s in re.findall(r'[^.?!]+[.?!]', text) if s.strip()]

def tokenize(s):
    # words only; keeps digits; drops apostrophes
    return re.findall(r'\b\w+\b', s.lower())

def preprocess(text, stopwords=None, min_sentence_len=0):
    stop = set(stopwords or [])
    sentences = split_sentences(text)
    tokens_list = []
    kept_sentences = []
    for s in sentences:
        toks = [w for w in tokenize(s) if w not in stop]
        if len(toks) >= min_sentence_len:
            kept_sentences.append(s)
            tokens_list.append(toks)
    return kept_sentences, tokens_list  # sentences with punctuation preserved

def compute_tfidf(tokens_list):
    N = len(tokens_list)
    if N == 0:
        return [], {}
    df = Counter()
    for toks in tokens_list:
        df.update(set(toks))
    idf = {t: math.log((N + 1) / (df[t] + 1)) for t in df}  # smoothed
    scores = []
    for toks in tokens_list:
        tf = Counter(toks)
        raw = sum(tf[t] * idf.get(t, 0.0) for t in tf)
        score = raw / max(1, len(toks))  # length normalization
        scores.append(score)
    return scores, idf

def jaccard(a, b):
    A, B = set(a), set(b)
    if not A and not B:
        return 0.0
    return len(A & B) / len(A | B)

def summarize_tfidf(
    text,
    K=3,
    stopwords=None,
    min_sentence_len=0,
    percent=None,
    redundancy_lambda=0.0  # 0 = off; e.g., 0.15 gives light penalty
):
    sentences, tokens_list = preprocess(text, stopwords, min_sentence_len)
    N = len(sentences)
    if N == 0:
        return ""
    if percent is not None:
        K = max(1, round((percent / 100.0) * N))
    K = max(1, min(K, N))

    scores, _ = compute_tfidf(tokens_list)

    # candidate order: score desc, then earlier index first (stable tie-break)
    candidates = sorted(range(N), key=lambda i: (-scores[i], i))

    if redundancy_lambda <= 0 or K == 1:
        top = sorted(candidates[:K])  # restore original order
        return " ".join(sentences[i] for i in top)

    # light redundancy-aware selection
    selected = []
    selected_tokens = []
    pool = candidates[: min(len(candidates), 10 * K)]
    while len(selected) < K and pool:
        best_i = None
        best_val = float("-inf")
        for i in pool:
            if not selected:
                val = scores[i]
            else:
                red = max(jaccard(tokens_list[i], tokens_list[j]) for j in selected)
                val = (1 - redundancy_lambda) * scores[i] - redundancy_lambda * red
            if val > best_val or (val == best_val and i < (best_i if best_i is not None else i+1)):
                best_i, best_val = i, val
        selected.append(best_i)
        selected_tokens.append(tokens_list[best_i])
        pool.remove(best_i)

    selected.sort()
    return " ".join(sentences[i] for i in selected)

if __name__ == "__main__":
    stop = {'the','is','at','on','of','and','a','an','to','in'}
    text = (
    "AI is changing many industries. "
    "Students study algorithms to solve complex problems. "
    "Traveling can broaden perspectives. "
    "Music helps people relax. "
    "Electric cars reduce pollution. "
    "Wildlife conservation is becoming more important. "
    "Space exploration teaches us about the universe. "
    "Healthy diets improve quality of life."
)

    summary1 = summarize_tfidf(
        text,
        K=3,
        stopwords=stop,
        min_sentence_len=3,
        redundancy_lambda=0.15
    )

    summary2 = summarize_tfidf(
        text,
        K=3,
        stopwords=stop,
        min_sentence_len=3,
        redundancy_lambda=0.15
    )

    print("Run 1:", summary1)
    print("Run 2:", summary2)
    print("Same output?", summary1 == summary2)


