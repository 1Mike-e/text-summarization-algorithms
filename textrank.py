import re
import math
from collections import Counter, defaultdict

def split_sentences(text):
    # Keeps end punctuation with sentence
    return [s.strip() for s in re.findall(r'[^.?!]+[.?!]', text) if s.strip()]

def tokenize(s):
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
    return kept_sentences, tokens_list

def build_tfidf_vectors(tokens_list):
    N = len(tokens_list)
    df = Counter()
    for toks in tokens_list:
        df.update(set(toks))
    idf = {t: math.log((N + 1) / (df[t] + 1)) for t in df}  # smoothed
    vectors = []
    for toks in tokens_list:
        tf = Counter(toks)
        vec = {t: tf[t] * idf.get(t, 0.0) for t in tf}
        vectors.append(vec)
    return vectors

def cosine_similarity(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum(vec1[x] * vec2[x] for x in intersection)
    sum1 = sum(v ** 2 for v in vec1.values())
    sum2 = sum(v ** 2 for v in vec2.values())
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    return numerator / denominator if denominator else 0.0

def build_similarity_graph(vectors, threshold=0.1):
    N = len(vectors)
    graph = defaultdict(list)
    for i in range(N):
        for j in range(i+1, N):
            sim = cosine_similarity(vectors[i], vectors[j])
            if sim >= threshold:
                graph[i].append((j, sim))
                graph[j].append((i, sim))
    return graph

def power_iteration(graph, N, d=0.85, max_iter=100, tol=1e-4):
    rank = [1.0 / N] * N
    prev = [1.0 / N] * N
    for it in range(max_iter):
        for i in range(N):
            acc = 0
            denom = sum(w for _, w in graph[i]) or 1e-6
            for j, w in graph[i]:
                denom_j = sum(w_ for _, w_ in graph[j]) or 1e-6
                acc += (w / denom_j) * prev[j]
            rank[i] = (1 - d) / N + d * acc
        if sum(abs(rank[i] - prev[i]) for i in range(N)) < tol:
            break
        prev = rank[:]
    return rank

def jaccard(a, b):
    A, B = set(a), set(b)
    if not A and not B:
        return 0.0
    return len(A & B) / len(A | B)

def summarize_textrank(
    text,
    K=3,
    stopwords=None,
    min_sentence_len=0,
    threshold=0.1,
    redundancy_lambda=0.0
):
    sentences, tokens_list = preprocess(text, stopwords, min_sentence_len)
    N = len(sentences)
    if N == 0:
        return ""
    K = max(1, min(K, N))
    vectors = build_tfidf_vectors(tokens_list)
    graph = build_similarity_graph(vectors, threshold)
    ranks = power_iteration(graph, N)
    # Candidates are those with highest rank (tie-break by order)
    candidates = sorted(range(N), key=lambda i: (-ranks[i], i))
    if redundancy_lambda <= 0 or K == 1:
        top = sorted(candidates[:K])
        return " ".join(sentences[i] for i in top)
    # Redundancy-aware selection (greedy, like in your TFIDF)
    selected = []
    pool = candidates[: min(len(candidates), 10 * K)]
    while len(selected) < K and pool:
        best_i = None
        best_val = float('-inf')
        for i in pool:
            if not selected:
                val = ranks[i]
            else:
                red = max(jaccard(tokens_list[i], tokens_list[j]) for j in selected)
                val = (1 - redundancy_lambda) * ranks[i] - redundancy_lambda * red
            if val > best_val or (val == best_val and (best_i is None or i < best_i)):
                best_i, best_val = i, val
        selected.append(best_i)
        pool.remove(best_i)
    selected.sort()
    return " ".join(sentences[i] for i in selected)

if __name__ == "__main__":
    stop = {'the','is','at','on','of','and','a','an','to','in'}

    text = """   Amazing news!!!     
    
    The team won the championship.         
    
       Wow!!!   """

    summary = summarize_textrank(
        text,
        K=1,
        stopwords=stop,
        min_sentence_len=3,    
        redundancy_lambda=0.0  # no need for redundancy with K=1
    )

    print(summary)

