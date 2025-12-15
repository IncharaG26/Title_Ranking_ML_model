# src/preprocess.py
import re
import nltk
nltk.download('punkt', quiet=True)
from nltk.tokenize import word_tokenize

def simple_clean(text):
    if not isinstance(text, str):
        return ""
    text = text.replace("\n", " ").strip()
    # keep punctuation minimal (titles may contain special tokens)
    text = re.sub(r'\s+', ' ', text)
    return text

def title_length(title):
    return len(title.split())

def abstract_length(abstract):
    return len(abstract.split())

def word_overlap_ratio(title, abstract):
    ts = set(title.lower().split())
    as_ = set(abstract.lower().split())
    if not ts:
        return 0.0
    return len(ts & as_) / len(ts)

# small ROUGE-L (longest common subsequence ratio) approx
def rouge_l_score(s1, s2):
    # simple LCS dynamic programming
    a = s1.split()
    b = s2.split()
    n, m = len(a), len(b)
    if n == 0 or m == 0: return 0.0
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n-1, -1, -1):
        for j in range(m-1, -1, -1):
            if a[i] == b[j]:
                dp[i][j] = 1 + dp[i+1][j+1]
            else:
                dp[i][j] = max(dp[i+1][j], dp[i][j+1])
    lcs = dp[0][0]
    return lcs / n  # recall-based normalization
