import unicodedata

def damerau_levenshtein(s1: str, s2: str) -> int:
    s1 = s1.lower()
    s2 = s2.lower()
    len1, len2 = len(s1), len(s2)

    if len1 == 0:
        return len2
    if len2 == 0:
        return len1

    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,        # deletion
                dp[i][j - 1] + 1,        # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )
            # Transposition
            if i > 1 and j > 1 and s1[i - 1] == s2[j - 2] and s1[i - 2] == s2[j - 1]:
                dp[i][j] = min(dp[i][j], dp[i - 2][j - 2] + cost)

    return dp[len1][len2]


def similarity_score(s1: str, s2: str) -> float:
    """Normalisasi DL distance menjadi skor similarity [0, 1]."""
    dist = damerau_levenshtein(s1, s2)
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    return 1.0 - dist / max_len


def _edit_distance(ref: list, hyp: list) -> int:
    r, h = len(ref), len(hyp)
    dp = [[0] * (h + 1) for _ in range(r + 1)]
    for i in range(r + 1):
        dp[i][0] = i
    for j in range(h + 1):
        dp[0][j] = j
    for i in range(1, r + 1):
        for j in range(1, h + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[r][h]


def compute_wer(reference: str, hypothesis: str) -> float:
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
    return _edit_distance(ref_words, hyp_words) / len(ref_words)


def compute_cer(reference: str, hypothesis: str) -> float:
    ref_chars = list(reference.lower())
    hyp_chars = list(hypothesis.lower())
    if len(ref_chars) == 0:
        return 0.0 if len(hyp_chars) == 0 else 1.0
    return _edit_distance(ref_chars, hyp_chars) / len(ref_chars)


def normalize(text: str) -> str:
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    return text.strip().lower()


def find_best_ocr_token(ocr_text: str, true_label: str) -> str:
    if not ocr_text or not ocr_text.strip():
        return ""
    tokens = [t.strip() for t in ocr_text.split('|') if t.strip()]
    if not tokens:
        return ""
    ref = true_label.lower()
    best_token = tokens[0]
    best_dist = _edit_distance(list(ref), list(tokens[0].lower()))
    for token in tokens[1:]:
        dist = _edit_distance(list(ref), list(token.lower()))
        if dist < best_dist:
            best_dist = dist
            best_token = token
    return best_token
