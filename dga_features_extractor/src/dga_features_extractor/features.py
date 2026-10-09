from __future__ import annotations

import math
from collections import Counter
from types import MappingProxyType
from typing import Dict, Iterable, List, Set

from .feature_registry import FeatureContext, FeatureRegistry, default_feature_registry

VOWELS = set("aeiou")
LETTERS = set("abcdefghijklmnopqrstuvwxyz")


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def _mean(xs: List[float]) -> float:
    return _safe_div(sum(xs), len(xs))


def _qmean(xs: List[float]) -> float:
    return math.sqrt(_safe_div(sum(x * x for x in xs), len(xs)))


def _var_sample(xs: List[float]) -> float:
    n = len(xs)
    if n <= 1:
        return 0.0
    m = _mean(xs)
    return sum((x - m) ** 2 for x in xs) / float(n - 1)


def _var_pop(xs: List[float]) -> float:
    n = len(xs)
    if n == 0:
        return 0.0
    m = _mean(xs)
    return sum((x - m) ** 2 for x in xs) / float(n)


def _std_sample(xs: List[float]) -> float:
    return math.sqrt(_var_sample(xs))


def _std_pop(xs: List[float]) -> float:
    return math.sqrt(_var_pop(xs))


def _skewness(xs: List[float]) -> float:
    n = len(xs)
    if n == 0:
        return 0.0
    m = _mean(xs)
    s = _std_pop(xs)
    if s == 0.0:
        return 0.0
    return _safe_div(sum((x - m) ** 3 for x in xs), n * (s ** 3))


def _kurtosis(xs: List[float]) -> float:
    n = len(xs)
    if n == 0:
        return 0.0
    m = _mean(xs)
    s = _std_pop(xs)
    if s == 0.0:
        return 0.0
    return _safe_div(sum((x - m) ** 4 for x in xs), n * (s ** 4)) - 3.0


def _jarque_bera(xs: List[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    s = _skewness(xs)
    k_raw = _kurtosis(xs) + 3.0
    return (n / 6.0) * (s ** 2) + (n / 24.0) * ((k_raw - 3.0) ** 2)


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    pos = q * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(xs[lo])
    w = pos - lo
    return float(xs[lo] * (1.0 - w) + xs[hi] * w)


def _entropy_from_probs(ps: Iterable[float]) -> float:
    filtered = [p for p in ps if p > 0.0]
    if not filtered:
        return 0.0
    return -sum(p * math.log(p, 2) for p in filtered)


def _shannon_entropy_chars(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    total = float(len(s))
    entropy = 0.0
    for n in counts.values():
        p = n / total
        entropy -= p * (math.log(p, 2) if p > 0 else 0.0)
    return entropy


def _max_run_same_char(s: str) -> int:
    if not s:
        return 0
    best = 1
    cur = 1
    prev = s[0]
    for ch in s[1:]:
        if ch == prev:
            cur += 1
            best = max(best, cur)
        else:
            prev = ch
            cur = 1
    return best


def _rankdata(a: List[float]) -> List[float]:
    n = len(a)
    order = sorted(range(n), key=lambda i: a[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and a[order[j + 1]] == a[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(x: List[float], y: List[float]) -> float:
    n = len(x)
    if n == 0:
        return 0.0
    mx, my = _mean(x), _mean(y)
    vx, vy = _var_pop(x), _var_pop(y)
    if vx == 0.0 or vy == 0.0:
        return 0.0
    cov = _safe_div(sum((a - mx) * (b - my) for a, b in zip(x, y)), n)
    return cov / math.sqrt(vx * vy)


def _spearman(x: List[float], y: List[float]) -> float:
    if not x:
        return 0.0
    return _pearson(_rankdata(x), _rankdata(y))


def _kendall_tau(x: List[float], y: List[float]) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    conc = disc = ties_x = ties_y = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_x += 1
                continue
            if dy == 0:
                ties_y += 1
                continue
            if dx * dy > 0:
                conc += 1
            else:
                disc += 1
    denom = math.sqrt((conc + disc + ties_x) * (conc + disc + ties_y))
    return (conc - disc) / denom if denom else 0.0


def _char_masks(s: str) -> tuple[List[bool], List[bool], List[bool], List[bool], List[bool]]:
    is_letter = [ch in LETTERS for ch in s]
    is_digit = [ch.isdigit() for ch in s]
    is_vowel = [ch in VOWELS for ch in s]
    is_consonant = [ch in LETTERS and ch not in VOWELS for ch in s]
    is_symbol = [not (ch in LETTERS or ch.isdigit()) for ch in s]
    return is_letter, is_digit, is_vowel, is_consonant, is_symbol


def _longest_consecutive(mask: List[bool]) -> int:
    best = 0
    cur = 0
    for bit in mask:
        if bit:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def _normalize_dist(d: Dict[str, float]) -> Dict[str, float]:
    total = float(sum(d.values()))
    if not total:
        return {}
    return {k: float(v) / total for k, v in d.items()}


def _ngrams(s: str, n: int) -> Counter[str]:
    if n <= 0 or len(s) < n:
        return Counter()
    return Counter(s[i : i + n] for i in range(len(s) - n + 1))


def _vectorize(obs_probs: Dict[str, float], tgt_probs: Dict[str, float]) -> tuple[List[str], List[float], List[float]]:
    keys = sorted(set(obs_probs.keys()) | set(tgt_probs.keys()))
    p = [obs_probs.get(k, 0.0) for k in keys]
    q = [tgt_probs.get(k, 0.0) for k in keys]
    return keys, p, q


def _kl(p: List[float], q: List[float], eps: float = 1e-12) -> float:
    total = 0.0
    for pi, qi in zip(p, q):
        if pi <= 0.0:
            continue
        total += pi * math.log((pi + eps) / (qi + eps))
    return total


def _l1(p: List[float], q: List[float]) -> float:
    return sum(abs(a - b) for a, b in zip(p, q))


def _l2(p: List[float], q: List[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p, q)))


def _linf(p: List[float], q: List[float]) -> float:
    return max((abs(a - b) for a, b in zip(p, q)), default=0.0)


def _canberra(p: List[float], q: List[float], eps: float = 1e-12) -> float:
    return sum(abs(a - b) / (abs(a) + abs(b) + eps) for a, b in zip(p, q))


def _emd_1d(p: List[float], q: List[float]) -> float:
    cumulative = 0.0
    total = 0.0
    for a, b in zip(p, q):
        cumulative += a - b
        total += abs(cumulative)
    return total


def _jaccard(obs_keys: set[str], tgt_keys: set[str]) -> float:
    universe = obs_keys | tgt_keys
    if not universe:
        return 0.0
    return float(len(obs_keys & tgt_keys)) / float(len(universe))


def _pronounceability(obs_probs: Dict[str, float], tgt_probs: Dict[str, float], eps: float = 1e-12) -> float:
    if not obs_probs:
        return 0.0
    total = 0.0
    for ng, p in obs_probs.items():
        q = tgt_probs.get(ng, 0.0)
        total += p * math.log(q + eps)
    return total


def _baseline_target_ngrams_counts() -> Dict[int, Dict[str, float]]:
    unigram = {
        "e": 12.0,
        "t": 9.1,
        "a": 8.2,
        "o": 7.5,
        "i": 7.0,
        "n": 6.7,
        "s": 6.3,
        "h": 6.1,
        "r": 6.0,
        "d": 4.3,
        "l": 4.0,
        "c": 2.8,
        "u": 2.8,
        "m": 2.4,
        "w": 2.4,
        "f": 2.2,
        "g": 2.0,
        "y": 2.0,
        "p": 1.9,
        "b": 1.5,
        "v": 1.0,
        "k": 0.8,
        "j": 0.15,
        "x": 0.15,
        "q": 0.10,
        "z": 0.07,
    }
    bigram = {"th": 10, "he": 9, "in": 8, "er": 7, "an": 7, "re": 6, "on": 6, "at": 5, "en": 5, "nd": 5}
    trigram = {"the": 10, "and": 8, "ing": 7, "her": 5, "ion": 5, "ent": 4, "tha": 4, "nth": 3}
    return {1: unigram, 2: bigram, 3: trigram}


def _nlp_features_for_level(s: str, prefix: str) -> Dict[str, float]:
    length = len(s)
    is_letter, is_digit, is_vowel, is_consonant, is_symbol = _char_masks(s)
    return {
        f"N_LEN_{prefix}": float(length),
        f"N_CON_{prefix}": float(_safe_div(sum(is_consonant), length)),
        f"N_LET_{prefix}": float(_safe_div(sum(is_letter), length)),
        f"N_NUM_{prefix}": float(_safe_div(sum(is_digit), length)),
        f"N_SYM_{prefix}": float(_safe_div(sum(is_symbol), length)),
        f"N_VOW_{prefix}": float(_safe_div(sum(is_vowel), length)),
    }


def _lexical_features(fqdn_str: str, labels: List[str], tld: str) -> Dict[str, float]:
    length = float(len(fqdn_str))
    hyphens = fqdn_str.count("-")
    letters = [ch for ch in fqdn_str if ch.isalpha()]
    n_letters = float(len(letters)) if letters else 0.0
    n_vowel = float(sum(1 for ch in letters if ch in VOWELS))
    n_con = float(sum(1 for ch in letters if ch not in VOWELS))

    vc = 0
    for a, b in zip(letters, letters[1:]):
        if (a in VOWELS) != (b in VOWELS):
            vc += 1

    max_vowel_run = 0
    max_con_run = 0
    cur_v = 0
    cur_c = 0
    for ch in letters:
        if ch in VOWELS:
            cur_v += 1
            cur_c = 0
        else:
            cur_c += 1
            cur_v = 0
        max_vowel_run = max(max_vowel_run, cur_v)
        max_con_run = max(max_con_run, cur_c)

    return {
        "L_NUM_DOTS": float(max(0, len(labels) - 1)),
        "L_LEN_LABEL_MAX": float(max((len(x) for x in labels), default=0)),
        "L_LEN_LABEL_MIN": float(min((len(x) for x in labels), default=0)),
        "L_LEN_TLD": float(len(tld)),
        "L_HYPHEN_RATIO": float(_safe_div(hyphens, length)),
        "L_HAS_HYPHEN": float(1 if hyphens > 0 else 0),
        "L_UNIQUE_CHARS": float(len(set(fqdn_str))) if fqdn_str else 0.0,
        "L_UNIQUE_RATIO": float(_safe_div(len(set(fqdn_str)), length)),
        "L_MAX_CHAR_RUN": float(_max_run_same_char(fqdn_str)),
        "L_VC_ALTERNATIONS": float(vc),
        "L_VOWEL_RATIO_LETTERS": float(_safe_div(n_vowel, n_letters)) if n_letters else 0.0,
        "L_CONSONANT_RATIO_LETTERS": float(_safe_div(n_con, n_letters)) if n_letters else 0.0,
        "L_MAX_VOWEL_RUN_LETTERS": float(max_vowel_run),
        "L_MAX_CONSONANT_RUN_LETTERS": float(max_con_run),
        "L_ENTROPY": float(_shannon_entropy_chars(fqdn_str)),
    }


def _ngram_features(
    core_letters: str,
    n: int,
    tgt_probs: Dict[str, float],
    selected_names: Set[str] | None = None,
) -> Dict[str, float]:
    level_prefix = f"{n}G_"
    level_selected: Set[str] | None = None
    if selected_names is not None:
        level_selected = {name for name in selected_names if name.startswith(level_prefix)}
        if not level_selected:
            return {}

    def want(name: str) -> bool:
        return level_selected is None or name in level_selected

    counts = _ngrams(core_letters, n)
    total = sum(counts.values())

    obs_probs: Dict[str, float] | None = None
    freq: List[float] | None = None
    p_vec: List[float] | None = None
    q_vec: List[float] | None = None

    def ensure_obs_probs() -> Dict[str, float]:
        nonlocal obs_probs
        if obs_probs is None:
            obs_probs = {k: float(v) / float(total) for k, v in counts.items()} if total else {}
        return obs_probs

    def ensure_freq() -> List[float]:
        nonlocal freq
        if freq is None:
            freq = list(ensure_obs_probs().values())
        return freq

    def ensure_vectors() -> tuple[List[float], List[float]]:
        nonlocal p_vec, q_vec
        if p_vec is None or q_vec is None:
            _, p_vec_local, q_vec_local = _vectorize(ensure_obs_probs(), tgt_probs)
            p_vec = p_vec_local
            q_vec = q_vec_local
        return p_vec, q_vec

    out: Dict[str, float] = {}
    if want(f"{n}G_DIST"):
        out[f"{n}G_DIST"] = float(len(counts))
    if want(f"{n}G_REP"):
        out[f"{n}G_REP"] = float(sum(1 for v in counts.values() if v > 1))

    if want(f"{n}G_25P"):
        out[f"{n}G_25P"] = float(_percentile(ensure_freq(), 0.25))
    if want(f"{n}G_50P"):
        out[f"{n}G_50P"] = float(_percentile(ensure_freq(), 0.50))
    if want(f"{n}G_75P"):
        out[f"{n}G_75P"] = float(_percentile(ensure_freq(), 0.75))
    if want(f"{n}G_MEAN"):
        out[f"{n}G_MEAN"] = float(_mean(ensure_freq()))
    if want(f"{n}G_QMEAN"):
        out[f"{n}G_QMEAN"] = float(_qmean(ensure_freq()))
    if want(f"{n}G_SUMSQ"):
        out[f"{n}G_SUMSQ"] = float(sum(x * x for x in ensure_freq()))
    if want(f"{n}G_VAR"):
        out[f"{n}G_VAR"] = float(_var_sample(ensure_freq()))
    if want(f"{n}G_PVAR"):
        out[f"{n}G_PVAR"] = float(_var_pop(ensure_freq()))
    if want(f"{n}G_STD"):
        out[f"{n}G_STD"] = float(_std_sample(ensure_freq()))
    if want(f"{n}G_PSTD"):
        out[f"{n}G_PSTD"] = float(_std_pop(ensure_freq()))
    if want(f"{n}G_SKE"):
        out[f"{n}G_SKE"] = float(_skewness(ensure_freq()))
    if want(f"{n}G_KUR"):
        out[f"{n}G_KUR"] = float(_kurtosis(ensure_freq()))
    if want(f"{n}G_E"):
        out[f"{n}G_E"] = float(_entropy_from_probs(ensure_freq()))
    if want(f"{n}G_NORM"):
        out[f"{n}G_NORM"] = float(_jarque_bera(ensure_freq()))
    if want(f"{n}G_PRO"):
        out[f"{n}G_PRO"] = float(_pronounceability(ensure_obs_probs(), tgt_probs))

    if want(f"{n}G_COV"):
        p_values, q_values = ensure_vectors()
        out[f"{n}G_COV"] = float(
            _safe_div(
                sum((a - _mean(p_values)) * (b - _mean(q_values)) for a, b in zip(p_values, q_values)),
                len(p_values),
            )
            if p_values
            else 0.0
        )
    if want(f"{n}G_PEA"):
        p_values, q_values = ensure_vectors()
        out[f"{n}G_PEA"] = float(_pearson(p_values, q_values))
    if want(f"{n}G_SPE"):
        p_values, q_values = ensure_vectors()
        out[f"{n}G_SPE"] = float(_spearman(p_values, q_values))
    if want(f"{n}G_KEN"):
        p_values, q_values = ensure_vectors()
        out[f"{n}G_KEN"] = float(_kendall_tau(p_values, q_values))

    if any(
        want(name)
        for name in (
            f"{n}G_TSUM",
            f"{n}G_TSUMSQ",
            f"{n}G_TVAR",
            f"{n}G_TPVAR",
            f"{n}G_TSTD",
            f"{n}G_TPSTD",
            f"{n}G_TSKE",
            f"{n}G_TKUR",
        )
    ):
        _, target_values = ensure_vectors()
        if want(f"{n}G_TSUM"):
            out[f"{n}G_TSUM"] = float(sum(target_values))
        if want(f"{n}G_TSUMSQ"):
            out[f"{n}G_TSUMSQ"] = float(sum(x * x for x in target_values))
        if want(f"{n}G_TVAR"):
            out[f"{n}G_TVAR"] = float(_var_sample(target_values))
        if want(f"{n}G_TPVAR"):
            out[f"{n}G_TPVAR"] = float(_var_pop(target_values))
        if want(f"{n}G_TSTD"):
            out[f"{n}G_TSTD"] = float(_std_sample(target_values))
        if want(f"{n}G_TPSTD"):
            out[f"{n}G_TPSTD"] = float(_std_pop(target_values))
        if want(f"{n}G_TSKE"):
            out[f"{n}G_TSKE"] = float(_skewness(target_values))
        if want(f"{n}G_TKUR"):
            out[f"{n}G_TKUR"] = float(_kurtosis(target_values))

    if want(f"{n}G_DST_KL"):
        p_values, q_values = ensure_vectors()
        out[f"{n}G_DST_KL"] = float(_kl(p_values, q_values))
    if want(f"{n}G_DST_CA"):
        p_values, q_values = ensure_vectors()
        out[f"{n}G_DST_CA"] = float(_canberra(p_values, q_values))
    if want(f"{n}G_DST_CH"):
        p_values, q_values = ensure_vectors()
        out[f"{n}G_DST_CH"] = float(_linf(p_values, q_values))
    if want(f"{n}G_DST_EU"):
        p_values, q_values = ensure_vectors()
        out[f"{n}G_DST_EU"] = float(_l2(p_values, q_values))
    if want(f"{n}G_DST_MA"):
        p_values, q_values = ensure_vectors()
        out[f"{n}G_DST_MA"] = float(_l1(p_values, q_values))
    if want(f"{n}G_DST_EM"):
        p_values, q_values = ensure_vectors()
        out[f"{n}G_DST_EM"] = float(_emd_1d(p_values, q_values))
    if want(f"{n}G_DST_JI"):
        out[f"{n}G_DST_JI"] = float(_jaccard(set(ensure_obs_probs().keys()), set(tgt_probs.keys())))
    return out


def default_target_probs_by_n() -> Dict[int, Dict[str, float]]:
    baseline = _baseline_target_ngrams_counts()
    return {
        1: _normalize_dist(baseline[1]),
        2: _normalize_dist(baseline[2]),
        3: _normalize_dist(baseline[3]),
    }


def compute_domain_features_subset(
    domain: str,
    selected_feature_names: Iterable[str],
    tgt_probs_by_n: Dict[int, Dict[str, float]] | None = None,
) -> Dict[str, float]:
    selected = selected_feature_names if isinstance(selected_feature_names, set) else set(selected_feature_names)
    if not selected:
        return {}

    d = domain.lower().strip().strip(".")
    labels = [lbl for lbl in d.split(".") if lbl]
    n_labels = len(labels)

    sld = labels[-2] if n_labels >= 2 else (labels[0] if n_labels == 1 else "")
    old_labels = labels[:-2] if n_labels >= 2 else []
    fqdn_str = "".join(labels)
    old_str = "".join(old_labels)

    feats: Dict[str, float] = {}

    if "N_LEN_OLD" in selected or "N_CON_OLD" in selected:
        old_feats = _nlp_features_for_level(old_str, "OLD")
        if "N_LEN_OLD" in selected:
            feats["N_LEN_OLD"] = float(old_feats.get("N_LEN_OLD", 0.0))
        if "N_CON_OLD" in selected:
            feats["N_CON_OLD"] = float(old_feats.get("N_CON_OLD", 0.0))

    if "N_CON_2LD" in selected or "N_LET_2LD" in selected or "N_VOW_2LD" in selected:
        sld_feats = _nlp_features_for_level(sld, "2LD")
        if "N_CON_2LD" in selected:
            feats["N_CON_2LD"] = float(sld_feats.get("N_CON_2LD", 0.0))
        if "N_LET_2LD" in selected:
            feats["N_LET_2LD"] = float(sld_feats.get("N_LET_2LD", 0.0))
        if "N_VOW_2LD" in selected:
            feats["N_VOW_2LD"] = float(sld_feats.get("N_VOW_2LD", 0.0))

    if any(name.startswith("L_") for name in selected):
        length = float(len(fqdn_str))
        if "L_NUM_DOTS" in selected:
            feats["L_NUM_DOTS"] = float(max(0, len(labels) - 1))
        if "L_LEN_LABEL_MAX" in selected:
            feats["L_LEN_LABEL_MAX"] = float(max((len(x) for x in labels), default=0))
        if "L_LEN_LABEL_MIN" in selected:
            feats["L_LEN_LABEL_MIN"] = float(min((len(x) for x in labels), default=0))
        if "L_HAS_HYPHEN" in selected:
            feats["L_HAS_HYPHEN"] = float(1 if "-" in fqdn_str else 0)
        if "L_UNIQUE_RATIO" in selected:
            feats["L_UNIQUE_RATIO"] = float(_safe_div(len(set(fqdn_str)), length))
        if "L_MAX_CHAR_RUN" in selected:
            feats["L_MAX_CHAR_RUN"] = float(_max_run_same_char(fqdn_str))
        if "L_ENTROPY" in selected:
            feats["L_ENTROPY"] = float(_shannon_entropy_chars(fqdn_str))

        needs_letter_stats = any(
            name in selected
            for name in (
                "L_VC_ALTERNATIONS",
                "L_CONSONANT_RATIO_LETTERS",
                "L_MAX_VOWEL_RUN_LETTERS",
                "L_MAX_CONSONANT_RUN_LETTERS",
            )
        )
        if needs_letter_stats:
            letters = [ch for ch in fqdn_str if ch.isalpha()]
            n_letters = float(len(letters))
            n_con = float(sum(1 for ch in letters if ch not in VOWELS))

            if "L_CONSONANT_RATIO_LETTERS" in selected:
                feats["L_CONSONANT_RATIO_LETTERS"] = float(_safe_div(n_con, n_letters))

            if (
                "L_VC_ALTERNATIONS" in selected
                or "L_MAX_VOWEL_RUN_LETTERS" in selected
                or "L_MAX_CONSONANT_RUN_LETTERS" in selected
            ):
                vc = 0
                max_vowel_run = 0
                max_con_run = 0
                cur_v = 0
                cur_c = 0
                for idx, ch in enumerate(letters):
                    if ch in VOWELS:
                        cur_v += 1
                        cur_c = 0
                    else:
                        cur_c += 1
                        cur_v = 0
                    max_vowel_run = max(max_vowel_run, cur_v)
                    max_con_run = max(max_con_run, cur_c)
                    if idx > 0 and ((letters[idx - 1] in VOWELS) != (ch in VOWELS)):
                        vc += 1

                if "L_VC_ALTERNATIONS" in selected:
                    feats["L_VC_ALTERNATIONS"] = float(vc)
                if "L_MAX_VOWEL_RUN_LETTERS" in selected:
                    feats["L_MAX_VOWEL_RUN_LETTERS"] = float(max_vowel_run)
                if "L_MAX_CONSONANT_RUN_LETTERS" in selected:
                    feats["L_MAX_CONSONANT_RUN_LETTERS"] = float(max_con_run)

        if "L_LC_C" in selected or "L_LC_V" in selected or "L_LC_D" in selected:
            _, is_digit, is_vowel, is_consonant, _ = _char_masks(fqdn_str)
            if "L_LC_C" in selected:
                feats["L_LC_C"] = float(_longest_consecutive(is_consonant))
            if "L_LC_V" in selected:
                feats["L_LC_V"] = float(_longest_consecutive(is_vowel))
            if "L_LC_D" in selected:
                feats["L_LC_D"] = float(_longest_consecutive(is_digit))

    if tgt_probs_by_n is None:
        tgt_probs_by_n = default_target_probs_by_n()

    needs_ngram = any(name.startswith("1G_") or name.startswith("2G_") or name.startswith("3G_") for name in selected)
    if needs_ngram:
        core_letters = "".join(ch for ch in fqdn_str if ch in LETTERS)
        feats.update(_ngram_features(core_letters, 1, tgt_probs_by_n.get(1, {}), selected))
        feats.update(_ngram_features(core_letters, 2, tgt_probs_by_n.get(2, {}), selected))
        feats.update(_ngram_features(core_letters, 3, tgt_probs_by_n.get(3, {}), selected))

    for feature_name in selected:
        feats.setdefault(feature_name, 0.0)

    return feats


def compute_registered_features(domain: str, registry: FeatureRegistry) -> Dict[str, float]:
    """Compute registered features while reusing the unchanged legacy formulas."""
    default_names = set(default_feature_registry().names)
    requested_defaults = [name for name in registry.names if name in default_names]
    default_values = compute_domain_features_subset(domain, requested_defaults)
    context = FeatureContext(
        normalized_domain=domain,
        default_feature_values=MappingProxyType(default_values),
    )
    return {definition.name: definition.extractor(context) for definition in registry.definitions}


def compute_domain_features(domain: str, tgt_probs_by_n: Dict[int, Dict[str, float]] | None = None) -> Dict[str, float]:
    d = domain.lower().strip().strip(".")
    labels = [lbl for lbl in d.split(".") if lbl]
    n_labels = len(labels)

    tld = labels[-1] if n_labels >= 1 else ""
    sld = labels[-2] if n_labels >= 2 else (labels[0] if n_labels == 1 else "")
    old_labels = labels[:-2] if n_labels >= 2 else []

    fqdn_str = "".join(labels)
    sld_str = sld
    old_str = "".join(old_labels)

    feats: Dict[str, float] = {"L_NUM_LABELS": float(n_labels)}
    feats.update(_nlp_features_for_level(fqdn_str, "FQDN"))
    feats.update(_nlp_features_for_level(sld_str, "2LD"))
    feats.update(_nlp_features_for_level(old_str, "OLD"))
    feats.update(_lexical_features(fqdn_str, labels, tld))

    _, is_digit, is_vowel, is_consonant, _ = _char_masks(fqdn_str)
    feats["L_LC_C"] = float(_longest_consecutive(is_consonant))
    feats["L_LC_V"] = float(_longest_consecutive(is_vowel))
    feats["L_LC_D"] = float(_longest_consecutive(is_digit))

    if tgt_probs_by_n is None:
        tgt_probs_by_n = default_target_probs_by_n()

    core_letters = "".join(ch for ch in fqdn_str if ch in LETTERS)
    feats.update(_ngram_features(core_letters, 1, tgt_probs_by_n.get(1, {})))
    feats.update(_ngram_features(core_letters, 2, tgt_probs_by_n.get(2, {})))
    feats.update(_ngram_features(core_letters, 3, tgt_probs_by_n.get(3, {})))
    return feats


def default_feature_names() -> List[str]:
    return sorted(compute_domain_features("example.com").keys())
