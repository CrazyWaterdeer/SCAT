"""Unsupervised clustering of deposits for labeling assistance.

Pure core (no image/CLI I/O): build a scaled feature matrix from deposit features, cluster
with HDBSCAN (kmeans optional), pick representative samples, summarise clusters, validate a
user cluster->label mapping, propagate, and gate training.

A cluster is NOT a class. The same visual features can reflect lighting / segmentation size /
plate batch as much as biology, and a cluster's tails can be mixed even when its medoid looks
clean. This module *assists* labeling (cut hundreds of deposit labels to a handful of cluster
decisions) — it does not replace the human judgement, and `training_readiness` blocks the
degenerate cases that would poison or crash training. See the spec.
"""
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Intrinsic shape+colour+density features (identity/position deliberately excluded).
_LOG_COLS = ["area_px", "perimeter", "iod"]          # skewed positive magnitudes -> log1p
_LINEAR_COLS = ["circularity", "aspect_ratio", "mean_saturation", "mean_lightness", "pigment_density"]

VALID_LABELS = {"normal", "rod", "artifact"}

_PROFILE_COLS = ["area_px", "circularity", "aspect_ratio", "solidity", "mean_hue",
                 "mean_saturation", "mean_lightness", "pigment_density", "iod"]


def _num(df, col, default=1.0):
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index)


def line_flag(df: pd.DataFrame) -> pd.Series:
    """Rotation-invariant 'thin line' (film-boundary) flag: very elongated AND poorly filling its
    rotated bounding box. `aspect_ratio` alone comes from an axis-aligned bbox and misses DIAGONAL
    boundaries, so prefer the minAreaRect-based `elongation`/`rect_fill`; fall back to aspect+circ."""
    if "elongation" in df.columns and "rect_fill" in df.columns:
        thin = (_num(df, "elongation") > 4) & (_num(df, "rect_fill") < 0.4)
        # a SOLID straight line fills its rotated box (rect_fill≈1) yet is extremely elongated with
        # near-zero circularity — catch those too, else they'd surface as false 'unusual' ROD.
        straight = (_num(df, "elongation") > 8) & (_num(df, "circularity") < 0.15)
        return thin | straight
    return (_num(df, "aspect_ratio") > 8) & (_num(df, "circularity") < 0.15)


def unusual_flag(df: pd.DataFrame) -> pd.Series:
    """Shape-atypical (irregular / elongated) deposits that are NOT line artifacts."""
    elong = _num(df, "elongation") if "elongation" in df.columns else _num(df, "aspect_ratio")
    return ((_num(df, "circularity") < 0.5) | (elong > 2.5)) & ~line_flag(df)


def cluster_kind(circularity, aspect_ratio, solidity=None, pct_line=0.0, pct_unusual=0.0) -> str:
    """A short label to guide labeling. Prefers per-cluster FRACTIONS (a median alone marks a mixed
    or giant-noise cluster 'common' while its tail holds exactly the interesting deposits); falls
    back to the scalar medians for single-deposit callers."""
    if pct_line > 0.5:
        return "line-artifact?"
    if pct_unusual > 0.4:
        return "unusual?"
    if pct_unusual > 0.1:
        return "mixed (has unusual)"
    if aspect_ratio is not None and circularity is not None and aspect_ratio > 8 and circularity < 0.15:
        return "line-artifact?"
    if (circularity is not None and circularity < 0.5) or (aspect_ratio is not None and aspect_ratio > 2.5):
        return "unusual?"
    return "common"


def unusual_ranking(df: pd.DataFrame) -> pd.Series:
    """Per-deposit 'shape-unusualness' score (elongated + irregular + big). Line artifacts are
    EXCLUDED (−inf) rather than softly penalised, so an extreme line can never outrank a real
    atypical deposit even after z-scoring."""
    def z(s):
        s = pd.to_numeric(s, errors="coerce").fillna(0.0)
        return (s - s.mean()) / (s.std(ddof=0) + 1e-9)  # ddof=0 so a single deposit -> 0, not NaN
    elong = df["elongation"] if "elongation" in df.columns else df["aspect_ratio"]
    score = z(elong) + z(-df["circularity"]) + 0.6 * z(df["area_px"])
    return score.mask(line_flag(df), float("-inf"))


# --------------------------------------------------------------------------- feature matrix
def build_feature_matrix(df: pd.DataFrame):
    """Assemble a standardized feature matrix. Returns (X, feature_names).

    - area_px/perimeter/iod: log1p (skewed and correlated; raw values let a few huge deposits
      dominate the structure several times over).
    - hue: (sin, cos) of mean_hue, each weighted by mean_saturation, so low-saturation
      (near-gray) deposits don't get arbitrary circular coordinates. Clustering-only; the
      model's mean_hue is untouched.
    - NaN -> column median, then StandardScaler.
    """
    from sklearn.preprocessing import StandardScaler

    cols, names = [], []
    for c in _LOG_COLS:
        cols.append(np.log1p(pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)))
        names.append(c)
    for c in _LINEAR_COLS:
        cols.append(pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float))
        names.append(c)
    if "solidity" in df.columns:  # optional: separates solid deposits from branchy/concave ones,
        cols.append(pd.to_numeric(df["solidity"], errors="coerce").to_numpy(dtype=float))
        names.append("solidity")  # consolidating the irregular/unusual deposits into a real cluster
    hue = np.deg2rad(pd.to_numeric(df["mean_hue"], errors="coerce").to_numpy(dtype=float))
    sat = pd.to_numeric(df["mean_saturation"], errors="coerce").to_numpy(dtype=float)
    cols.append(np.sin(hue) * sat); names.append("hue_sin")
    cols.append(np.cos(hue) * sat); names.append("hue_cos")

    X = np.column_stack(cols).astype(float)
    med = np.nanmedian(X, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    bad = np.where(~np.isfinite(X))
    X[bad] = np.take(med, bad[1])
    X = StandardScaler().fit_transform(X)
    return X, names


# --------------------------------------------------------------------------- clustering
@dataclass
class ClusterResult:
    labels: np.ndarray
    method: str
    n_clusters: int
    n_noise: int
    health: list = field(default_factory=list)


def _default_min_cluster_size(n: int) -> int:
    # Softer than n/40 (which over-noised real data ~73%); the health warnings + --min-cluster-size
    # let the user tune from here.
    return int(min(100, max(10, round(n / 80))))


def cluster_deposits(X, method="hdbscan", min_cluster_size=None, min_samples=None,
                     k=None, random_state=0) -> ClusterResult:
    """Cluster the (standardized) matrix. HDBSCAN default (auto #clusters, -1 = noise); kmeans
    when a fixed count is wanted. Returns labels + health WARNINGS for degenerate results."""
    n = X.shape[0]
    if n < 3:  # too few to cluster — HDBSCAN/kmeans would raise; mark all noise
        return ClusterResult(labels=np.full(n, -1, dtype=int), method=method, n_clusters=0,
                             n_noise=n, health=["too few deposits to cluster — label manually"])
    mcs = max(2, min(int(min_cluster_size or _default_min_cluster_size(n)), n))  # HDBSCAN needs <= n
    if method == "kmeans":
        from sklearn.cluster import KMeans
        kk = int(k or max(2, min(8, round(n / 50))))
        kk = max(1, min(kk, n))
        labels = KMeans(n_clusters=kk, random_state=random_state, n_init=10).fit_predict(X)
    else:
        from sklearn.cluster import HDBSCAN
        ms = min(int(min_samples), n) if min_samples else None  # min_samples must be <= n
        labels = HDBSCAN(min_cluster_size=mcs, min_samples=ms,
                         metric="euclidean", cluster_selection_method="eom",
                         copy=True).fit_predict(X)

    uniq = [c for c in np.unique(labels) if c != -1]
    n_clusters = len(uniq)
    n_noise = int(np.sum(labels == -1))
    health = []
    if n < 4 * mcs:
        health.append("too few deposits to cluster meaningfully — consider labeling manually")
    if n_clusters < 2:
        health.append(f"only {n_clusters} cluster(s) found — try --min-cluster-size or --method kmeans")
    if n and n_noise / n > 0.5:
        health.append(f"high noise fraction ({n_noise/n:.0%}) — many deposits left unclustered")
    if n_clusters:
        biggest = max(int(np.sum(labels == c)) for c in uniq)
        if biggest / max(1, n - n_noise) > 0.8:
            health.append("one cluster holds >80% of clustered deposits — likely under-segmented")
    return ClusterResult(labels=np.asarray(labels), method=method, n_clusters=n_clusters,
                         n_noise=n_noise, health=health)


# --------------------------------------------------------------------------- representatives
def representatives(X, labels, per_kind=6, random_state=0):
    """Per non-noise cluster, row indices for three sample kinds: 'medoid' (nearest the
    centroid), 'boundary' (farthest from it), and 'random'. Boundary + random expose mixed
    clusters that a medoid alone would hide."""
    rng = np.random.RandomState(random_state)
    out = {}
    for cid in sorted(int(c) for c in np.unique(labels) if c != -1):
        members = np.where(labels == cid)[0]
        centroid = X[members].mean(axis=0)
        dist = np.linalg.norm(X[members] - centroid, axis=1)
        order = members[np.argsort(dist)]
        medoid = list(order[:per_kind])
        boundary = list(order[::-1][:per_kind])
        used = set(int(i) for i in medoid) | set(int(i) for i in boundary)
        pool = [int(m) for m in members if int(m) not in used]
        random = list(rng.choice(pool, size=min(per_kind, len(pool)), replace=False)) if pool else []
        out[cid] = {"medoid": [int(i) for i in medoid],
                    "boundary": [int(i) for i in boundary],
                    "random": [int(i) for i in random]}
    return out


# --------------------------------------------------------------------------- profile
def cluster_profile(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """One row per cluster id (incl. -1 noise): size + mean of each raw feature. High per-feature
    spread is a mixed-cluster hint (surfaced in the report)."""
    work = df.copy()
    work["cluster_id"] = np.asarray(labels)
    work["_line"] = line_flag(work).astype(float)
    work["_unusual"] = unusual_flag(work).astype(float)
    cols = [c for c in _PROFILE_COLS if c in work.columns]
    # median (robust to the outliers that define these clusters) + line/unusual member FRACTIONS,
    # so a mixed / giant-noise cluster is not mislabeled 'common' when its tail is the interesting part.
    agg = work.groupby("cluster_id").agg(
        size=("cluster_id", "size"),
        pct_line=("_line", "mean"),
        pct_unusual=("_unusual", "mean"),
        **{c: (c, "median") for c in cols}).reset_index()
    agg["kind"] = agg.apply(
        lambda r: cluster_kind(r.get("circularity"), r.get("aspect_ratio"), r.get("solidity"),
                               pct_line=r["pct_line"], pct_unusual=r["pct_unusual"]), axis=1)
    return agg.sort_values("cluster_id").reset_index(drop=True)


# --------------------------------------------------------------------------- propagation
def parse_cluster_labels_csv(path) -> dict:
    """Read cluster_labels.csv -> {cluster_id: label}. Normalizes case/whitespace; keeps only
    non-blank labels; rejects invalid labels and duplicate cluster_ids."""
    df = pd.read_csv(path)
    if "cluster_id" not in df.columns or "label" not in df.columns:
        raise ValueError("cluster_labels.csv must have 'cluster_id' and 'label' columns")
    mapping, seen = {}, set()
    for _, row in df.iterrows():
        cid = int(row["cluster_id"])
        if cid in seen:
            raise ValueError(f"duplicate cluster_id {cid} in labels CSV")
        seen.add(cid)
        raw = row["label"]
        lab = "" if pd.isna(raw) else str(raw).strip().lower()
        if not lab:
            continue  # blank = skip this cluster
        if lab not in VALID_LABELS:
            raise ValueError(f"invalid label {raw!r} for cluster {cid} (use {sorted(VALID_LABELS)})")
        mapping[cid] = lab
    return mapping


def propagate_labels(assignments: pd.DataFrame, mapping: dict):
    """Map each deposit's cluster_id to a label via `mapping`; noise/unmapped -> 'unknown'.
    Returns ({(filename, deposit_id): label}, summary)."""
    labels, n_labeled = {}, 0
    for _, r in assignments.iterrows():
        key = (str(r["filename"]), int(r["deposit_id"]))
        lab = mapping.get(int(r["cluster_id"]), "unknown")
        labels[key] = lab
        if lab != "unknown":
            n_labeled += 1
    return labels, {"n_labeled": n_labeled, "n_skipped": len(labels) - n_labeled}


# --------------------------------------------------------------------------- training guard
@dataclass
class ReadinessReport:
    verdict: str
    reasons: list
    class_counts: dict
    n_labeled: int
    n_skipped: int


def training_readiness(member_labels, largest_cluster_share=None, test_size=0.2) -> ReadinessReport:
    """Gate before `scat train`. BLOCKS configs that would crash or poison the RF: <2 labeled
    classes, any class with <2 samples, or too few labels for the trainer's stratified
    `train_test_split(test_size=0.2)` — which needs the test split (int(0.2*n)) to hold at least
    one of every class. WARNS on extreme imbalance or most labels from a single cluster."""
    counts = Counter(l for l in member_labels if l in VALID_LABELS)
    n_labeled = sum(counts.values())
    n_skipped = sum(1 for l in member_labels if l not in VALID_LABELS)
    reasons, verdict = [], "ok"
    if len(counts) < 2:
        verdict = "block"
        reasons.append(f"only {len(counts)} labeled class(es); need >=2 to train")
    singletons = [c for c, m in counts.items() if m < 2]
    if singletons:
        verdict = "block"
        reasons.append(f"class(es) with <2 samples: {singletons} (stratified split fails)")
    if verdict != "block" and len(counts) >= 2 and int(test_size * n_labeled) < len(counts):
        verdict = "block"
        need = int(len(counts) / test_size) + 1
        reasons.append(f"only {n_labeled} labeled deposits for {len(counts)} classes; the stratified "
                       f"20% test split needs >= {len(counts)} — label ~{need}+ (more clusters)")
    if verdict != "block" and counts:
        lo, hi = min(counts.values()), max(counts.values())
        if hi >= 10 * max(1, lo):
            verdict = "warn"
            reasons.append(f"extreme class imbalance ({dict(counts)})")
        if largest_cluster_share and largest_cluster_share > 0.8:
            verdict = "warn"
            reasons.append(f"{largest_cluster_share:.0%} of labels come from one cluster")
    return ReadinessReport(verdict=verdict, reasons=reasons, class_counts=dict(counts),
                           n_labeled=n_labeled, n_skipped=n_skipped)
