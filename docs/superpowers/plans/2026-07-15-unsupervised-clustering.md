# Unsupervised Clustering (Labeling Assistance) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a user cluster deposits from unlabeled images, label whole clusters (with QA), propagate to members, and train the existing RF — cutting labeling from hundreds of deposits to a handful of cluster decisions, guarded against poisoned training.

**Architecture:** A pure-math core `scat/clustering.py` (feature matrix → HDBSCAN → representatives/profile/propagation/readiness), two I/O services in `scat/pipeline.py` (`cluster_folder_service`, `propagate_service`) that reuse the existing analyzer/report machinery, and two CLI subcommands (`scat cluster`, `scat propagate`). `cluster_assignments.csv` is the propagation source of truth (decoupled from the GUI). Everything is additive — no existing output changes.

**Tech Stack:** Python 3.14, scikit-learn 1.8 (`HDBSCAN`, `KMeans`, `StandardScaler`), numpy, pandas, existing `scat` detector/features/analyzer/trainer. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-07-15-scat-unsupervised-clustering-design.md`

---

## File Structure

- **Create `scat/clustering.py`** — pure core: `build_feature_matrix`, `cluster_deposits`+`ClusterResult`, `representatives`, `cluster_profile`, `parse_cluster_labels_csv`, `propagate_labels`, `training_readiness`+`ReadinessReport`. No image/CLI I/O.
- **Modify `scat/pipeline.py`** — add `cluster_folder_service`, `propagate_service`, and private helpers `_export_cluster_thumbnails`, `_write_cluster_report_html`, `_write_cluster_labels_json`. All the disk/image I/O lives here.
- **Modify `scat/cli.py`** — add `cluster_command`, `propagate_command`, and their `add_parser` wiring.
- **Create `tests/test_clustering.py`** — unit tests for the pure core (incl. adversarial cases).
- **Modify `tests/test_cli.py`** — end-to-end `cluster → fill CSV → propagate → train` round-trip.

Naming contract (used across tasks):
- Feature DataFrame rows carry `filename` (str) and `deposit_id` (int) plus feature columns.
- `ClusterResult(labels: np.ndarray[int], method: str, n_clusters: int, n_noise: int, health: list[str])`.
- Cluster ids are ints; `-1` = noise. `mapping: dict[int, str]` maps cluster_id → one of `{"normal","rod","artifact"}`.
- `ReadinessReport(verdict: str in {"ok","warn","block"}, reasons: list[str], class_counts: dict[str,int], n_labeled: int, n_skipped: int)`.

---

## Task 1: Feature matrix (`build_feature_matrix`)

**Files:**
- Create: `scat/clustering.py`
- Test: `tests/test_clustering.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_clustering.py
import numpy as np
import pandas as pd
import pytest
from scat import clustering as C


def _df(n=30, seed=0):
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "filename": [f"img_{i%3}.tif" for i in range(n)],
        "deposit_id": list(range(n)),
        "area_px": rng.randint(30, 500, n).astype(float),
        "perimeter": rng.randint(20, 200, n).astype(float),
        "circularity": rng.uniform(0.2, 1.0, n),
        "aspect_ratio": rng.uniform(1.0, 3.0, n),
        "mean_hue": rng.uniform(0, 360, n),
        "mean_saturation": rng.uniform(0, 1, n),
        "mean_lightness": rng.uniform(0, 1, n),
        "pigment_density": rng.uniform(0, 1, n),
        "iod": rng.uniform(0, 300, n),
    })


def test_build_feature_matrix_shape_and_hue_encoding():
    df = _df()
    X, names = C.build_feature_matrix(df)
    assert X.shape[0] == len(df)
    # hue is encoded as two saturation-weighted columns (sin/cos), raw mean_hue is dropped
    assert "mean_hue" not in names
    assert "hue_sin" in names and "hue_cos" in names
    assert np.isfinite(X).all()  # standardized, no NaN/inf


def test_build_feature_matrix_median_impute():
    df = _df()
    df.loc[0, "iod"] = np.nan
    X, names = C.build_feature_matrix(df)
    assert np.isfinite(X).all()  # NaN imputed, not propagated
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_clustering.py -q`
Expected: FAIL (`No module named 'scat.clustering'`).

- [ ] **Step 3: Write minimal implementation**

```python
# scat/clustering.py
"""Unsupervised clustering of deposits for labeling assistance.

Pure core (no image/CLI I/O): build a scaled feature matrix from deposit features, cluster
with HDBSCAN (kmeans optional), pick representative samples, summarise clusters, validate a
user cluster→label mapping, propagate, and gate training. A cluster is NOT a class — see the
spec: this assists labeling, it does not replace it.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

# Intrinsic shape+colour+density features (identity/position deliberately excluded).
_LOG_COLS = ["area_px", "perimeter", "iod"]          # skewed positive magnitudes → log1p
_LINEAR_COLS = ["circularity", "aspect_ratio", "mean_saturation", "mean_lightness", "pigment_density"]


def build_feature_matrix(df: pd.DataFrame):
    """Assemble a standardized feature matrix. Returns (X, feature_names).

    - area_px/perimeter/iod: log1p (they are skewed and correlated; raw values let a few huge
      deposits dominate the structure several times over).
    - hue: (sin, cos) of mean_hue, each weighted by mean_saturation, so low-saturation (near-gray)
      deposits don't get arbitrary circular coordinates. Clustering-only; the model's mean_hue is
      untouched.
    - NaN → column median. Then StandardScaler.
    """
    from sklearn.preprocessing import StandardScaler

    cols, names = [], []
    for c in _LOG_COLS:
        cols.append(np.log1p(pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)))
        names.append(c)
    for c in _LINEAR_COLS:
        cols.append(pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float))
        names.append(c)
    hue = np.deg2rad(pd.to_numeric(df["mean_hue"], errors="coerce").to_numpy(dtype=float))
    sat = pd.to_numeric(df["mean_saturation"], errors="coerce").to_numpy(dtype=float)
    cols.append(np.sin(hue) * sat); names.append("hue_sin")
    cols.append(np.cos(hue) * sat); names.append("hue_cos")

    X = np.column_stack(cols)
    # column-median impute
    med = np.nanmedian(X, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    inds = np.where(~np.isfinite(X))
    X[inds] = np.take(med, inds[1])
    X = StandardScaler().fit_transform(X)
    return X, names
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_clustering.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add scat/clustering.py tests/test_clustering.py
git commit -m "feat(clustering): standardized feature matrix (log1p + sat-weighted hue + median impute)"
```

---

## Task 2: Clustering (`cluster_deposits` + `ClusterResult` + health flags)

**Files:**
- Modify: `scat/clustering.py`
- Test: `tests/test_clustering.py`

- [ ] **Step 1: Write the failing test**

```python
def _blobs():
    # three well-separated blobs in feature space → deterministic clustering
    rng = np.random.RandomState(1)
    centers = np.array([[0, 0], [8, 8], [-8, 8]], dtype=float)
    X = np.vstack([c + rng.randn(40, 2) * 0.3 for c in centers])
    return X


def test_cluster_deposits_hdbscan_finds_blobs():
    res = C.cluster_deposits(_blobs(), min_cluster_size=10)
    assert res.method == "hdbscan"
    assert res.n_clusters == 3
    assert res.labels.shape[0] == 120


def test_cluster_deposits_health_flags_one_giant():
    rng = np.random.RandomState(2)
    X = rng.randn(200, 2) * 0.2  # one dense blob → one giant cluster / much noise
    res = C.cluster_deposits(X, min_cluster_size=10)
    assert any("cluster" in h or "noise" in h for h in res.health)


def test_cluster_deposits_kmeans():
    res = C.cluster_deposits(_blobs(), method="kmeans", k=3)
    assert res.method == "kmeans"
    assert res.n_clusters == 3
    assert res.n_noise == 0  # kmeans labels everything
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_clustering.py -k cluster_deposits -q`
Expected: FAIL (`cluster_deposits` not defined).

- [ ] **Step 3: Write minimal implementation**

```python
# append to scat/clustering.py

@dataclass
class ClusterResult:
    labels: np.ndarray
    method: str
    n_clusters: int
    n_noise: int
    health: list = field(default_factory=list)


def _default_min_cluster_size(n: int) -> int:
    return int(min(200, max(15, round(n / 40))))


def cluster_deposits(X, method="hdbscan", min_cluster_size=None, min_samples=None,
                     k=None, random_state=0) -> ClusterResult:
    """Cluster the (already standardized) matrix. HDBSCAN default (auto #clusters, -1 = noise);
    kmeans when a fixed count is wanted. Returns labels + health warnings for degenerate results."""
    n = X.shape[0]
    if method == "kmeans":
        from sklearn.cluster import KMeans
        kk = int(k or max(2, min(8, round(n / 50))))
        labels = KMeans(n_clusters=kk, random_state=random_state, n_init=10).fit_predict(X)
    else:
        from sklearn.cluster import HDBSCAN
        mcs = int(min_cluster_size or _default_min_cluster_size(n))
        labels = HDBSCAN(min_cluster_size=mcs, min_samples=min_samples,
                         metric="euclidean", cluster_selection_method="eom").fit_predict(X)

    uniq = [c for c in np.unique(labels) if c != -1]
    n_clusters = len(uniq)
    n_noise = int(np.sum(labels == -1))
    health = []
    if n < 4 * (min_cluster_size or _default_min_cluster_size(n)):
        health.append("too few deposits to cluster meaningfully — consider labeling manually")
    if n_clusters < 2:
        health.append(f"only {n_clusters} cluster(s) found — try --min-cluster-size or --method kmeans")
    if n and n_noise / n > 0.5:
        health.append(f"high noise fraction ({n_noise/n:.0%}) — many deposits unclustered")
    if n_clusters:
        biggest = max(int(np.sum(labels == c)) for c in uniq)
        if biggest / max(1, n - n_noise) > 0.8:
            health.append("one cluster holds >80% of clustered deposits — likely under-segmented")
    return ClusterResult(labels=labels, method=method, n_clusters=n_clusters,
                         n_noise=n_noise, health=health)
```

- [ ] **Step 4: Run to verify passes**

Run: `.venv/bin/python -m pytest tests/test_clustering.py -k cluster_deposits -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add scat/clustering.py tests/test_clustering.py
git commit -m "feat(clustering): HDBSCAN/kmeans cluster_deposits with health warnings"
```

---

## Task 3: Representatives (medoid / random / boundary)

**Files:**
- Modify: `scat/clustering.py`
- Test: `tests/test_clustering.py`

- [ ] **Step 1: Write the failing test**

```python
def test_representatives_three_kinds_near_and_far():
    X = _blobs()
    res = C.cluster_deposits(X, min_cluster_size=10)
    reps = C.representatives(X, res.labels, per_kind=3)
    # a rep set per non-noise cluster, each with the three kinds
    for cid, kinds in reps.items():
        assert cid != -1
        assert set(kinds) == {"medoid", "random", "boundary"}
        assert 1 <= len(kinds["medoid"]) <= 3
        # medoid rows are closer to the cluster centroid than boundary rows
        members = np.where(res.labels == cid)[0]
        centroid = X[members].mean(axis=0)
        d_med = np.linalg.norm(X[kinds["medoid"][0]] - centroid)
        d_bnd = np.linalg.norm(X[kinds["boundary"][0]] - centroid)
        assert d_med <= d_bnd
```

- [ ] **Step 2: Run to verify fails**

Run: `.venv/bin/python -m pytest tests/test_clustering.py -k representatives -q`
Expected: FAIL.

- [ ] **Step 3: Write implementation**

```python
# append to scat/clustering.py

def representatives(X, labels, per_kind=6, random_state=0):
    """Per non-noise cluster, return row indices for three sample kinds: 'medoid' (nearest the
    centroid), 'boundary' (farthest from it), and 'random'. Boundary + random expose mixed
    clusters that a medoid alone would hide."""
    rng = np.random.RandomState(random_state)
    out = {}
    for cid in sorted(c for c in np.unique(labels) if c != -1):
        members = np.where(labels == cid)[0]
        centroid = X[members].mean(axis=0)
        dist = np.linalg.norm(X[members] - centroid, axis=1)
        order = members[np.argsort(dist)]
        medoid = list(order[:per_kind])
        boundary = list(order[::-1][:per_kind])
        pool = [m for m in members if m not in set(medoid) | set(boundary)]
        random = list(rng.choice(pool, size=min(per_kind, len(pool)), replace=False)) if pool else []
        out[int(cid)] = {"medoid": [int(i) for i in medoid],
                         "boundary": [int(i) for i in boundary],
                         "random": [int(i) for i in random]}
    return out
```

- [ ] **Step 4: Run to verify passes**

Run: `.venv/bin/python -m pytest tests/test_clustering.py -k representatives -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scat/clustering.py tests/test_clustering.py
git commit -m "feat(clustering): medoid/random/boundary representatives per cluster"
```

---

## Task 4: Cluster profile

**Files:**
- Modify: `scat/clustering.py`
- Test: `tests/test_clustering.py`

- [ ] **Step 1: Write failing test**

```python
def test_cluster_profile_rows_and_size():
    df = _df(60)
    res = C.cluster_deposits(*(C.build_feature_matrix(df)[:1] * 1), min_cluster_size=10) \
        if False else None  # placeholder to keep import order; real call below
    X, _ = C.build_feature_matrix(df)
    res = C.cluster_deposits(X, min_cluster_size=10)
    prof = C.cluster_profile(df, res.labels)
    assert "cluster_id" in prof.columns and "size" in prof.columns
    assert prof["size"].sum() == len(df)  # every deposit accounted for (incl. noise row)
    assert "area_px" in prof.columns  # mean feature columns present
```

- [ ] **Step 2: Run to verify fails**

Run: `.venv/bin/python -m pytest tests/test_clustering.py -k cluster_profile -q`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
# append to scat/clustering.py
_PROFILE_COLS = ["area_px", "circularity", "aspect_ratio", "mean_hue",
                 "mean_saturation", "mean_lightness", "pigment_density", "iod"]


def cluster_profile(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """One row per cluster id (incl. -1 noise): size + mean of each raw feature. High per-feature
    spread is a mixed-cluster hint (surfaced in the report)."""
    work = df.copy()
    work["cluster_id"] = labels
    cols = [c for c in _PROFILE_COLS if c in work.columns]
    agg = work.groupby("cluster_id").agg(size=("cluster_id", "size"),
                                         **{c: (c, "mean") for c in cols}).reset_index()
    return agg.sort_values("cluster_id").reset_index(drop=True)
```

- [ ] **Step 4: Run to verify passes**

Run: `.venv/bin/python -m pytest tests/test_clustering.py -k cluster_profile -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scat/clustering.py tests/test_clustering.py
git commit -m "feat(clustering): per-cluster feature profile"
```

---

## Task 5: CSV parsing/validation + propagation

**Files:**
- Modify: `scat/clustering.py`
- Test: `tests/test_clustering.py`

- [ ] **Step 1: Write failing test**

```python
def test_parse_cluster_labels_csv_validates(tmp_path):
    p = tmp_path / "cl.csv"
    p.write_text("cluster_id,size,label\n0,10, Normal \n1,5,rod\n2,3,\n")
    mapping = C.parse_cluster_labels_csv(p)
    assert mapping == {0: "normal", 1: "rod"}  # normalized; blank(2) skipped


def test_parse_cluster_labels_csv_rejects_bad_label(tmp_path):
    p = tmp_path / "cl.csv"
    p.write_text("cluster_id,label\n0,banana\n")
    with pytest.raises(ValueError):
        C.parse_cluster_labels_csv(p)


def test_parse_cluster_labels_csv_rejects_duplicate(tmp_path):
    p = tmp_path / "cl.csv"
    p.write_text("cluster_id,label\n0,normal\n0,rod\n")
    with pytest.raises(ValueError):
        C.parse_cluster_labels_csv(p)


def test_propagate_labels_maps_and_defaults_unknown():
    asg = pd.DataFrame({"filename": ["a", "a", "b"], "deposit_id": [0, 1, 0],
                        "cluster_id": [0, -1, 1]})
    labels, summary = C.propagate_labels(asg, {0: "normal", 1: "rod"})
    assert labels[("a", 0)] == "normal"
    assert labels[("a", 1)] == "unknown"   # noise
    assert labels[("b", 0)] == "rod"
    assert summary["n_labeled"] == 2 and summary["n_skipped"] == 1
```

- [ ] **Step 2: Run to verify fails**

Run: `.venv/bin/python -m pytest tests/test_clustering.py -k "csv or propagate" -q`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
# append to scat/clustering.py
VALID_LABELS = {"normal", "rod", "artifact"}


def parse_cluster_labels_csv(path) -> dict:
    """Read cluster_labels.csv → {cluster_id: label}. Normalizes case/whitespace; keeps only
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
    """Map each deposit's cluster_id to a label via `mapping`; noise/unmapped → 'unknown'.
    Returns ({(filename, deposit_id): label}, summary)."""
    labels, n_labeled = {}, 0
    for _, r in assignments.iterrows():
        key = (str(r["filename"]), int(r["deposit_id"]))
        lab = mapping.get(int(r["cluster_id"]), "unknown")
        labels[key] = lab
        if lab != "unknown":
            n_labeled += 1
    return labels, {"n_labeled": n_labeled, "n_skipped": len(labels) - n_labeled}
```

- [ ] **Step 4: Run to verify passes**

Run: `.venv/bin/python -m pytest tests/test_clustering.py -k "csv or propagate" -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add scat/clustering.py tests/test_clustering.py
git commit -m "feat(clustering): strict cluster_labels.csv parsing + label propagation"
```

---

## Task 6: Training-readiness guard

**Files:**
- Modify: `scat/clustering.py`
- Test: `tests/test_clustering.py`

- [ ] **Step 1: Write failing test**

```python
def test_training_readiness_blocks_one_class():
    rep = C.training_readiness(["normal"] * 10)
    assert rep.verdict == "block"
    assert any("class" in r for r in rep.reasons)


def test_training_readiness_blocks_singleton_class():
    rep = C.training_readiness(["normal"] * 5 + ["rod"])  # rod has 1 sample → stratify crashes
    assert rep.verdict == "block"


def test_training_readiness_ok():
    rep = C.training_readiness(["normal"] * 20 + ["rod"] * 20 + ["artifact"] * 10)
    assert rep.verdict in ("ok", "warn")
    assert rep.class_counts == {"normal": 20, "rod": 20, "artifact": 10}
```

- [ ] **Step 2: Run to verify fails**

Run: `.venv/bin/python -m pytest tests/test_clustering.py -k readiness -q`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
# append to scat/clustering.py
from collections import Counter


@dataclass
class ReadinessReport:
    verdict: str
    reasons: list
    class_counts: dict
    n_labeled: int
    n_skipped: int


def training_readiness(member_labels, largest_cluster_share=None) -> ReadinessReport:
    """Gate before `scat train`. Blocks configs that would crash or poison the RF: <2 labeled
    classes, or any class with <2 samples (the trainer's stratified split needs ≥2 per class).
    Warns on extreme imbalance or most labels coming from a single cluster."""
    counts = Counter(l for l in member_labels if l in VALID_LABELS)
    n_labeled = sum(counts.values())
    n_skipped = sum(1 for l in member_labels if l not in VALID_LABELS)
    reasons, verdict = [], "ok"
    if len(counts) < 2:
        verdict = "block"; reasons.append(f"only {len(counts)} labeled class(es); need ≥2 to train")
    singthan = [c for c, n in counts.items() if n < 2]
    if singthan:
        verdict = "block"; reasons.append(f"class(es) with <2 samples: {singthan} (stratified split fails)")
    if verdict != "block" and counts:
        lo, hi = min(counts.values()), max(counts.values())
        if hi >= 10 * max(1, lo):
            verdict = "warn"; reasons.append(f"extreme class imbalance ({dict(counts)})")
        if largest_cluster_share and largest_cluster_share > 0.8:
            verdict = "warn"; reasons.append(f"{largest_cluster_share:.0%} of labels from one cluster")
    return ReadinessReport(verdict=verdict, reasons=reasons, class_counts=dict(counts),
                           n_labeled=n_labeled, n_skipped=n_skipped)
```

- [ ] **Step 4: Run to verify passes**

Run: `.venv/bin/python -m pytest tests/test_clustering.py -k readiness -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add scat/clustering.py tests/test_clustering.py
git commit -m "feat(clustering): training-readiness guard (blocks <2 classes / singleton classes)"
```

---

## Task 7: `cluster_folder_service` (pipeline I/O)

**Files:**
- Modify: `scat/pipeline.py`
- Test: `tests/test_clustering.py` (integration, uses `synth_dir`)

- [ ] **Step 1: Write failing test**

```python
# add to tests/test_clustering.py
def test_cluster_folder_service_writes_outputs(synth_dir, tmp_path):
    from scat.pipeline import cluster_folder_service
    out = tmp_path / "clus"
    summ = cluster_folder_service(str(synth_dir), output_dir=str(out), min_cluster_size=3)
    assert (out / "cluster_assignments.csv").exists()
    assert (out / "cluster_labels.csv").exists()
    assert (out / "cluster_report.html").exists()
    labels_jsons = list((out / "deposits").glob("*.labels.json"))
    assert labels_jsons
    import json
    data = json.loads(labels_jsons[0].read_text())
    assert all(d["label"] == "unknown" for d in data["deposits"])          # unlabeled
    assert all("cluster_id" in d for d in data["deposits"])                # informational field
    assert summ.n_deposits > 0
```

- [ ] **Step 2: Run to verify fails**

Run: `.venv/bin/python -m pytest tests/test_clustering.py -k cluster_folder_service -q`
Expected: FAIL (`cannot import cluster_folder_service`).

- [ ] **Step 3: Implement**

Add to `scat/pipeline.py` (near the other services). Reuses `list_images`, `resolve_model_type`, `Analyzer`, `DepositDetector`, `ClassifierConfig`, `FeatureExtractor`. Cluster works from in-memory results so it includes ALL deposits (incl. RF "artifact").

```python
# scat/pipeline.py  (imports already present at top of file; add dataclass + numpy/json locally)
from dataclasses import dataclass


@dataclass
class ClusterSummary:
    output_dir: str
    n_images: int
    n_deposits: int
    n_clusters: int
    n_noise: int
    health: list


def cluster_folder_service(path, output_dir=None, method="hdbscan", min_cluster_size=None,
                           k=None, reps_per_cluster=6) -> ClusterSummary:
    import json
    import numpy as np
    import pandas as pd
    from PIL import Image
    from . import clustering as C
    from .detector import DepositDetector
    from .classifier import ClassifierConfig
    from .analyzer import Analyzer
    from .features import FeatureExtractor

    images = list_images(str(path))
    mtype, mpath = resolve_model_type(None, None)
    az = Analyzer(detector=DepositDetector(),
                  classifier_config=ClassifierConfig(model_type=mtype, model_path=mpath))
    results = az.analyze_batch(images)

    # one feature row per deposit across all images
    rows, keys = [], []
    for res in results:
        fx = FeatureExtractor(dpi=res.dpi)
        for d in res.deposits:
            fd = fx.to_feature_dict(d)
            fd["filename"] = res.filename
            fd["deposit_id"] = d.id
            rows.append(fd)
            keys.append((res.filename, d.id))
    df = pd.DataFrame(rows)

    out = Path(output_dir) if output_dir else get_timestamped_output_dir(Path(path).parent, "clusters")
    out.mkdir(parents=True, exist_ok=True)

    if df.empty:
        (out / "cluster_assignments.csv").write_text("filename,deposit_id,cluster_id\n")
        return ClusterSummary(str(out), len(images), 0, 0, 0, ["no deposits detected"])

    X, feat_names = C.build_feature_matrix(df)
    cres = C.cluster_deposits(X, method=method, min_cluster_size=min_cluster_size, k=k)
    df["cluster_id"] = cres.labels

    # 1) assignments (authoritative for propagation)
    df[["filename", "deposit_id", "cluster_id"]].to_csv(out / "cluster_assignments.csv", index=False)

    # 2) labels.json per image (label="unknown" + informational cluster_id)
    _write_cluster_labels_json(out / "deposits", results, df)

    # 3) representatives + thumbnails
    reps = C.representatives(X, cres.labels, per_kind=reps_per_cluster)
    _export_cluster_thumbnails(out / "clusters", df, reps, images)

    # 4) report + labels CSV template
    profile = C.cluster_profile(df, cres.labels)
    _write_cluster_report_html(out / "cluster_report.html", profile, reps, df, cres)
    _write_cluster_labels_csv(out / "cluster_labels.csv", profile, cres)

    return ClusterSummary(str(out), len(images), len(df), cres.n_clusters, cres.n_noise, cres.health)
```

Add the private helpers (same file). Keep them small and focused:

```python
def _write_cluster_labels_json(deposits_dir, results, df):
    import json
    deposits_dir.mkdir(parents=True, exist_ok=True)
    cid_by_key = {(r.filename, int(r.deposit_id)): int(r.cluster_id) for r in df.itertuples()}
    for res in results:
        stem = Path(res.filename).stem
        deps = []
        for d in res.deposits:
            deps.append({
                "id": d.id,
                "contour": d.contour.squeeze().tolist() if d.contour is not None else [],
                "x": d.centroid[0], "y": d.centroid[1], "width": d.width, "height": d.height,
                "area": float(d.area), "circularity": float(d.circularity),
                "label": "unknown", "confidence": 0.0,
                "cluster_id": cid_by_key.get((res.filename, d.id), -1),
                "merged": getattr(d, "merged", False), "group_id": getattr(d, "group_id", None),
            })
        with open(deposits_dir / f"{stem}.labels.json", "w") as f:
            json.dump({"image_file": res.filename, "next_group_id": 1, "deposits": deps}, f, indent=2)


def _export_cluster_thumbnails(clusters_dir, df, reps, images):
    import numpy as np
    from PIL import Image
    img_by_name = {Path(p).name: p for p in images}
    row_by_pos = {i: row for i, row in enumerate(df.itertuples())}
    cache = {}
    for cid, kinds in reps.items():
        cdir = clusters_dir / f"cluster_{cid}"
        cdir.mkdir(parents=True, exist_ok=True)
        for kind, idxs in kinds.items():
            for j, pos in enumerate(idxs):
                r = row_by_pos[pos]
                fn = r.filename
                if fn not in cache:
                    cache[fn] = np.array(Image.open(img_by_name[fn]))
                arr = cache[fn]
                x, y, w, h = int(r.x), int(r.y), int(getattr(r, "width", 20)), int(getattr(r, "height", 20))
                pad = 4
                y0, y1 = max(0, y - h // 2 - pad), min(arr.shape[0], y + h // 2 + pad)
                x0, x1 = max(0, x - w // 2 - pad), min(arr.shape[1], x + w // 2 + pad)
                crop = arr[y0:y1, x0:x1]
                if crop.size:
                    Image.fromarray(crop).save(cdir / f"{kind}_{j}.png")
```

> Note: `df.itertuples()` exposes `x`/`y`/`width`/`height` only if those columns exist. `to_feature_dict` provides `x, y, width, height` (verified in features.py). Keep the getattr fallbacks for safety.

```python
def _write_cluster_labels_csv(path, profile, cres):
    import pandas as pd
    keep = ["cluster_id", "size", "area_px", "circularity", "mean_hue", "pigment_density"]
    cols = [c for c in keep if c in profile.columns]
    out = profile[profile["cluster_id"] != -1][cols].copy()
    out["noise_frac"] = round(cres.n_noise / max(1, len(profile["cluster_id"])), 3)
    out["label"] = ""  # user fills
    out.to_csv(path, index=False)


def _write_cluster_report_html(path, profile, reps, df, cres):
    import base64
    def _b64(p):
        return base64.b64encode(Path(p).read_bytes()).decode() if Path(p).exists() else ""
    parts = ["<!DOCTYPE html><meta charset='utf-8'><title>SCAT clusters</title>",
             "<style>body{font-family:sans-serif;max-width:1000px;margin:2rem auto}"
             ".c{border:1px solid #ddd;border-radius:8px;padding:12px;margin:12px 0}"
             "img{height:64px;margin:2px;border:1px solid #eee}.warn{color:#b26a00}</style>",
             f"<h1>Cluster report — {cres.n_clusters} clusters, {cres.n_noise} noise</h1>"]
    for h in cres.health:
        parts.append(f"<p class='warn'>⚠ {h}</p>")
    clus_root = path.parent / "clusters"
    for _, row in profile.iterrows():
        cid = int(row["cluster_id"])
        title = "Outliers (noise −1) — review individually, NOT auto-artifact" if cid == -1 else f"Cluster {cid}"
        parts.append(f"<div class='c'><h3>{title} — size {int(row['size'])}</h3>")
        if cid in reps:
            for kind in ("medoid", "random", "boundary"):
                for j in range(len(reps[cid][kind])):
                    b = _b64(clus_root / f"cluster_{cid}" / f"{kind}_{j}.png")
                    if b:
                        parts.append(f"<img title='{kind}' src='data:image/png;base64,{b}'>")
        feats = ", ".join(f"{c}={row[c]:.2f}" for c in ("area_px", "circularity", "mean_hue",
                          "pigment_density") if c in profile.columns)
        parts.append(f"<p>{feats}</p></div>")
    Path(path).write_text("".join(parts), encoding="utf-8")
```

- [ ] **Step 4: Run to verify passes**

Run: `.venv/bin/python -m pytest tests/test_clustering.py -k cluster_folder_service -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scat/pipeline.py tests/test_clustering.py
git commit -m "feat(pipeline): cluster_folder_service (assignments + labels.json + thumbnails + report + CSV)"
```

---

## Task 8: `propagate_service` (pipeline I/O)

**Files:**
- Modify: `scat/pipeline.py`
- Test: `tests/test_clustering.py`

- [ ] **Step 1: Write failing test**

```python
def test_propagate_service_populates_labels_and_guard(synth_dir, tmp_path):
    import json, pandas as pd
    from scat.pipeline import cluster_folder_service, propagate_service
    out = tmp_path / "clus"
    cluster_folder_service(str(synth_dir), output_dir=str(out), min_cluster_size=3)
    # label two clusters as different classes so the guard passes
    cl = pd.read_csv(out / "cluster_labels.csv")
    assert len(cl) >= 2
    cl.loc[0, "label"] = "normal"
    cl.loc[1, "label"] = "rod"
    cl.to_csv(out / "cluster_labels.csv", index=False)
    summ = propagate_service(str(out))
    assert summ.n_labeled > 0
    assert summ.readiness in ("ok", "warn", "block")
    # labels.json now has some normal/rod labels
    seen = set()
    for p in (out / "deposits").glob("*.labels.json"):
        for d in json.loads(p.read_text())["deposits"]:
            seen.add(d["label"])
    assert seen & {"normal", "rod"}
```

- [ ] **Step 2: Run to verify fails**

Run: `.venv/bin/python -m pytest tests/test_clustering.py -k propagate_service -q`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
# scat/pipeline.py
@dataclass
class PropagateSummary:
    n_labeled: int
    n_skipped: int
    readiness: str
    reasons: list
    class_counts: dict


def propagate_service(results_dir, csv_path=None) -> PropagateSummary:
    import json
    import pandas as pd
    from . import clustering as C

    rd = Path(results_dir)
    assignments = pd.read_csv(rd / "cluster_assignments.csv")
    mapping = C.parse_cluster_labels_csv(Path(csv_path) if csv_path else rd / "cluster_labels.csv")
    labels, summary = C.propagate_labels(assignments, mapping)

    # rewrite labels.json labels from the assignments-derived mapping (keyed by filename+id)
    for p in (rd / "deposits").glob("*.labels.json"):
        data = json.loads(p.read_text())
        fn = data.get("image_file", p.stem.replace(".labels", ""))
        for d in data["deposits"]:
            d["label"] = labels.get((fn, int(d["id"])), d.get("label", "unknown"))
        p.write_text(json.dumps(data, indent=2))

    # readiness guard: which cluster contributes the most labels?
    share = None
    if summary["n_labeled"]:
        by_cluster = assignments.assign(_lab=[labels.get((str(f), int(i)), "unknown")
                                              for f, i in zip(assignments["filename"], assignments["deposit_id"])])
        lab_counts = by_cluster[by_cluster["_lab"] != "unknown"].groupby("cluster_id").size()
        share = float(lab_counts.max() / lab_counts.sum()) if len(lab_counts) else None
    rep = C.training_readiness(list(labels.values()), largest_cluster_share=share)
    return PropagateSummary(summary["n_labeled"], summary["n_skipped"], rep.verdict,
                            rep.reasons, rep.class_counts)
```

- [ ] **Step 4: Run to verify passes**

Run: `.venv/bin/python -m pytest tests/test_clustering.py -k propagate_service -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scat/pipeline.py tests/test_clustering.py
git commit -m "feat(pipeline): propagate_service (assignments→labels.json + training-readiness guard)"
```

---

## Task 9: CLI subcommands `cluster` / `propagate`

**Files:**
- Modify: `scat/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_cli.py  (append)
def test_cli_cluster_and_propagate_roundtrip(synth_dir, tmp_path, capsys):
    import pandas as pd
    from scat.cli import main
    out = tmp_path / "clus"
    main(["cluster", str(synth_dir), "--output", str(out), "--min-cluster-size", "3"])
    assert (out / "cluster_labels.csv").exists()
    cl = pd.read_csv(out / "cluster_labels.csv")
    if len(cl) >= 2:
        cl.loc[0, "label"] = "normal"; cl.loc[1, "label"] = "rod"
        cl.to_csv(out / "cluster_labels.csv", index=False)
        main(["propagate", str(out)])
        txt = capsys.readouterr().out
        assert "labeled" in txt.lower()
```

- [ ] **Step 2: Run to verify fails**

Run: `.venv/bin/python -m pytest tests/test_cli.py -k cluster_and_propagate -q`
Expected: FAIL (unknown subcommand `cluster`).

- [ ] **Step 3: Implement**

```python
# scat/cli.py
def cluster_command(args):
    from .pipeline import cluster_folder_service
    s = cluster_folder_service(args.folder, output_dir=args.output, method=args.method,
                               min_cluster_size=args.min_cluster_size, k=args.k,
                               reps_per_cluster=args.reps_per_cluster)
    print(f"Clustered {s.n_deposits} deposits from {s.n_images} images -> {s.n_clusters} clusters, "
          f"{s.n_noise} noise")
    for h in s.health:
        print(f"  ⚠ {h}")
    print(f"Next: open {s.output_dir}/cluster_report.html, fill the 'label' column of "
          f"{s.output_dir}/cluster_labels.csv, then: scat propagate {s.output_dir}")


def propagate_command(args):
    from .pipeline import propagate_service
    s = propagate_service(args.results_dir, csv_path=args.labels)
    print(f"Propagated: {s.n_labeled} deposits labeled, {s.n_skipped} left unknown. "
          f"Classes: {s.class_counts}")
    print(f"Training readiness: {s.readiness.upper()}")
    for r in s.reasons:
        print(f"  - {r}")
    if s.readiness == "block":
        print("Refusing to recommend training — fix the above (label more/other clusters).")
    else:
        print(f"Next: scat train {args.results_dir}")
```

Wire the parsers (inside the existing `main()` arg-parser block, next to `train`):

```python
    # cluster
    cp = subparsers.add_parser('cluster', help='Unsupervised-cluster deposits to assist labeling')
    cp.add_argument('folder')
    cp.add_argument('--output')
    cp.add_argument('--method', choices=['hdbscan', 'kmeans'], default='hdbscan')
    cp.add_argument('--k', type=int, default=None, help='n_clusters for kmeans')
    cp.add_argument('--min-cluster-size', type=int, default=None, dest='min_cluster_size')
    cp.add_argument('--reps-per-cluster', type=int, default=6, dest='reps_per_cluster')
    cp.set_defaults(func=cluster_command)
    # propagate
    pp = subparsers.add_parser('propagate', help='Propagate cluster labels to deposits, guard training')
    pp.add_argument('results_dir')
    pp.add_argument('--labels', default=None, help='cluster_labels.csv (default: in results_dir)')
    pp.set_defaults(func=propagate_command)
```

- [ ] **Step 4: Run to verify passes**

Run: `.venv/bin/python -m pytest tests/test_cli.py -k cluster_and_propagate -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scat/cli.py tests/test_cli.py
git commit -m "feat(cli): scat cluster + scat propagate subcommands"
```

---

## Task 10: Adversarial + smoke coverage, full suite

**Files:**
- Modify: `tests/test_clustering.py`, `tests/test_smoke.py`

- [ ] **Step 1: Add adversarial unit tests**

```python
# tests/test_clustering.py
def test_propagate_unknown_cluster_id_defaults_unknown():
    asg = pd.DataFrame({"filename": ["a"], "deposit_id": [0], "cluster_id": [7]})
    labels, summary = C.propagate_labels(asg, {0: "normal"})  # cluster 7 not mapped
    assert labels[("a", 0)] == "unknown" and summary["n_labeled"] == 0


def test_all_noise_clustering_is_handled():
    rng = np.random.RandomState(9)
    X = rng.randn(50, 2) * 5  # sparse → mostly noise
    res = C.cluster_deposits(X, min_cluster_size=25)
    prof = C.cluster_profile(_df(50), res.labels)  # must not crash with a -1-only profile
    assert prof["size"].sum() == 50
```

- [ ] **Step 2: Register the new module in smoke**

Add `"scat.clustering"` to `MODULES` in `tests/test_smoke.py`.

- [ ] **Step 3: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS (all prior + new).

- [ ] **Step 4: Real end-to-end sanity**

Run: `.venv/bin/python -m scat.cli cluster <synth folder> --output /tmp/clus --min-cluster-size 3`
Expected: prints cluster counts + writes report/CSV; then fill 2 labels and `scat propagate /tmp/clus` prints readiness.

- [ ] **Step 5: Commit**

```bash
git add tests/test_clustering.py tests/test_smoke.py
git commit -m "test(clustering): adversarial cases + smoke registration"
```

---

## Task 11: Docs

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document the workflow**

Add a "Cluster-assisted labeling" section: `scat cluster <folder>` → review `cluster_report.html` → fill `cluster_labels.csv` (one label per cluster, blank to skip) → `scat propagate <dir>` → `scat train <dir>`. State the honest caveat: clusters are suggestions; skip mixed clusters; the readiness guard blocks degenerate training.

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: cluster-assisted labeling workflow"
```

---

## Self-Review Notes

- **Spec coverage:** feature eng (T1), HDBSCAN+health+kmeans (T2), representatives 3-kind (T3), profile (T4), CSV validation + propagation (T5), readiness guard (T6), service + assignments/labels.json/thumbnails/report/CSV (T7), propagate service + guard (T8), CLI (T9), adversarial tests + smoke (T10), docs (T11). All spec sections mapped.
- **`to_feature_dict` keys** used (`x,y,width,height,area_px,perimeter,circularity,aspect_ratio,mean_hue,mean_saturation,mean_lightness,pigment_density,iod,id`) verified against `scat/features.py`.
- **Signatures consistent:** `build_feature_matrix→(X,names)`, `cluster_deposits→ClusterResult`, `representatives→{cid:{kind:[idx]}}`, `parse_cluster_labels_csv→dict`, `propagate_labels→(labels,summary)`, `training_readiness→ReadinessReport` — used identically in T7/T8/T9.
- **Additive-only:** new files + new labels.json under a NEW results dir; no change to existing CSV outputs → parity gate unaffected.
- **Known follow-up during impl:** confirm `to_feature_dict` emits `perimeter` (used by feature matrix); if absent, drop it from `_LOG_COLS` — verify in Task 1.
