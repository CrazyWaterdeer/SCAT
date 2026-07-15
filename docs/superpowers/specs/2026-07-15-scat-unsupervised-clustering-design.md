# SCAT — Unsupervised clustering for labeling assistance

**Status:** design (Codex-reviewed) · 2026-07-15 · branch `feat/unsupervised-clustering`

## Problem / goal

SCAT's classification (Normal / ROD / Artifact) is fully supervised. The user has **piles of
unlabeled images** and wants to avoid labeling every deposit.

**Honest framing (sharpened after Codex review):** unsupervised clustering is **labeling
*assistance*, not "train without labels."** A cluster is not a class — the same visual features
can reflect lighting/segmentation-size/plate-batch as much as biology, and a cluster's tails and
boundaries can be mixed even when its medoid looks clean. So the design is a **human-in-the-loop,
QA-gated** flow: clustering proposes groups + shows purity evidence; the user labels clusters they
judge pure and **skips** mixed ones; a pre-train guard blocks poisoned/degenerate training.

> unlabeled images → detect deposits (already unsupervised) → cluster deposit features →
> per cluster the user sees medoid + random + boundary thumbnails and either assigns ONE label or
> **skips** → propagate the label to members → **quality guard** → train the existing RF.

Primary goal: **cut labeling from hundreds of deposits to a handful of cluster decisions, without
silently poisoning the model.** Bonus: the cluster report is the "natural group discovery." Scope
(user): **CLI-first**, reuse existing detection/features/training; GUI button + deep embeddings are
follow-ups.

## Non-goals

- Deep / self-supervised embeddings (pluggable later behind the feature-matrix seam).
- A labeling-GUI "cluster" button (follow-up).
- Iterative active learning (one-shot cluster→label→propagate; re-run manually).
- Batch-effect correction; changing detection, feature math (`mean_hue` stays non-circular for the
  RF model — ROADMAP), or the training code.

## Data flow

```
<folder> of unlabeled images
  → analyzer.analyze_batch (detect + features; parallel engine)  → in-memory AnalysisResults
      (ALL deposits incl. RF-called "artifact"; each Deposit carries geometry + features)
  → clustering.build_feature_matrix → transform+scale → cluster_deposits (HDBSCAN)
  → writes <results_dir>/:
      • cluster_assignments.csv     (filename, deposit_id, cluster_id)  ← propagation source of truth
      • deposits/<img>.labels.json  geometry + cluster_id(informational) + label="unknown"
      • clusters/cluster_<id>/{medoid,random,boundary}_*.png   QA thumbnails
      • cluster_report.html         per cluster: size, 3 sample sets, feature profile, purity hints
      • cluster_labels.csv          cluster_id,size,noise_frac,<feature means>,label(BLANK)
  → USER fills `label` per cluster (normal/rod/artifact) or leaves BLANK to skip — the manual step
  → scat propagate <results_dir>   validate CSV → assignments → member labels; prints QUALITY GUARD
  → scat train <results_dir>       existing RF (guard already confirmed ≥2 classes, enough samples)
```

## Components

### 1. `scat/clustering.py` (pure core; arrays/DataFrames in, results out)

**Feature engineering** (`build_feature_matrix(df) -> (X, ids, feat_names)`):
- Columns (intrinsic shape+colour+density; NOT id/x/y/width/height/label/confidence):
  `area_px, perimeter, circularity, aspect_ratio, mean_saturation, mean_lightness,
  pigment_density, iod`, plus **hue** and **area/perimeter/iod** handled specially:
  - **Skewed positive magnitudes** (`area_px, perimeter, iod`) → `log1p` before scaling (else a few
    huge deposits dominate the structure several times over — they are correlated). *(Codex #4)*
  - **Hue** → `(sin, cos)` of `mean_hue`, each **multiplied by `mean_saturation`** so near-gray,
    low-saturation deposits don't get arbitrary hue coordinates. *(Codex #5)* Clustering-only; does
    NOT touch the model's `mean_hue`.
  - NaN → column **median** impute.
- Then `StandardScaler` on the assembled matrix; return the fitted scaler + names for the report.

**Clustering** (`cluster_deposits(X, method="hdbscan", min_cluster_size=None, min_samples=None,
k=None, random_state=0) -> ClusterResult`):
- Defaults (concrete, *Codex #2*): `min_cluster_size = max(15, round(n/40))` clamped to `[15, 200]`;
  `min_samples = None` (HDBSCAN default = min_cluster_size); `metric="euclidean"`;
  `cluster_selection_method="eom"`. Label `-1` = noise/outlier.
- **kmeans** (`--method kmeans --k`): escape hatch when HDBSCAN over-noises; labels EVERY point
  (no noise) so junk gets absorbed — the report/CSV flag this and recommend HDBSCAN. *(Codex #14)*
- **Health flags** returned + surfaced as WARNINGS (not just summary): `n < 4*min_cluster_size`
  (too few deposits to cluster meaningfully → suggest labeling manually), `noise_frac > 0.5`,
  `n_clusters < 2`, or one cluster holding `> 0.8` of clustered points. *(Codex #2/#11)*

**Representatives** (`representatives(Xs, labels, per_kind=6) -> {cluster_id: {medoid, random,
boundary}}`): three sample sets per cluster — **medoid** (nearest centroid), **random**, and
**boundary** (farthest-from-medoid within the cluster) — so mixed clusters reveal themselves
(medoids alone hide impurity). Stratify random samples across source images where possible.
*(Codex #1/#3/#10)* Capped per kind for huge clusters. *(Codex #11)*

**Profile / propagation**:
- `cluster_profile(df, labels) -> DataFrame` — per cluster size, noise flag, per-feature mean/std
  (purity hint: high spread ⇒ likely mixed).
- `propagate_labels(assignments_df, mapping) -> dict` — validate + map. **Validation** *(Codex #7)*:
  normalize labels (`strip().lower()`); accept only `{normal, rod, artifact}` or blank(=skip);
  error on any other value, duplicate `cluster_id`, or a `cluster_id` absent from the run; `-1`
  and unmapped clusters → `unknown`. Returns per-deposit label + a coverage/skip summary.
- `training_readiness(labels) -> ReadinessReport` — the **quality guard** *(Codex #8/#12)*: counts
  of labeled vs skipped/unknown, per-class counts, #clusters contributing to each class, the
  largest single cluster's share of labels, noise fraction. Verdict `ok | warn | block` with
  reasons; **block** when `<2 classes` or **any labeled class has `<2` samples** (the RF trainer's
  `stratify=y` split crashes otherwise — verified in trainer.py), warn on extreme imbalance or
  `>0.8` of labels from one cluster.

### 2. `scat/pipeline.py`

- `cluster_folder_service(path, output_dir=None, method, min_cluster_size, k, reps_per_cluster)` —
  runs `analyze_batch` (all deposits, incl. artifacts), builds the matrix from `to_feature_dict`,
  clusters, writes assignments.csv + labels.json (`label="unknown"` + informational `cluster_id`)
  + QA thumbnails + report.html + cluster_labels.csv. Returns a summary incl. health flags.
- `propagate_service(results_dir, csv_path=None) -> PropagateSummary` — read+validate
  `cluster_labels.csv`, join to `cluster_assignments.csv` (NOT the GUI-mutable labels.json —
  *Codex #9*), write member labels into the labels.json, run `training_readiness`, return coverage
  + the readiness verdict.

### 3. `scat/cli.py`

- `scat cluster <folder> [--output DIR] [--method hdbscan|kmeans] [--k N] [--min-cluster-size M]
  [--reps-per-cluster R]` — prints n clusters / sizes / **noise fraction** / health WARNINGS + where
  to fill `cluster_labels.csv`.
- `scat propagate <results_dir> [--labels cluster_labels.csv]` — prints coverage + the **quality
  guard verdict**; on `block`, refuses and explains; on `warn`, proceeds but flags. Reminds next
  step `scat train <results_dir>`.

## labels.json compatibility

`cluster_assignments.csv` (filename, deposit_id, cluster_id) is the **authoritative** cluster
membership — propagation reads IT, never a `cluster_id` round-tripped through the GUI (whose
`_save_to_path` rebuilds deposits from a fixed field set and would drop it — verified). The
`cluster_id` written into labels.json is purely informational. `cluster` writes `label="unknown"`;
`propagate` sets `{normal,rod,artifact}` for labeled clusters, leaves the rest `unknown`; the
trainer keeps only `{normal,rod,artifact}` (verified `load_labeled_data`), so skipped/noise are
excluded — no trainer change. *(Codex #6/#8/#9)*

## Reports / the one manual step

- `cluster_report.html` (self-contained base64, like `report.py`): one card per cluster with its
  **medoid + random + boundary** thumbnails, size, noise flag, and feature profile; a separate
  "outliers (noise, `-1`) — review individually, NOT auto-artifact" card. Thumbnails capped;
  clusters beyond a cap are summarized. *(Codex #1/#6/#11)*
- `cluster_labels.csv`: `cluster_id,size,noise_frac,area_px,circularity,mean_hue,pigment_density,
  label` (label BLANK). The user types `normal`/`rod`/`artifact` or leaves blank to skip.

## Testing (`tests/test_clustering.py` + `tests/test_cli.py`)

Beyond happy-path blobs, adversarial cases *(Codex #13)*: one giant cluster; all-noise; a
deliberately **mixed** cluster (propagation still one-label, guard/report flags spread); invalid
CSV label; duplicate/unknown `cluster_id` rows; blank(skip) rows; tiny `n` (health warn); the
`training_readiness` verdicts (block on 1 class / a class with 1 sample; warn on imbalance);
feature engineering (log1p, sat-weighted hue, NaN impute); and a full CLI round-trip
`cluster → fill CSV → propagate → train` on `synth_dir`. Full suite + parity gate unaffected
(additive, new files only).

## Risks (residual, honest)

| risk | mitigation |
|---|---|
| cluster ≠ class → poisoned RF labels | 3-sample QA per cluster, skip path, `training_readiness` guard that blocks degenerate training; framed as assistance not automation |
| features reflect lighting/batch not biology | image-stratified samples + feature profile in report; documented limitation (no batch correction) |
| HDBSCAN over-noises / one giant cluster / tiny n | concrete defaults + health WARNINGS + `--min-cluster-size`/`--k`/kmeans escape hatches |
| skewed/correlated features dominate | log1p on area/perimeter/iod, StandardScaler, sat-weighted hue |
| GUI drops informational `cluster_id` | propagation uses the sidecar assignments.csv, not labels.json |
| trainer crashes on <2 classes / <2 samples per class | `training_readiness` blocks before `scat train` |

## Follow-ups (deferred)
Deep self-supervised embeddings behind `build_feature_matrix`; a labeling-GUI "cluster & label"
button (with the 3-sample QA view); iterative active learning; an agent `cluster_folder` tool;
batch-effect diagnostics.
