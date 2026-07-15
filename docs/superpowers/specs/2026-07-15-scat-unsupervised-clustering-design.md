# SCAT — Unsupervised clustering for labeling assistance

**Status:** design · 2026-07-15 · branch `feat/unsupervised-clustering`

## Problem / goal

SCAT's classification (Normal / ROD / Artifact) is fully supervised — every classifier
(Threshold / RF / CNN) trains from hand-labeled `*.labels.json`. The user has **piles of
unlabeled images** and wants to "train without labeling everything."

Unsupervised learning cannot invent the biological classes (Normal/ROD/Artifact are the
user's definitions), but it can **discover the natural groups** in the data and thereby
**collapse the labeling job from hundreds of deposits to a handful of clusters**:

> unlabeled images → detect deposits (already unsupervised) → cluster deposits by their
> features → the user labels each *cluster* once (5–20 labels) → propagate the cluster label
> to all its member deposits → train the existing RF classifier.

Primary goal: **reduce labeling burden.** Bonus: the cluster report **is** the "natural
group discovery." Chosen scope (user): **CLI-first**, reuse existing detection/features/
training; GUI button + deep embeddings are explicit follow-ups.

## Non-goals

- Deep / self-supervised embeddings (pluggable later behind the feature-matrix seam).
- A new labeling GUI or GUI "cluster" button (follow-up).
- Iterative active-learning loops (one-shot cluster→label→propagate; re-run manually).
- Changing detection, feature math (`mean_hue` stays non-circular for the RF model — see
  ROADMAP), or the training code.

## Data flow

```
unlabeled images/<folder>
  → DepositDetector.detect  (unsupervised, existing)
  → FeatureExtractor        (existing, per-deposit feature dicts)
  → clustering.build_feature_matrix → StandardScaler → clustering.cluster_deposits (HDBSCAN)
  → writes into <results_dir>/:
      • deposits/<img>.labels.json   each deposit: contour/pos/features + cluster_id + label="unknown"
      • clusters/cluster_<id>/rep_*.png   representative (medoid) deposit thumbnails
      • cluster_report.html          per-cluster: size, medoid thumbnails, feature profile, outliers
      • cluster_labels.csv           columns: cluster_id, size, <key feature means>, label(BLANK)
  → USER fills the `label` column of cluster_labels.csv  (looking at the report) — the only manual step
  → scat propagate <results_dir>    cluster label → every member deposit's label in the labels.json
  → scat train <results_dir>        existing RF training on the now-labeled labels.json
```

## Components (each single-purpose, testable)

### 1. `scat/clustering.py` (pure core — arrays/DataFrames in, results out; no I/O of images/CLI)

- `FEATURE_COLUMNS` — the intrinsic per-deposit features used for clustering (shape + colour +
  density), deliberately **excluding** identity/position (`id, x, y, width, height, label,
  confidence`): `area_px, perimeter, circularity, aspect_ratio, mean_saturation,
  mean_lightness, pigment_density, iod` + **hue encoded as `(sin, cos)`** of `mean_hue`
  (hue is circular 0–360; raw Euclidean distance would wrongly split reds at the 0/360 wrap —
  this transform is clustering-only and does NOT touch the model's `mean_hue`).
- `build_feature_matrix(df, columns=FEATURE_COLUMNS) -> (X, ids, feat_names)` — assemble the
  numeric matrix from an `all_deposits`-style DataFrame; NaN → column **median** impute;
  returns row→deposit id mapping so results map back.
- `cluster_deposits(X, method="hdbscan", min_cluster_size=None, k=None, random_state=0) ->
  ClusterResult` — `StandardScaler().fit_transform(X)` then:
  - **hdbscan (default):** `sklearn.cluster.HDBSCAN(min_cluster_size=…)`; auto #clusters,
    label `-1` = noise/outlier (→ artifact candidates). Default `min_cluster_size =
    max(5, n // 50)` (tunable via CLI).
  - **kmeans:** `KMeans(n_clusters=k)` when the user prefers a fixed count.
  `ClusterResult` = `{labels, method, n_clusters, n_noise, scaler, feat_names}`.
- `representatives(Xs, labels, per_cluster=6) -> {cluster_id: [row_idx...]}` — medoids: the
  `per_cluster` rows nearest to each cluster's centroid (in scaled space).
- `cluster_profile(df, labels) -> DataFrame` — per cluster: size + mean of each raw feature
  (the discovery summary; drives the report + CSV template).
- `propagate_labels(cluster_ids, mapping) -> list[str]` — map each deposit's `cluster_id` to
  `mapping[cluster_id]`; unmapped / `-1` → `"unknown"`. (Majority is not needed — the user
  labels whole clusters — but noise and unlabeled clusters resolve to `unknown`, which the
  trainer already skips.)

### 2. `scat/pipeline.py` — `cluster_folder_service(path, output_dir=None, method, min_cluster_size, k, reps_per_cluster) -> ClusterResultSummary`

Mirrors the other services (single seam for CLI/GUI/agent). Runs detection+features via the
existing analyzer (parallel engine included — free speedup), writes the labels.json (reusing
`ReportGenerator._save_contour_json`, extended with an additive `cluster_id` + `label="unknown"`),
crops medoid thumbnails, writes `cluster_report.html` + `cluster_labels.csv`. Returns counts.

`propagate_service(results_dir, csv_path=None) -> PropagateSummary` — read `cluster_labels.csv`
(default in the dir), set each deposit's `label` from its `cluster_id`, rewrite labels.json,
report coverage (deposits labeled / clusters still blank).

### 3. `scat/cli.py` — two subcommands

- `scat cluster <folder> [--output DIR] [--method hdbscan|kmeans] [--k N] [--min-cluster-size M]
  [--reps-per-cluster R]` → runs the service, prints the discovery summary (n clusters, sizes,
  noise count) + where to fill `cluster_labels.csv`.
- `scat propagate <results_dir> [--labels cluster_labels.csv]` → runs propagate, prints coverage,
  and reminds the next step is `scat train <results_dir>`.

## labels.json compatibility

The existing schema (`{image_file, next_group_id, deposits:[{id, contour, x, y, width, height,
area, circularity, label, confidence, group_id, merged}]}`) is extended with an **additive**
`cluster_id` field per deposit; existing readers ignore it. `cluster` writes `label="unknown"`;
`propagate` sets it to the cluster's label. The trainer already keeps only deposits whose label
∈ `{normal, rod, artifact}`, so noise/unlabeled clusters are naturally excluded from training —
no trainer change. The labeling GUI can still open a propagated labels.json for manual
correction (must verify it round-trips the additive `cluster_id`; if it drops unknown keys, add a
one-line preserve — verified during implementation).

## Reports / UX of the one manual step

- `cluster_report.html` (self-contained, base64 like `report.py`): one card per cluster with its
  medoid thumbnails, size, and mean feature profile; a "noise / outliers" card. This is what the
  user looks at to decide each cluster's label. Reuses the base64-figure pattern.
- `cluster_labels.csv`: `cluster_id,size,area_px,circularity,mean_hue,pigment_density,label` with
  `label` blank — the user types `normal`/`rod`/`artifact` (or leaves blank to skip). 5–20 rows.

## Testing

- **Unit (`tests/test_clustering.py`)** on synthetic feature matrices with known groups:
  `build_feature_matrix` (NaN impute, hue sin/cos, column selection), `cluster_deposits`
  (recovers well-separated blobs; noise handling), `representatives` (near centroid),
  `propagate_labels` (mapping + unknown fallback).
- **CLI round-trip (`tests/test_cli.py`)** on `synth_dir`: `scat cluster` → labels.json carry
  `cluster_id` + a report + CSV exist; write a CSV mapping; `scat propagate` → labels.json labels
  populated; `scat train` runs on the result.
- Full suite + parity gate unaffected (additive, new files only).

## Risks & mitigations

| risk | mitigation |
|---|---|
| clusters don't map 1:1 to Normal/ROD/Artifact (a cluster mixes classes) | medoid thumbnails + feature profile let the user judge; `--min-cluster-size`/`--k` to re-cluster; mixed clusters can be left blank and fixed in the GUI. Honest: propagation quality = cluster↔class alignment |
| HDBSCAN dumps most points into noise | default `min_cluster_size` scales with n; report the noise fraction; `--min-cluster-size`/`--method kmeans --k` escape hatches |
| labeling GUI drops the additive `cluster_id` on save | verify round-trip in implementation; add a preserve if needed |
| hue circularity distorts distances | encode hue as (sin, cos) in the matrix (clustering-only) |
| feature scale imbalance | StandardScaler before clustering |

## Follow-ups (deferred)
Deep self-supervised embeddings behind `build_feature_matrix`'s seam; a labeling-GUI "cluster &
label" button; iterative active learning; an agent `cluster_folder` tool.
