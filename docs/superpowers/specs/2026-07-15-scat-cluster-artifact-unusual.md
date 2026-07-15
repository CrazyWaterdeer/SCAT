# SCAT clustering — surface unusual deposits + separate artifacts

**Status:** design (empirically grounded) · 2026-07-15 · branch `feat/cluster-artifact-unusual`
**Builds on:** `2026-07-15-scat-unsupervised-clustering-design.md` (PR #18)

## Problem

Reviewing the real cluster report, the user (domain expert) noted the report shows only the
**common, small, round ("dust")** deposits, while the biologically interesting **shape-unusual
(elongated / irregular, ROD-like)** deposits were missing. Investigation on real data
(`2026_01_29`, 1069 deposits) confirmed:

- Detection is fine (nothing clipped by `max_area`); the ~200–736px² and irregular deposits ARE
  detected.
- But HDBSCAN clusters the dense common dust and dumps the **rare** shape-unusual deposits into the
  **noise bucket (69%)** — and the report thumbnailed only clusters, so **the unusual deposits were
  hidden.** 94% of the largest deposits and 65% of the least-round were in noise.

## Empirical findings that shape the design

- **Adding `solidity` (contour area / convex-hull area) to the feature matrix consolidates the
  irregular/unusual deposits into their OWN labelable cluster** (measured: a 54-deposit cluster with
  median circularity 0.25) instead of scattering them into noise. This is the core lever.
- **`solidity` does NOT separate the film-boundary LINE artifacts** — straight lines have solidity
  ≈ 1 (a line ≈ its own convex hull). Lines are identified by **extreme aspect_ratio + very low
  circularity** instead (they are few — the user confirmed "the lines are mostly film boundaries,
  and there are only a few real unusual deposits").
- So: `solidity` surfaces the unusual deposits (as a cluster); a shape heuristic flags the few line
  artifacts; both are cheap and additive.

## Goals

1. **Surface the unusual deposits** — they must appear in the report (not hidden in noise).
2. **Guide labeling** — per-cluster hints (common / unusual / line-artifact) so the user labels
   quickly (round→normal, irregular→rod, line→artifact/skip), reusing the existing
   `cluster_labels.csv → propagate` flow.
3. Additive; no change to detection, training, or existing non-cluster outputs.

## Non-goals

- New detection modes / `sensitive_mode` tuning (detection is adequate for these).
- Noise recursive sub-clustering (adding `solidity` already forms the unusual cluster — simpler).
- Auto-deleting deposits (the user decides via labels; the tool only flags/surfaces).

## Design

### 1. `solidity` feature (`scat/clustering.py` + `scat/pipeline.py`)
- `cluster_folder_service` computes `solidity = contourArea(contour) / contourArea(convexHull)` per
  deposit (guard hull area > 0 → else 0.0) and adds it to the feature DataFrame.
- `build_feature_matrix` includes `solidity` **only when the column is present** (graceful: existing
  callers/tests without it are unchanged). Added to `_LINEAR_COLS`-style handling (no log; it is a
  0–1 ratio) and standardized with the rest.

### 2. Per-cluster "kind" hint (`scat/clustering.py`)
- `cluster_kind(row) -> str` from a cluster's median features:
  - `"line-artifact?"` when `aspect_ratio > 8 and circularity < 0.15` (film-boundary lines),
  - else `"unusual?"` when `circularity < 0.5 or aspect_ratio > 2.5` (irregular / elongated),
  - else `"common"` (round).
- `cluster_profile` already returns per-cluster medians; add a `kind` column. Noise row (`-1`) is
  profiled too and gets a kind from its own medians.

### 3. "Unusual deposits" report section + labelability
- A new `unusual_ranking(df) -> Series` = z(aspect_ratio) + z(−circularity) + 0.6·z(area_px), with
  **line-artifacts down-ranked** (subtract a penalty when `aspect_ratio > 8 and circularity < 0.15`).
- `cluster_folder_service` exports thumbnails for the **top-N unusual deposits drawn from the noise
  bucket** (default 24) and adds an **"Unusual deposits (review — likely ROD/atypical)"** section to
  `cluster_report.html` showing them ranked, plus keeps the existing per-cluster cards now annotated
  with the `kind` hint. A separate small **"Likely line artifacts (film boundaries)"** note lists the
  flagged-line count.
- `cluster_labels.csv` gains a **`kind`** column (informational, from §2) so the user sees, per row,
  whether it's common/unusual/line and labels accordingly. (The unusual deposits are now inside a
  real cluster via `solidity`, so they are labeled through the normal per-cluster CSV flow — no new
  labeling mechanism.)

### 4. Nothing else changes
`propagate` / `training_readiness` / labels.json are unchanged; the `kind` column is ignored by
`parse_cluster_labels_csv` (it only reads `cluster_id` + `label`).

## Testing

- Unit (`tests/test_clustering.py`): `build_feature_matrix` uses `solidity` when present and ignores
  it when absent (both paths); `cluster_kind` returns the three kinds for representative rows;
  `unusual_ranking` penalizes line-artifacts (an extreme-aspect/low-circularity row ranks below a
  solid irregular one of equal size).
- Integration: `cluster_folder_service` on `synth_dir` writes the report with an "Unusual deposits"
  section and a `kind` column in `cluster_labels.csv`; propagate still works (kind ignored).
- Real-data check: on `2026_01_29`, the unusual/irregular deposits appear as a cluster + in the
  gallery (verified during implementation).
- Full suite green; additive (parity gate untouched).

## Risks

| risk | mitigation |
|---|---|
| adding solidity shifts clustering for existing users | clustering is a new feature (PR #18); this is an intended improvement, not a parity-locked output; graceful when solidity absent |
| line-artifact heuristic mislabels a real elongated ROD as a line | it only *down-ranks / hints*, never deletes; the user sees the thumbnail and decides |
| very few unusual deposits in some datasets | the section simply shows fewer; health/kind hints still inform |

---

## Codex review → incorporated

- **solidity is not a general "ROD" feature** (Codex #2): it captures *concavity/irregularity*; a
  smooth convex elongated ROD can have solidity ≈ 1. So `solidity` is added to help *irregular*
  deposits form a cluster, while `aspect_ratio`/`circularity` (already present) + a rotation-invariant
  `elongation` carry elongated-but-solid ROD. Framing corrected.
- **Line flag is rotation-invariant** (Codex #4/#6): `aspect_ratio` from an axis-aligned bbox misses
  DIAGONAL film boundaries, so lines are flagged from minAreaRect `elongation` + `rect_fill`
  (very elongated AND poorly filling its rotated box), which does not flag a filled ROD. Measured:
  ~14 lines flagged on real data vs 1 with the axis-aligned rule.
- **Unusual ranking excludes line artifacts** (−inf), not a fixed penalty (Codex #5); ranks across
  ALL deposits and stamps each thumbnail with its `cluster_id` so the unusual cohort that `solidity`
  consolidates into a cluster is shown and is labelable via `cluster_labels.csv` (Codex #1).
- **`kind` uses member fractions, not just the median** (Codex #3): `pct_line`/`pct_unusual` per
  cluster, so the giant noise bucket reads `mixed (has unusual)` instead of `common` when its tail is
  the interesting part. Measured: on real data the noise card now reads `mixed (has unusual)`.
- **Tests strengthened** (Codex #8): rotation-invariant line flag, ranking excludes lines,
  fraction-based kind on a round-median/unusual-tail cluster.
- **Measured before/after** (Codex #9; `2026_01_29`, 1069 deposits): unusual/irregular cohort moved
  from scattered-in-noise to a labelable 54-deposit cluster (`unusual?`); ~14 diagonal/line artifacts
  flagged; gallery top-24 now cluster-tagged and line-free.
