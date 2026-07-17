# SCAT redesign — Plan 5: group-conditional pooled data (Phase 5)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop presenting pooled biological means/distributions as if they were meaningful for a
**grouped** experiment (where the pooled value is the grand mean of distinct conditions — an artefact
that describes no condition and hides the between-group signal). For grouped runs, replace the report's
pooled Summary cards + distribution histograms with **one compact per-group means table** (keep the
pooled *scope counts*); for ungrouped/single-group runs, keep the pooled cards + distributions (there
they are meaningful). Make the result-window hero **group-aware** (flag the pooled value + show the
per-group range).

**Architecture:** A new pure `scat/metrics.per_group_means(...)` computes the group × metric mean table.
`_html_population_overview` becomes group-conditional (needs `film_summary` + `group_by`, threaded from
`_build_html`). `ResultsTab.load_results` adds a group-aware note to the hero subtitle when the run has
groups. No new dependencies.

**Rationale (why this is correct):** for a grouped run the report ALREADY shows per-group truth (the
Figure 1/2 boxplots + the appendix Group-Statistics tables). The pooled Population overview is therefore
both **redundant** and **misleading** — so for grouped runs we drop the pooled biological cards +
histograms (their per-group detail lives above / in the appendix) and add ONE consolidated per-group
means table (more scannable than six boxplots). Scope counts (N images, N deposits, class counts) stay
pooled — they describe the dataset, not a condition.

**Spec:** extends `docs/.../2026-07-17-scat-results-report-redesign-design.md` §2.1 (experimental unit /
honest presentation). **Depends on** Plan 1 (`scat/metrics.py`) + Plan 4b (`_html_population_overview`).

---

## File structure

- **Modify `scat/metrics.py`** — add `per_group_means(film_summary, group_by, keys=None)`.
- **Create `tests/test_metrics.py` additions** (extend the existing file).
- **Modify `scat/report.py`** — `_html_population_overview(self, summary, inline_plots, film_summary,
  group_by)` group-conditional; new `_html_per_group_table(film_summary, group_by)`; thread the two
  new args from `_build_html`.
- **Modify `scat/main_gui.py`** — `ResultsTab.load_results` hero subtitle gains a group-aware note.
- **Test:** `tests/test_report_per_group.py`, extend `tests/test_results_*`.

---

## Task 1: `metrics.per_group_means` — the per-group means table data

**Files:** Modify `scat/metrics.py`; extend `tests/test_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_metrics.py
def _film_grouped():
    return pd.DataFrame({
        "filename": ["a", "b", "c", "d"],
        "group": ["Ctrl", "Ctrl", "Treat", "Treat"],
        "n_normal": [8, 12, 4, 6], "n_rod": [2, 0, 6, 4], "n_artifact": [1, 1, 1, 1],
        "rod_fraction": [0.2, 0.0, 0.6, 0.4], "mean_area": [80.0, 90.0, 70.0, 60.0],
        "mean_hue": [160.0, 170.0, 150.0, 140.0], "total_iod": [1000.0, 2000.0, 500.0, 700.0],
        "mean_circularity": [0.8, 0.9, 0.7, 0.6],
    })


def test_per_group_means_rows_and_values():
    rows = metrics.per_group_means(_film_grouped(), "group",
                                   keys=["total_deposits", "rod_fraction"])
    by = {r["group"]: r for r in rows}
    assert by["Ctrl"]["n"] == 2 and by["Treat"]["n"] == 2
    # deposits/image mean for Ctrl = mean(10, 12) = 11.0 ; Treat = mean(10, 10) = 10.0
    assert round(by["Ctrl"]["means"]["total_deposits"], 1) == 11.0
    assert round(by["Treat"]["means"]["total_deposits"], 1) == 10.0
    # rod fraction (percent) mean for Ctrl = mean(20, 0) = 10.0
    assert round(by["Ctrl"]["means"]["rod_fraction"], 1) == 10.0


def test_per_group_means_defaults_to_all_registry_metrics_present():
    rows = metrics.per_group_means(_film_grouped(), "group")
    assert set(rows[0]["means"]) <= set(metrics.METRICS)   # only real metrics
    assert "total_deposits" in rows[0]["means"] and "mean_area" in rows[0]["means"]
```

- [ ] **Step 2: Run** → FAIL (`AttributeError: ... 'per_group_means'`).

- [ ] **Step 3: Implement (append to `scat/metrics.py`)**

```python
def per_group_means(film_summary, group_by, keys=None):
    """Per-group mean of each metric (spec: grouped runs get per-group context, not a pooled grand
    mean). Returns [{"group": str, "n": int, "means": {key: float}}] in group order. `keys` defaults
    to every registry metric whose columns are present in film_summary."""
    if group_by is None or group_by not in getattr(film_summary, "columns", []):
        return []
    if keys is None:
        keys = [k for k in METRICS
                if k == "total_deposits" or METRICS[k].values.__name__ != "_deposits"]
        # keep only metrics whose underlying columns exist (avoid KeyError on partial data)
        keys = [k for k in keys if _metric_available(film_summary, k)]
    rows = []
    for g, sub in film_summary.groupby(group_by):
        if g == "ungrouped" or (isinstance(g, float) and g != g):   # skip NaN/ungrouped
            continue
        means = {}
        for k in keys:
            vals = metric_values(sub, k).dropna()
            if len(vals):
                means[k] = float(vals.mean())
        rows.append({"group": str(g), "n": int(len(sub)), "means": means})
    return rows


def _metric_available(film, key):
    m = METRICS[resolve_metric(key)]
    try:
        m.values(film.head(1))
        return True
    except Exception:
        return False
```

- [ ] **Step 4: Run** → PASS. **Step 5: Commit**

```bash
git add scat/metrics.py tests/test_metrics.py
git commit -m "feat(metrics): per_group_means — per-group mean of each metric for grouped runs"
```

---

## Task 2: Report Population overview — group-conditional (per-group table)

**Files:** Modify `scat/report.py` (`_html_population_overview` ~1144 + `_build_html` call + new
`_html_per_group_table`); Test `tests/test_report_per_group.py`

- [ ] **Step 1: Read** — `sed -n '1144,1200p' scat/report.py` (the current pooled cards + the call to
  `_html_distributions`), and the `_build_html` call site (`grep -n "_html_population_overview" scat/report.py`).

- [ ] **Step 2: Write the failing test**

```python
# tests/test_report_per_group.py
from pathlib import Path


def _report(tmp_path, synth_dir, grouped=True):
    from scat.pipeline import analyze_folder_service, generate_report_service, run_statistics_service
    kw = {}
    if grouped:
        g = {f"ctrl_{i}.tif": "Control" for i in range(3)}
        g.update({f"treat_{i}.tif": "Treatment" for i in range(3)})
        kw["groups"] = g
    res = analyze_folder_service(str(synth_dir), output_dir=str(tmp_path / "out"), annotate=False, **kw)
    rd = Path(res.output_dir)
    stats = run_statistics_service(str(rd)) if grouped else None
    generate_report_service(str(rd), statistical_results=stats, group_by="group" if grouped else None)
    return (rd / "report.html").read_text()


def test_grouped_population_overview_is_a_per_group_table(synth_dir, tmp_path):
    html = _report(tmp_path, synth_dir, grouped=True)
    i = html.lower().find("population overview")
    pop = html[i:i + 8000]
    # scope counts stay; the biological summary is a per-group table with the group names as rows
    assert "Control" in pop and "Treatment" in pop
    assert "Total Images" in pop or "Total Deposits" in pop
    # the pooled biological hero cards + pooled distribution histograms are gone from the overview
    assert "stat-card rod" not in pop
    assert "Distribution of Deposit Counts" not in pop


def test_ungrouped_population_overview_keeps_pooled_cards(synth_dir, tmp_path):
    html = _report(tmp_path, synth_dir, grouped=False)
    i = html.lower().find("population overview")
    pop = html[i:i + 8000]
    assert "stat-card" in pop            # pooled cards retained when there are no groups
```

- [ ] **Step 3: Run** → FAIL (grouped overview still shows pooled cards + histograms).

- [ ] **Step 4: Implement**
  - Thread `film_summary` + `group_by` into `_html_population_overview` from `_build_html`.
  - In `_html_population_overview`: `grouped = bool(group_by) and group_by in film_summary.columns and
    film_summary[group_by].dropna().nunique() >= 2`. If grouped → render the section with the **scope
    count cards only** (Total Images, Total Deposits, Normal/ROD/Artifact counts, Total IOD) + a call to
    `_html_per_group_table(film_summary, group_by)`; **omit** the pooled biological mean cards and the
    `inline_plots` distribution body. If not grouped → the CURRENT behavior (all pooled cards + the
    distributions body) unchanged.
  - `_html_per_group_table(self, film_summary, group_by)`: use `metrics.per_group_means(...)` + a
    metric label map (reuse `self.get_metric_label` / `metrics.METRICS[k].label`) to render a
    `class="data-table"` — header row `Group · n · {metric labels}`, one row per group, values formatted
    via each `metrics.METRICS[k].fmt` (+ unit). Escape group names.

- [ ] **Step 5: Run + full suite** → PASS; `QT_QPA_PLATFORM=offscreen /home/lab/SCAT/.venv/bin/python -m pytest -q` all green (keep `test_html_structure_is_balanced` green — the table is inside one `.section`).
- [ ] **Step 6: Verify** — Chrome render a grouped report: Population overview = scope counts + a
  per-group means table (no pooled biological cards/histograms); render an ungrouped report: pooled
  cards + histograms retained.
- [ ] **Step 7: Commit**

```bash
git add scat/report.py tests/test_report_per_group.py
git commit -m "feat(report): grouped Population overview shows a per-group means table (not a pooled grand mean)"
```

---

## Task 3: Result-window hero — group-aware note

**Files:** Modify `scat/main_gui.py` (`ResultsTab.load_results` hero subtitle); Test
`tests/test_results_primary_metric.py` (extend)

- [ ] **Step 1: Read** — `grep -n "hero_sub.setText" scat/main_gui.py` (the current subtitle
  "±SD · across N images").

- [ ] **Step 2: Write the failing test**

```python
# append to tests/test_results_primary_metric.py
def test_hero_is_group_aware_when_grouped(synth_dir, tmp_path):
    QApplication.instance() or QApplication([])
    g = {f"ctrl_{i}.tif": "Control" for i in range(3)}
    g.update({f"treat_{i}.tif": "Treatment" for i in range(3)})
    res = analyze_folder_service(str(synth_dir), output_dir=str(tmp_path / "out"),
                                 primary_metric="total_deposits", groups=g, annotate=False)
    tab = ResultsTab(); tab.load_results(_results_dict_from_output(Path(res.output_dir)))
    sub = tab.hero_sub.text().lower()
    assert "pooled" in sub and "group" in sub          # flags the pooled value + the grouping
```

- [ ] **Step 3: Run** → FAIL (subtitle has no "pooled"/group note).

- [ ] **Step 4: Implement** — in `load_results`, when the results have >=2 groups, append a group-aware
  note to the hero subtitle so the pooled headline is honestly flagged:

```python
# after building hero_sub's base "±SD · across N images":
gcol = "group"
n_groups = int(film_summary[gcol].dropna().nunique()) if gcol in film_summary.columns else 0
if n_groups >= 2:
    vals_by_group = _metrics.metric_values  # per-image; compute per-group means for the range
    gm = film_summary.groupby(gcol)
    per_group = [_metrics.metric_values(sub, pm).dropna().mean() for _, sub in gm]
    per_group = [v for v in per_group if v == v]
    m = _metrics.METRICS[_metrics.resolve_metric(pm)]
    if per_group:
        lo, hi = min(per_group), max(per_group)
        base = self.hero_sub.text()
        self.hero_sub.setText(
            f"{base}  ·  pooled across {n_groups} groups "
            f"(per-group {m.fmt.format(lo)}–{m.fmt.format(hi)}{m.unit})")
```

(`pm` is the resolved primary metric already computed for the hero in `load_results`.)

- [ ] **Step 5: Run + full suite** → PASS.
- [ ] **Step 6: Verify** — render a grouped ResultsTab; the hero subtitle reads e.g.
  "±31.3 · across 30 images · pooled across 6 groups (per-group 25.8–69.4)".
- [ ] **Step 7: Commit**

```bash
git add scat/main_gui.py tests/test_results_primary_metric.py
git commit -m "feat(gui): result-window hero flags the pooled value + per-group range for grouped runs"
```

---

## Self-review checklist

- **Rationale honored:** grouped runs no longer present pooled biological means/distributions as if
  meaningful — replaced by ONE per-group means table (scope counts stay pooled); ungrouped keeps the
  pooled view. The hero honestly flags pooling + the per-group range. Per-group detail is not duplicated
  wholesale — the table consolidates what the boxplots/appendix show, in a scannable grid.
- **Placeholder scan:** T2/T3 give precise contracts + read-the-real-code guidance for the HTML/GUI
  weaving; T1 has complete code. **Type consistency:** `per_group_means` → `[{group,n,means}]`, consumed
  in `_html_per_group_table`; `_html_population_overview(summary, inline_plots, film_summary, group_by)`.
- **Structure guard:** the grouped overview keeps everything inside one `<div class="section">` — the
  balanced-tag test from Plan 4b stays green.
- **Deferred:** per-group small-multiple distributions (option not chosen); the composition strip's
  pooled ROD-fraction (counts are legit scope; a light note is a possible later polish).
```
