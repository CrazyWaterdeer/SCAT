# SCAT redesign — Plan 5: group-conditional pooled data (Phase 5, robust)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop presenting pooled biological means/distributions as if meaningful for a **grouped**
experiment (the pooled value is a grand mean of distinct conditions — describes no condition, hides the
signal). For grouped runs: replace the report's pooled biological Summary cards + distribution
histograms with ONE compact **per-group means table** (keep the pooled *scope counts*); ungrouped keeps
the pooled view. Make the result-window hero **group-aware** (flag the pooled value + per-group range).

**Reworked after a codex review (verified against real data):**
- `image_summary.csv` has `n_normal/n_rod/n_artifact/rod_fraction/mean_area/total_iod` and the SPLIT
  columns `normal_mean_hue/normal_mean_circularity` (…and rod_* variants) — but NO plain `mean_hue`/
  `mean_circularity`. So the per-group table is computed IN THE REPORT from the columns that actually
  exist (matching the report's own group-comparison boxplots), NOT via the `scat.metrics` registry
  (whose `mean_hue`/`mean_circularity` reference absent columns — a latent Plan-1 bug, flagged below as
  a separate follow-up, NOT fixed here).
- Groups use an **"ungrouped" sentinel**; grouped-detection must exclude `NaN`/blank/`"ungrouped"` and
  count **effective** groups (≥2), else one real group + ungrouped images falsely triggers grouped mode.
- Existing tests assume distributions live in the grouped Population overview — they are updated here.
- `mean_hue` is **circular** (0/360) — excluded from any min–max range.

**Rationale:** grouped runs ALREADY show per-group truth (Figure 1/2 boxplots + appendix Group-Stats).
The pooled overview is redundant + misleading; the per-group table consolidates it in one scannable
grid. Scope counts stay pooled (they describe the dataset, not a condition).

**Spec:** extends §2.1. **Depends on** Plan 4b (`_html_population_overview`).

**KNOWN FOLLOW-UP (not this plan):** `scat/metrics.py` `mean_hue`/`mean_circularity` metrics reference
`film["mean_hue"]`/`["mean_circularity"]`, which don't exist in `image_summary.csv` — they only work on
synthetic frames. Fix (map to `normal_mean_*` or enrich film_summary) is a separate bug ticket.

---

## File structure

- **Modify `scat/report.py`** — new `_effective_groups(film, group_by)`; new `_html_per_group_table(
  film, group_by)`; `_html_population_overview(self, summary, inline_plots, film_summary, group_by)`
  group-conditional; thread `film_summary`+`group_by` from `_build_html`.
- **Modify `scat/main_gui.py`** — `ResultsTab.load_results` hero subtitle group-aware note.
- **Tests:** `tests/test_report_per_group.py`; UPDATE `tests/test_report.py` +
  `tests/test_report_findings_note.py` where they assume distributions in the grouped overview; extend
  `tests/test_results_primary_metric.py`.

---

## Task 1: `_effective_groups` + `_html_per_group_table` (report-native, real columns)

**Files:** Modify `scat/report.py`; Test `tests/test_report_per_group.py`

- [ ] **Step 1: Read** — `head -1 <a results_dir>/image_summary.csv` (confirm columns: `n_normal,
  n_rod, mean_area, total_iod, normal_mean_hue, normal_mean_circularity, rod_fraction, group`) and
  `sed -n '1258,1268p' scat/report.py` (the group_metrics labels to mirror).

- [ ] **Step 2: Write the failing test**

```python
# tests/test_report_per_group.py
import pandas as pd
from scat.report import ReportGenerator


def _film():
    return pd.DataFrame({
        "filename": list("abcd"),
        "group": ["Ctrl", "Ctrl", "Treat", "ungrouped"],   # a stray ungrouped row
        "n_normal": [8, 12, 4, 5], "n_rod": [2, 0, 6, 1], "n_artifact": [1, 1, 1, 1],
        "rod_fraction": [0.2, 0.0, 0.6, 0.1], "mean_area": [80.0, 90.0, 70.0, 60.0],
        "total_iod": [1000.0, 2000.0, 500.0, 700.0],
        "normal_mean_hue": [160.0, 170.0, 150.0, 155.0],
        "normal_mean_circularity": [0.8, 0.9, 0.7, 0.75],
    })


def test_effective_groups_excludes_ungrouped():
    g = ReportGenerator._effective_groups(_film(), "group")
    assert g == ["Ctrl", "Treat"]        # 'ungrouped' + order preserved (no sort)


def test_per_group_table_has_real_columns_and_values(tmp_path):
    rg = ReportGenerator(tmp_path)
    html = rg._html_per_group_table(_film(), "group")
    assert "Ctrl" in html and "Treat" in html and "ungrouped" not in html
    assert "ROD" in html and "Deposits" in html          # metric column headers
    # Ctrl deposits/img mean = mean(10, 12) = 11.0
    assert "11.0" in html
    # a pH column exists (from normal_mean_hue) — pH is present, not silently dropped
    assert "pH" in html
```

- [ ] **Step 3: Run** → FAIL (`AttributeError`).

- [ ] **Step 4: Implement (in `scat/report.py`, as `@staticmethod`/methods)**

```python
@staticmethod
def _effective_groups(film, group_by):
    """Real group labels in first-seen order, excluding NaN/blank/the 'ungrouped' sentinel."""
    if group_by is None or group_by not in getattr(film, "columns", []):
        return []
    seen = []
    for g in film[group_by].tolist():
        if g is None or (isinstance(g, float) and g != g):
            continue
        s = str(g).strip()
        if not s or s == "ungrouped" or s in seen:
            continue
        seen.append(s)
    return seen

# columns rendered in the per-group table — real image_summary columns, mirroring the report's
# group-comparison metrics. (mean_hue/circularity use the normal_* columns that actually exist.)
_PER_GROUP_COLS = [
    ("Deposits / img", lambda s: (s["n_normal"] + s["n_rod"]).mean(), "{:.1f}"),
    ("ROD %",          lambda s: (s["rod_fraction"] * 100).mean(),    "{:.1f}%"),
    ("Mean area (px²)", lambda s: s["mean_area"].mean(),              "{:.0f}"),
    ("Total IOD",      lambda s: s["total_iod"].mean(),               "{:.0f}"),
    ("pH (hue °)",     lambda s: s["normal_mean_hue"].mean(),         "{:.0f}"),
    ("Circularity",    lambda s: s["normal_mean_circularity"].mean(), "{:.3f}"),
]

def _html_per_group_table(self, film, group_by):
    import html as _h
    groups = self._effective_groups(film, group_by)
    # stable column set: keep a column only if its source columns exist in film
    cols = []
    for label, fn, fmt in self._PER_GROUP_COLS:
        try:
            fn(film.head(1)); cols.append((label, fn, fmt))
        except Exception:
            pass
    head = "".join(f"<th>{_h.escape(c[0])}</th>" for c in cols)
    rows = ""
    for g in groups:
        sub = film[film[group_by].astype(str).str.strip() == g]
        cells = f"<td>{_h.escape(g)}</td><td class='num'>{len(sub)}</td>"
        for _label, fn, fmt in cols:
            try:
                v = fn(sub)
                cells += f"<td class='num'>{fmt.format(v)}</td>" if v == v else "<td class='num'>—</td>"
            except Exception:
                cells += "<td class='num'>—</td>"
        rows += f"<tr>{cells}</tr>"
    return (f'<table class="data-table"><thead><tr><th>Group</th><th class="num">n</th>{head}</tr>'
            f'</thead><tbody>{rows}</tbody></table>')
```

- [ ] **Step 5: Run** → PASS. **Step 6: Commit**

```bash
git add scat/report.py tests/test_report_per_group.py
git commit -m "feat(report): per-group means table + effective-group detection (excludes ungrouped)"
```

---

## Task 2: Population overview — group-conditional

**Files:** Modify `scat/report.py` (`_html_population_overview` + `_build_html` call); Tests:
`tests/test_report_per_group.py` + UPDATE `tests/test_report.py`, `tests/test_report_findings_note.py`

- [ ] **Step 1: Read** — `sed -n '1144,1200p' scat/report.py`; the `_build_html` call to
  `_html_population_overview`.

- [ ] **Step 2: Write the failing tests + fix the ones the change invalidates**

```python
# append to tests/test_report_per_group.py
from pathlib import Path


def _report(tmp_path, synth_dir, grouped=True):
    from scat.pipeline import analyze_folder_service, generate_report_service, run_statistics_service
    kw = {}
    if grouped:
        gm = {f"ctrl_{i}.tif": "Control" for i in range(3)}
        gm.update({f"treat_{i}.tif": "Treatment" for i in range(3)})
        kw["groups"] = gm
    res = analyze_folder_service(str(synth_dir), output_dir=str(tmp_path / "out"), annotate=False, **kw)
    rd = Path(res.output_dir)
    stats = run_statistics_service(str(rd)) if grouped else None
    generate_report_service(str(rd), statistical_results=stats, group_by=("group" if grouped else None))
    return (rd / "report.html").read_text()


def test_grouped_overview_is_per_group_table_no_pooled_biology(synth_dir, tmp_path):
    html = _report(tmp_path, synth_dir, grouped=True)
    i = html.lower().find("population overview")
    pop = html[i:html.find("</div>\n    </div>", i) + 20] if i != -1 else ""
    assert "Control" in pop and "Treatment" in pop            # per-group rows
    assert "Total Images" in pop and "Total Deposits" in pop  # scope counts kept
    assert "stat-card rod" not in pop                          # no pooled ROD hero card
    assert "Distribution of Deposit Counts" not in pop         # pooled histograms gone


def test_ungrouped_overview_keeps_pooled(synth_dir, tmp_path):
    html = _report(tmp_path, synth_dir, grouped=False)
    i = html.lower().find("population overview")
    pop = html[i:i + 12000]
    assert "stat-card rod" in pop and "Distribution of Deposit Counts" in pop
```

Then UPDATE the two existing tests that assumed distributions are always in the overview:
- `tests/test_report.py::test_generate_html_report_with_deposits_and_groups`: it builds a grouped
  report and asserts `"Distributions" in html`. Change that assertion to the grouped truth — assert the
  per-group table appears (e.g. `'class="data-table"' in html` inside the overview / the group names).
- `tests/test_report_findings_note.py::test_finding_leads_summary_demotes_to_population_overview`: its
  `_report` helper is grouped, and it asserts `"Distributions" in pop_slice`. Change to assert the
  per-group table (grouped mode no longer renders pooled distributions in the overview). Keep the
  finding→group→overview ORDER assertions.

- [ ] **Step 3: Run** → FAIL (grouped overview still pooled).

- [ ] **Step 4: Implement** — thread `film_summary`+`group_by` into `_html_population_overview`. Compute
  `grouped = len(self._effective_groups(film_summary, group_by)) >= 2`. If grouped: render the section
  with **scope cards only** (Total Images, Total Deposits — from `summary`; + Normal/ROD/Artifact counts
  computed from `film_summary[['n_normal','n_rod','n_artifact']].sum()`) + `self._html_per_group_table(
  film_summary, group_by)`, and OMIT the pooled biological mean cards + the distributions body. If not
  grouped: the CURRENT behavior (pooled cards + distributions) unchanged. Keep it all inside the one
  `<div class="section">` (balanced-tag test stays green).

- [ ] **Step 5: Run + full suite** → PASS (incl. the two updated tests + `test_html_structure_is_balanced`).
- [ ] **Step 6: Verify** — Chrome render grouped (per-group table, no pooled biology) + ungrouped (pooled).
- [ ] **Step 7: Commit**

```bash
git add scat/report.py tests/test_report_per_group.py tests/test_report.py tests/test_report_findings_note.py
git commit -m "feat(report): grouped Population overview = scope counts + per-group table (no pooled grand mean)"
```

---

## Task 3: Result-window hero — group-aware note (honest, non-circular)

**Files:** Modify `scat/main_gui.py` (`ResultsTab.load_results`); Test `tests/test_results_primary_metric.py`

- [ ] **Step 1: Read** — `grep -n "hero_sub.setText\|group_by" scat/main_gui.py` (the subtitle + whether
  the results dict carries `group_by`; `_results_dict_from_output` sets it).

- [ ] **Step 2: Write the failing test**

```python
# append to tests/test_results_primary_metric.py
def test_hero_group_aware_note(synth_dir, tmp_path):
    QApplication.instance() or QApplication([])
    gm = {f"ctrl_{i}.tif": "Control" for i in range(3)}
    gm.update({f"treat_{i}.tif": "Treatment" for i in range(3)})
    res = analyze_folder_service(str(synth_dir), output_dir=str(tmp_path / "out"),
                                 primary_metric="total_deposits", groups=gm, annotate=False)
    tab = ResultsTab(); tab.load_results(_results_dict_from_output(Path(res.output_dir)))
    sub = tab.hero_sub.text().lower()
    assert "pooled across 2 groups" in sub and "group image-means" in sub


def test_hero_no_group_note_when_ungrouped(synth_dir, tmp_path):
    QApplication.instance() or QApplication([])
    res = analyze_folder_service(str(synth_dir), output_dir=str(tmp_path / "out"), annotate=False)
    tab = ResultsTab(); tab.load_results(_results_dict_from_output(Path(res.output_dir)))
    assert "pooled across" not in tab.hero_sub.text().lower()
```

- [ ] **Step 3: Run** → FAIL.

- [ ] **Step 4: Implement** — in `load_results`, after the base subtitle, use the results' own
  `group_by` (not a hardcoded "group"), count EFFECTIVE groups (exclude NaN/blank/"ungrouped"), and add
  an honest note. Skip the min–max **range** when the primary metric is circular (`mean_hue`):

```python
gcol = results.get("group_by")
groups = []
if gcol and gcol in film_summary.columns:
    for g in film_summary[gcol].dropna().unique():
        s = str(g).strip()
        if s and s != "ungrouped" and s not in groups:
            groups.append(s)
if len(groups) >= 2:
    m = _metrics.METRICS[_metrics.resolve_metric(pm)]
    note = f"pooled across {len(groups)} groups"
    if pm != "mean_hue":     # hue is circular — a min/max range is misleading
        per_group = [_metrics.metric_values(film_summary[film_summary[gcol].astype(str).str.strip() == g], pm).dropna().mean()
                     for g in groups]
        per_group = [v for v in per_group if v == v]
        if per_group:
            note += (f" (group image-means {m.fmt.format(min(per_group))}"
                     f"–{m.fmt.format(max(per_group))}{m.unit})")
    self.hero_sub.setText(self.hero_sub.text() + "  ·  " + note)
```

- [ ] **Step 5: Run + full suite** → PASS.
- [ ] **Step 6: Verify** — grouped ResultsTab hero subtitle: "…across 30 images · pooled across 6 groups
  (group image-means 25.8–69.4)".
- [ ] **Step 7: Commit**

```bash
git add scat/main_gui.py tests/test_results_primary_metric.py
git commit -m "feat(gui): hero flags pooled value + per-group range for grouped runs (circular-safe)"
```

---

## Self-review checklist

- **Codex fixes folded:** per-group table computed in-report from REAL columns (`normal_mean_hue/
  circularity`, not the registry's missing `mean_hue`), so pH/circularity aren't silently dropped;
  effective-group detection excludes the "ungrouped" sentinel (report + hero); stable column set with
  "—" for missing; existing distribution-assuming tests updated; hero uses `results["group_by"]` and
  skips the circular-hue range, labeled "group image-means" (not a CI/raw range).
- **Scope-count cards** (Total Images/Deposits + Normal/ROD/Artifact) computed from `film_summary`
  (not the summary dict, which lacks `total_artifact`).
- **Structure guard:** grouped overview stays inside one `.section` (Plan 4b balanced-tag test green).
- **Latent bug flagged, not fixed here:** `metrics.mean_hue/mean_circularity` reference absent columns.
- **Placeholder scan:** none — T1 has full code; T2/T3 give precise contracts + read-real-code steps.
```
