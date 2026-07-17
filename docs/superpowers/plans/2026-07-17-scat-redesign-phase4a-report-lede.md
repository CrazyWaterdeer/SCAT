# SCAT redesign — Plan 4a: report Finding lede + Methods (Phase 4a)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the HTML report an answer-first opening — a **Finding lede** (an auto-composed,
conservative, verdict-driven finding sentence keyed off the predeclared primary metric + a test/scope
trio + a factual trust line) and a **Methods appendix** explaining how the numbers (incl. the
confidence score) are produced. Additive; figure reordering is Plan 4b.

**Architecture:** A new pure `scat/findings.py` (`compose_finding`) turns the report's per-metric stats
mapping + the primary metric into a sentence + trio, with honest wording. The run's `analysis` block
(`primary_metric`, `normalization`, `confidence_threshold`) is read from the manifest and threaded into
the report path; two new `ReportGenerator` methods render the lede (top) and the Methods appendix.

**Tech Stack:** Python 3.10+, pandas, pytest. No new deps.

**Spec:** `docs/.../2026-07-17-scat-results-report-redesign-design.md` (§2.1, §5 items 2 + 6).
**Depends on** Plan 1 (manifest carries the `analysis` block), `scat/metrics.py`, `scat/confidence.py`.

**Ground truth (verified in `scat/report.py` `_html_group_comparison` ~1219-1268 + `scat/statistics/
common.py`):** the flat stats mapping is keyed by **stats key**, and each entry has
`overall_test`/`overall_p_value`/`overall_significant` at the **top level** for 3+ groups, or
`test_name`/`p_value`/`significant` for two groups. The primary-metric → stats-key map (from the
report's own `group_metrics`):
`total_deposits→n_total`, `rod_fraction→rod_fraction`, `mean_area→mean_area`, `mean_hue→mean_hue`,
`total_iod→total_iod`, `mean_circularity→mean_circularity`. **All allowed primary metrics have a group
comparison** (they are all in `group_metrics`).

**Honesty (spec §2.1 + codex):** finding sentence uses **"differed"/"showed a difference across …"**
(significant) vs **"showed no statistically detected difference"** (n.s.) — NEVER "did not differ"/
causal; omnibus (3+ groups) says **"across … groups"** (not "between", which implies all pairs differ);
uses ONLY the predeclared primary metric; descriptive when ungrouped/single-group/skipped. Tiny p →
**`p < 0.001`**. The trio field is **"Test"**, not "Effect" (a test statistic is not an effect size).
Methods is **conditional** (the test depends on normality + group count; Holm only for >2 groups) — no
false blanket claim. Trust line via `confidence.run_trust` (factual). All lede fields HTML-escaped.

---

## File structure

- **Create `scat/findings.py`** + **`tests/test_findings.py`** — `compose_finding(...)` (pure).
- **Modify `scat/pipeline.py`** `generate_report_service` — read the manifest `analysis` block, pass it.
- **Modify `scat/report.py`** — thread an `analysis` dict into `generate_report` → `generate_html_report`
  → `_build_html` (both HTML **and** PDF branches, keyword arg at the end of each explicit signature);
  add `_html_finding_lede(...)` (first in `_build_html`) + `_html_methods(...)` (appended); add lede CSS.
- **Test:** `tests/test_report_lede.py`.

---

## Task 1: `scat/findings.py` — the finding-sentence composer (pure, honest)

**Files:** Create `scat/findings.py`, `tests/test_findings.py`

- [ ] **Step 1: Write the failing test** (uses the REAL top-level stats shape, not nested)

```python
# tests/test_findings.py
from scat import findings


def _stats(key, *, test, p, sig):
    return {key: {"overall_test": test, "overall_p_value": p, "overall_significant": sig}}


def test_default_metric_maps_to_n_total_stats_key():
    # primary_metric total_deposits must look up stats key n_total (NOT "total_deposits")
    s = _stats("n_total", test="Kruskal-Wallis", p=0.008, sig=True)
    f = findings.compose_finding(stats=s, primary_metric="total_deposits", headline="40.8 deposits / image",
                                 n_images=30, n_groups=6, group_label="condition")
    assert "Total deposits showed a difference across condition groups" in f["sentence"]
    assert "Kruskal-Wallis" in f["sentence"] and "p = 0.008" in f["sentence"]
    assert f["test"].startswith("Kruskal-Wallis") and f["scope"] == "30 images · 6 groups"


def test_omnibus_significant_says_across_not_between():
    s = _stats("rod_fraction", test="Kruskal-Wallis", p=0.0004, sig=True)
    f = findings.compose_finding(stats=s, primary_metric="rod_fraction", headline="8.3%",
                                 n_images=30, n_groups=6, group_label="condition")
    assert "across condition groups" in f["sentence"] and "between" not in f["sentence"]
    assert "p < 0.001" in f["sentence"]              # tiny p formatting


def test_two_group_significant_uses_between_and_names_the_test():
    s = {"rod_fraction": {"test_name": "Mann-Whitney U", "p_value": 0.03, "significant": True}}
    f = findings.compose_finding(stats=s, primary_metric="rod_fraction", headline="8.3%",
                                 n_images=10, n_groups=2, group_label="group")
    assert "differed between the group groups" in f["sentence"] or "differed between" in f["sentence"]
    assert "Mann-Whitney U" in f["sentence"]


def test_nonsignificant_never_claims_equivalence():
    s = _stats("rod_fraction", test="Kruskal-Wallis", p=0.29, sig=False)
    f = findings.compose_finding(stats=s, primary_metric="rod_fraction", headline="8.3%",
                                 n_images=30, n_groups=6, group_label="condition")
    assert "no statistically detected difference" in f["sentence"].lower()
    assert "did not differ" not in f["sentence"].lower()


def test_ungrouped_is_descriptive():
    f = findings.compose_finding(stats=None, primary_metric="total_deposits",
                                 headline="40.8 deposits / image", n_images=30, n_groups=0, group_label=None)
    assert f["sentence"] == "Across 30 images, Total deposits averaged 40.8 deposits / image."
    assert f["test"] == "no group comparison" and f["scope"] == "30 images"


def test_missing_stats_key_degrades_to_descriptive():
    # grouped run but the primary metric's stats are absent -> descriptive, never a crash
    f = findings.compose_finding(stats={"mean_area": {"overall_test": "x", "overall_p_value": 0.1,
                                 "overall_significant": False}}, primary_metric="rod_fraction",
                                 headline="8.3%", n_images=10, n_groups=2, group_label="group")
    assert "averaged 8.3%" in f["sentence"] and f["test"] == "no group comparison"
```

- [ ] **Step 2: Run** → FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement**

```python
# scat/findings.py
"""Compose the report's answer-first finding (spec §2.1/§5). PURE. Conservative + verdict-driven:
never "did not differ"/causal; omnibus (3+ groups) says "across … groups"; uses ONLY the predeclared
primary metric; descriptive when ungrouped/single-group/stats missing."""
from __future__ import annotations

from scat import metrics as _metrics

# primary-metric (registry) -> stats mapping key (verified against report.py group_metrics)
_STATS_KEY = {
    "total_deposits": "n_total", "rod_fraction": "rod_fraction", "mean_area": "mean_area",
    "mean_hue": "mean_hue", "total_iod": "total_iod", "mean_circularity": "mean_circularity",
}


def _fmt_p(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def _primary_comparison(stats, primary_metric):
    """{test, p, significant, is_omnibus} for the primary metric, or None if unavailable/skipped."""
    if not isinstance(stats, dict) or stats.get("skipped"):
        return None
    entry = stats.get(_STATS_KEY.get(_metrics.resolve_metric(primary_metric)))
    if not isinstance(entry, dict) or "error" in entry:
        return None
    if "overall_test" in entry:                       # 3+ groups (omnibus)
        test, p = entry.get("overall_test"), entry.get("overall_p_value")
        sig, omni = bool(entry.get("overall_significant")), True
    else:                                             # two groups
        test, p = entry.get("test_name"), entry.get("p_value")
        sig, omni = bool(entry.get("significant")), False
    if test is None or p is None:
        return None
    return {"test": test, "p": float(p), "significant": sig, "is_omnibus": omni}


def compose_finding(*, stats, primary_metric, headline, n_images, n_groups, group_label) -> dict:
    label = _metrics.METRICS[_metrics.resolve_metric(primary_metric)].label
    comp = _primary_comparison(stats, primary_metric) if (n_groups or 0) >= 2 else None
    if comp is None:
        return {"sentence": f"Across {n_images} images, {label} averaged {headline}.",
                "metric": label, "test": "no group comparison", "scope": f"{n_images} images"}
    gl = group_label or "group"
    pstr = _fmt_p(comp["p"])
    if comp["significant"]:
        verb = (f"showed a difference across {gl} groups" if comp["is_omnibus"]
                else f"differed between the {gl} groups")
    else:
        conn = "across" if comp["is_omnibus"] else "between"
        verb = f"showed no statistically detected difference {conn} {gl} groups"
    return {"sentence": f"{label} {verb} ({comp['test']}, {pstr}).",
            "metric": f"{label} · {headline}",
            "test": f"{comp['test']}, {pstr}" + (" (significant)" if comp["significant"] else " (n.s.)"),
            "scope": f"{n_images} images · {n_groups} groups"}
```

- [ ] **Step 4: Run** → PASS (6). **Step 5: Commit**

```bash
git add scat/findings.py tests/test_findings.py
git commit -m "feat(findings): finding-sentence composer (real stats shape, metric->stats-key map, honest wording)"
```

---

## Task 2: Thread the manifest `analysis` block into the report builder

**Files:** Modify `scat/pipeline.py` (`generate_report_service` ~250) + `scat/report.py`
(`generate_report`, `generate_html_report` ~601, `_build_html` ~1054). Test: monkeypatch (no lede yet).

- [ ] **Step 1: Read** — `sed -n '250,275p' scat/pipeline.py` (does it `import json` at top? the spatial
  block imports it locally — reuse that pattern); `sed -n '601,612p'` and `sed -n '1054,1075p' scat/report.py`
  (the explicit signatures of `generate_html_report` / `_build_html` and the section list); and
  `grep -n "def generate_report\b" -A25 scat/report.py` (its HTML **and** PDF branches).

- [ ] **Step 2: Write the failing test** (threading only — assert the value REACHES `_build_html`)

```python
# tests/test_report_lede.py
import json
from pathlib import Path
import scat.report as report_mod


def test_analysis_block_reaches_build_html(monkeypatch, synth_dir, tmp_path):
    from scat.pipeline import analyze_folder_service, generate_report_service, run_statistics_service
    res = analyze_folder_service(str(synth_dir), output_dir=str(tmp_path / "out"),
                                 primary_metric="rod_fraction", annotate=False)
    rd = Path(res.output_dir)
    seen = {}
    orig = report_mod.ReportGenerator._build_html
    def spy(self, *a, **kw):
        seen.update(kw)
        return orig(self, *a, **kw)
    monkeypatch.setattr(report_mod.ReportGenerator, "_build_html", spy)
    stats = run_statistics_service(str(rd))
    generate_report_service(str(rd), statistical_results=stats, group_by="group")
    assert seen.get("analysis", {}).get("primary_metric") == "rod_fraction"
```

- [ ] **Step 3: Run** → FAIL (`_build_html` gets no `analysis`).

- [ ] **Step 4: Implement** — read the manifest `analysis` block in `generate_report_service` and thread
  a single `analysis` dict (default `{}`) through `generate_report` (BOTH html + pdf calls) →
  `generate_html_report` → `_build_html` (add `analysis: dict = None` at the END of each signature,
  passed by keyword). Do NOT render the lede yet.

```python
# generate_report_service, after `rd = Path(results_dir)`:
import json
analysis = {}
mpath = rd / "run_manifest.json"
if mpath.exists():
    try:
        analysis = (json.loads(mpath.read_text()).get("analysis") or {})
    except Exception:
        analysis = {}
# pass analysis=analysis into generate_report(...)
```

- [ ] **Step 5: Run + full suite** → PASS; `QT_QPA_PLATFORM=offscreen /home/lab/SCAT/.venv/bin/python -m pytest -q` all pass (defaulted param breaks nothing; `test_report.py` still green).

- [ ] **Step 6: Commit**

```bash
git add scat/pipeline.py scat/report.py tests/test_report_lede.py
git commit -m "feat(report): thread the manifest analysis block into the report builder"
```

---

## Task 3: `_html_finding_lede` — the answer-first lede

**Files:** Modify `scat/report.py` (`_build_html` + new `_html_finding_lede` + lede CSS);
Test `tests/test_report_lede.py`

- [ ] **Step 1: Write the failing test** (grouped, real stats — covers total_deposits→n_total)

```python
# append to tests/test_report_lede.py
def _grouped_report(tmp_path, synth_dir, primary):
    from scat.pipeline import analyze_folder_service, generate_report_service, run_statistics_service
    res = analyze_folder_service(str(synth_dir), output_dir=str(tmp_path / "out"),
                                 primary_metric=primary, annotate=False)
    rd = Path(res.output_dir)
    stats = run_statistics_service(str(rd))     # synth_dir has 2 groups -> real comparison
    generate_report_service(str(rd), statistical_results=stats, group_by="group")
    return (rd / "report.html").read_text()


def test_lede_leads_with_finding_for_default_metric(synth_dir, tmp_path):
    html = _grouped_report(tmp_path, synth_dir, "total_deposits")
    assert 'class="finding"' in html
    assert "Total deposits" in html
    # grouped run -> a real verdict sentence, not the descriptive fallback
    assert ("difference" in html.lower()) and ("group" in html.lower())
    assert "confidence-score threshold" in html          # factual trust line
    for bad in ("did not differ", "high-confidence", "reviewed in the app", "caused"):
        assert bad not in html.lower()


def test_lede_fields_are_escaped(monkeypatch, synth_dir, tmp_path):
    # a malicious group label must not inject markup into the lede
    html = _grouped_report(tmp_path, synth_dir, "rod_fraction")
    assert "<script>" not in html.lower()
```

- [ ] **Step 2: Run** → FAIL (no `class="finding"`).

- [ ] **Step 3: Implement** — insert `_html_finding_lede` FIRST in `_build_html` (after
  `_html_document_head`, before `_html_distributions`); read the run's values from the threaded
  `analysis` dict (NOT global config), fall back to defaults:

```python
def _html_finding_lede(self, film_summary, deposit_data, statistical_results, group_by, analysis):
    from scat import metrics as _metrics, confidence as _confidence, findings as _findings
    import html as _h
    analysis = analysis or {}
    pm = _metrics.resolve_metric(analysis.get("primary_metric"))
    norm = analysis.get("normalization") or _metrics.DEFAULT_NORMALIZATION
    thr = float(analysis.get("confidence_threshold", _metrics.DEFAULT_THRESHOLD))
    headline = _metrics.format_headline(film_summary, pm, norm, meta={})
    n_images = len(film_summary)
    grouped = bool(group_by) and group_by in film_summary.columns
    n_groups = int(film_summary[group_by].dropna().nunique()) if grouped else 0
    group_label = self.get_metric_label(group_by) if grouped else None
    if group_label and group_label.strip().lower() in ("group", "groups", "condition"):
        group_label = group_label.lower()
    f = _findings.compose_finding(stats=statistical_results, primary_metric=pm, headline=headline,
                                  n_images=n_images, n_groups=n_groups, group_label=group_label)
    trust = _confidence.run_trust(deposit_data, thr)
    return f'''
    <div class="section">
      <div class="lede">
        <div class="finding">{_h.escape(f["sentence"])}</div>
        <div class="lede-trio">
          <span><b>Primary metric</b>{_h.escape(f["metric"])}</span>
          <span><b>Test</b>{_h.escape(f["test"])}</span>
          <span><b>Scope</b>{_h.escape(f["scope"])}</span>
        </div>
        <div class="lede-trust">{_h.escape(trust["line"])}</div>
      </div>
    </div>
'''
```

Wire it in `_build_html` (which now receives `analysis`):
```python
self._html_finding_lede(film_summary, deposit_data, statistical_results, group_by, analysis),
```
Add to the `<style>` block (near `.stat-card`):
```css
.lede{border-left:4px solid var(--rod);background:var(--surface);border:1px solid var(--hair);border-radius:8px;padding:22px 24px;margin:20px 0}
.finding{font-family:var(--serif);font-size:1.4rem;font-weight:600;line-height:1.35}
.lede-trio{display:flex;gap:32px;margin-top:14px;font-size:0.95rem}
.lede-trio b{color:var(--muted);font-weight:600;text-transform:uppercase;font-size:0.7rem;letter-spacing:var(--track-caps);display:block;margin-bottom:2px}
.lede-trust{color:var(--muted);font-size:0.8rem;margin-top:12px;border-top:1px solid var(--hair);padding-top:10px}
```

- [ ] **Step 4: Run + full suite** → PASS; `pytest -q` all green.
- [ ] **Step 5: Verify** — Chrome render of a regenerated grouped report; lede leads with the finding.
- [ ] **Step 6: Commit**

```bash
git add scat/report.py tests/test_report_lede.py
git commit -m "feat(report): answer-first Finding lede (verdict sentence + test/scope trio + trust line)"
```

---

## Task 4: `_html_methods` — the Methods appendix (conditional, honest)

**Files:** Modify `scat/report.py` (`_build_html` + new `_html_methods`); Test `tests/test_report_lede.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_report_lede.py
def test_methods_appendix_is_honest(synth_dir, tmp_path):
    html = _grouped_report(tmp_path, synth_dir, "total_deposits").lower()
    assert "methods" in html
    assert "image-level" in html                                     # experimental unit
    assert ("uncalibrated" in html) or ("not a calibrated probability" in html)  # confidence honesty
    assert "reject class" in html                                    # artifacts = reject class
    # conditional test wording — must NOT hard-claim one test always applies
    assert "depending on" in html or "normality" in html
```

- [ ] **Step 2: Run** → FAIL (no methods section).

- [ ] **Step 3: Implement** — append `_html_methods()` to `_build_html`'s section list; wording is
  CONDITIONAL (the test depends on normality + group count; Holm only for >2 groups):

```python
def _html_methods(self):
    return '''
    <div class="section">
      <h2>Appendix — Methods</h2>
      <p class="section-intro">How the numbers were produced.</p>
      <p><b>Detection &amp; classification.</b> Deposits are detected, then a Random-Forest (or
      rule-based) classifier labels each Normal, ROD, or Artifact. ROD Fraction = ROD / (Normal + ROD);
      Artifacts are the reject class, excluded from deposit counts and metrics.</p>
      <p><b>Confidence.</b> The per-deposit confidence is the classifier score (the RF class
      probability, or a circularity-derived score in rule-based mode). It is <b>uncalibrated</b> — not
      a calibrated probability of correctness — and covers classification only, not detection. The
      "below the confidence-score threshold" counts are a review/workload signal, not a reliability
      measure.</p>
      <p><b>Statistics.</b> Group comparisons test <b>image-level</b> aggregates (the experimental unit
      is the image, not the deposit — avoiding pseudoreplication). The omnibus test is chosen by
      normality and group count (one-way ANOVA or Kruskal-Wallis for three or more groups; an
      independent t-test or Mann-Whitney U for two), with Holm-corrected pairwise comparisons when
      there are more than two groups; effect sizes are reported alongside.</p>
    </div>
'''
```

- [ ] **Step 4: Run + full suite** → PASS. **Step 5: Verify** — Chrome render shows Methods. **Step 6: Commit**

```bash
git add scat/report.py tests/test_report_lede.py
git commit -m "feat(report): Methods appendix (conditional test wording, uncalibrated confidence, image-level unit)"
```

---

## Self-review checklist

- **Codex fixes folded:** `_STATS_KEY` map (total_deposits→n_total) + real top-level shape (T1, tested
  for the default metric + omnibus/2-group/ns/ungrouped/missing); "Test" not "Effect", `p < 0.001`
  (T1); omnibus says "across" not "between" (T1); Methods is conditional, never a false blanket claim
  (T4); threading is an `analysis` dict through both HTML+PDF branches with `import json`, reading
  normalization+threshold from the manifest not global config (T2); Task 2 tests threading by
  monkeypatch (valid checkpoint), the grouped integration test lives in T3; lede fields escaped +
  an injection test (T3).
- **Spec coverage (§5 items 2 + 6):** lede sentence + trio + trust line (T1/T3); Methods (T4). Figure
  reordering = Plan 4b.
- **Placeholder scan:** none. **Type consistency:** `compose_finding` → `{sentence, metric, test, scope}`,
  consumed identically in `_html_finding_lede`; `analysis` dict keys match Plan 1's manifest block.
