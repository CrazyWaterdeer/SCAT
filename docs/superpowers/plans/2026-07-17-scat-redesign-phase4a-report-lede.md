# SCAT redesign — Plan 4a: report Finding lede + Methods (Phase 4a)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the HTML report an answer-first opening: a **Finding lede** (an auto-composed,
conservative, verdict-driven finding sentence keyed off the predeclared primary metric + an effect
trio + a factual trust line) and a **Methods appendix** that explains how the numbers — including the
confidence score — are produced. Additive: this inserts a lede at the top and an appendix; the figure
reordering (Fig 1 primary / Fig 2 exploratory / demote the population overview) is **Plan 4b**.

**Architecture:** A new pure `scat/findings.py` (`compose_finding`) turns the statistical results +
primary metric into a sentence + effect trio, with honest wording (never "did not differ"/causal;
descriptive when ungrouped or stats are skipped). `primary_metric` is threaded from the manifest into
the report path; two new `ReportGenerator` methods (`_html_finding_lede`, `_html_methods`) render the
lede (top of the document) and the Methods appendix.

**Tech Stack:** Python 3.10+, pandas, pytest. No new deps.

**Spec:** `docs/.../2026-07-17-scat-results-report-redesign-design.md` (§2.1, §5 items 2 + 6).
**Depends on Plan 1** (manifest carries `primary_metric`) + `scat/metrics.py` (labels/headline) +
`scat/confidence.py` (trust line).

**Honesty (spec §2.1 — binding):** the finding sentence uses **"differed"** vs **"showed no
statistically detected difference under [test]"** — NEVER "did not differ" (implies equivalence),
never causal/biological-strength wording; it uses ONLY the predeclared primary metric (never the
most-significant one); it is **descriptive, not inferential**, for ungrouped / single-group / skipped-
stats runs. The trust line reuses `confidence.run_trust` (factual, "confidence-score threshold", no
"reviewed"/"reliable"). Methods states: the experimental unit is the image (tests on image-level
aggregates — no pseudoreplication); confidence is an uncalibrated classifier score (RF probability OR
rule-based heuristic) covering classification only.

---

## File structure

- **Create `scat/findings.py`** + **`tests/test_findings.py`** — `compose_finding(...)` (pure).
- **Modify `scat/pipeline.py`** `generate_report_service` — read `primary_metric` from the manifest and
  pass it through.
- **Modify `scat/report.py`** — thread `primary_metric` into `generate_html_report` + `_build_html`;
  add `_html_finding_lede(...)` (inserted first in `_build_html`) and `_html_methods(...)` (appended).
- **Test:** `tests/test_report_lede.py`.

---

## Task 1: `scat/findings.py` — the finding-sentence composer (pure, honest)

**Files:** Create `scat/findings.py`, `tests/test_findings.py`

The composer takes the report's flat per-metric stats mapping + the primary metric + scope, and returns
a sentence + an effect trio. It must be robust to missing/odd stats (return a descriptive sentence).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_findings.py
from scat import findings


def _grouped_stats(sig):
    # Mirrors the report's per-metric comparison shape: metric -> {"comparison": {...}}.
    return {"rod_fraction": {"comparison": {
        "overall_test": "Kruskal-Wallis", "overall_p_value": 0.008 if sig else 0.29,
        "overall_significant": sig}}}


def test_significant_finding_is_verdict_driven_not_causal():
    f = findings.compose_finding(stats=_grouped_stats(True), primary_metric="rod_fraction",
                                 headline="8.3%", n_images=30, n_groups=6, group_label="condition")
    assert "differed between condition groups" in f["sentence"]
    assert "Kruskal-Wallis" in f["sentence"] and "0.008" in f["sentence"]
    for bad in ("caused", "because", "did not differ", "due to"):
        assert bad not in f["sentence"].lower()
    assert f["effect"].startswith("Kruskal-Wallis") and f["scope"] == "30 images · 6 groups"


def test_nonsignificant_says_no_detected_difference_not_equivalence():
    f = findings.compose_finding(stats=_grouped_stats(False), primary_metric="rod_fraction",
                                 headline="8.3%", n_images=30, n_groups=6, group_label="condition")
    assert "no statistically detected difference" in f["sentence"].lower()
    assert "did not differ" not in f["sentence"].lower()   # never claim equivalence


def test_ungrouped_is_descriptive_not_inferential():
    f = findings.compose_finding(stats=None, primary_metric="total_deposits",
                                 headline="40.8 deposits / image", n_images=30, n_groups=0,
                                 group_label=None)
    assert f["sentence"] == "Across 30 images, Total deposits averaged 40.8 deposits / image."
    assert f["effect"] == "no group comparison" and f["scope"] == "30 images"


def test_skipped_or_missing_metric_stats_degrades_to_descriptive():
    f = findings.compose_finding(stats={"skipped": True}, primary_metric="rod_fraction",
                                 headline="8.3%", n_images=10, n_groups=2, group_label="group")
    assert "averaged 8.3%" in f["sentence"] and f["effect"] == "no group comparison"
```

- [ ] **Step 2: Run** — `QT_QPA_PLATFORM=offscreen /home/lab/SCAT/.venv/bin/python -m pytest tests/test_findings.py -q` → FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement**

```python
# scat/findings.py
"""Compose the report's answer-first finding (spec §2.1/§5). PURE. Conservative + verdict-driven:
never "did not differ" (implies equivalence) or causal wording; uses ONLY the predeclared primary
metric; descriptive (not inferential) when ungrouped / single-group / stats skipped."""
from __future__ import annotations

from scat import metrics as _metrics


def _primary_comparison(stats, primary_metric):
    """Pull {test, p, significant} for the primary metric from the report's flat stats mapping, or
    None if unavailable/skipped. Mirrors _html_group_comparison's reading (report.py)."""
    if not isinstance(stats, dict) or stats.get("skipped"):
        return None
    entry = stats.get(primary_metric)
    if not isinstance(entry, dict):
        return None
    comp = entry.get("comparison", entry)
    test = comp.get("overall_test") or comp.get("test_name")
    p = comp.get("overall_p_value", comp.get("p_value"))
    if test is None or p is None:
        return None
    return {"test": test, "p": float(p),
            "significant": bool(comp.get("overall_significant", comp.get("significant", False)))}


def compose_finding(*, stats, primary_metric, headline, n_images, n_groups, group_label) -> dict:
    label = _metrics.METRICS[_metrics.resolve_metric(primary_metric)].label
    comp = _primary_comparison(stats, primary_metric) if (n_groups or 0) >= 2 else None
    if comp is None:
        # Descriptive: no inference (ungrouped / single group / skipped / missing).
        return {"sentence": f"Across {n_images} images, {label} averaged {headline}.",
                "metric": label, "effect": "no group comparison",
                "scope": f"{n_images} images"}
    verdict = ("differed" if comp["significant"]
               else "showed no statistically detected difference")
    gl = group_label or "group"
    sentence = (f"{label} {verdict} between {gl} groups "
                f"({comp['test']}, p = {comp['p']:.3f}).")
    return {"sentence": sentence, "metric": f"{label} · {headline}",
            "effect": f"{comp['test']}, p = {comp['p']:.3f}"
                      + (" (significant)" if comp["significant"] else " (n.s.)"),
            "scope": f"{n_images} images · {n_groups} groups"}
```

- [ ] **Step 4: Run** → PASS. **Step 5: Commit**

```bash
git add scat/findings.py tests/test_findings.py
git commit -m "feat(findings): conservative verdict-driven finding-sentence composer (pure)"
```

---

## Task 2: Thread `primary_metric` from the manifest into the report

**Files:** Modify `scat/pipeline.py` (`generate_report_service` ~250) + `scat/report.py`
(`generate_html_report` ~601, `_build_html` ~1054); Test `tests/test_report_lede.py`

- [ ] **Step 1: Read** — `sed -n '250,275p' scat/pipeline.py`; `sed -n '1054,1075p' scat/report.py`
  (`_build_html` param list + the assembled sections list).

- [ ] **Step 2: Write the failing test**

```python
# tests/test_report_lede.py
import json
from pathlib import Path
from scat.pipeline import analyze_folder_service, generate_report_service, run_statistics_service


def _report(tmp_path, synth_dir, primary="total_deposits"):
    res = analyze_folder_service(str(synth_dir), output_dir=str(tmp_path / "out"),
                                 primary_metric=primary, annotate=False)
    rd = Path(res.output_dir)
    stats = run_statistics_service(str(rd))
    generate_report_service(str(rd), statistical_results=stats, group_by="group")
    return (rd / "report.html").read_text()


def test_report_reads_primary_metric_from_manifest(synth_dir, tmp_path):
    html = _report(tmp_path, synth_dir, primary="rod_fraction")
    # the lede names the primary metric's label (proof it was threaded from the manifest)
    assert "ROD fraction" in html
```

(This test also covers Task 3's lede rendering; it will fully pass after Task 3.)

- [ ] **Step 3: Run** → FAIL (no lede yet / primary metric not threaded).

- [ ] **Step 4: Implement threading (no lede HTML yet — that's Task 3)**

- `generate_report_service`: read the manifest and forward `primary_metric`:

```python
# in generate_report_service, after rd is set:
primary_metric = None
mpath = rd / "run_manifest.json"
if mpath.exists():
    try:
        primary_metric = (json.loads(mpath.read_text()).get("analysis") or {}).get("primary_metric")
    except Exception:
        primary_metric = None
# pass primary_metric=... into generate_report(...)
```

- `generate_report` / `generate_html_report` / `_build_html`: add a `primary_metric: str = None`
  parameter threaded through each call (default None → `metrics.DEFAULT_METRIC` downstream). Do NOT
  render the lede yet; just make the parameter reach `_build_html`.

- [ ] **Step 5: Run** → the test may still fail (no lede) — that's expected; it passes after Task 3.
  Run the FULL suite `QT_QPA_PLATFORM=offscreen /home/lab/SCAT/.venv/bin/python -m pytest -q` → all
  pass (threading a defaulted param breaks nothing).

- [ ] **Step 6: Commit**

```bash
git add scat/pipeline.py scat/report.py
git commit -m "feat(report): thread primary_metric from the manifest into the report builder"
```

---

## Task 3: `_html_finding_lede` — the answer-first lede (top of the document)

**Files:** Modify `scat/report.py` (`_build_html` sections list + new `_html_finding_lede`);
Test `tests/test_report_lede.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_report_lede.py
def test_lede_has_finding_sentence_effect_trio_and_trust_line(synth_dir, tmp_path):
    html = _report(tmp_path, synth_dir, primary="total_deposits")
    assert 'class="lede"' in html or 'class="finding"' in html
    assert "Total deposits" in html
    assert "confidence-score threshold" in html          # factual trust line
    for bad in ("did not differ", "high-confidence", "reviewed in the app", "caused"):
        assert bad not in html.lower()
```

- [ ] **Step 2: Run** → FAIL (no lede markup).

- [ ] **Step 3: Implement** — add `_html_finding_lede` and insert it FIRST in `_build_html`'s section
  list (right after `_html_document_head`, before `_html_distributions`):

```python
# _build_html: add primary_metric to the signature, and insert the lede:
sections = [
    self._html_document_head(title, summary),
    self._html_finding_lede(film_summary, deposit_data, statistical_results, group_by, primary_metric),
    self._html_distributions(inline_plots),
    self._html_group_comparison(inline_plots, statistical_results, group_by),
    # ... spatial ...
]

def _html_finding_lede(self, film_summary, deposit_data, statistical_results, group_by, primary_metric):
    from scat import metrics as _metrics, confidence as _confidence, findings as _findings
    from scat.config import config
    pm = _metrics.resolve_metric(primary_metric)
    norm = "per_image"
    headline = _metrics.format_headline(film_summary, pm, norm, meta={})
    n_images = len(film_summary)
    n_groups = int(film_summary[group_by].dropna().nunique()) if group_by and group_by in film_summary.columns else 0
    group_label = self.get_metric_label(group_by) if group_by else None
    if group_label and group_label.strip().lower() in ("group", "groups", "condition"):
        group_label = group_label.lower()
    f = _findings.compose_finding(stats=statistical_results, primary_metric=pm, headline=headline,
                                  n_images=n_images, n_groups=n_groups, group_label=group_label)
    thr = float(config.get("analysis.confidence_threshold", _metrics.DEFAULT_THRESHOLD))
    trust = _confidence.run_trust(deposit_data, thr)
    import html as _h
    return f'''
    <div class="section">
      <div class="lede">
        <div class="finding">{_h.escape(f["sentence"])}</div>
        <div class="lede-trio">
          <span><b>Primary metric</b> {_h.escape(f["metric"])}</span>
          <span><b>Effect</b> {_h.escape(f["effect"])}</span>
          <span><b>Scope</b> {_h.escape(f["scope"])}</span>
        </div>
        <div class="lede-trust">{_h.escape(trust["line"])}</div>
      </div>
    </div>
'''
```

Add minimal CSS to the document `<style>` (near `.stat-card`): `.lede{border-left:4px solid var(--rod);
background:var(--surface);border:1px solid var(--hair);border-radius:8px;padding:22px 24px;margin:20px 0}`
`.finding{font-family:var(--serif);font-size:1.4rem;font-weight:600;line-height:1.35}`
`.lede-trio{display:flex;gap:32px;margin-top:14px;font-size:0.9rem}` `.lede-trio b{color:var(--muted);
font-weight:600;text-transform:uppercase;font-size:0.7rem;letter-spacing:var(--track-caps);display:block}`
`.lede-trust{color:var(--muted);font-size:0.8rem;margin-top:12px;border-top:1px solid var(--hair);padding-top:10px}`

- [ ] **Step 4: Run + full suite** → PASS (Task 2's test now passes too); `pytest -q` all green,
  including `tests/test_report.py`.
- [ ] **Step 5: Verify** — regenerate a report via the render harness + Chrome; the lede leads with the
  finding sentence + trio + trust line.
- [ ] **Step 6: Commit**

```bash
git add scat/report.py tests/test_report_lede.py
git commit -m "feat(report): answer-first Finding lede (sentence + effect trio + factual trust line)"
```

---

## Task 4: `_html_methods` — the Methods appendix (NEW)

**Files:** Modify `scat/report.py` (`_build_html` + new `_html_methods`); Test `tests/test_report_lede.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_report_lede.py
def test_methods_appendix_explains_confidence_and_unit(synth_dir, tmp_path):
    html = _report(tmp_path, synth_dir).lower()
    assert "methods" in html
    assert "image-level" in html                 # experimental unit stated (no pseudoreplication)
    assert "uncalibrated" in html or "not a calibrated probability" in html  # confidence honesty
    assert "reject class" in html                # artifacts = reject class
```

- [ ] **Step 2: Run** → FAIL (no methods section).

- [ ] **Step 3: Implement** — append `_html_methods()` to `_build_html`'s section list (after the
  existing sections / appendix), returning static templated text:

```python
def _html_methods(self):
    return '''
    <div class="section">
      <h2>Appendix — Methods</h2>
      <p class="section-intro">How the numbers were produced.</p>
      <p><b>Detection &amp; classification.</b> Deposits are detected, then a Random-Forest (or
      rule-based) classifier labels each as Normal, ROD, or Artifact. ROD Fraction = ROD / (Normal +
      ROD); Artifacts are the reject class, excluded from deposit counts and metrics.</p>
      <p><b>Confidence.</b> The per-deposit confidence is the classifier score (the RF class
      probability, or a circularity-derived score in rule-based mode). It is <b>uncalibrated</b> — it
      is not a calibrated probability of correctness, and it covers classification only, not detection.
      The "below the confidence-score threshold" counts are a review/workload signal, not a reliability
      measure.</p>
      <p><b>Statistics.</b> Group comparisons test <b>image-level</b> aggregates (the experimental unit
      is the image, not the deposit — avoiding pseudoreplication): Kruskal-Wallis / Mann-Whitney with
      Holm-corrected pairwise comparisons and effect sizes.</p>
    </div>
'''
```

- [ ] **Step 4: Run + full suite** → PASS.
- [ ] **Step 5: Verify** — Chrome render shows the Methods appendix.
- [ ] **Step 6: Commit**

```bash
git add scat/report.py tests/test_report_lede.py
git commit -m "feat(report): Methods appendix explaining confidence + the experimental unit"
```

---

## Self-review checklist

- **Spec coverage (§5 items 2 + 6):** finding sentence + effect trio + trust line (T1 composer, T3
  render); Methods appendix (T4). Figure reordering (Fig 1 primary / Fig 2 exploratory / demote the
  population overview / spatial→Exploratory / per-image ledger collapse / figure numbering) is Plan 4b.
- **Honesty:** composer forbids "did not differ"/causal, uses only the primary metric, descriptive when
  ungrouped (tests assert this); trust line via `confidence.run_trust`; Methods states uncalibrated +
  image-level unit + reject class (tests assert these strings).
- **Threading:** `primary_metric` flows manifest → `generate_report_service` → `generate_report` →
  `generate_html_report` → `_build_html` → `_html_finding_lede`, defaulting to `metrics.DEFAULT_METRIC`.
- **Stats shape risk:** `_primary_comparison` reads `overall_test/overall_p_value/overall_significant`
  (with `test_name`/`p_value`/`significant` fallbacks) — the implementer MUST confirm this against
  `_html_group_comparison`'s own extraction (report.py ~1240-1270) and align the keys; the grouped test
  uses that shape. **Placeholder scan:** none. **Type consistency:** `compose_finding` returns
  `{sentence, metric, effect, scope}`, consumed identically in `_html_finding_lede`.
