# SCAT redesign — Plan 4b: report figure reordering (Phase 4b)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the "Findings Note" reorder of the HTML report: the finding **leads** (the pooled
Summary + distribution histograms **demote** to a "Population overview" below the evidence); the
primary metric's group comparison becomes **Figure 1** and the rest **Figure 2 — Secondary metrics
(exploratory)**; the spatial section is tagged **Exploratory**; the per-image ledger becomes a
**collapsible** appendix.

**Architecture:** Reorder `_build_html`'s section list and split the Summary stat-cards out of
`_html_document_head` (so the masthead can stand alone at the top and the cards can render lower under a
"Population overview" heading). `_html_group_comparison` gains a `primary_metric` argument and renders
the primary metric's panel first under a "Figure 1" heading, the rest under "Figure 2 — Secondary
metrics (exploratory)". Small tags/wrappers for spatial + the film table. No new modules.

**Tech Stack:** Python 3.10+, pandas, pytest. No new deps.

**Spec:** `docs/.../2026-07-17-scat-results-report-redesign-design.md` (§5 items 3-9). **Depends on
Plan 4a** (lede + `analysis` threaded to `_build_html`; primary-metric → stats-key map in
`scat/findings._STATS_KEY`).

**Current section order (post-4a), from `_build_html`:** `_html_document_head` (masthead **+ Summary
cards**), `_html_finding_lede`, `_html_distributions`, `_html_group_comparison`, `_html_stats_appendix`,
`_html_spatial_section`, `_html_film_table`, `_html_methods`, footer.

**Honesty:** "Secondary metrics (exploratory)" and the spatial "Exploratory" tag are explicit
(non-primary comparisons are not protected endpoints — spec §2.1 / codex). The primary metric is the
predeclared one from the manifest; Figure 1 is always it (never the most-significant).

---

## File structure

- **Modify `scat/report.py`:**
  - Split `_html_document_head` → keep masthead; new `_html_summary_cards(summary)` (the stat grid).
  - Reorder `_build_html`'s list; wrap Summary-cards + distributions in a "Population overview" section.
  - `_html_group_comparison(..., primary_metric)` — Figure 1 (primary) then Figure 2 (exploratory).
  - `_html_spatial_section` — add an "Exploratory" tag; `_html_film_table` — wrap in `<details>`.
- **Test:** `tests/test_report_findings_note.py` (reuse Plan 4a's `_grouped_report`-style helper).

---

## Task 1: Split the Summary cards out of `_html_document_head`

**Files:** Modify `scat/report.py` (`_html_document_head` ~1118 + new `_html_summary_cards`);
Test `tests/test_report_findings_note.py`

- [ ] **Step 1: Read** — `sed -n '1118,1175p' scat/report.py` (find where the `<!-- Summary Section -->`
  `<div class="section"><h2>Summary</h2>...` block starts and where it closes — that whole block moves).

- [ ] **Step 2: Write the failing test**

```python
# tests/test_report_findings_note.py
from pathlib import Path


def _report(tmp_path, synth_dir, primary="total_deposits"):
    from scat.pipeline import analyze_folder_service, generate_report_service, run_statistics_service
    groups = {f"ctrl_{i}.tif": "Control" for i in range(3)}
    groups.update({f"treat_{i}.tif": "Treatment" for i in range(3)})
    res = analyze_folder_service(str(synth_dir), output_dir=str(tmp_path / "out"),
                                 primary_metric=primary, groups=groups, annotate=False)
    rd = Path(res.output_dir)
    stats = run_statistics_service(str(rd))
    generate_report_service(str(rd), statistical_results=stats, group_by="group")
    return (rd / "report.html").read_text()


def test_summary_cards_render_via_dedicated_method(synth_dir, tmp_path):
    from scat.report import ReportGenerator
    import inspect
    assert hasattr(ReportGenerator, "_html_summary_cards")
    # the Summary stat grid still renders in the report
    html = _report(tmp_path, synth_dir)
    assert 'class="stats-grid"' in html and "ROD Fraction" in html
```

- [ ] **Step 3: Run** — `QT_QPA_PLATFORM=offscreen /home/lab/SCAT/.venv/bin/python -m pytest tests/test_report_findings_note.py -q` → FAIL (no `_html_summary_cards`).

- [ ] **Step 4: Implement** — move the `<!-- Summary Section --> <div class="section"><h2>Summary</h2>
  ...</div>` block out of `_html_document_head` into a new `def _html_summary_cards(self, summary):`
  that returns exactly that block. `_html_document_head` returns everything up to and INCLUDING
  `</div>` of `.header` (the masthead) but NOT the Summary block. Wire `_html_summary_cards(summary)`
  into `_build_html` in the SAME position for now (right after `_html_document_head`) so this task is a
  pure refactor with no reordering yet.

- [ ] **Step 5: Run + full suite** → PASS; `QT_QPA_PLATFORM=offscreen /home/lab/SCAT/.venv/bin/python -m pytest -q` all pass (`test_report.py` unchanged output — the same HTML, just assembled from two methods).

- [ ] **Step 6: Commit**

```bash
git add scat/report.py tests/test_report_findings_note.py
git commit -m "refactor(report): split Summary stat-cards out of _html_document_head"
```

---

## Task 2: Reorder — lede leads, Summary+distributions demote to "Population overview"

**Files:** Modify `scat/report.py` (`_build_html` section list + wrap Summary+distributions);
Test `tests/test_report_findings_note.py`

- [ ] **Step 1: Write the failing test** (assert the DOCUMENT ORDER by string position)

```python
# append to tests/test_report_findings_note.py
def test_finding_leads_and_summary_demotes_below_evidence(synth_dir, tmp_path):
    html = _report(tmp_path, synth_dir)
    i_finding = html.find('class="finding"')
    i_summary = html.find('class="stats-grid"')
    i_group = html.find("Group Comparison")
    i_pop = html.lower().find("population overview")
    assert i_finding != -1 and i_summary != -1 and i_group != -1 and i_pop != -1
    # finding leads; the group-comparison evidence comes BEFORE the pooled Summary cards;
    # the Summary sits inside the demoted Population overview.
    assert i_finding < i_group < i_summary
    assert i_pop < i_summary
```

- [ ] **Step 2: Run** → FAIL (Summary currently precedes the finding/evidence).

- [ ] **Step 3: Implement** — reorder `_build_html`'s list to the Findings-Note order and wrap the
  demoted pooled content in a "Population overview" section header:

```python
return "".join([
    self._html_document_head(title, summary),                                   # masthead only
    self._html_finding_lede(film_summary, deposit_data, statistical_results, group_by, analysis),
    self._html_group_comparison(inline_plots, statistical_results, group_by, analysis),  # evidence
    '<div class="section"><h2>Population overview</h2>'
    '<p class="section-intro">Pooled characteristics across all images (context, not the headline).</p>',
    self._html_summary_cards(summary),
    self._html_distributions(inline_plots),
    '</div>',
    self._html_stats_appendix(statistical_results),
    self._html_spatial_section(spatial_stats),
    self._html_film_table(film_summary),
    self._html_methods(),
    _REPORT_FOOTER,
])
```

Note `_html_group_comparison` now takes `analysis` (Task 3 uses it; add the param now, default None).
`_html_summary_cards`/`_html_distributions` return `<div class="section">…</div>` blocks — wrapping
them inside another `<div class="section">` is acceptable (nested sections render fine); if the visual
nesting is ugly in the render, drop the inner sections' own `.section` wrapper in Task 2 (note it).

- [ ] **Step 4: Run + full suite** → PASS; all green.
- [ ] **Step 5: Verify** — Chrome render: masthead → finding lede → group-comparison figures →
  "Population overview" (Summary cards + histograms) → appendices.
- [ ] **Step 6: Commit**

```bash
git add scat/report.py tests/test_report_findings_note.py
git commit -m "feat(report): finding leads; pooled Summary + distributions demote to Population overview"
```

---

## Task 3: Figure 1 (primary) + Figure 2 (secondary, exploratory)

**Files:** Modify `scat/report.py` (`_html_group_comparison` ~1242); Test `tests/test_report_findings_note.py`

- [ ] **Step 1: Read** — `sed -n '1242,1330p' scat/report.py` (the `group_metrics` list + the
  two-column cell loop + the omnibus captions). Note the metric→stats-key pairs match
  `findings._STATS_KEY`.

- [ ] **Step 2: Write the failing test**

```python
# append to tests/test_report_findings_note.py
def test_primary_metric_is_figure_1_rest_are_exploratory(synth_dir, tmp_path):
    html = _report(tmp_path, synth_dir, primary="rod_fraction")
    assert "Figure 1" in html
    i_fig1 = html.find("Figure 1")
    i_rodplot = html.find("ROD Fraction by Group")     # primary metric's plot title
    i_secondary = html.lower().find("secondary metrics")
    assert i_fig1 != -1 and i_secondary != -1
    assert "exploratory" in html.lower()
    # the primary metric's figure appears in the Figure-1 region, before the Secondary block
    assert i_fig1 < i_secondary
```

(`"ROD Fraction by Group"` is the matplotlib title the existing code sets; confirm the exact string in
`_generate_metric_boxplot`/`_generate_all_group_comparisons` while reading, and use whatever the plot
alt/title actually is.)

- [ ] **Step 3: Implement** — give `_html_group_comparison(self, inline_plots, statistical_results,
  group_by, analysis=None)` the primary metric (via `analysis`), map it to its stats/plot key using
  `findings._STATS_KEY`, and render:
  - a `<h3>Figure 1 — {primary label} (primary endpoint)</h3>` + the primary metric's single
    plot-container (with its omnibus caption), then
  - a `<h3>Figure 2 — Secondary metrics (exploratory)</h3>` + the remaining metrics in the existing
    two-column grid.
  Keep the existing omnibus-caption logic; just partition `group_metrics` into `[primary] + rest` and
  emit two headed blocks. If the primary metric isn't in `group_metrics` (shouldn't happen — all are),
  fall back to the current single-list rendering under one heading.

- [ ] **Step 4: Run + full suite** → PASS.
- [ ] **Step 5: Verify** — Chrome render: Figure 1 = the chosen metric, full-width lead; Figure 2 =
  the others, labeled exploratory.
- [ ] **Step 6: Commit**

```bash
git add scat/report.py tests/test_report_findings_note.py
git commit -m "feat(report): Figure 1 = primary endpoint, Figure 2 = secondary metrics (exploratory)"
```

---

## Task 4: Spatial "Exploratory" tag + collapsible per-image ledger

**Files:** Modify `scat/report.py` (`_html_spatial_section` ~1564, `_html_film_table`);
Test `tests/test_report_findings_note.py`

- [ ] **Step 1: Read** — `sed -n '1564,1580p' scat/report.py` (spatial `<h2>`), and
  `grep -n "def _html_film_table" -A20 scat/report.py` (the film-table wrapper).

- [ ] **Step 2: Write the failing test**

```python
# append to tests/test_report_findings_note.py
def test_spatial_tagged_exploratory_and_film_table_collapsible(synth_dir, tmp_path):
    html = _report(tmp_path, synth_dir)
    # spatial only renders when spatial_stats exist; if absent this run has no spatial section —
    # so assert the film-table collapse unconditionally and spatial-tag only when present.
    assert "<details" in html.lower()                 # per-image ledger is collapsible
    if "Spatial Analysis" in html:
        # the spatial heading (or an adjacent tag) marks it Exploratory
        i = html.find("Spatial Analysis")
        assert "exploratory" in html[max(0, i - 60):i + 200].lower()
```

- [ ] **Step 3: Implement** — in `_html_spatial_section`, add an "Exploratory" tag beside the
  `<h2>Spatial Analysis</h2>` (e.g. `<h2>Spatial Analysis <span class="exp-tag">Exploratory</span></h2>`
  + a `.exp-tag` CSS rule near `.appendix-ref`). In `_html_film_table`, wrap the table in
  `<details><summary>Per-image ledger (N images)</summary>…</details>`; add `@media print{details{...}}`
  so print force-opens it (or a `open` attribute is acceptable for v1 — note the choice).

- [ ] **Step 4: Run + full suite** → PASS.
- [ ] **Step 5: Verify** — Chrome render.
- [ ] **Step 6: Commit**

```bash
git add scat/report.py tests/test_report_findings_note.py
git commit -m "feat(report): spatial tagged Exploratory; per-image ledger collapsible"
```

---

## Self-review checklist

- **Spec coverage (§5 items 3-9):** finding leads + Summary demote (T2); Figure 1 primary / Figure 2
  exploratory (T3); spatial Exploratory + per-image collapse (T4). Figure↔appendix numbered cross-refs
  (A1↔Fig1) are deferred — a small later polish; the "Figure 1/2" labels ship here.
- **Honesty:** Figure 1 is always the predeclared primary metric (not the most-significant); Figure 2 +
  spatial are explicitly "exploratory".
- **Risk:** T1 is a pure refactor (same HTML from two methods) with a full-suite gate; T2/T3 reorder +
  partition — verified by document-order string tests + Chrome render. `_html_group_comparison` gains an
  `analysis` param (default None → current behavior) so callers/tests that don't pass it still work.
- **Placeholder scan:** T3/T4 give contract + guidance (the implementer reads the real
  `_html_group_comparison`/`_html_film_table` first) because they weave into existing loops; every other
  step has concrete code. **Type consistency:** `_html_summary_cards(summary)` /
  `_html_group_comparison(..., analysis)` signatures used identically in `_build_html`.
```
