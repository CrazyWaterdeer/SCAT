# SCAT redesign — Plan 4b: report figure reordering (Phase 4b)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the "Findings Note" reorder: the finding **leads**; the pooled Summary + distribution
histograms **demote** to a self-contained "Population overview" below the evidence; the primary metric's
group comparison is **Figure 1** and the rest **Figure 2 — Secondary metrics (exploratory)**; spatial is
tagged **Exploratory**; the per-image ledger is **collapsible**.

**CRITICAL structure fact (verified — a codex review caught this):** the report's section tags are
**tangled across methods**. `_html_document_head` opens `<html><body>`, the masthead, AND a
`<div class="section"><h2>Summary</h2>…` that it **never closes**; `_html_distributions` (docstring:
"closes the Summary section") emits the closing `</div>` for it (`scat/report.py:1239`). So Summary +
Distributions are **one section split across two methods**. `_REPORT_FOOTER` closes `</body></html>`.
Therefore we must **normalize fragment ownership first**: fold the Summary cards + distributions into a
single self-contained `_html_population_overview(...)` and make `_html_document_head` masthead-only. A
naive "extract + wrap" would emit stray/mismatched `</div>`.

**Architecture:** Task 1 does the ownership-normalizing refactor **and** the reorder atomically (a
partial split would produce malformed HTML). Task 2 = Figure 1/2 in `_html_group_comparison` (with an
explicit primary→plot-key map). Task 3 = spatial Exploratory tag + `<details>` film table.

**Spec:** §5 items 3-9. **Depends on Plan 4a** (`analysis` threaded to `_build_html`;
`findings._STATS_KEY`). **Self-contained methods today:** `_html_group_comparison`,
`_html_stats_appendix`, `_html_spatial_section`, `_html_film_table`, `_html_methods`, `_html_finding_lede`
(each returns its own `<div class="section">…</div>`). **NOT self-contained:** `_html_document_head`
(opens Summary, no close) + `_html_distributions` (only closes it).

**Honesty:** Figure 1 is always the predeclared primary metric (never the most-significant); Figure 2 +
spatial are explicitly "exploratory".

---

## Task 1: Population-overview refactor + Findings-Note reorder (atomic, structure-checked)

**Files:** Modify `scat/report.py` (`_html_document_head` ~1118, `_html_distributions` ~1176, new
`_html_population_overview`, `_build_html` ~1074); Test `tests/test_report_findings_note.py`

- [ ] **Step 1: Read the tangle** — `sed -n '1118,1175p' scat/report.py` (document_head: the boilerplate
  + masthead + the UNCLOSED `<div class="section"><h2>Summary</h2><div class="stats-grid">…cards…</div>`);
  `sed -n '1176,1241p' scat/report.py` (distributions: `<h3>Distributions</h3>` + 6 histograms, then the
  trailing `html += '    </div>\n'` that closes the Summary section). Note the EXACT card-grid + histogram
  markup so you move it verbatim.

- [ ] **Step 2: Write the failing test** (structure sanity + order via bounded slices)

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


def test_html_structure_is_balanced(synth_dir, tmp_path):
    html = _report(tmp_path, synth_dir)
    assert html.count("<body") == 1 and html.count("</body>") == 1
    assert html.count("<html") == 1 and html.count("</html>") == 1
    # no premature imbalance: every <div ...> is closed
    assert html.count("<div") == html.count("</div>")


def test_finding_leads_summary_demotes_to_population_overview(synth_dir, tmp_path):
    html = _report(tmp_path, synth_dir)
    i_finding = html.find('class="finding"')
    i_group = html.find("Group Comparison")
    i_pop = html.lower().find("population overview")
    i_grid = html.find('class="stats-grid"')
    assert -1 < i_finding < i_group < i_pop < i_grid       # finding → evidence → demoted pooled cards
    # the Summary grid + Distributions live INSIDE the Population overview slice
    pop_slice = html[i_pop:i_pop + 60000]
    assert 'class="stats-grid"' in pop_slice and "Distributions" in pop_slice
```

- [ ] **Step 3: Run** → FAIL (Summary precedes the finding; no "Population overview").

- [ ] **Step 4: Implement (atomic)**
  1. `_html_document_head(title, summary)` → return ONLY through the masthead: the `<!DOCTYPE …><head>…
     </head><body>` + `<div class="header">…</div>`. **Delete** the `<!-- Summary Section --> <div
     class="section"><h2>Summary</h2> … stats-grid … </div>` open+cards from it.
  2. New `_html_population_overview(self, summary, inline_plots)` → ONE self-contained section:
     `<div class="section"><h2>Population overview</h2><p class="section-intro">Pooled characteristics
     across all images (context, not the headline).</p>` + the **stats-grid cards markup** (moved from
     document_head) + the **distributions body** (moved from `_html_distributions`, WITHOUT its trailing
     `</div>`) + `</div>`.
  3. Delete `_html_distributions` (its body now lives in population_overview) OR keep it returning just
     the distributions body and call it from population_overview — your choice; if kept, it must NOT emit
     the trailing `</div>` anymore (that close moves to population_overview). State which you did.
  4. `_build_html` new list:
     ```python
     return "".join([
         self._html_document_head(title, summary),                                  # masthead only
         self._html_finding_lede(film_summary, deposit_data, statistical_results, group_by, analysis),
         self._html_group_comparison(inline_plots, statistical_results, group_by, analysis),  # analysis added (Task 2)
         self._html_population_overview(summary, inline_plots),                      # demoted pooled context
         self._html_stats_appendix(statistical_results),
         self._html_spatial_section(spatial_stats),
         self._html_film_table(film_summary),
         self._html_methods(),
         _REPORT_FOOTER,
     ])
     ```
     Add `analysis=None` to `_html_group_comparison`'s signature now (Task 2 uses it; default = current behavior).

- [ ] **Step 5: Run + full suite** → PASS; `QT_QPA_PLATFORM=offscreen /home/lab/SCAT/.venv/bin/python -m pytest -q` all pass (esp. `test_report.py` — the same cards/plots, new order + one heading rename; if it asserts the literal "Summary" heading, update it to "Population overview" and note it).
- [ ] **Step 6: Verify** — Chrome render: masthead → finding → group figures → "Population overview"
  (cards + histograms) → appendices. No stray boxes / broken layout.
- [ ] **Step 7: Commit**

```bash
git add scat/report.py tests/test_report_findings_note.py
git commit -m "feat(report): finding leads; Summary+distributions fold into a demoted Population overview"
```

---

## Task 2: Figure 1 (primary endpoint) + Figure 2 (secondary, exploratory)

**Files:** Modify `scat/report.py` (`_html_group_comparison` ~1242); Test `tests/test_report_findings_note.py`

- [ ] **Step 1: Read** — `sed -n '1242,1330p' scat/report.py`. Note `group_metrics` = list of
  `(plot_key, stat_key, title, desc)` where `plot_key == "group_" + stat_key`, the img `alt` is the
  `title` (e.g. `alt="ROD Fraction"`), and the two-column cell loop + omnibus captions.

- [ ] **Step 2: Write the failing test** (assert the primary plot is IN the Figure-1 slice — not the PNG)

```python
# append to tests/test_report_findings_note.py
def test_primary_metric_is_figure_1(synth_dir, tmp_path):
    html = _report(tmp_path, synth_dir, primary="rod_fraction")
    i_f1, i_f2 = html.find("Figure 1"), html.find("Figure 2")
    assert -1 < i_f1 < i_f2
    assert "exploratory" in html[i_f2:i_f2 + 400].lower()
    # the primary metric's plot (alt="ROD Fraction") sits in the Figure-1 slice, not Figure-2
    assert 'alt="ROD Fraction"' in html[i_f1:i_f2]
    assert 'alt="ROD Fraction"' not in html[i_f2:]
```

- [ ] **Step 3: Implement** — `_html_group_comparison(self, inline_plots, statistical_results, group_by,
  analysis=None)`: resolve the primary via `metrics.resolve_metric(analysis.get("primary_metric"))` →
  `findings._STATS_KEY[pm]` → `plot_key = "group_" + stats_key`. Partition `group_metrics` into
  `primary_first = [the tuple whose plot_key matches] ` and `rest = [others in the current order]`.
  Render:
  - `<h3>Figure 1 — {primary title} (primary endpoint)</h3>` then that ONE metric's plot-container
    (reuse the existing cell builder + omnibus caption), then
  - `<h3>Figure 2 — Secondary metrics (exploratory)</h3>` then `rest` in the existing two-column grid.
  If the primary's `plot_key` is not in `inline_plots` (e.g. no groups), fall back to the current
  single-list rendering under the existing heading (no Figure labels). Keep the omnibus/appendix logic.

- [ ] **Step 4: Run + full suite** → PASS.
- [ ] **Step 5: Verify** — Chrome render: Figure 1 = the chosen metric; Figure 2 = the rest (exploratory).
- [ ] **Step 6: Commit**

```bash
git add scat/report.py tests/test_report_findings_note.py
git commit -m "feat(report): Figure 1 = primary endpoint, Figure 2 = secondary metrics (exploratory)"
```

---

## Task 3: Spatial "Exploratory" tag + collapsible per-image ledger

**Files:** Modify `scat/report.py` (`_html_spatial_section` ~1564, `_html_film_table`, CSS);
Test `tests/test_report_findings_note.py`

- [ ] **Step 1: Read** — `sed -n '1564,1580p' scat/report.py` (spatial `<h2>`) and
  `grep -n "def _html_film_table" -A25 scat/report.py` (its self-contained `<div class="section">` +
  table). Note the exact table wrapper so `<details>` goes INSIDE the section, around the table only.

- [ ] **Step 2: Write the failing test**

```python
# append to tests/test_report_findings_note.py
def test_spatial_exploratory_and_film_table_collapsed(synth_dir, tmp_path):
    html = _report(tmp_path, synth_dir)
    assert "<details" in html.lower()                        # per-image ledger collapsible
    assert "<details open" not in html.lower()               # collapsed by default (not open)
    if "Spatial Analysis" in html:
        i = html.find("Spatial Analysis")
        assert "exploratory" in html[max(0, i - 40):i + 260].lower()
```

- [ ] **Step 3: Implement** — in `_html_spatial_section`, add an Exploratory tag beside the heading:
  `<h2>Spatial Analysis <span class="exp-tag">Exploratory</span></h2>` + a `.exp-tag` CSS rule near
  `.appendix-ref` (small, muted, uppercase). In `_html_film_table`, wrap the `<table>…</table>` INSIDE
  its existing `.section` in `<details><summary>Per-image ledger ({n} images)</summary> … </details>`
  (NO `open` attribute — collapsed). Add `@media print { details { } details > summary { } }` +
  `details[open]`-independent print handling, or simply a `@media print` rule that forces the table
  visible; if print-open is non-trivial, ship collapsed + note it as a follow-up.

- [ ] **Step 4: Run + full suite** → PASS.
- [ ] **Step 5: Verify** — Chrome render: spatial tagged Exploratory (if present); the per-image table
  is a collapsed `<details>`.
- [ ] **Step 6: Commit**

```bash
git add scat/report.py tests/test_report_findings_note.py
git commit -m "feat(report): spatial tagged Exploratory; per-image ledger collapsible"
```

---

## Self-review checklist

- **Codex blocker fixed:** the tangled Summary/Distributions ownership is normalized into a single
  self-contained `_html_population_overview` (Task 1), and Task 1 does the refactor + reorder **atomically**
  (no intermediate malformed state); a **structure-sanity test** (one body/html, balanced `<div>`) guards it.
- **Codex fixes folded:** explicit primary→stats-key→`group_`+key plot-key mapping (T2); the Figure test
  asserts `alt="ROD Fraction"` inside the Figure-1 slice (not a PNG title) and its absence from Figure-2;
  bounded-slice order test (T1); `<details>` without `open` = collapsed (T3); `_html_group_comparison`
  gains `analysis=None` (back-compatible).
- **Spec coverage (§5 3-9):** finding leads + demote (T1); Figure 1 primary / Figure 2 exploratory (T2);
  spatial Exploratory + per-image collapse (T3). Numbered Fig↔appendix cross-refs deferred (later polish).
- **Placeholder scan:** T1/T2/T3 give precise contracts + guidance for the tag surgery (the implementer
  moves verbatim markup after reading the real lines); the structure test is the guardrail.
