# SCAT — Result window + HTML report, redesigned from first principles

**Date:** 2026-07-17
**Status:** Design (approved in brainstorming; pending written-spec review → implementation plan)
**Surfaces:** `scat/main_gui.py` `ResultsTab`; `scat/report.py` `ReportGenerator`
**Supersedes the polish pass** shipped in `85ae757` / `9400e0f` where the two conflict — this
re-questions the information architecture itself, not just the styling.

## 1. Why

The elevation + polish work matched both surfaces to one visual language, but left the *layout and
composition* untouched — and that IA does not fit the job either surface actually does. Two problems:

- The result window is a **dashboard** (hero number → KPI tiles → per-image table → a ~20-chart
  gallery → stats). But its job is **interactive triage + correction**, and it never surfaces the one
  signal that job needs (the classifier already emits a per-deposit **confidence**, unused).
- The report **opens with an 8-card pooled Summary + 6 histograms** — it never *states a finding*,
  buries the comparison, has no Methods, and treats spatial as a peer of the headline. Its job is a
  **shareable narrative + rigorous appendix** for a reader who wasn't present.

Both hard-code **ROD Fraction** as "the answer," which is wrong: depending on the experiment the
metric of interest is often deposit count, area, pH indicator (hue), pigment (IOD), or circularity.

## 2. Confirmed direction (from the design dialogue)

- The result window (in-app) and the report serve **two different jobs** — they may diverge in
  *structure*, not only styling.
  - **Result window = interactive TRIAGE + FIX.** Reader = the operator, mid-analysis, can act.
  - **Report = shareable NARRATIVE + rigorous APPENDIX** (progressive disclosure). Reader = someone
    who wasn't present; static self-contained HTML.
- **Primary metric is configurable, chosen in the AI chat; default = total deposit count** (not ROD
  Fraction). Both surfaces lead with the *primary* metric.
- **Per-deposit confidence is surfaced** as a triage signal (result window) and a trust line (report),
  with the report's Methods explaining how it's measured.

## 3. Shared mechanism — Primary metric + confidence

### 3.1 Primary metric
- A `primary_metric` selection flows into the results (config + the results dict). Candidates are the
  per-image metrics SCAT already computes: `n_total` (Total deposits — **default**), `rod_fraction`,
  `mean_area`, `mean_hue` (pH indicator), `total_iod`, `mean_circularity`.
- **Selected in the conversational agent**: the chat agent already infers grouping from filenames;
  extend it to capture "what is this experiment measuring?" and set `primary_metric` (a tool
  parameter / inferred, confirmable in chat). For non-agent runs (Run Analysis button, Load previous)
  it **falls back to the default** (`n_total`); a lightweight in-GUI switch is a *v2* nicety, not v1.
- The result-window hero, the report lede/finding sentence, and report **Figure 1** all key off
  `primary_metric`. "Deposits" everywhere = Normal + ROD (artifacts are the reject class), consistent
  with the shipped `9400e0f` decision.

### 3.2 Confidence
- `all_deposits.csv` already carries per-deposit `label` + `confidence` (RF class probability).
- **Review(N)** for an image = count of that image's deposits with `confidence < threshold`
  (default threshold **0.60**, sourced from config; see open items). Aggregated by joining
  `all_deposits` (grouped by filename) onto the film-summary-driven table. `deposit_data` must be
  loaded into `ResultsTab` (today the table is film_summary-only).
- **Run trust state** = derived from the flagged population: e.g. *Trustworthy* (0 flagged) /
  *Review recommended* (some) / *Low confidence* (many). Drives the header traffic-light.
- **"% high-confidence"** (trust line) = share of deposits with `confidence ≥ threshold`.
- **Honest range band** = the primary metric recomputed at the two extremes of reclassifying the
  flagged (low-confidence) deposits — the min/max the metric could take if every flagged call went
  the other way. It is a sensitivity bound, **not** a statistical confidence interval, and is labeled
  as such. (The exact reclassification rule — flip to the single most-likely alternative label vs. the
  worst-case label — is finalized in the implementation plan.)

## 4. Result window — "Verdict + Worklist"

A triage surface, not a dashboard: **verdict → the images that need you → fix.** Everything
report-grade moves to the report. Top-to-bottom:

1. **Verdict header.** LEFT: the **primary-metric hero** (e.g. "Total deposits · 40.8 / image", or
   "Mean ROD Fraction 8.3%" when ROD is chosen) at display size; a **traffic-light trust line**
   ("● Review recommended — 6 of 30 images have low-confidence calls"); an **honest range band**
   ("could read 6.1%–10.8% if the flagged deposits were reclassified") **explicitly labeled so it is
   never mistaken for a statistical CI**; a subtitle folding in ±SD · N images · N deposits. RIGHT:
   the **state-driven primary action** ("Review next (6)" while dirty → coral "Open report" once
   clean) plus Open folder.
2. **Composition strip.** One slim, semantically-tinted line — "Normal 1,412 · ROD 128 · ROD 8.3% ·
   Artifact 96 · Total IOD 43,311" — **replacing the six KPI tiles** (composition is context, not the
   work, so it doesn't earn a full band).
3. **Worklist controls.** A thin row docked above the table: an **All images (N) / Needs review (N)**
   segmented toggle + the **confidence threshold** control (recomputes Review(N) live).
4. **Per-image worklist** — the ONE table, the heart. Columns: Filename · **Review(N)** · Normal ·
   ROD · Artifact · ROD % · Total IOD. Review(N) cell hover → reason hint ("lowest confidence 0.52").
   **Default sort = filename** (confirmed); the Review column is sortable to bubble flagged images up.
   Double-click / Enter → drill-in.
5. **Drill-in edit** (existing deposit editor, one level deeper): deposits sorted **worst-confidence
   first** + a "show only low-confidence" filter; correcting a label **decrements Review(N)**. This is
   where per-deposit confidence literally shows.
6. **Quiet footer.** A muted pointer — "Full distributions, group comparisons and statistics are in
   the report → [Open report]" — plus **Rebuild after edits** (lit when edits are pending) and
   **Load a previous results folder…**.

**Dropped / demoted from the working view:** the six KPI tiles (→ composition line); the entire
visualization gallery, descriptive-stats table, group-comparison text, and spatial text (→ the
report; the working view keeps NO charts); "Rebuild" as a co-equal top button (→ footer + edits-
pending cue); ROD Fraction as the fixed hero (→ configurable primary metric).

**Added:** the primary-metric hero; the trust traffic-light + honest range band; the Review(N)
column + reason hint; the All/Needs-review toggle + threshold control; worst-confidence-first +
filter in the drill-in; the report pointer.

## 5. Report — "Findings Note" (research letter)

Lead with the finding for someone who wasn't present; rigor beneath via progressive disclosure.

1. **Masthead** — keep the pH-gradient hairline signature; a provenance dateline (run · N images ·
   N deposits · grouped-by · N groups · date) replaces the bare "GENERATED" line.
2. **The Finding** (lede band, self-contained / screenshot-able): an **auto-composed, conservative,
   verdict-driven finding sentence** (uses "differed / did not differ", **never causal**) keyed off
   the primary metric, as the largest type; an **effect trio** in words (Primary metric · Effect
   [test, p] · Scope [n, groups]); a **trust line** ("1,224 deposits · 91% high-confidence · reviewed
   in the app before export").
3. **Figure 1** — the primary-metric evidence, full width (group boxplot with significance brackets;
   distribution fallback when ungrouped). Takeaway-first caption.
4. **Figure 2** — the other metrics as a compact grid of **ranked** small-multiple group comparisons,
   significant ones first, each verdict-led.
5. **Population overview** (reference, demoted) — the six pooled histograms + descriptive means that
   today masquerade as the opening "Summary".
6. **Appendix A — Methods [NEW]** — detector + RF classifier; ROD Fraction = ROD/(Normal+ROD),
   artifacts = reject class; **confidence = class probability (explains the trust line)**; tests
   (Kruskal-Wallis + Holm-corrected pairwise) + effect sizes. Frames every table below.
7. **Appendix B — Full statistics** — existing per-metric Holm-corrected pairwise + group-stats
   tables, kept verbatim, **numbered to the figures** (A1 ↔ Fig 1) with cross-refs.
8. **Appendix C — Spatial analysis** — demoted from a mid-document peer to a section tagged
   **Exploratory** (honest lower evidentiary weight).
9. **Appendix D — Per-image ledger** — the film-summary Image Summary table as the audit trail, in a
   single collapsible `<details>` (print CSS force-opens it).
10. **Footer** — SCAT version + timestamp + run id (as today).

**Dropped / demoted:** the 8-card pooled Summary as the opener (→ demoted population overview);
6 equal histograms as the opener (→ ranked Figure 2 + demoted overview); spatial as a peer
(→ Exploratory appendix). **Added:** the finding sentence, effect trio, trust line, Figure numbering
+ figure/appendix cross-refs, and the Methods appendix.

## 6. Open items — resolved defaults (v1) and deferred (v2)

- **Total IOD in the worklist:** keep as the right-most table column (v1).
- **Flag signal scope:** low-confidence deposits **only** (the literal, restrained reading) for v1;
  fusing statistical outliers (ROD-fraction / count outliers) is a v2 enrichment.
- **Threshold:** default **0.60**, read from config; the control retunes Review(N) live. The honest
  range band always shows regardless of threshold (so a green verdict can't be gamed).
- **Reviewed state:** v1 — Review(N) simply **decrements as real edits land**. A persisted per-run
  "reviewed ledger" + an "accept / looks right" in-place clear (true burn-down) is **v2** (real
  plumbing; risks rubber-stamp trust-theater — needs its own go/no-go).
- **Keyboard throughput** (j/k/Enter on the worklist): desirable, v1 if cheap, else v2.
- **Headline-metric policy:** the *primary* metric is fixed per run (chosen in chat); the report
  headline verb flips to "did not differ" rather than letting the most-significant metric hijack the
  lede (avoids multiplicity cherry-picking). The real mover still surfaces in ranked Figure 2.
- **Report collapse policy:** only the per-image ledger is `<details>`; everything else stays linear
  and visible for a printable/static file.

## 7. Architecture / data notes

- `ResultsTab.load_results` must receive `deposit_data` (all_deposits) to compute Review(N); wire it
  through `_results_dict_from_output` and the live-run path.
- Add `primary_metric` to the analysis config + the results dict + `run_manifest.json`; thread it into
  the agent (a tool param / inference) and into both renderers. Default `n_total`.
- The finding-sentence composer is a small pure function over the statistical results + primary metric
  (verdict-driven templating; unit-testable; null-result and no-grouping templates required).
- Report figures gain numbering + a primary-metric-driven Figure 1 selection; the Methods appendix is
  static templated text.

## 8. Testing / verification

- Unit: Review(N) aggregation from a synthetic all_deposits; finding-sentence composer across
  grouped / ungrouped / null / single-group cases; primary-metric selection + fallback.
- Offscreen `ResultsTab.grab()` renders (the established harness) for the verdict header, worklist,
  and empty state; Chrome render of a regenerated report for the Findings-Note structure.
- Full suite stays green; the existing `test_chat_widget` / `test_gui_slimdown` contracts are updated
  where the IA changes them (e.g. KPI tiles → composition line).

## 9. Suggested phasing (for the implementation plan)

1. **Primary-metric mechanism** (config + results dict + default + agent hook + both renderers key off
   it). Unblocks the "answer" on both surfaces.
2. **Confidence in the result window** (deposit_data wiring, Review(N) column, threshold, trust line +
   range, drill-in worst-first).
3. **Result-window recomposition** (verdict header, composition strip, chart/stat exile, footer).
4. **Report "Findings Note"** (finding sentence + effect trio + trust line, Figure 1/2, demotions,
   Methods appendix, figure numbering).

Each phase ships green + verified before the next.
