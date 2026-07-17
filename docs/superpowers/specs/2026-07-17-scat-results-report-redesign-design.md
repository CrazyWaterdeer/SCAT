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

## 2.1 Statistical honesty contract (Phase 0 — precedes all UI work)

A codex review flagged that the trust/confidence language could make **classifier-label uncertainty
look like scientific certainty**. These constraints are binding and must be settled *before* the UI
copy hardens (they are the first implementation phase):

- **Confidence is classifier-label sensitivity, not correctness.** RF `predict_proba` is uncalibrated,
  so never call it "high-confidence / reliable". Use "**above the classification threshold**" and, in
  the report, state that calibration is not validated. Confidence covers **classification only** — the
  detector emits no detection-confidence, so it says nothing about missed/merged/split deposits; the
  default metric (count) is *detection*-driven, so the trust line must not read as certifying the count.
- **The "range band" is a sensitivity bound to low-confidence classifier labels only** — not a bound on
  the true biological metric (it ignores detection FN, segmentation, lighting, drift, high-confidence
  wrong calls). Label it exactly that; never phrase it as total measurement uncertainty or a CI.
- **Threshold provenance.** The report / trust state uses a **fixed threshold from config + the
  manifest**; the result-window slider **filters only** and never changes the recorded trust state.
  Persist the threshold used; mark "changed from default" if a non-default was recorded.
- **No unearned "reviewed" claim.** v1 has no reviewed ledger, so nothing may say "reviewed in the app".
  The trust line states only **factual counts** ("N of M deposits above threshold 0.60"). A real
  "reviewed" claim requires the v2 audit trail (who/what/when/accepted-or-changed).
- **Experimental unit + no pseudoreplication.** *Confirmed in code*: `_compare_metric_between_groups`
  (`scat/statistics/common.py`) groups `film_summary` (one row per image) and tests **per-image**
  values, so group `n` = number of images — the unit is the image, not the deposit. The Methods must
  state this explicitly (it is currently implicit).
- **Primary metric is a predeclared endpoint** — confirmed in chat *before* analysis, persisted in the
  manifest, and shown in Methods. Never silently inferred (that invites post-hoc endpoint selection).
- **Wording guardrails.** Never "did not differ" (implies equivalence) → "**no statistically detected
  difference under [test]**". Never causal/biological-strength wording. The lede uses **only** the
  predeclared primary metric — never the largest/most-significant result. Single-group / ungrouped runs
  are **descriptive, not inferential**. Secondary metrics (report Figure 2) are labeled **exploratory**.
- **Metric-specific flip rules.** Reclassifying a flagged deposit to artifact vs normal vs ROD changes
  each metric's denominator differently; the per-metric sensitivity rule is defined in Phase 0 before
  any range-band copy is written.

## 3. Shared mechanism — Primary metric + confidence

### 3.1 Primary metric
- A `primary_metric` selection flows into the results (config + the results dict). Candidates are the
  per-image metrics SCAT already computes: `n_total` (Total deposits — **default**), `rod_fraction`,
  `mean_area`, `mean_hue` (pH indicator), `total_iod`, `mean_circularity`.
- **Selected in the conversational agent as a predeclared endpoint**: the chat agent captures "what is
  this experiment measuring?" and sets `primary_metric`, but must **surface it for explicit
  confirmation before running** (not silently infer) and **persist it in `run_manifest.json`**. For
  non-agent runs it **falls back to the default** (`n_total`); a lightweight in-GUI switch is a *v2*
  nicety, not v1.
- **Normalization (chosen: support it).** A count-based metric is sensitive to image area, assay
  duration, and fly count, so the primary metric carries a **normalization mode** —
  `per_image` (default), `per_fly`, `per_area`, `per_time`. `per_image` needs no extra metadata and is
  always available; the others require run metadata (fly count / ROI area / duration) and degrade
  gracefully to `per_image` with an explicit note when that metadata is absent. The chosen
  normalization is persisted in the manifest and shown in Methods. Count is **displayed as a rate**
  (e.g. "40.8 deposits / image"), never a bare pooled total, with N images explicit.
- The result-window hero, the report lede/finding sentence, and report **Figure 1** all key off the
  predeclared `primary_metric` (+ its normalization). "Deposits" everywhere = Normal + ROD (artifacts
  are the reject class), consistent with the shipped `9400e0f` decision.

### 3.2 Confidence
- `all_deposits.csv` already carries per-deposit `label` + `confidence` (RF class probability).
- **Review(N)** for an image = count of that image's deposits with `confidence < threshold`
  (default **0.60**, from config/manifest). Because count scales with deposits/image, the cell also
  carries the **flagged fraction** (and, on hover, the flagged deposits' potential **impact on the
  primary metric**) so one pivotal low-confidence call isn't lost among many harmless ones. Aggregated
  by joining `all_deposits` (grouped by filename) onto the film-summary-driven table; `deposit_data`
  must be loaded into `ResultsTab` (today the table is film_summary-only).
- **Run trust state** = derived from the flagged population at the **fixed manifest threshold** (the UI
  slider filters the worklist only and never moves this state): e.g. *Above threshold* (0 flagged) /
  *Review recommended* (some) / *Many low-confidence* (many). Drives the header traffic-light, worded
  as classifier-label sensitivity, not reliability.
- **"% above the classification threshold"** (trust line) = share of deposits with
  `confidence ≥ threshold`. **Not** called "high-confidence" (uncalibrated; see §2.1).
- **Sensitivity range band** = the primary metric recomputed at the extremes of reclassifying the
  flagged (low-confidence) deposits, per the **metric-specific flip rule** (§2.1). Labeled
  "**sensitivity to low-confidence classifier labels only**" — a sensitivity bound, **not** a CI and
  **not** total measurement uncertainty. Computed at the fixed threshold so it can't be gamed.

## 4. Result window — "Verdict + Worklist"

A triage surface, not a dashboard: **verdict → the images that need you → fix.** Everything
report-grade moves to the report. Top-to-bottom:

1. **Verdict header.** LEFT: the **primary-metric hero** (e.g. "Total deposits · 40.8 / image", or
   "Mean ROD Fraction 8.3%" when ROD is chosen) shown as a **rate** at display size; a **traffic-light
   trust line** ("● Review recommended — 6 of 30 images have low-confidence classifier calls"); a
   **sensitivity range band** labeled "sensitivity to low-confidence labels only" ("6.1%–10.8% if the
   flagged calls were reclassified") — never worded as a CI; a subtitle folding in ±SD · N images ·
   N deposits. RIGHT:
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
**report-grade** visualization gallery, descriptive-stats table, group-comparison text, and spatial
text (→ the report). The working view keeps no report charts, but QC visuals stay — the **annotated
overlay is in the drill-in** (its existing role), so systematic image-quality problems are still
visible (codex #10); a dedicated run-level QC cue is a later option. "Rebuild" as a co-equal top
button (→ footer + edits-pending cue); ROD Fraction as the fixed hero (→ configurable primary metric).

**Added:** the primary-metric hero; the trust traffic-light + honest range band; the Review(N)
column + reason hint; the All/Needs-review toggle + threshold control; worst-confidence-first +
filter in the drill-in; the report pointer.

## 5. Report — "Findings Note" (research letter)

Lead with the finding for someone who wasn't present; rigor beneath via progressive disclosure.

1. **Masthead** — keep the pH-gradient hairline signature; a provenance dateline (run · N images ·
   N deposits · grouped-by · N groups · date) replaces the bare "GENERATED" line.
2. **The Finding** (lede band, self-contained / screenshot-able): an **auto-composed, conservative,
   verdict-driven finding sentence** keyed off the predeclared primary metric, as the largest type.
   Wording per §2.1: "differed" / "**no statistically detected difference under [test]**" (never "did
   not differ", never causal); includes effect size, sample unit, N, and correction status; descriptive
   (not inferential) for single-group/ungrouped runs. An **effect trio** in words (Primary metric ·
   Effect [test, p, effect size] · Scope [n images, groups]); a **factual trust line** —
   "1,224 deposits · 91% above the classification threshold (0.60)" — **no** "high-confidence" and
   **no** "reviewed" claim in v1 (§2.1).
3. **Figure 1** — the primary-metric evidence, full width (group boxplot with significance brackets;
   distribution fallback when ungrouped). Takeaway-first caption.
4. **Figure 2 — Secondary metrics (exploratory)** — the other metrics as a compact grid of
   small-multiple group comparisons, verdict-led. Explicitly labeled **exploratory** so that ordering
   the panels by significance can't read as protected endpoints (codex #12).
5. **Population overview** (reference, demoted) — the six pooled histograms + descriptive means that
   today masquerade as the opening "Summary".
6. **Appendix A — Methods [NEW]** — detector + RF classifier; ROD Fraction = ROD/(Normal+ROD),
   artifacts = reject class; the **predeclared primary metric + normalization mode**; the
   **experimental unit** (tests run on **image-level aggregates**, not deposits — no pseudoreplication);
   **confidence = uncalibrated RF class probability** (explains the trust line; states calibration is
   not validated and covers classification only, not detection); the **classification threshold used**;
   tests (Kruskal-Wallis + Holm-corrected pairwise) + effect sizes. Frames every table below.
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
- **Threshold:** default **0.60**. The **trust state + sensitivity band + report use the fixed
  manifest threshold**; the result-window slider **filters the worklist only** (never moves the trust
  state), so a green verdict can't be gamed (§2.1). A non-default recorded threshold is marked
  "changed from default".
- **Normalization:** resolved — **support it** (`per_image` default; `per_fly`/`per_area`/`per_time`
  where metadata exists, else graceful `per_image` with a note). See §3.1.
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

0. **Statistical honesty contract (§2.1) — precedes all UI.** Settle, in code + docs: metric formulas
   + normalization modes; the experimental unit (verify image-level aggregation); confidence calibration
   status; threshold provenance (manifest-fixed vs UI-filter); metric-specific flip rules for the
   sensitivity band; the exact report/UI wording; and what the (v1) audit trail does/doesn't claim.
   Everything downstream renders these decisions, so ambiguity here hardens into false claims.
1. **Primary-metric mechanism** (config + manifest + results dict + default + agent *confirmation* hook
   + normalization + both renderers key off it). Unblocks the "answer" on both surfaces.
2. **Confidence in the result window** (deposit_data wiring, Review(N) + flagged fraction/impact,
   filter-only threshold, trust line + sensitivity band, drill-in worst-first).
3. **Result-window recomposition** (verdict header, composition strip, report-grade chart exile, footer).
4. **Report "Findings Note"** (finding sentence + effect trio + factual trust line, Figure 1 / Figure 2
   [exploratory], demotions, Methods appendix, figure numbering).

Each phase ships green + verified before the next.
