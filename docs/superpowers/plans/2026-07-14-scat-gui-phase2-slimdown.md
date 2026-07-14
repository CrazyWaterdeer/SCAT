# SCAT GUI Phase-2 Slimdown — Implementation Plan (v2, Codex-incorporated)

**Branch:** `feat/ai-agent` · **Scope:** the *slimdown* half of spec §8/§13 phase-2 (the
PySide6 chat dock is **deferred**). **Design source:** spec §8/§13. **Maps:** 7-reader
`map-gui-phase2` workflow (line-precise). **Review:** Codex (gpt-5) read-only pass folded
below — 10 findings, all verified firsthand; the two Criticals (multi-file picker, original-
path lookup) and the UI-crash stats-shape mismatch materially reshaped this v2.

## 0. Goal
Make the GUI call the **canonical `scat/pipeline.py` services** instead of its own copy of
the pipeline, and delete the UI the agent design retires:
1. **Rewire** AnalysisTab Run (`_run_analysis`/`_do_analysis`) + ResultsTab `_generate_report`
   onto the services.
2. **Remove** the drag-drop grouping stack → a **read-only review/correct panel** fed by a
   deterministic (no-LLM) grouping source.
3. **Remove** Quick/Full mode + the "report not generated" nag.
4. **Collapse** duplicate detection-param surfaces, the two stats entry points, and the two
   HTML-report entry points onto one service call each.

**KEEP (spec §8):** TrainingTab, Labeling launch, EDIT_MODE misclassification correction,
labeling shortcuts, worker-count/window config, PathSelector, I/O paths, group plumbing.
`WorkerThread` stays (shared with TrainingTab).

**Invariants:** core `import scat` + `python -m scat.cli analyze` work **without** `[agent]`;
GUI must not import `scat.agent`/`scat.tools`; parity gate `tests/test_pipeline_parity.py`
stays green (calls `analyze_folder_service` with **default** kwargs).

---

## 1. Design decisions (Codex-incorporated)

### D1 — Enrich `analyze_folder_service` to a true superset (parity-preserving defaults)
The service is a subset of the GUI's inline `_do_analysis`; naive swap regresses U-Net,
`sensitive_mode`, progress, worker-count, `save_json`, spatial. All underlying methods
already accept these. Add params, thread them through, **defaults reproduce today's
CLI/parity behavior** (Codex confirmed: parity test diffs only `image_summary.csv`/
`all_deposits.csv`, calls with defaults → unchanged).

### D2 — Deterministic grouping from the GUI's OWN file list (not `scan_folder_service`)
**[Codex F1/F3]** The GUI is a multi-file picker holding full paths in
`self._image_files_for_analysis` (main_gui.py:1273). Build the group map **directly from that
list** — `{Path(f).name: Path(f).parent.name}` — NOT from `scan_folder_service` (which caps
at 500 files, pipeline.py:54, and would also widen input). Run
`grouping_util.duplicate_basenames` on the **full** path list; on collision → all-ungrouped +
inline warning. The panel writes `self._group_data = {label: [basename,…]}` (real groups
only) — the exact shape `_do_analysis:1738-1746` already consumes. Manual correction in the
panel (reassign file → group, rename/create group) fully replaces the drag-drop editor.
**CSV import is deferred** (Codex F4: underspecified) — subfolder + manual correction covers
the use case; a "Load grouping CSV…" override is a follow-up. The panel imports nothing from
`scat.agent`/`scat.tools`.

### D3 — One report entry point; forward spatial_stats, ignore visualization_paths
`report._build_html` **ignores `visualization_paths`** (0 body refs; the HTML uses
self-generated `inline_plots`) — so dropping it is a no-op. **CORRECTION (2nd Codex pass, on
the diff):** `_build_html` **does** render a "🗺️ Spatial Analysis" section from `spatial_stats`
(report.py:1300-1324) — my first-pass grep capped at line 1120 and missed it. So
`generate_report_service` **must** forward spatial stats or the GUI report regresses (loses the
spatial section the old inline path produced). Fix: `generate_report_service` loads the
`spatial_stats.json` sidecar (written by `analyze_folder_service(spatial=True)`) from the
results dir and passes it to `report.generate_report(spatial_stats=…)` — backward-compatible
(CLI writes no sidecar → None → unchanged). Regression-tested (`test_report_forwards_spatial_stats`).

### D4 — One statistics entry point, with the correct UI shape
**[Codex F5 — verified crash]** `run_statistics_service` returns the nested
`{metadata,basic,ph,…}` dict; the Results-tab renderer `_generate_comparison_stats`
(main_gui.py:2241-2278) iterates `.items()` and indexes `result['group1_name']` — feeding it
the nested dict → **KeyError**. So: store `stats_results = stats.get('basic',{}).get(
'metrics',{})` (flat `{metric: comparison}`) for the UI, and pass the **whole** `stats` dict
to `generate_report_service` (it self-flattens `basic.metrics`, pipeline.py:152-153). Drop
`statistics.generate_statistics_report` from the Run path — **but keep the function + its
`scat/__init__.py` export** (public API; only the GUI *call* is removed).

### D5 — Regenerate-after-edit: lightweight JSON→annotation objects
**[Codex F8/F9]** `_generate_report` re-annotates from **edited** `deposits/<stem>.labels.json`.
`analyzer.generate_annotated_image` reads only `.contour/.label/.id/.centroid` (analyzer.py:
260-275), so the helper returns a **lightweight duck-typed object** (NOT a full `Deposit` —
avoids fabricating perimeter/aspect_ratio). labels.json already stores `x`,`y`
(=original centroid), `contour`, `id`, `label` (analyzer.py:382-396) → use saved `(x,y)` as
`.centroid`. **Preserve the `if json_path.exists()` guard** so a `save_json=False` run (no
labels.json) skips annotation instead of crashing.

### D6 — Original-image lookup must handle subfolders
**[Codex F2 — Critical]** The edit/regenerate paths find originals at
`config['last_input_dir']/stem.ext` **non-recursively** (main_gui.py:2322, 2497); subfolder
originals (`ctrl/foo.tif`) are unreachable. Add `_find_original_image(filename)`:
(1) match basename in `self.results.get('image_paths')` (exact full paths, present in-session);
(2) fall back to a **recursive** `rglob` under `last_input_dir` (for disk-loaded results).
The duplicate-basename guard keeps (2) unambiguous. Use it in both 2322 and 2497.

---

## 2. Phase A — Enrich `analyze_folder_service` (parity-gated, mechanical)
**File: `scat/pipeline.py`.** New signature:
```python
def analyze_folder_service(path, groups=None, model_type=None, model_path=None,
        min_area=20, max_area=10000, edge_margin=20, circularity=0.6,
        sensitive_mode=False, unet_model_path=None,
        annotate=True, visualize=False, spatial=False,
        parallel=True, max_workers=0, save_json=True,
        image_paths=None, progress_callback=None, output_dir=None) -> AnalyzeResult:
```
- `images = [Path(p) for p in image_paths] if image_paths is not None else list_images(path)` **[F1]**.
- `DepositDetector(min_area, max_area, edge_margin, sensitive_mode=sensitive_mode, unet_model_path=unet_model_path)`.
- `analyzer.analyze_batch(images, metadata=metadata, progress_callback=progress_callback, parallel=parallel, max_workers=max_workers)`.
- `reporter.save_all(results, metadata, group_by, save_json=save_json)`.
- **Spatial block** (`if spatial:`, after annotate): per-image `SpatialAnalyzer().analyze(res.deposits, img.shape[:2])`; `agg = aggregate_spatial_stats(spatial_results)`; `json.dump(agg, open(out/'spatial_stats.json','w'), default=str)`; `generate_spatial_visualizations(spatial_results, out/'visualizations')` inside a `try/except ImportError`. Lazy-import `spatial`/`visualization` (core modules — packaging-guard clean).
- **NO change to `generate_report_service`** (D3). CLI unaffected (all new params default).

**Parity/behavior defaults (Codex-confirmed):** `sensitive_mode=False`, `unet_model_path=None`,
`parallel=True`, `max_workers=0`, `save_json=True`, `spatial=False`, `image_paths=None`,
`progress_callback=None` → CSVs byte-identical; spatial writes only extra sidecar files.

**Tests:** (a) positive: `analyze_folder_service(synth_dir, spatial=True, visualize=True)`
writes `spatial_stats.json` + `visualizations/`, CSVs unchanged vs default run; (b)
`image_paths` subset: passing 1 of N synth files analyzes exactly that one; (c) full suite
green (baseline 49).

---

## 3. Phase B — GUI removals (delete bottom-up; final re-grep must be 0 hits)
**File: `scat/main_gui.py`.**
- **B1 drag-drop:** delete `GroupWidget` (719-885), `DroppableContainer` (886-916),
  `GroupEditorDialog` (917-1260); `create_groups_btn` create/connect/add (1345-1348);
  `create_groups_btn.setEnabled` (1563, KEEP `groups_tree.setEnabled` 1564). No external refs
  (grep-confirmed).
- **B2 Quick/Full:** `mode_group` block (1376-1409); `self.options_group` (1452);
  `_on_mode_changed(True)` init (1455-1456); `_on_mode_changed` (1534-1560); comment (1425);
  `is_quick_mode` (1832-1833,1844); `_report_pending` init (1893) / set (1989-1990) / summary
  nag (1997-2004 incl `{mode_text}` at 2004) / reset (2587-2588) / loaded-key (2659) /
  `_reload_results` set (2430-2431); **closeEvent nag (2810-2824)** → `self._save_window_state();
  event.accept()`. Update `generate_report_btn` tooltip (1939).
- **B3 SettingsDialog Detection tab:** delete build (229-249) + saves (275-278). AnalysisTab
  spinboxes (1458-1485) are the sole live source. (Grep: no other reader of `*_spin`.)

---

## 4. Phase C — Rewire onto services
**File: `scat/main_gui.py`.**

**C1 grouping panel** (replaces `_open_group_editor` 1628-1665): a read-only/correct widget in
the "Groups" QGroupBox. On input change (`_browse_input` 1607-1610 already resets
`_group_data`) → build `{basename: parent.name}` from `self._image_files_for_analysis`;
`duplicate_basenames(full_paths)` non-empty → all-ungrouped + warning label; render via
`_update_groups_list` (KEEP). Correction: reassign a file's group / rename / create group →
write `self._group_data = {label:[basename,…]}` (real groups only, basenames). **[F7]** files
left unassigned get no row → NaN on merge → excluded from stats (current behavior; documented).

**C2 Run flow** (`_run_analysis` 1667-1700 keeps UI bookkeeping; `_do_analysis` 1702-1845
replaced by a service closure):
1. `groups = invert(_group_data)` → `{basename: label}` or `None` when unchecked/empty.
2. `res = analyze_folder_service(path=Path(files[0]).parent, image_paths=self._image_files_for_analysis,
   groups=groups, model_type=…, model_path=self.model_path.path() or None, min_area=…, max_area=…,
   circularity=self.threshold.value(), sensitive_mode=True,
   unet_model_path=self.detection_model_path.path() or None, annotate=self.annotate.isChecked(),
   visualize=self.visualize.isChecked(), spatial=self.spatial.isChecked(),
   parallel=config.get('performance.parallel_enabled',True), max_workers=worker_count,
   save_json=self.save_json.isChecked(),
   progress_callback=lambda c,t: self.worker.progress.emit(c,t), output_dir=str(output_path))`
   **[F1: explicit `image_paths`; no `self.input_dir.path()` — that symbol doesn't exist]**.
3. `stats = run_statistics_service(res.output_dir) if (self.stats.isChecked() and len(res.groups)>=2) else None` (mirror cli.py:38-43).
4. `generate_report_service(res.output_dir, statistical_results=stats, group_by='group') if self.report.isChecked()`.
5. Rebuild the Results-tab dict from `res.output_dir` via a shared helper
   `_results_dict_from_output(output_dir, group_by, image_paths, stats)`: read `image_summary.csv`,
   `all_deposits.csv`, glob `visualizations/`, read `spatial_stats.json` (→ `spatial_stats`),
   set `stats_results = (stats or {}).get('basic',{}).get('metrics',{})` **[F5]**, carry
   `image_paths` **[F2/D6]**; NO `is_quick_mode`. Reuse this helper in `_load_previous_results`
   (2635-2660), which passes `image_paths=None`, `stats=None`, `group_by=None` (auto-detect).
- **Errors:** service raises `ValueError` on empty list / duplicate-basename grouping → already
  surfaced by the worker `error`→`_on_error` path; verify a dialog, not a crash.

**C3 `_generate_report`** (2447-2602): keep CSV reload; replace inline cv2 loop (2484-2544)
with `analyzer.generate_annotated_image(image, deposits_from_labels_json(json_path),
show_labels=True, skip_artifacts=True)` guarded by `if json_path.exists()` **[F8/F9]**; find
originals via `_find_original_image` **[F2/D6]**; stats → `run_statistics_service(str(output_dir),
group_col=self.results.get('group_by') or 'group')`; HTML → `generate_report_service(
str(output_dir), statistical_results=stats_results, group_by=self.results.get('group_by'))`
(pass the **whole** stats dict). Keep success-branch `self.results` updates; drop the
`is_quick_mode`/`_report_pending` resets. Update the Results-tab `stats_results` used for
re-render to `stats.get('basic',{}).get('metrics',{})` **[F5]**.

**C4:** `_do_analysis`'s inline stats (1804-1814) + report (1820-1830) die with the replacement;
confirm no other `generate_statistics_report` caller remains in main_gui.

**C5 edit double-click** (2308-2411, esp. 2322): route original lookup through
`_find_original_image` **[F2/D6]** so subfolder originals resolve.

---

## 5. Phase D — Helpers
- **`scat/analyzer.py`** (or `scat/deposit_io.py`): `deposits_from_labels_json(json_path) ->
  list` returning lightweight `types.SimpleNamespace(id=…, label=…, contour=np.array(c).reshape(-1,2),
  centroid=(x,y))` per deposit **[F8]** — annotate-only, prefers saved `x/y`.
- **`scat/main_gui.py`**: `_find_original_image(filename)` **[F2/D6]** (in-session `image_paths`
  match → recursive `rglob` fallback).

---

## 6. Verification (heavy, runtime)
1. `QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q` → all green (49 + new).
2. **Parity gate** explicitly green.
3. **Packaging guards:** existing core guard + **new** `test_gui_no_agent_import` **[F10]** —
   static assert `scat/main_gui.py` source has no `scat.agent`/`scat.tools` import.
4. `import scat` + `QT_QPA_PLATFORM=offscreen python -c "import scat.main_gui"` OK.
5. **Runtime GUI smoke (offscreen, driven):** synth blue-blobs in `ctrl/` + `treated/`
   subfolders; build `MainWindow`; set input to the 4 files → assert panel auto-grouped
   `{ctrl,treated}` from parent dirs → call the Run closure synchronously → assert
   `output_dir` has `image_summary.csv`, `annotated/`, `visualizations/`, `report.html`,
   `spatial_stats.json` → `load_results` renders stats WITHOUT KeyError (F5) → `_generate_report`
   re-runs (no cv2 block), rewrites `report.html`, and `_find_original_image` resolves a
   subfolder original.
6. `.venv/bin/python -m scat.cli analyze <synth> --stats --report` still works (service
   enrichment didn't break the CLI).
7. Re-grep for every removed symbol → 0 hits.

## 7. Risks
- **R1 dangling reads** (Quick/Full) → remove reads in lockstep; final re-grep = 0.
- **R2 stats-shape** (F5) → store flat `basic.metrics` for UI, whole dict for report; smoke test asserts no KeyError.
- **R3 subset input** (F1) → explicit `image_paths`; parity default `None`.
- **R4 subfolder originals** (F2) → `_find_original_image` in-session paths + recursive fallback.
- **R5 duplicate basenames** → `duplicate_basenames` in the panel → ungrouped + warn; service also raises.
- **R6 spatial_stats.json serialization** → `json.dump(default=str)` + load test.

## 8. Sequencing
A (services+tests, parity-safe) → B (removals) → C (rewire) + D (helpers) → verify.
Commit as reviewable chunks; do NOT merge to `main` (user reviews).
