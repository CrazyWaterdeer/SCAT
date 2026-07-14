# SCAT Roadmap

Status after the **AI-agent layer + phase-2 GUI** milestone merged to `main` (2026-07-14, PR #1).
Compiled from a 4-axis codebase survey (roadmap / code-debt / agent-followups / test-coverage).
Effort = S/M/L; Value = high/medium/low. Marker scan (`TODO/FIXME/XXX/HACK`) is clean — 0 hits.

Work through the tiers top-down. Items link to the spec (`docs/superpowers/specs/2026-07-14-scat-ai-agent-design.md`)
and the two phase-2 plans in `docs/superpowers/plans/`.

---

## 🔴 Tier 1 — Harden the shipped agent/GUI feature ✅ DONE (branch feat/hardening)

High-leverage: makes the just-merged agent + chat dock reach users reliably.

- [x] **T1.1 — Chat dock: per-image progress + cooperative cancel** · M · high
  A real folder analysis shows one `analyze_folder(...)` line then looks frozen with no feedback until
  the whole batch finishes, and **Stop does not halt the batch** (cancel is only checked at tool/loop
  boundaries, never inside `analyze_batch`). Plumbing is half-built: `analyze_folder_service` and
  `analyze_batch` already accept `progress_callback`, but the `@tool` wrapper drops it. Wire spec §10.1:
  a `report_progress` contextvar (rendered as "142/500" in the dock/CLI) + a `raise_if_cancelled`
  per-image token so a chat "Stop" halts at the next image; optional wall-clock/large-batch heads-up.
  *Where:* `tools/pipeline_tools.py:8-14`, `pipeline.py:71-112`, `analyzer.py:194-233`, `runner.py:95,136`,
  `chat_widget.py`, `cli.py` chat loop. Spec §10.1.

- [x] **T1.2 — Test coverage for the new surfaces + smoke guards** · S–M · high
  `cli.py` (incl. the new `chat` subcommand) has **zero tests**; `main_gui`/`labeling_gui`(1990 lines)/
  `ui_common` are not even in `test_smoke.MODULES`. Add `tests/test_cli.py` (drive `main(["analyze", …])`
  + arg-parser dispatch), extend `test_smoke.MODULES` (`scat.cli`, `scat.main_gui`, `scat.ui_common`,
  `scat.labeling_gui`), and add driven tests for the just-merged GUI edges: grouping-panel rename/merge
  (`main_gui.py:_on_groups_context_menu`), duplicate-basenames warn branch (`_autogroup_by_subfolder`),
  and the `_run_analysis`→WorkerThread→`_on_finished` signal bridge (only `_do_analysis` is driven today).
  *Where:* `cli.py`, `test_smoke.py:11-24`, `test_gui_slimdown.py`.

- [x] **T1.3 — Packaging decision + README docs for the agent layer** · M · high
  **Decision (user):** do NOT bundle the agent stack into a frozen exe — the subscription backend
  needs the external `claude` CLI, which PyInstaller can't bundle. Instead ship the Assistant via
  the source install and provide a **one-click Windows desktop shortcut** (`scripts/install_desktop_shortcut.sh`)
  that launches the WSL GUI via WSLg (no console). `SCAT.spec` left as a core-only exe.
  README now documents `pip install -e .[agent]`, `scat chat`, the Assistant dock + backends, and
  the desktop shortcut.

---

## 🟡 Tier 2 — Scientific reproducibility

- [x] **T2.1 — Provenance run-header enrichment** · M · high — DONE (feat/tier2): `scat/manifest.py`
  writes `run_manifest.json` (dataset fingerprint + version/commit + model+hash + grouping + params)
  into every results dir; `provenance.start_session` writes a redacted session header.
  `provenance.py` logs per-tool calls only; `read_session()` is never read back. Add a run header:
  dataset fingerprint (folder + file count + hash of sorted basenames/sizes), SCAT version + git commit,
  config snapshot, model id + classifier-model path/hash, and the grouping mapping. Spec §7 calls this
  the basis for a future "methods" section — and it's the natural first slice/manifest of T3.1 (bundles).
  *Where:* `provenance.py:11-18,80`, spec §7.

- [x] **T2.2 — statistics.py: unify CV + dedup the 5 `compare_*_between_groups`** · M · medium — DONE
  (feat/tier2). Canonical module-level `coefficient_of_variation` (n<2→NaN, NaN-skip, abs) replaces
  17 inline `std/mean*100` sites; the old inline formula reported a misleading `0` for n=1 groups.
  CV is descriptive (not inferential) so significance tests are unaffected; no prior data to preserve
  (user's call). Shared `compare_group_values` dispatch helper dedups all 5 compare_* methods.

---

## 🟢 Tier 3 — Larger features / lower urgency

- [ ] **T3.1 — Durable context ledger + resume + self-contained result bundles** · L · high
  spec §4.1/§7/§13 "Later". A `scat/agent/context.py` that rebuilds an "analysed vs pending" summary from
  on-disk results each turn and injects it into the system prompt → free resume / "don't re-analyze" /
  "7/12 done" that survives compaction. T2.1 is its first slice. (No `scat/agent/context.py` exists.)
- [ ] **T3.2 — Runner message compaction + one context-limit force-compact-and-retry** · M · medium (spec §7/§10; only tool *results* are compacted today, runner stops at the context limit).
- [ ] **T3.3 — Exponential backoff for 429/overloaded** · S · medium (spec §10; verify vs SDK default retries first).
- [ ] **T3.4 — Effort/thinking selector in the chat dock** · S · medium (API provider hardcodes `max_tokens=4096`, passes no thinking/effort; add a combo).
- [ ] **T3.5 — Live model catalog** (`client.models.list()`) vs hardcoded `LATEST_MODELS` · S · medium.
- [ ] **T3.6 — Ollama / OpenAI-compatible backends** behind `build_runner` · L · low (Provider Protocol is the clean seam).
- [ ] **T3.7 — GUI "Load grouping CSV…" override** · S · low (CLI already has `--metadata`; subfolder+manual already cover the case).

### Small cleanups (each S)
- [ ] Dedup the 4 identical deposit-from-contour blocks in `labeling_gui.py` (1120/1174/1237/1302).
- [ ] Resolve `visualization.effect_size_forest_plot` dead code (no caller passes `statistical_results`) — wire up or remove.
- [ ] Live subscription-bridge round-trip is CI-skipped (login-gated) — consider a recorded cassette test.
- [ ] Dedicated unit tests for `spatial.py` (Clark-Evans/NND/quadrant + edge guards), `trainer.py` round-trip, `segmentation.py`.

---

## ⛔ Do NOT touch without a model retrain
- **`features.mean_hue`** uses a plain (non-circular) mean of a circular hue channel (`features.py:47`).
  It feeds the RF classifier + pH estimation, and `models/model_rf.pkl` was **trained on this value** —
  changing the feature math silently shifts classifications. Retrain the model before fixing.
