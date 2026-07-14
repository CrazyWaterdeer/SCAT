# T3.1 — Durable context ledger + resume + result discovery

**Roadmap:** T3.1 (Tier 3, was L · high — Codex-informed rescope to ~M). **Spec refs:** §4.1 / §7 / §13 "Later".
**Branch:** `feat/t3.1-context-ledger`. **Builds on:** T2.1 (`run_manifest.json`).

## Problem

Both runners get a **static** `SYSTEM_PROMPT` at construction (`backend.py:36,45`) and nothing ever
tells the agent "what has already been analyzed." The prompt even ends with *"Treat any injected
session/progress context as authoritative — do not re-analyze images already done"* — a promise no
code keeps. Consequences:

1. **Re-analysis.** "Analyze /data/exp1"; later (new `scat chat`, or the same chat after compaction)
   "finish exp1" → the agent re-runs every image and silently writes a *second* results dir.
2. **No resume / progress memory** that survives a restart or compaction, though the durable record
   already exists on disk (T2.1's `run_manifest.json` + `image_summary.csv`).
3. **Prior results are undiscoverable to the agent** — "run stats on yesterday's exp1 results" forces
   the user to paste a timestamped `results_YYYYmmdd_HHMMSS/` path by hand.

## Ground facts (verified in code)

- **`image_summary.csv`** has one row per analyzed image, column `filename`, and that value is the
  image **basename** (`analyzer.py:169` `filename=image_path.name`; failure path `:228` likewise). It
  is the authoritative per-image record — but **basename-only**: there is no relpath/abspath anywhere,
  so resume matching is *inherently* basename-level.
- **`run_manifest.json`** (written **last**, `pipeline.py:179`, after the CSVs at `:131`): `dataset.path`,
  `dataset.sha256` (order-independent hash over sorted `relpath:size` — a **whole-dataset** fingerprint),
  `dataset.{n_images,sample[:10]}`, `model`, `grouping.{column,mapping}`, `detection`, `warnings`,
  `created_at`, `scat_version`, `git_commit`. Because it is written last, **manifest present ⇒ the run
  finished and the CSV exists**; a CSV with no manifest ⇒ a crashed/in-progress run.
- Results dirs: `get_timestamped_output_dir(Path(path).parent, "results")` →
  `results_<YYYYmmdd_HHMMSS>/` **as a sibling of the analyzed folder** (parent of the file for a single
  file).
- The agent recipe already calls `scan_folder(path)` **first, every turn** (`prompts.py:9`).
- `analyze_folder_service` **already accepts `image_paths`** and **refuses grouping on duplicate
  basenames** (`pipeline.py:88,92`). `scan_folder_service`, `run_statistics_service`,
  `generate_report_service` are all core (no agent deps) — the packaging guard applies to any new core
  module too.

## Design — discovery + tool-pull, not per-turn injection

The whole feature is a **core** on-disk index (`scat/results_index.py`) surfaced two ways the agent
already reaches for: enriched `scan_folder` output (the recipe's first call) and an explicit
`list_analyses` tool it can pull any turn. **No runner changes, no per-turn system/user-text injection,
no backend divergence** — that was the original plan's over-build (see "Codex review — incorporated",
points 4/5). The manifest+CSV on disk *are* the durable ledger; we read them back.

### `scat/results_index.py` (NEW, core — json + pandas + pathlib only, no agent/pydantic/anthropic)

- `find_analyses(search_roots) -> list[AnalysisRecord]` — for each root (a folder we might have written
  results beside), glob its **direct children** `results_*/run_manifest.json` (and accept a root that is
  itself a results dir). For each: parse the manifest, read the sibling `image_summary.csv` for the exact
  basename set. Record `{results_dir, created_at, dataset_path (resolved), dataset_sha256, n_images,
  analyzed_basenames:frozenset, groups, model, detection, warnings, status}` where `status` ∈
  {`complete`, `partial` (CSV but no/*unreadable* manifest), `unreadable`}. **Best-effort**: never
  raises; a dir that can't be read becomes a `skipped` entry with a reason (surfaced, not silently
  dropped — Codex trap). Deterministic order: parse `created_at`; sort by (parsed_ts or epoch-0, then
  `results_dir`) so missing/dup timestamps don't make ordering nondeterministic.
- `analysis_status(folder, *, image_paths=None, search_roots=None) -> dict` — the analysed-vs-pending
  delta for one target folder, computed in **two tiers of confidence**:
  1. **Fingerprint-verified complete (strong).** Compute `manifest.dataset_fingerprint(current_images)`;
     if any discovered run covering `folder` has `dataset_sha256 == that fingerprint`, the folder was
     analyzed **exactly** by that run → `{status:"complete", n_pending:0, verified:true,
     results_dir:…}`. Immune to basename ambiguity (whole-dataset hash). This is the headline resume
     case: reuse that dir for stats/report, re-analyze nothing.
  2. **Per-image delta (basename, guarded).** Otherwise, if the **current** image set has **unique
     basenames**, classify each current image analyzed/pending by basename membership across covering
     runs → `{status:"partial"|"none", n_current, n_analyzed, n_pending, pending:[…capped],
     latest_results_dir, verified:false}`. If the current set has **duplicate basenames**, refuse a
     numeric split: `{status:"ambiguous", reason:"duplicate basenames — cannot map results by basename",
     runs:[…]}` and let the agent handle it (never emit unsafe counts — Codex 1).
  - "Covers `folder`" is **exact resolved-path equality**: `Path(dataset_path).resolve() ==
    Path(folder).resolve()` (or single-file parent). No descendant/prefix matching in v1 — it invites
    false positives on nested experiments (Codex 2). Stale-same-name folders are handled by tier 1
    (fingerprint mismatch ⇒ not "complete") and tier 2's `verified:false` labeling.
- `format_ledger(records|status, *, max_chars=1500) -> str` — compact, deterministic, **hard char-cap**
  (not just item counts — Codex trap) plain text. Manifests carry no secrets (T2.1 F3), so nothing to
  redact.

### Slice 1 — `scan_folder` reports resume status (turn-1, both backends, zero runner changes)

`scan_folder_service(path)` gains an **additive** key `already_analyzed =
results_index.analysis_status(path)` (best-effort; omit the key on any error — never break scan; parity
gate untouched, no CSV change). Because the recipe scans first, the agent sees resume state on the very
first turn of a *fresh* process — no session state needed. Update `prompts.py` step 1: read
`already_analyzed`; if `status=="complete"`, **reuse `results_dir`** for stats/report and re-analyze
nothing; if `partial`, offer pending-only (see Slice 3) and be explicit it's a separate dir; if
`ambiguous`, tell the user about the basename clash.

### Slice 2 — `list_analyses` tool + `get_context_ledger` (pull the ledger any turn)

A `list_analyses(folder=None) -> list` **@tool** over `find_analyses` (default roots = cwd + any folder
the agent has scanned; explicit `folder` scans beside it). Returns each run's
`{results_dir, dataset_path, created_at, n_images, groups, status}` so the agent can answer "run stats
on yesterday's exp1 results" without a hand-pasted path, and can **re-pull the ledger after a
compaction** instead of relying on always-injected state (Codex 4's "explicit tool the agent can call
every turn"). Prompt guidance: when resuming/continuing prior work or unsure what's done, call
`list_analyses` / re-`scan_folder`. This replaces the dropped per-turn-injection design.

### Slice 3 (last, cuttable) — correct partial resume via strict `combine_results`

Pending-only analysis writes its own dir, so whole-experiment stats over "the original + the new"
images would span two dirs. Rather than "just re-run everything" (which defeats resume — Codex 3) or
silently reporting a partial dir as whole, add a **guarded** `combine_results_service(dirs, output_dir)`
(core) that concatenates `image_summary.csv` + `all_deposits.csv` from **compatible** runs into one new
dir + a merge manifest, then stats/report run on the merged dir normally. **Compatibility is enforced,
not assumed:** identical `model` + `detection` params, identical `grouping.column`, consistent
per-basename group label, and **no basename collisions across dirs** (last-writer-wins would corrupt the
join); on any mismatch it **refuses with a specific reason**. Exposed as a `combine_results` tool.
Prompt: for "analyze the N new images and give me combined stats", analyze pending-only → `combine_results`
→ stats/report. If Slice 3 slips, Slices 1–2 still ship a correct feature (complete-run reuse + discovery
+ honest partial handling); this slice is explicitly droppable.

## Verification

- `tests/test_results_index.py`: `find_analyses` discovers a written dir and reads the exact basename
  set; a malformed manifest → `skipped` with reason (no raise); a CSV-without-manifest → `partial`;
  deterministic order with missing/dup `created_at`. `analysis_status`: (a) fingerprint match on the
  same folder → `complete, verified, n_pending=0`; (b) a strict-subset prior run + unique basenames →
  correct `partial` split; (c) duplicate current basenames → `ambiguous`, no counts; (d) no prior
  results → `none`, all pending; (e) explicit `image_paths` honored; (f) resolved-path equality (a
  `./exp1` vs absolute `/…/exp1` still matches; a *different* same-named folder does **not**).
- `tests/test_pipeline.py` (extend): `scan_folder_service` includes a correct `already_analyzed` after a
  real `analyze_folder_service` run on `synth_dir`; `complete` after a full run; key omitted/empty and
  never raising when no results dirs exist. **Parity gate green** (no CSV output change).
- `tests/test_combine.py` (Slice 3): compatible dirs merge (row counts add, manifest records sources);
  mismatched model/params/grouping or colliding basenames → refused with reason; stats run on the merged
  dir.
- `tests/test_cli.py`/`test_chat_widget.py` (extend): `list_analyses` tool round-trips through the tool
  registry; the tool loop handles it.
- Packaging guard (`test_core_imports_without_agent.py`): `import scat.results_index` (+ `scat.combine`
  if added) works **without** the `[agent]` extra. Full suite green.

## Codex review — incorporated

Codex (gpt-5.5, xhigh) reviewed the first draft; folded in:
- **1 (basename delta unsafe):** verified `image_summary.csv.filename` is basename-only, so relpath
  matching is impossible. Added the **two-tier** model — fingerprint-verified `complete` (strong,
  ambiguity-immune) first, guarded basename delta only when **current basenames are unique**, and an
  explicit **`ambiguous`** status (no numeric counts) when they collide. No unsafe counts ever emitted.
- **2 (discovery misses/stale):** canonicalized to **`Path.resolve()`** equality for "covers", dropped
  descendant matching, label unverified basename matches `verified:false`, and use the manifest
  `dataset.sha256` as the strong guard against a stale same-named folder.
- **4 & 5 (subscription preamble weak / Slice 2 over-built):** **dropped per-turn injection entirely.**
  Primary durable mechanism is the enriched `scan_folder` (recipe calls it first) + an explicit
  `list_analyses` tool the agent pulls when resuming/after compaction. Removes the backend divergence,
  stale-preamble accumulation, and token growth; shrinks scope to ~M.
- **3 (pending-only fragmentation):** complete-run reuse is the correct headline; for partial work,
  added the strict, compatibility-checked **`combine_results`** (Slice 3) so whole-experiment stats are
  *correct across runs* instead of "just re-run everything" — while never presenting a partial dir as
  whole.
- **Traps:** surface `skipped` dirs with reasons; timestamp-safe deterministic ordering; manifest-last
  ⇒ presence implies a complete run (CSV-without-manifest = partial); char-hard-capped ledger; verify
  the `analyze_folder` **tool** forwards `image_paths` and still refuses dup basenames.

## Risks

- **R1 — moved/archived results.** Sibling scan misses relocated dirs. Honor `agent.results_search_roots`
  (config, default = input folder's parent) so a user can point at an archive.
- **R2 — stale same-named folder.** Tier-1 fingerprint mismatch prevents a false `complete`; tier-2
  labels basename matches `verified:false`. No silent false positives.
- **R3 — combine correctness.** `combine_results` refuses incompatible runs rather than producing a
  wrong merge; the failure mode is "won't combine, told you why", never a corrupt joined dataset.
- **R4 — ledger token cost.** `format_ledger` char-caps; `list_analyses` returns bounded rows.

## Out of scope (v1)

Cross-machine results registry; zip/export of a bundle (metadata-complete dirs + discovery already give
"self-contained"); descendant/nested-folder auto-aggregation; parsing user free-text to pre-seed the
ledger (scan-first covers turn-1 resume).
