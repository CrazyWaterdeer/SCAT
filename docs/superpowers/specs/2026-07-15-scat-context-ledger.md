# T3.1 — Durable context ledger + resume + self-contained result bundles

**Roadmap:** T3.1 (Tier 3, L · high). **Spec refs:** §4.1 / §7 / §13 "Later". **Branch:** `feat/t3.1-context-ledger`.
**Builds on:** T2.1 (`run_manifest.json` — this spec's on-disk source of truth is that manifest + `image_summary.csv`).

## Problem

Both runners get a **static** `SYSTEM_PROMPT` at construction (`backend.py:36,45`) and nothing ever
injects "what has already been analyzed." The prompt even ends with *"Treat any injected
session/progress context as authoritative — do not re-analyze images already done"* — a promise no
code keeps today. Consequences:

1. **Re-analysis.** Ask "analyze /data/exp1", come back later (new `scat chat`, or the same chat after
   context compaction) and say "finish exp1" → the agent re-runs every image from scratch. Wasteful and
   it silently produces a *second* results dir, fragmenting the outputs.
2. **No resume / no progress memory.** There is no "7/12 done, 5 pending" state that survives a restart
   or a compaction. The durable record already exists on disk (T2.1's `run_manifest.json` +
   `image_summary.csv`) but nothing reads it back.
3. **Prior results are undiscoverable to the agent.** To "run stats on the exp1 results I made
   yesterday" the user must paste the timestamped `results_YYYYmmdd_HHMMSS/` path by hand — the agent
   can't list what's on disk.

## What already exists (reuse, don't rebuild)

- **`run_manifest.json`** in every results dir (T2.1): `dataset.path`, `dataset.{n_images,sha256,sample}`,
  `model`, `grouping.{column,mapping}`, `detection`, `warnings`, `created_at`, `scat_version`,
  `git_commit`. The self-describing, portable record — a results dir is *already* a "self-contained
  bundle" of metadata. Its `dataset.sample` is only the first 10 relpaths, so it is **not** the
  authoritative per-image list.
- **`image_summary.csv`** in every results dir: one row per analyzed image, column `filename` — the
  **authoritative exact list** of images that run analyzed.
- Results dirs are written by `get_timestamped_output_dir(Path(path).parent, "results")` →
  `results_<YYYYmmdd_HHMMSS>/` **as a sibling of the analyzed folder** (or its parent when a single file).
- The agent's recipe already calls `scan_folder(path)` **first, every time** (`prompts.py:9`).

## Design

Three slices, smallest-blast-radius first. Slices 1 + 3 are core (no agent/LLM deps, uniform across
both backends, purely derived from on-disk artifacts — the roadmap's "rebuild from on-disk results").
Slice 2 is the runner-level always-on ledger the roadmap literally asks for.

### `scat/results_index.py` (NEW, core — json + pandas + pathlib only)

The single on-disk "ledger builder." No agent, pydantic, or anthropic imports (packaging guard applies).

- `find_analyses(search_roots) -> list[AnalysisRecord]` — for each root, glob for `*/run_manifest.json`
  (results dirs are direct children of the folders we search; also accept the root itself being a
  results dir). Read each manifest; read the sibling `image_summary.csv` for the exact `filename` set.
  Returns records `{results_dir, created_at, dataset_path, n_images, analyzed_filenames:set, groups,
  model, sha256, warnings}`. Best-effort: a dir with a malformed manifest or missing CSV is skipped,
  never raises. Deterministic order (sort by `created_at` then `results_dir`).
- `analysis_status(folder, *, image_paths=None) -> dict` — the "analysed vs pending" delta for one
  target folder. Enumerate current images via `pipeline.list_images(folder)` (or the explicit
  `image_paths`); discover analyses whose `dataset_path` covers `folder` (scan `folder`'s parent for
  sibling results dirs); classify each current image as **analyzed** (its basename ∈ some covering
  run's `analyzed_filenames`) or **pending**. Return
  `{folder, n_current, n_analyzed, n_pending, pending:[names…capped], latest_results_dir, runs:[…]}`.
  Matching is by **basename** — consistent with how SCAT joins group metadata (`grouping_util`,
  `analyze_folder_service`'s duplicate-basename guard), so the same key space is used end to end.
- `format_ledger(records|status) -> str` — compact, deterministic, **token-bounded** plain text for
  injection (cap the number of runs and the pending list; summarize the rest as "+N more"). No secrets
  (manifests carry none — T2.1 F3 already redacts the provenance snapshot; manifests never held keys).

### Slice 1 — enrich `scan_folder` with resume status (turn 1, both backends, zero runner changes)

`scan_folder_service(path)` (`pipeline.py:57`) gains an **additive** key
`already_analyzed = results_index.analysis_status(path)` (best-effort; on any error omit the key, never
break scan). Because the recipe calls `scan_folder` first, the agent sees "7/12 already analyzed, 5
pending, prior results at <dir>" on the **very first turn of a fresh session** — resume works after a
full restart with no session state at all. Update `prompts.py` step 1 to tell the agent to **read
`already_analyzed` and, when images are already done, offer to analyze only the pending ones / reuse the
existing results dir for stats+report** instead of blindly re-running.

Note: `analyze_folder` today always analyzes *all* images it globs. Resuming "only the pending"
requires passing the pending subset. `analyze_folder_service` **already accepts `image_paths`** (added
for the GUI multi-file picker) — so the agent can call `analyze_folder(path, image_paths=<pending>,
groups=…)`. No new pipeline capability needed; only the `analyze_folder` **tool** must expose
`image_paths` (it currently doesn't forward it — verify and add). **Open question for review:** merging
a pending-only run with the prior run's results dir is out of scope for v1 — a pending-only run writes
its *own* results dir, and stats/report run per-dir. Document this limitation; do not silently produce
partial stats. (See Risk R3.)

### Slice 2 — always-on ledger injected each turn (survives mid-session compaction)

Slice 1 covers a fresh start but not the case where, mid-session, the agent already scanned and the tool
result was **compacted away** (`runner.py` caps tool results at 6000 chars). Inject a compact ledger
every turn so it can't be lost:

- **`AgentRunner` (API path).** Add `context_provider: Callable[[], str] | None = None`. At the top of
  `turn()`, compute `ledger = context_provider()` (best-effort; empty on error) and stream with an
  **effective system prompt** `system_prompt + "\n\n<session_context>\n{ledger}\n</session_context>"`.
  The system prompt is **ephemeral** — passed to `provider.stream(...)`, never appended to
  `self.messages` — so it is recomputed fresh each turn and never bloats history. ✅
- **`ClaudeSubscriptionRunner`.** `system_prompt` is baked into `ClaudeAgentOptions` at connect and the
  session persists via `resume`; we must **not** reconnect per turn (that discards the session). Inject
  the ledger as a `<session_context>…</session_context>` **preamble prepended to `user_text`** inside
  `turn()` before `client.query(...)`. The chat widget / CLI render the *typed* text themselves (the
  runner only drives assistant-side events), so the preamble is **invisible in the UI**. The SDK owns
  its own history; re-sending the small bounded ledger each turn is mildly redundant but always current,
  and the system prompt already instructs "treat the latest injected context as authoritative."
- **Scope of the ledger.** The runner tracks the set of folder paths seen in `scan_folder` /
  `analyze_folder` tool-call arguments this session (a cheap `set`), and `context_provider` =
  `lambda: results_index.format_ledger(results_index.find_analyses(seen_roots ∪ their parents))`. Empty
  set (turn 1, nothing scanned yet) → empty ledger → no-op; Slice 1 covers turn 1.

**This is the one deliberate backend divergence** (system-prompt vs user-text injection) — it exists
because the two runners manage conversation history differently. Flagged for Codex.

### Slice 3 — discovery tool (make prior results addressable) + bundle self-containedness

- A `list_analyses(folder=None)` **@tool** over `results_index.find_analyses` so the agent can answer
  "run stats on yesterday's exp1 results" without the user pasting a timestamped path. Returns each
  run's `{results_dir, dataset_path, created_at, n_images, groups}`.
- **Result bundles:** T2.1 already made a results dir self-describing (manifest + CSVs + optional
  report/annotations/spatial). For v1 "self-contained" = *discoverable + reproducible-from-metadata*,
  which `run_manifest.json` + `image_summary.csv` already deliver. An explicit export/zip of a bundle is
  **deferred** (note it in ROADMAP as T3.1-followup) — it adds packaging surface without new
  reproducibility information. **Open question for review:** is a zip/export in-scope for "self-contained
  result bundles," or is metadata-completeness + discovery the right v1 cut?

## Verification

- `tests/test_results_index.py`: `find_analyses` discovers a written results dir and reads the exact
  `filename` set from `image_summary.csv`; skips a malformed/partial dir without raising; deterministic
  order. `analysis_status` on a folder with a prior partial run reports the right analyzed/pending split
  by basename; a folder with no prior results → all pending; explicit `image_paths` honored.
- `tests/test_pipeline.py` (extend): `scan_folder_service` includes `already_analyzed` after a real
  `analyze_folder_service` run on the synth dir; the pending set shrinks to 0 after a full run; the key
  is omitted/empty (never raises) when there are no results dirs. **Parity gate untouched** — no CSV
  output changes.
- `tests/test_runner_context.py`: `AgentRunner.turn()` with a stub provider that echoes the system
  prompt shows the `<session_context>` block present and updated across turns, and **absent from
  `self.messages`** (ephemeral). Subscription: unit-test the `user_text` preamble assembly in isolation
  (no live SDK) — a helper `_with_ledger(user_text, ledger)` so it's testable without a CLI spawn.
- `tests/test_chat_widget.py` / `test_cli.py` (extend): a `list_analyses` tool call round-trips; the
  injected preamble is **not** echoed into the visible transcript.
- Packaging guard (`test_core_imports_without_agent.py`): `import scat.results_index` works **without**
  the `[agent]` extra. Full suite green.

## Risks

- **R1 — cross-folder / moved trees.** `analysis_status` finds results dirs as *siblings of the input
  folder*; if the user moved the results elsewhere, discovery misses them. Mitigation: also honor an
  explicit `search_roots` (config `agent.results_search_roots`, default = input folder's parent) so the
  user can point at an archive. Manifest's `dataset.sha256` lets us confirm a found run really matches
  the current folder's contents (guard against a stale same-named folder).
- **R2 — basename collisions across subfolders.** Matching by basename mirrors SCAT's existing join key,
  and `analyze_folder_service` already refuses grouping on duplicate basenames — so within a groupable
  dataset basenames are unique. For an ungrouped tree with dup basenames, `analysis_status` may
  over-count "analyzed"; surface a `basename_collision: true` flag rather than silently miscount.
- **R3 — pending-only resume fragments outputs.** Analyzing only the pending images writes a *new*
  results dir; stats/report over "the whole experiment" then need *both* dirs, which v1 does not merge.
  Mitigation for v1: the agent, on seeing partial prior work, **recommends re-running the full folder
  into one dir** (correct, simple) OR explicitly tells the user the pending-only run is separate. A
  proper merge/append is deferred. Do **not** let the agent emit stats over a partial dir as if whole.
- **R4 — ledger token cost.** A workspace with hundreds of runs → a huge ledger. `format_ledger` caps
  runs + pending names and summarizes the tail; the injected block is hard-bounded (like the runner's
  6000-char tool-result cap).
- **R5 — stale git_commit / created_at in manifest** — informational only; never used for matching
  (we match on path + basename + optional sha256). No correctness impact.

## Deliberately out of scope (v1)

Merging/appending pending runs into an existing results dir; zip/export of a bundle; a cross-machine
results registry; parsing the user's free-text message to pre-seed the ledger (the `scan_folder`-first
recipe covers turn-1 resume without it).
