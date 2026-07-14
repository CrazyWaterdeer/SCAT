# SCAT AI-Agent Layer — Design Spec

**Date:** 2026-07-14
**Status:** Approved (design); ready for implementation planning
**Author:** Jin (with Claude)

## 1. Summary

Add a conversational AI-agent layer to SCAT so a user can say *"analyze this
folder"* and the agent autonomously runs the full pipeline — scan → detect →
classify → **auto-assign experimental groups** → statistics → HTML report — with
no manual clicking through tabs, no drag-drop group assignment, and no groups
CSV. The agent drives SCAT's existing analysis code through a shared **`@tool`
registry**, so a chat turn, a CLI command, and (in phase 2) a GUI button all
execute the *same* code path.

The architecture is ported from the sibling project **Imajin** (`/home/lab/Imajin`),
a conversational confocal-microscopy assistant with a mature, napari-free agent
core (provider abstraction, tool-loop runner, tool registry, provenance).

This is not merely additive: SCAT's analysis pipeline is currently hand-written
**three times** (GUI `_do_analysis`, `cli.analyze_command`, GUI post-edit
`_generate_report`) and those copies have drifted. Introducing one shared tool
path is also the fix for that duplication.

## 2. Goals / Non-goals

### Goals (v1)
- Natural-language driver: *"analyze /data/exp1"* → complete analysis + report.
- **Reliable auto-grouping** from filename/subfolder structure (a first-class
  deterministic tool, not LLM improvisation).
- One shared tool registry that the CLI and the agent both call (button==chat==CLI).
- Frictionless auth: works via a local `claude` subscription login (no API key),
  falling back to `ANTHROPIC_API_KEY`.
- Ship a working **CLI chat** (`python -m scat.cli chat`) as the v1 milestone.

### Non-goals (v1 — deferred to phase 2+)
- **GUI chat dock** in the PySide6 app (Qt-main-thread tool marshalling).
- Removal/rewire of GUI manual-orchestration widgets (`_do_analysis`,
  `GroupEditorDialog`, Quick/Full mode) — happens when the GUI dock lands.
- Self-contained result bundles / crash-resume ledger.
- Local Ollama / OpenAI-compatible backends.
- Live model-catalog resolution (v1 hardcodes the current model id + a fallback).

## 3. Decisions (locked)

| # | Decision | Choice |
|---|----------|--------|
| 1 | LLM backend | **Claude subscription (Agent SDK, no key) preferred; `ANTHROPIC_API_KEY` fallback.** |
| 2 | Interface | **Shared core; CLI chat in v1; GUI chat dock in phase 2.** |
| 3 | Group assignment | **Auto-infer + auto-apply + show the inferred mapping** (non-blocking; user can correct in chat). Backed by a deterministic `infer_groups` tool. |
| 4 | v1 scope | **Core automation via CLI** (see Non-goals). Also: reimplement CLI `analyze` on the shared tools (collapse one duplicate) + delete 4 dead config keys + add `agent` config section. |

Default model: `claude-opus-4-8` (API-key path; adaptive thinking, prompt
caching). Subscription path lets the `claude` CLI resolve the model alias.

## 4. Architecture

### 4.1 Module layout (greenfield — nothing exists in `scat/` yet)

```
scat/agent/
  events.py         # Event union (TextDelta/ToolUseStart/ToolUse/Stop) + Provider Protocol  (port Imajin base.py)
  providers/
    __init__.py
    anthropic_api.py       # API-key provider: client.messages.stream(); prompt caching on system + last tool
    claude_subscription.py # Claude Agent SDK bridge (no key); tools exposed as in-process MCP server
  runner.py         # AgentRunner: the agentic loop (stream→collect tool_use→execute→tool_result→reloop)
  registry.py       # @tool decorator + Pydantic-schema-from-signature + call_tool + tools_for_anthropic  (port Imajin registry.py)
  prompts.py        # SCAT system prompt (bias-to-action, pipeline recipes, stats guardrails)  BUILD FRESH
  context.py        # durable per-turn ledger built from on-disk results (done/pending images)  BUILD FRESH (pattern from Imajin context.py)
  provenance.py     # session JSONL log of every tool call  (port Imajin provenance.py)
  backend.py        # provider/runner selection: subscription if available else API key
scat/tools/
  __init__.py       # eager import barrel (registration is an import side-effect)
  scan.py           # scan_folder
  grouping.py       # infer_groups (tool) + _build_group_metadata (shared helper)   ← the new intelligence
  pipeline.py       # analyze_folder (the single orchestration, worker=True)
  stats.py          # run_statistics
  report.py         # generate_report
scat/cli.py         # add `chat` subcommand; reimplement `analyze` on the shared tools
docs/superpowers/specs/2026-07-14-scat-ai-agent-design.md  (this file)
```

### 4.2 Two-layer agent

- **Provider** = *one* model turn: `stream(messages, tools, system) -> Iterator[Event]`.
  Emits `TextDelta`, `ToolUseStart`, `ToolUse(id,name,input)`, `Stop(reason,usage)`.
  It does **not** execute tools. Messages use Anthropic content-block format as
  the canonical internal representation.
- **AgentRunner** = the agentic loop: call `provider.stream()`, accumulate
  assistant text + `tool_use` blocks; when `stop_reason == "tool_use"`, execute
  each tool via an injected `tool_caller` (defaults to `registry.call_tool`),
  append Anthropic-style `tool_result` blocks, and re-stream — up to `max_loops`.
  Exposes `turn(text) -> Iterator[RunEvent]`, `reset()`, `cancel()`.
- **Subscription exception:** the Claude Agent SDK owns its own loop (it drives
  the `claude` CLI, which executes tools itself). So `ClaudeSubscriptionRunner`
  fuses provider+runner and re-exposes the *same* `turn()/reset()/cancel()`
  surface. The CLI/agent front-ends treat both runners identically.

**Runner parity contract (the two paths are NOT automatically equivalent).**
The same `turn()` surface hides different internals (the API path's `AgentRunner`
owns the loop, compaction, provenance, retries, `max_loops`; the subscription
path delegates the loop and tool execution to the SDK/CLI). Both runners MUST
guarantee the same *observable* behavior: (a) `cancel()` stops at the next tool
boundary; (b) **every** tool invocation is provenance-logged regardless of who
drives the loop (so the subscription bridge must record calls itself); (c) a tool
exception surfaces as an `is_error` result, not a crashed turn; (d) `max_loops`
is enforced. What the subscription path **cannot** match and is documented as
such: server-side context compaction and token/cost accounting are owned by the
SDK, so the durable-ledger-in-system-prompt (§7) is the portable substitute, and
cost metering exists only on the API path. This divergence is called out so
callers never assume identical semantics.

### 4.3 Backend selection (`backend.build_runner()`)
1. If `subscription_available()` (Claude Agent SDK importable + `claude` on PATH
   + a login: `CLAUDE_CODE_OAUTH_TOKEN` or `~/.claude/.credentials.json`) →
   `ClaudeSubscriptionRunner`. During CLI spawn, pop `ANTHROPIC_API_KEY` /
   `ANTHROPIC_AUTH_TOKEN` so a stray key can't silently bill the API instead of
   the subscription.
2. Elif `ANTHROPIC_API_KEY` set → `AgentRunner(AnthropicApiProvider(model="claude-opus-4-8"))`.
3. Else → a clear error explaining both options.

**Observability (required).** The selection is silent-surprise-prone (auto-
preferring subscription, popping API env vars). On startup, loudly log the
chosen backend, model, and **whether API billing is possible** ("using Claude
subscription — no API charges" vs "using ANTHROPIC_API_KEY — requests are
billed"), so a user never unknowingly bills the API. `agent.backend` can be
forced to `subscription`/`api` to override auto.

### 4.4 Plain services beneath thin tools (avoid the "@tool == domain API" trap)

The reusable orchestration is **plain Python**, not a tool. `scat/pipeline.py`
exposes `analyze_folder(path, groups=None, ...) -> AnalyzeResult` (and the
stats/report/scan services) as ordinary typed functions with rich return types.
The `@tool` in `scat/tools/` is a **thin adapter**: it calls the plain service,
then compacts the return into an LLM-sized dict. So:
- The CLI `analyze` command and (phase 2) the GUI Run button call the **plain
  services**, never the `@tool` — no JSON-schema/LLM-argument-shape/compact-
  return constraints leak into CLI/GUI internals.
- Only the agent path imports the registry/pydantic/anthropic (see §12).
This keeps "collapse to one path" honest: the *service* is the single canonical
implementation; chat, CLI, and GUI are three thin front doors onto it.

## 5. Tool set

Each tool is a thin `@tool`-decorated wrapper over SCAT's **existing** pipeline
code. Type hints on every parameter (the JSON schema is auto-derived; untyped →
`Any`). **Return compact JSON-able dicts** (counts, paths, per-class tallies) —
never raw DataFrames/arrays, or the context window fills after a few images.

| Tool | Wraps | Key params (LLM-fillable) | Returns |
|------|-------|---------------------------|---------|
| `scan_folder(path)` | glob (from `cli.analyze_command`) | `path` | image count, extensions, sample filename patterns |
| `infer_groups(path)` | **new** deterministic parser | `path` | `{file: group}` mapping, tokens matched, confidence, `unmatched` |
| `analyze_folder(path, ...)` | `Analyzer.analyze_batch` + `ReportGenerator.save_all` | `groups: dict[str,str]\|None`, `model_type`, `model_path`, `min_area`, `max_area`, `circularity`, `annotate` | output dir, Normal/ROD/Artifact counts, n_failed |
| `run_statistics(results_dir, group_col)` | `statistics.run_comprehensive_analysis` | `results_dir`, `group_col='group'` | per-metric test/p/effect summary |
| `generate_report(results_dir, ...)` | `report.ReportGenerator.generate_html_report` | `results_dir`, `visualize`, `format='html'` | `report.html` path |

`analyze_folder` is `worker=True` (CPU-heavy). Parameters that are manual
flags/widgets today become agent-inferred tool params with config-backed defaults.

**Group data passing (stateless).** `infer_groups` returns the `{file: group}`
mapping *in the conversation*; the agent passes it straight into
`analyze_folder(path, groups=<mapping>)`. `analyze_folder` builds the metadata
DataFrame internally via a single shared helper `_build_group_metadata(mapping)`
— the deduped `_generate_metadata` logic, standardized on the `'group'` column
and the `'ungrouped'` sentinel. There is no separate stateful `apply_groups`
tool holding a DataFrame; the mapping threads through the tool arguments. (An
explicit `--metadata` CSV remains a supported alternative input to
`analyze_folder` for user-authored groupings.)

### 5.1 `infer_groups` — the core new capability

Deterministic, not prompt-driven, so grouping is reliable:
1. If images live in **subfolders**, subfolder name = group (highest priority).
2. Else tokenize each filename on separators (`_ - space .`); find the token
   position(s) that **vary** across files; match against a condition vocabulary
   (control/ctrl/wt/treated/treatment/mutant/dose/timepoint/genotype + numeric
   replicate suffixes) to pick the grouping token.
3. Group files by that token; return the canonical **`{file: group}`** mapping
   (per-file, so `unmatched` files are explicit), plus the tokens matched, the
   inferred **role** of the varying token (condition vs replicate vs dose vs
   timepoint — from the vocabulary match), and a **confidence** score.
4. Fallback: no structure found → single cohort (no comparison).

**Join key.** SCAT merges metadata on the `filename` column, which is the image
**basename** (`Path(path).name`, per `analyzer.py`). So `_build_group_metadata`
keys on basename. **Risk:** subfolder-based grouping can produce duplicate
basenames across subfolders, which would mis-join — in that case
`infer_groups`/`analyze_folder` must disambiguate (prefix the group, or key on a
relative path and expose it as `filename`). This is called out as a required
edge case, not an assumption.

**Confidence & the bias-to-action exception.** Numeric suffixes are ambiguous
(replicate vs dose vs timepoint vs plate-row vs image index), and real names mix
delimiters (`ctrl_rep1_flyA`, `drug10uM_rep2`). The parser is conservative:
below a confidence threshold it does **not** silently group — it surfaces the
proposal prominently and recommends confirmation or an explicit metadata CSV.
Grouping ambiguity is the one sanctioned exception to §6's "don't ask" stance.

Unassigned files are labeled with the sentinel `'ungrouped'` so
`statistics.run_all_tests`' existing filter keeps working.

The agent calls `infer_groups`, **states the mapping in chat**, and passes it to
`analyze_folder(groups=...)` (auto, per Decision 3); the `group` column then flows
into the **unchanged** `save_all → condition_summary/groups/ → statistics` machinery.

### 5.2 The stable seam (why grouping "just works")

The entire downstream is source-agnostic and **kept verbatim**:
`metadata = DataFrame with a 'group' column` + `group_by=['group']` →
`ReportGenerator.generate_condition_summary` / `save_all` group branches
(`analyzer.py`) → `statistics.run_all_tests` / `analyze_groups`. A single
`_build_group_metadata(mapping) -> (metadata_df, group_by)` helper is all that
connects `infer_groups` output to this existing flow — **zero downstream
changes**. Column name is standardized to `'group'` (the old customizable
`'condition'` name never reached downstream).

## 6. System prompt (`prompts.py`)

Reuse Imajin's prompt *structure*, none of its confocal/napari/Korean content:
1. Role: an action-oriented analysis agent.
2. **Bias to action** — given a folder, run the whole pipeline to completion;
   don't ask clarifying questions unless genuinely ambiguous.
3. **Pipeline recipe**: "analyze this folder" → `scan_folder` → `infer_groups` →
   `analyze_folder(groups=<mapping>)` → (if ≥2 groups) `run_statistics` →
   `generate_report`. State the inferred grouping to the user before analyzing.
4. **Statistics guardrail** (domain-neutral, high value): you assert the design
   (paired vs independent); for 3+ groups report the omnibus p + a
   multiplicity-corrected post-hoc, not uncorrected pairwise; relay warnings
   (small n, non-normal) instead of a bare p-value.
5. Forbidden: no menus, no re-asking, no inventing group names beyond what
   filename structure supports.
6. Treat the injected batch-progress ledger as authoritative.

## 7. Context management & provenance

- **Durable ledger** (`context.py`): each turn, rebuild an "analysed vs pending"
  summary from on-disk results (`image_summary.csv`, `deposits/*.json`) and inject
  it into the system prompt. Gives free resume / "don't re-analyze" / "7/12 done"
  that survives context compaction because it reads disk, not chat.
- **Tool-result compaction** in the runner (cap ~4000 chars; SCAT payloads are
  small dicts, so mostly a safety net); message compaction if the transcript
  grows; **orphaned-`tool_result` backfill** preserved (Anthropic 400s otherwise).
- **Provenance** (`provenance.py`): append every tool call to a session JSONL.
  For real reproducibility the run header records more than calls: **dataset
  fingerprint** (folder path + file count + a hash of sorted basenames+sizes),
  **SCAT version + git commit**, **config snapshot**, **model id / classifier
  model path + hash**, and the **grouping mapping + confidence**. This is the
  basis for a future "methods" section.
- **Idempotency / no overwrite:** `analyze_folder` writes to a **timestamped**
  output dir (`get_timestamped_output_dir`), so re-running never overwrites or
  mixes results. The durable ledger and `run_statistics`/`generate_report` take
  an explicit `results_dir`, so "regenerate the report" targets a specific run.

## 8. Agent-first slimdown (what the new design removes/simplifies)

The pipeline is hand-written 3× and drifted (different stats functions, a
hand-rolled cv2 annotator, different metric lists, two `statistical_results`
shapes). Verdicts from the code sweep (73 findings): 13 REMOVE, 25 SIMPLIFY, 18
KEEP-BUT-ROUTE-THROUGH-TOOL, 17 KEEP.

### v1 (this milestone)
- **New** shared tools = the single canonical orchestration.
- **Reimplement `cli.analyze_command`** on the shared tools (collapse one of the
  three copies; drop the CLI's inline stats `.txt` writer and duplicate glob).
  Establish one canonical default: `model_type = rf` if a model exists in
  `models/`, else `threshold` (removes the CLI-only `threshold` default surprise).
- **Delete 4 dead config keys**: `detection.sensitive_mode` (hardcoded `True`
  everywhere, never read), `detection.edge_margin` (GUI never reads),
  `analysis.group_by` (never read), `last_metadata_path` (zero references).
- **Add `agent` config section** (see §9).
- Keep the `Config` singleton and dot-notation get/set/save.

### Phase 2 (with the GUI chat dock)
- Rewire GUI `Run` button + `Ctrl+R` + `ResultsTab._generate_report` to call the
  shared tools (collapses copies 2 and 3; deletes the inline cv2 annotator in
  favor of `analyzer.generate_annotated_image`).
- **Remove** the drag-drop grouping stack (~550 lines): `GroupEditorDialog`,
  `GroupWidget`, `DroppableContainer`, "Create Groups" button — replaced by
  `infer_groups` + a thin read-only review/correct panel bound to the tool output.
- **Remove** Quick/Full mode radios, `is_quick_mode`, `_on_mode_changed`, the
  close-time "report not generated" nag (the agent sequences detect→review→report
  as conversation turns; human correction between them stays via the KEEP'd edit
  flow).
- Collapse the duplicate detection-parameter surfaces (SettingsDialog Detection
  tab + AnalysisTab spinboxes + CLI flags → one tool signature).
- Collapse the two statistics entry points to `run_comprehensive_analysis` and
  the two report entry points to `generate_html_report`.

### Always KEEP (human value / infra — NOT orchestration)
Human labeling (`TrainingTab`, Labeling launch), misclassification correction
(`Results → LabelingWindow EDIT_MODE`), labeling keyboard shortcuts, worker-count
& window config, `PathSelector`, I/O path config, and the **group plumbing**
(`generate_condition_summary`, `save_all` group branches,
`statistics.run_all_tests/analyze_groups`) — source-agnostic; `infer_groups` feeds
it identically.

## 9. New `agent` config + secrets

No `api_key/provider/anthropic/openai` reference exists in `scat/` today. Add a
config section for **non-secret** selection only:

```json
"agent": { "backend": "auto",  // auto | subscription | api
           "model": "claude-opus-4-8",
           "max_loops": 40 }
```

**Secrets never go in plaintext `~/.scat/config.json`.** The `ANTHROPIC_API_KEY`
is read from the environment (or an OS keyring later); config stores only the
provider/model selection. The subscription path needs no secret in config.

## 10. Error handling

- Per-tool `try/except` → `tool_result` with `is_error: true` so the agent can
  recover (try a different arg, report to the user).
- One context-limit force-compact-and-retry (as in Imajin).
- **Add** exponential backoff for `429`/`overloaded` (Imajin punts on this).
- A bad image never aborts the batch (`analyze_batch` already produces a
  placeholder result).
- If `< 2` groups, statistics is **explicitly skipped with a note** ("stats
  skipped: N groups"), never a silent no-op.

### 10.1 Cost, latency, and cancellation

Cost is inherently **low**: because tools are coarse, a whole folder is **one**
`analyze_folder` tool call (pure local compute, no per-image LLM), so a 500-image
run is ~5 LLM turns total, not 500. `max_loops=40` is a runaway backstop, not the
expected depth (~5). Still add:
- **Progress independent of model turns:** `analyze_folder` streams per-image
  progress via a `report_progress` contextvar (rendered by the CLI/chat), so the
  user sees "142/500" during the pure-compute turn, not just at tool return.
- **Cooperative cancellation:** the batch loop checks a `raise_if_cancelled`
  token per image, so a chat "stop" actually halts analysis at the next image
  (documented as cooperative, not preemptive).
- **Wall-clock guardrails:** an overall turn timeout; a large-batch heads-up
  (e.g. ">N images — proceeding, this may take a while") rather than a blocking
  confirm (bias-to-action), configurable.
- On the API path only, surface token usage per turn (from `Stop.usage`).

## 11. Testing (pytest; no real LLM required)

- `registry`: a sample `@tool` produces the expected JSON schema from its hints.
- `infer_groups`: cases for `control/treated`, `dose`, subfolder-as-group, and
  the no-structure single-cohort fallback; verify `'ungrouped'` sentinel.
- `_build_group_metadata`: `{file:group}` → `(DataFrame with 'group', ['group'])`.
- **FakeProvider**: a scripted provider that emits a fixed sequence of `ToolUse`
  blocks, driving `AgentRunner` end-to-end through `analyze_folder` on the
  synthetic-image fixture (reused from the bugfix test suite) with **no network**.
- `backend`: selection logic (subscription available / only key / neither).
- **Parity / anti-drift gate (required for v1):** the new plain-service
  `analyze_folder` path must produce output **byte-identical** to the current
  pipeline on the synthetic fixture (`image_summary.csv`, `all_deposits.csv`) —
  the same diff-based check used in the bugfix pass. This is the guard against
  the v1/phase-2 sequencing trap (CLI/agent on the new path while the GUI is
  still on the old one).
- **Subscription bridge contract test:** a focused test against the real Claude
  Agent SDK bridge (guarded/skipped when the SDK/`claude` login is absent) that
  verifies a tool round-trips: schema is accepted, a compact dict result returns,
  a raised exception maps to an error result, and a `worker=True` tool runs. The
  `FakeProvider` cannot cover the bridge; this closes that gap.

## 12. Dependencies

Add an optional extra in `pyproject.toml`:
```toml
[project.optional-dependencies]
agent = ["anthropic>=0.40", "pydantic>=2.0", "claude-agent-sdk"]
```
`anthropic` + `pydantic` are required for the agent layer; `claude-agent-sdk` is
needed only for the subscription path (import guarded).

**Packaging guard (required).** Core SCAT must import and run without the
`[agent]` extra. `pydantic`/`anthropic`/`claude-agent-sdk` are imported **only**
under `scat/agent/` and `scat/tools/`, never by `scat/analyzer.py`,
`scat/pipeline.py`, or the CLI `analyze` command (which calls the plain services
per §4.4). `python -m scat.cli analyze ...` and `import scat` must work with just
the base dependencies; only `python -m scat.cli chat` requires `[agent]`. A smoke
test asserts core import without the extra installed.

## 13. Phasing summary

- **v1 (this spec):** agent core + tools (`scan_folder`, `infer_groups`,
  `analyze_folder`, `run_statistics`, `generate_report`) +
  system prompt + context ledger + provenance + backend selection + CLI `chat` +
  CLI `analyze` reimplemented on the tools + `agent` config + dead-key cleanup +
  tests.
- **Phase 2:** PySide6 chat dock (Qt-main-thread tool marshalling) + rewire GUI
  Run/Ctrl+R/regenerate onto the tools + remove drag-drop grouping + remove
  Quick/Full mode + collapse duplicate detection/stats/report surfaces.
- **Later:** result bundles + resume; Ollama/OpenAI backends; live model catalog.

## 14. Open risks

- Claude Agent SDK block shapes are duck-typed in Imajin's bridge (no `type`
  field in that SDK version) — an SDK upgrade can silently break translation.
  Mitigate by pinning the SDK version and covering the bridge with the FakeProvider
  test where possible.
- `infer_groups` heuristics won't cover every naming scheme; the mapping is always
  surfaced in chat so the user can correct, and an explicit metadata CSV path
  remains a supported override.
- **Two-runner non-equivalence** (subscription vs API): mitigated by the runner
  parity contract (§4.2) and the documented divergences (compaction/cost). Track:
  keep provenance/cancel/error-shape behavior tested on both.
- **Subscription MCP bridge is external and version-fragile**: mitigated by
  pinning the SDK, the bridge contract test (§11), and duck-typed translation
  matching Imajin's working bridge. An SDK upgrade requires re-running that test.

> This revision folds an independent Codex (gpt-5.5) review: fixed the
> `apply_groups` recipe/return-shape inconsistencies; added the join-key contract
> and conservative grouping confidence (§5.1); the runner parity contract (§4.2);
> auth observability (§4.3); the plain-services-under-thin-tools rule (§4.4);
> provenance enrichment + idempotency (§7); cost/progress/cancel controls (§10.1);
> the parity gate + bridge contract test (§11); and the packaging guard (§12).
