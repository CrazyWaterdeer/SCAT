# T3.1 implementation plan — context ledger + resume + discovery

**Spec:** `docs/superpowers/specs/2026-07-15-scat-context-ledger.md` · **Branch:** `feat/t3.1-context-ledger`.
Order = Slice 1 → 2 → 3; Slice 3 (combine) is cuttable (1+2 already ship a correct feature). Full suite +
parity green after each slice. Two Codex passes folded in (see the two "incorporated" sections).

## Confirmed before coding
- Parity gate (`test_pipeline_parity.py:16-19`) diffs only `all_deposits.csv` + `image_summary.csv` from
  `analyze_folder_service` → a `scan_folder_service` key is safe. ✅
- `config.get` is **dot-notation** (`config.get("agent.results_search_roots", [])`). ✅
- `run_statistics_service`/`generate_report_service` read the **CSVs only**, never `run_manifest.json`
  (`pipeline.py:194-226`) → a combined dir needs valid CSVs; its manifest only matters to discovery. ✅
- Tier-1 fingerprint is stable on rescanned folder, flips when a file is added (verified). ✅

## 1. `scat/results_index.py` (NEW, core — stdlib + pandas + `scat.manifest`/`scat.pipeline` only)

No `scat.agent`/pydantic/anthropic imports (packaging test covers it). pandas is already a `scat.pipeline`
dep, so this adds no new core dependency.

```python
@dataclass(frozen=True)
class AnalysisRecord:
    results_dir: str            # resolved abspath
    status: str                 # "complete" | "partial"      (unreadable dirs go to the skip channel)
    created_at: str | None
    dataset_path: str | None    # resolved, from manifest.dataset.path (None if manifest missing)
    dataset_sha256: str | None
    n_images: int
    analyzed_basenames: frozenset[str]
    basename_dupes: bool        # image_summary.csv had duplicate 'filename' rows (delta math unsafe)
    groups: list[str]           # sorted labels from manifest.grouping.mapping ([] if ungrouped)
    group_column: str | None
    model: dict | None          # exact manifest.model
    detection: dict | None      # exact manifest.detection
    warnings: list[str]
```

Helpers:
- `_resolve(p) -> Path` — `Path(p).expanduser().resolve()`; on OSError return `Path(p)`.
- `_norm_roots(roots) -> list[Path]` — map through `_resolve`, **drop falsy/`""`**, drop non-existent,
  **dedupe** (stable order). (Codex-2 #5.)
- `_parse_ts(s) -> float` — ISO8601 → epoch; `0.0` on missing/malformed (stable sort key, never used for
  correctness).
- `_read_manifest(dir) -> dict | None` — json load, swallow errors.
- `_read_basenames(dir) -> tuple[frozenset[str], bool] | None` — read `image_summary.csv` col `filename`;
  `names=[str(x) for x in df["filename"]]`; return `(frozenset(names), len(names)!=len(set(names)))`; None
  if missing/unreadable. (Values are basenames — `analyzer.py:169`. Dup detection = Codex-2 #4.)
- `_iter_result_dirs(root) -> Iterable[Path]` — `root` itself if it holds `run_manifest.json` or
  `image_summary.csv`, plus its **direct children** that do. (Direct children only; results dirs are
  siblings of inputs, never nested — no rglob.)

Public API:
- `find_analyses(search_roots) -> list[AnalysisRecord]` and
  `find_analyses_with_skips(search_roots) -> (records, skipped: list[{dir, reason}])` — `find_analyses`
  = the records half. Per dir: manifest+csv→`complete` (manifest is written last ⇒ finished run);
  csv, no/broken manifest→`partial` (crashed/in-progress, `dataset_path=None`); neither→a `skipped`
  entry (surfaced, not silently dropped — Codex-1 trap). Sort by `(_parse_ts, results_dir)`.
- `run_brief(record) -> dict` — **public** (Codex-2 #14) JSON-safe subset:
  `{results_dir, dataset_path, created_at, n_images, groups, group_column, model, detection, status}`
  (includes params so the agent can judge run compatibility — Codex-2 #3).
- `analysis_status(folder, *, image_paths=None, search_roots=None) -> dict`:
  1. `current = [Path(p) for p in image_paths] if image_paths is not None else pipeline.list_images(folder)`;
     `n_current=len(current)`; empty → `{"status":"empty","folder":str(folder)}`.
  2. `roots = _norm_roots(search_roots or [_resolve(folder).parent])`. **Covers-match is exact resolved
     equality**, one rule for dir *and* single-file input (a single-file run writes `dataset.path=<file>`
     into a `results_*` dir under the file's folder = `folder.parent` — so both the root and the equality
     work unchanged; no special "compare parent" branch — Codex-2 #2):
     `records = [r for r in find_analyses(roots) if r.dataset_path and _resolve(r.dataset_path)==_resolve(folder)]`.
     `runs = [run_brief(r) for r in records]`.
  3. **Tier 1 — fingerprint-verified complete (strong, ambiguity-immune).**
     `fp = manifest.dataset_fingerprint([str(p) for p in current])["sha256"]`; if any
     `r.dataset_sha256 == fp` → `{"status":"complete","verified":True,"n_current":n_current,
     "n_analyzed":n_current,"n_pending":0,"results_dir":r.results_dir,"runs":runs}`. **Semantics
     (Codex-2 #1):** answers "was *exactly this image set* analyzed in one run?" For `scan_folder`
     (no `image_paths`) that is folder-level completeness; with an explicit subset it is subset-level and
     labeled by `n_current`.
  4. **Tier 2 — guarded basename delta (seen / not-seen).** `cur_bn=[p.name for p in current]`.
     - duplicate **current** basenames (`len(set)!=len`) OR any covering record with `basename_dupes`
       → `{"status":"ambiguous","reason":…,"n_current":n_current,"runs":runs}` (no numeric split —
       Codex-1 #1 / Codex-2 #4).
     - else `seen = set().union(*[r.analyzed_basenames for r in records]) if records else set()`;
       `analyzed=[b for b in cur_bn if b in seen]`; `pending=[b for b in cur_bn if b not in seen]`;
       `status = "partial" if records else "none"`; return `{status,"verified":False,"n_current",
       "n_analyzed":len(analyzed),"n_pending":len(pending),"pending":pending[:_PENDING_CAP],
       "pending_truncated":len(pending)>_PENDING_CAP,"latest_results_dir":records[-1].results_dir if
       records else None,"runs":runs,"note":"analyzed = basename seen in a prior run; runs may use
       different params — not reusable for whole-experiment stats without combine_results"}`.
       (Reframed as "seen", params exposed in `runs` — Codex-2 #3.)
- `format_ledger(payload, *, max_chars=1500) -> str` — deterministic lines from a `find_analyses` list or
  an `analysis_status` dict; **hard char-cap** with "+N more".
- Caps: `_PENDING_CAP=50`, `_MAX_RUNS_IN_LEDGER=20`.

## 2. Slice 1 — enrich `scan_folder`

`scat/pipeline.py::scan_folder_service` — after building `result`, best-effort (lazy import keeps a core
`import scat.pipeline` free of the read path):
```python
try:
    from . import results_index
    result["already_analyzed"] = results_index.analysis_status(path)
except Exception:
    pass   # never break scan; scan output isn't in the parity gate
```
`scat/agent/prompts.py` step 1: read `already_analyzed`; `complete`→reuse `results_dir` for stats/report,
re-analyze nothing; `partial`/`none` with pending→analyze pending-only into a new dir (say it's separate,
or offer `combine_results`); `ambiguous`→surface the basename clash; **never** treat a tier-2 `partial`
as reusable-for-stats.

## 3. Slice 2 — `list_analyses` tool + `analyze_folder` gains `image_paths`

`scat/tools/pipeline_tools.py`:
- **Add `image_paths: Optional[list[str]] = None`** to the `analyze_folder` tool and forward it (today
  dropped). Service keeps the dup-basename refusal. Concrete generic for a proper schema (Codex-2 #12).
- **New `list_analyses` tool** (fixes the sibling-discovery bug, Codex-2 #6 — search `folder.parent`, not
  just `folder`):
  ```python
  @tool(description="List prior SCAT results dirs discoverable near a folder (or default roots): each "
        "run's results_dir, dataset_path, created_at, n_images, groups, model/detection, status. Use to "
        "resume/reuse prior analyses or to find a results dir for stats/report.")
  def list_analyses(folder: Optional[str] = None) -> dict:
      from scat.results_index import find_analyses, run_brief
      roots = _search_roots(folder)
      return {"analyses": [run_brief(r) for r in find_analyses(roots)],
              "search_roots": [str(x) for x in roots]}
  ```
  `_search_roots(folder)` (tool module): start from `[folder, Path(folder).parent]` when `folder` given
  else `[os.getcwd()]`; extend with `config.get("agent.results_search_roots", [])`; `results_index`
  normalizes/dedupes. (folder.parent is where sibling `results_*` live — the correctness fix.)
- `scat/config.py`: add `"results_search_roots": []` to the `agent` default section (R1).
- `prompts.py`: "When continuing prior work, reusing earlier results, or unsure what's done after a long
  chat, call `list_analyses` (or re-`scan_folder`) before analyzing."

## 4. Slice 3 (cuttable) — `scat/combine.py` + `combine_results` tool

`scat/combine.py` (NEW, core): `combine_results_service(results_dirs: list[str], output_dir: str|None=None) -> dict`.
- Load each dir's manifest + `image_summary.csv` (+ `all_deposits.csv` if present).
- **Compatibility gate — refuse (`ValueError` with a specific reason) on any failure (Codex-2 #7/#8/#9):**
  - **same resolved `dataset.path`** across all sources (prevents merging unrelated folders — Codex-2 #8);
  - **exact-equal `model` dict** and **exact-equal `detection` dict** after canonical normalization
    (compare whole objects, don't hand-pick fields — Codex-2 #7);
  - **same `grouping.column`** and, for any shared basename, the **same group label**;
  - **basenames disjoint across sources** (the resume case: full run A..F + pending run G..I). If a
    basename appears in >1 source, its `image_summary` row **and** its `all_deposits` rows must be
    value-identical across sources → keep one; otherwise **refuse** (no created_at "newest wins" —
    Codex-2 #9/#10). Disjoint is the primary supported mode and needs no dedup.
- Concat `image_summary.csv` and `all_deposits.csv` (deposits filtered to the retained filenames), write
  to a fresh `output_dir` (default timestamped sibling of `results_dirs[0]`), and write a
  **`run_manifest.json` superset** (valid for discovery + any consumer, Codex-2 #11): normal manifest
  fields (dataset.path, recomputed dataset fingerprint over the union, model, detection, grouping) **plus**
  `"combined_from": [resolved source dirs]`.
- Return `{output_dir, n_images, groups, sources}`. `combine_results` **@tool** wraps it
  (`results_dirs: list[str]`, `output_dir: Optional[str]=None`). Prompt: "analyze pending →
  combine_results([old, new]) → run_statistics/generate_report on the combined dir."

## 5. Tests

- `tests/test_results_index.py` (NEW): `find_analyses` discovers a real dir (built via
  `analyze_folder_service(synth_dir, output_dir=…)`), reads exact basenames; malformed manifest →
  `skipped` w/ reason (no raise); csv-without-manifest → `partial`; deterministic order w/ missing/dup
  `created_at`. `analysis_status`: (a) fingerprint match → `complete/verified/n_pending=0`; (b) strict
  subset prior run + unique basenames → correct `partial` seen/pending; (c) duplicate current basenames →
  `ambiguous`; (c') covering run with dup CSV rows → `ambiguous`; (d) no prior → `none`; (e) explicit
  `image_paths` honored; (f) resolved-path equality: `./exp1` vs `/abs/exp1` matches, a *different*
  same-named folder does **not**; (g) single-file input matches its run; (h) `_norm_roots` drops
  `""`/relative/nonexistent + dedupes.
- `tests/test_pipeline.py` (extend): `scan_folder_service` `already_analyzed` correct pre/post full run;
  `complete` after full; key omitted/no-raise with no results.
- `tests/test_cli.py`/`test_chat_widget.py` (extend): `call_tool("list_analyses", …)` round-trips AND
  finds sibling dirs when given the analyzed **folder** (guards Codex-2 #6); `call_tool("analyze_folder",
  …, image_paths=[…])` works; `tools_for_anthropic()` emits a nullable schema for the new optional params
  (guards Codex-2 #13).
- `tests/test_combine.py` (NEW, Slice 3): disjoint compatible merge → row counts add, manifest
  `combined_from` set, discoverable by `find_analyses`, `run_statistics_service` runs on it; refuse on
  mismatched dataset.path / model / detection / grouping; refuse on overlapping basename with differing
  rows; overlapping **identical** rows → kept once.
- `tests/test_core_imports_without_agent.py` (extend): `import scat.results_index` and `import
  scat.combine` succeed without the `[agent]` extra; `import scat.pipeline` still does too.
- Full `pytest -q` green incl. parity.

## 6. Wiring / ROADMAP / memory
- No GUI change (agent surface). `docs/ROADMAP.md`: mark T3.1 done (branch/PR); add `T3.1-followup`
  (zip/export bundles + multi-dataset/overlapping-merge aggregation beyond `combine_results`). Update
  [[scat-agent-v1]] Tier-3 progress after merge.

## Codex plan review — incorporated (pass 2, gpt-5.5 xhigh)
1 subset-vs-folder fingerprint semantics → documented on tier 1. 2 single-file equality → one resolved-
equality rule, dropped the "compare parent" branch. 3 tier-2 union across incompatible runs → reframed as
"seen", params exposed in `runs`, prompt forbids reuse-for-stats. 4 legacy dup CSV rows → `_read_basenames`
detects, degrades to `ambiguous`. 5 root hygiene → `_norm_roots` drops ""/relative/nonexistent + dedupes.
6 **`list_analyses(folder)` missed sibling dirs → search `folder.parent`** (real bug, fixed). 7 combine
gate compares whole normalized model+detection. 8 combine requires same `dataset.path`. 9/10 combine =
disjoint-or-identical basenames, no created_at "newest wins", deposits filtered to retained filenames.
11 combined dir writes a `run_manifest.json` **superset** (+`combined_from`) so discovery/consumers work.
12 concrete generics `Optional[list[str]]`/`list[str]`. 13 tests exercise `call_tool` + bridge schema.
14 `run_brief` is public. 15/16 packaging + edge tests enumerated above.
