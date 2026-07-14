# T2.1 — Provenance run-header enrichment (reproducibility manifest)

**Branch:** `feat/tier2` · **Roadmap:** T2.1 · **Spec:** §7.
Today `provenance.py` logs only per-tool `CallRecord`s and `read_session()` is never read back.
Spec §7 wants a run header recording more than calls: dataset fingerprint, SCAT version + git
commit, config snapshot, model id + classifier hash, grouping mapping — the basis for a future
"methods" section and the first slice of the deferred "result bundles" (T3.1).

## Design — a core `scat/manifest.py` used by both the pipeline and provenance
Two artifacts, one primitive module (core, no agent/GUI deps):

1. **`run_manifest.json` written into every analysis output dir** (the reproducibility artifact —
   travels with the results, seeds result bundles). Written by `analyze_folder_service`, which
   already holds every field.
2. **A `session_header` line in the agent provenance JSONL** (spec §7's literal ask) — static run
   context + config snapshot, written by `provenance.start_session`.

### `scat/manifest.py`
- `run_context()` (cached): `{scat_version (scat.__version__), git_commit (git rev-parse --short
  HEAD, best-effort None off-tree), python, platform}`.
- `sha256_file(path)` → hex | None (OSError-safe, chunked).
- `dataset_fingerprint(image_paths)` → `{n_images, sha256(of sorted "name:size" lines), sample[:10]}`
  — deterministic, order-independent (sorted), content-agnostic (size not bytes → cheap).
- `write_run_manifest(output_dir, *, path, image_paths, model_type, model_path, groups,
  group_column, detection, warnings)` → writes `<output_dir>/run_manifest.json`
  `{schema:"scat.run_manifest/1", created_at, **run_context, dataset:{path,**fingerprint},
  model:{type,path,sha256}, grouping:{column,mapping}|None, detection:{...}, warnings}`; OSError-safe.

### `analyze_folder_service` (pipeline.py)
After `save_all`, before `return`, call `manifest.write_run_manifest(out, path=path,
image_paths=[str(p) for p in images], model_type=mtype, model_path=mpath, groups=groups,
group_column=(group_by[0] if group_by else None), detection={min_area,max_area,edge_margin,
circularity,sensitive_mode,unet_model_path}, warnings=warnings)`. New file only — does NOT touch
`image_summary.csv`/`all_deposits.csv`, so the **parity gate stays green**.

### `provenance.py`
- Refactor the append+OSError-fallback into `_append_line(obj)`.
- `start_session` writes a first line `{"type":"session_header", session_id, started_at, driver,
  **run_context(), "config": dict(config.data)}` (config has no secrets — spec §9). Tag call
  records `{"type":"call", **asdict(rec)}` so the two line kinds are distinguishable.

## Verification
- `tests/test_manifest.py`: `run_context` has the keys; `dataset_fingerprint` is deterministic +
  order-independent + changes when a file size changes; `sha256_file` on a temp file; None off-tree.
- Service: `analyze_folder_service(...)` writes a valid `run_manifest.json` with the right
  n_images/model/grouping; parity gate unchanged (new file only).
- Provenance: `start_session` writes a `session_header` line with run_context + config; a subsequent
  `record_call` writes a `type:"call"` line; `read_session` returns both.
- Full suite green.

## Codex review — incorporated
Codex confirmed the parity gate and `read_session`/consumer safety, and flagged 5 gaps (all folded in):
- **F1 (fingerprint collision):** hash `relpath-to-common-root:size`, not `basename:size`, so same-named
  files in different subfolders don't collide (still portable if the tree moves). Test added.
- **F2 (wrong git commit off-tree):** only report a commit when `git rev-parse --show-toplevel` equals
  the SCAT root — an install inside another repo now records `None`, not that repo's HEAD.
- **F3 (config secret leak):** redact secret-looking keys (`key/token/secret/password/credential/auth`)
  in the session-header config snapshot.
- **F4 (fallback read mismatch):** `read_session(sid)` now also checks the temp fallback dir.
- **F5 (param placement):** `circularity` (a classifier setting) moved under `model`, not `detection`.

## Risks
- **R1 non-determinism in the manifest** (created_at, git_commit) — it's a NEW sidecar, never
  diffed by parity; fine. Tests assert structure/fields, not byte-equality.
- **R2 git_commit in an installed/packaged env** — no `.git` → returns None (best-effort). No crash.
- **R3 sha256 of a large model file** — model_rf.pkl is ~small; chunked read; done once per run.
- **R4 config snapshot leaking a secret** — none: spec §9 keeps API keys in env, not config.
