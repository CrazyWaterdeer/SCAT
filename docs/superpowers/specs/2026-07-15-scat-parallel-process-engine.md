# SCAT — Hardware-aware parallel batch engine

**Status:** spec · 2026-07-15 · branch `feat/parallel-process-engine`

## Problem

Batch image analysis (`Analyzer.analyze_batch`) uses a `ThreadPoolExecutor` and asks the
user (GUI Setup tab → "Worker threads") or caller to pick a worker count, defaulting to an
"Auto" that calls `_get_safe_worker_count`. Two measured problems on a 24-core WSL box:

1. **Auto is broken in the common case.** `_get_safe_worker_count` needs `psutil` for its
   memory term and falls back to `memory_workers = 4` when it is missing (it is not a
   dependency and is not installed). `min(cpu//2, 4, 20, n) = 4` → **auto uses 4 of 24 cores.**
2. **Threads cannot use the hardware.** The per-image pipeline is **GIL-bound** (pure-Python
   per-deposit construction + feature glue dominate; cv2/numpy release the GIL but are the
   minority of wall time). Measured throughput vs a 1-worker baseline (48×1600px images):

   | engine | speedup |
   |---|---|
   | threads, 2 workers | **1.3× (peak)** — degrades past 2 (GIL contention) |
   | processes, spawn + per-worker model load | ~3× |
   | **processes, fork + parent model inherited (COW)** | **~12.6× (peak @16, plateau to 24)** |

The user's ask ("thread 개수를 수동 지정 → 자동 결정") plus the deeper goal ("하드웨어를 제대로 활용") ⇒
**auto-size the workers AND switch to a process engine that can actually use the cores.**

## Measured facts (this machine, grounding the design)

- 24 logical cores (`os.sched_getaffinity(0)`), 60 GB MemAvailable.
- OpenBLAS + OpenCV each default to 24 intra-op threads → N pool workers × 24 = oversubscription;
  workers must pin intra-op threads to 1.
- `AnalysisResult`/`Deposit` (numpy contours) pickle round-trip cleanly.
- `fork` `ProcessPoolExecutor` works when created from a **non-main thread** (agent tool + GUI
  QThread both run analysis off the main thread).
- `/proc/meminfo` `MemAvailable` gives a psutil-free memory probe on Linux/WSL (the runtime).

## Goals

- **Auto worker count** that reflects real hardware (cores + free memory + batch size), no manual
  input required; the agent/CLI path decides and proceeds. Manual override still honored.
- **Process-based execution** for the real speedup, defaulting to `fork` + inherited model on
  POSIX, with a safe cross-platform fallback.
- **Preserve everything that works today:** per-image progress, cooperative cancel (T1.1),
  result order, per-image failure isolation, and **byte-identical CSV/JSON output** (parity gate).
- Transparent: log the chosen engine + worker count.
- Parallelize the currently-sequential **annotate** loop (audit rank 16).

## Non-goals

- Free-threaded (no-GIL) Python — not this build.
- GPU. Distributed/multi-machine.
- Changing feature math / model (parity + `mean_hue` remain untouched).

## Design

### 1. Worker-count heuristic — `scat/parallel.py::auto_worker_count(n_tasks, per_worker_gb=0.5)`

```
cores  = len(os.sched_getaffinity(0)) or os.cpu_count() or 1
mem_gb = _available_gb()            # psutil if present else /proc/meminfo else None
mem_bound = int(mem_gb / per_worker_gb) if mem_gb else cores   # no psutil dep required
return max(1, min(cores, mem_bound, n_tasks, HARD_CAP))        # HARD_CAP = 32
```

- Uses **all** usable cores (not `//2`); the fork benchmark plateaus at/after core count with no
  penalty at 24, so `cores` is a safe target. Memory term prevents OOM on big images / small RAM.
- `per_worker_gb` default 0.5 (measured working set ≪ 0.5 GB for 1600px; conservative). Callers
  can raise it (e.g. CNN/U-Net) later; not required now.

### 2. Execution — refactor `Analyzer.analyze_batch`

Keep the public signature (`image_paths, metadata, progress_callback, parallel, max_workers`).
Add internal engine selection:

```
n = len(image_paths)
if not parallel or n <= 1:            # sequential (unchanged)
if max_workers <= 0: max_workers = auto_worker_count(n)
engine = choose_engine(n, max_workers) # 'process' | 'thread' | 'sequential'
```

- **`choose_engine`**: use `process` when `n >= MIN_BATCH_FOR_PROCESS` (=4, below which
  spawn/fork overhead ≥ benefit) and multiprocessing is usable; else fall back to the existing
  thread pool (which is ≥ sequential only up to ~2 workers, so thread path caps workers at 2).
- **Process path** (`_run_process_pool`):
  - Start method: `fork` if available (POSIX) else `spawn`.
  - **fork:** set module global `scat.parallel._WORKER_ANALYZER = self` before creating the pool;
    worker = module-level `_worker_analyze(path)` that calls `_WORKER_ANALYZER.analyze_image(path)`.
    Model is inherited COW — never pickled. Only the path (in) and `AnalysisResult` (out) cross.
  - **spawn:** `initializer=_init_worker(config)` builds a per-process Analyzer from the pickled
    `ClassifierConfig` + dpi (model loaded once per worker, not per task).
  - Every worker pins intra-op threads: `cv2.setNumThreads(1)` and `OPENBLAS/OMP=1` (set in
    initializer / via env for spawn; via `cv2.setNumThreads(1)` at parent for fork children).
  - Submit all; consume with `as_completed`; **increment a main-process counter and call
    `progress_callback(done, n)`** exactly as the thread path does today (progress + cancel are
    entirely main-process-side, so T1.1 is preserved verbatim).
  - **Cancel:** `progress_callback` raising → `executor.shutdown(wait=False, cancel_futures=True)`
    then re-raise. Pending tasks dropped; in-flight finish (bounded by worker count) — identical
    semantics to today's threads.
  - **Robustness:** wrap per-future `.result()`; on worker exception → placeholder empty result +
    warning (today's behavior). On `BrokenProcessPool` (a worker crashed, e.g. segfault) → log and
    **fall back to sequential** for the *unfinished* images so a batch never dies wholesale.
- **Determinism / parity:** results are re-assembled into input order via the `future→index` map
  (as today). `analyze_image` is pure per image, so process vs thread vs sequential produce the
  same `AnalysisResult`s → CSVs stay byte-identical. Guarded by an explicit before/after parity
  check in verification.

### 3. Concurrency-safety of the fork global

`_WORKER_ANALYZER` is process-global and only read in forked children. SCAT runs a single active
analysis at a time (GUI serializes on one WorkerThread; CLI/agent are one call). The global is set
immediately before pool creation and cleared in `finally`. Document the single-active-analysis
assumption (same assumption `scat/progress.py` already documents for T1.1).

### 4. Annotate loop (rank 16) — `scat/pipeline.py`

The annotate loop re-decodes each image and writes a PNG sequentially. It needs only the image
path + `res.deposits` (already in hand) — no model. Parallelize with a **thread pool** sized by
`auto_worker_count`: decode (PIL), draw (cv2), and PNG encode all release the GIL, so threads
overlap the I/O-and-native-encode work without process overhead. Each iteration writes a distinct
file → order-independent, output unchanged. (Spatial's header-only `Image.open().size` loop is
already cheap; leave it.)

### 5. Config / GUI

- `analyze_folder_service` already threads `parallel` + `max_workers`; default `max_workers=0`
  now means "hardware-aware auto" for the agent/CLI. No signature change.
- GUI Setup: keep the combo; "Auto" now maps to the new heuristic (drop the psutil-only path so
  the displayed Auto value matches what runs). Manual counts still override. Show detected cores.
- New config keys are optional; existing `performance.parallel_enabled` / `performance.worker_count`
  keep working (`worker_count=0` → auto).

### 6. Logging

On each batch: one line — `engine=process(fork) workers=16 images=240` (via warnings list / logger)
so users and the agent can see the machine is being used.

## Risks & mitigations

| risk | mitigation |
|---|---|
| fork from a multithreaded Qt process deadlocks | pin intra-op threads to 1 in parent before pool; create pool at quiescent batch start; fork verified from a non-main thread; spawn fallback selectable |
| Windows-native (no fork) | `spawn` fallback (still ~3×, improves with batch size) |
| worker segfault → BrokenProcessPool | catch → sequential fallback for unfinished images |
| model not picklable under spawn init | init builds from `ClassifierConfig` (already picklable) + reloads model file, not the object |
| small batches slower under processes | `MIN_BATCH_FOR_PROCESS` gate → thread/sequential |
| memory blow-up (many big images) | memory-bound term in `auto_worker_count` |
| output drift vs today | before/after byte parity of CSV/JSON in verification; result-order re-assembly |

## Verification

- **Parity:** byte-identical `image_summary.csv` / `all_deposits.csv` / per-image / `groups/` /
  `*.labels.json` for the same inputs: sequential vs process(fork) vs process(spawn) vs thread.
- **Benchmark:** report speedup vs sequential for the process engine at the auto count.
- **Cancel:** a `progress_callback` that raises mid-batch stops promptly; ≤ worker-count extra
  images complete; partial results consistent.
- **Failure isolation:** a deliberately corrupt image → placeholder result, batch completes.
- **Full test suite + parity gate green;** add unit tests for `auto_worker_count` (mocked
  cores/memory) and an engine-equivalence test (sequential vs process results equal).
- **Real end-to-end** CLI + agent-tool run on synthetic images.

---

## Codex review → incorporated decisions (2026-07-15)

Codex (gpt-5.5, xhigh) flagged the fork+inherited-model+cancel+parity surface as large and
risky (18 findings). The design is tightened so the change is **purely additive** — it can only
ADD a fast path, never regress what runs today:

- **No spawn path.** (Codex #7 spawn-state-reconstruction parity risk, #16 spawn edge cases.)
  The engine either forks (fast path) or falls back to **today's unchanged thread/sequential
  path**. Cross-platform "3× spawn" is dropped as not worth the parity risk right now.
- **Fork only when provably safe:** POSIX + `hasattr(os,'fork')` + **`threading.active_count()==1`**
  + not frozen + start-method compatible. This deterministically enables fork for the clean
  single-threaded CLI/headless caller and disables it for ANY multithreaded context (GUI QThread,
  agent worker thread, parallel tests) — which keep today's behavior exactly. Eliminates the
  Qt/multithreaded-fork hazard (Codex #1) instead of hand-waving it.
- **`analyze_image` is side-effect-free** (decode→detect→extract→classify→return; writes nothing;
  uses a *local* `FeatureExtractor`, never mutates `self`). So: the inherited model is read-only
  under COW (Codex #2), and a `BrokenProcessPool` fallback can **re-run only the `None` result
  slots sequentially** with no duplicate side effects and no cancel violation (Codex #3/#5). A
  module `Lock` + the active_count gate make concurrent fork-analysis impossible (Codex #6).
- **Cancel granularity is per-image-completion — same as today.** The current thread path also
  only checks cancel in the `as_completed` loop; processes match it (in-flight bounded by worker
  count). No regression; the spec's "identical to threads" stands (Codex #4 clarified, not a bug).
- **Worker intra-op threads pinned to 1** via `cv2.setNumThreads(1)` in the worker (runtime-
  effective). BLAS is a minor cost in this pipeline (RF = tree traversal, features = tiny ROIs),
  which is why the fork benchmark already hit 12× without perfect BLAS control (Codex #13 noted;
  impact measured-negligible).
- **cgroup-aware sizing** (Codex #11): `usable_cores` clamps `sched_getaffinity` by cgroup-v2
  `cpu.max`; memory probe reads psutil→`/proc/meminfo`→cgroup `memory.max`. Cap + `per_worker_gb`
  are named, documented constants (Codex #10/#12).
- **Byte-parity is verified, not assumed** (Codex #8): the acceptance gate byte-diffs
  fork-engine vs sequential CSV/JSON output on identical inputs.
- **Annotate** parallelized with a small thread pool; distinct output files, no warnings in the
  loop → order-independent; verified PNG-byte-identical (Codex #15).

Deferred (not worth the risk now): spawn/forkserver for GUI/Windows speedup; cgroup memory
scaling by image size; making thresholds config-driven. Logged as follow-ups.

---

## Final implementation notes (what shipped, and why it differs from the spec)

Two empirical findings during implementation changed the engine choice:

1. **`threading.active_count()` is not a usable fork-safety gate.** Inside a GUI QThread it
   reports 1 (Qt threads are C-level, invisible to Python's `threading`), yet CPython's own
   `os.fork()` correctly warns there. And after importing numpy/cv2 the process already has
   idle BLAS helper threads, so a "single-threaded process" gate (`/proc/self/task == 1`)
   disables the pool for the *entire real pipeline*. So there is no cheap reliable "safe to
   fork" gate — the choice is fork-and-mitigate or don't-fork.

2. **forkserver/spawn reconstruct `__main__` in workers → fragile for an app.** A forkserver
   trial crashed every worker with `FileNotFoundError: .../<stdin>` (it runpath-imports the
   parent `__main__`), silently falling back to sequential; for a GUI/CLI entry point that
   reconstruction can even re-enter the app. **`fork` does none of this** (it inherits the
   live process, `__main__` included), which makes it the *more robust* choice here despite
   the py3.12+ multi-threaded-fork deprecation.

**Shipped:** a **fork** `ProcessPoolExecutor` with the parent model inherited COW (module
docstring in `scat/parallel.py`). Mitigations for the deprecation: pin cv2 to 1 thread in the
parent before forking; suppress the advisory warning (verified stable across CLI / plain-thread
/ Qt-QThread callers via byte-parity + behavioural tests); `SCAT_PARALLEL_ENGINE` env escape
hatch (`auto|process|thread|sequential`). The active_count gate and the spawn/forkserver
fallback are **not** used.

**Measured (24-core WSL, 48×1400px):** analyze_batch alone **6.9s → 0.73s (9.4×)**; full
`analyze_folder_service` (analysis + per-image CSV/JSON + annotate) **10.6s → 4.1s (2.6×)** —
the end-to-end ratio rises with batch size as the parallel analysis dominates the serial I/O.
Byte-identical to the pre-change `main` output across sequential/thread/process engines.

**GUI/agent also benefit:** because fork is used regardless of caller thread-count (not gated),
the GUI QThread and agent worker get the same speedup — resolving the user's original concern
that the GUI required a manual worker-count. Deferred follow-ups: forkserver/spawn for
Windows-native speedup; running the GUI analysis in a child process (belt-and-braces isolation).
