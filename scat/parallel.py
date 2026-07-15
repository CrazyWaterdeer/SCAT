"""Hardware-aware parallel execution for batch image analysis.

Why processes, not threads: the per-image pipeline is GIL-bound — per-deposit Python
construction + feature glue dominate wall time; cv2/numpy release the GIL but are the
minority — so a ThreadPoolExecutor tops out around 1.3x and *degrades* past ~2 workers.
A fork-based ProcessPoolExecutor whose workers inherit the parent-loaded model
copy-on-write reaches ~12x on 24 cores (measured, 1600px images).

Start method is **fork** (not spawn/forkserver): fork inherits the live parent — the
loaded model included, copy-on-write — so nothing is pickled per task and, crucially,
workers do NOT reconstruct ``__main__`` (spawn/forkserver do, which breaks on stdin /
unguarded scripts and could re-enter a GUI/CLI entry point). ``analyze_image`` is
side-effect-free and never mutates the Analyzer, so the inherited model is read-only
under COW and re-running a slot (broken-pool recovery) is safe and non-duplicative.

Forking a multi-threaded process is deprecated in Python 3.12+ (it can deadlock a child
on an inherited native lock). We mitigate — pin cv2 to one thread in the parent before
forking, and it is empirically stable across CLI, plain-thread, and Qt-QThread callers —
and suppress the advisory warning; the ``SCAT_PARALLEL_ENGINE`` env var
(``auto`` | ``process`` | ``thread`` | ``sequential``) is the escape hatch if a specific
environment ever misbehaves.
"""
import os
import sys
import threading
import warnings
import multiprocessing
from pathlib import Path

# Worker count is bounded by usable cores, free memory (per_worker_gb), batch size, and CAP.
HARD_CAP = 32                 # backstop; min(cores, ...) is the real bound
MIN_BATCH_FOR_PROCESS = 4     # below this, pool startup ≥ benefit → threads/sequential
THREAD_CAP = 4                # threads are GIL-bound; more never helps this pipeline
DEFAULT_PER_WORKER_GB = 0.5   # measured working set ≪ 0.5GB at 1600px; conservative

_fork_lock = threading.Lock()
_WORKER_ANALYZER = None       # parent sets before fork; workers inherit COW


# --------------------------------------------------------------------------- probes
def _available_gb():
    """Available RAM in GB (psutil → /proc/meminfo), then clamped by cgroup-v2 memory.max.
    None if unknown. The cgroup clamp applies to *whichever* source so a container with
    psutil installed still can't over-size the pool past its memory limit."""
    val = None
    try:
        import psutil
        val = psutil.virtual_memory().available / (1024 ** 3)
    except Exception:
        pass
    if val is None:
        try:
            with open('/proc/meminfo') as f:
                for line in f:
                    if line.startswith('MemAvailable:'):
                        val = int(line.split()[1]) / (1024 ** 2)  # kB → GB
                        break
        except Exception:
            pass
    try:  # a container/WSL cgroup cap can be lower than host free memory
        with open('/sys/fs/cgroup/memory.max') as f:
            raw = f.read().strip()
        if raw and raw != 'max':
            headroom = int(raw) / (1024 ** 3)
            try:  # subtract what the cgroup is already using
                with open('/sys/fs/cgroup/memory.current') as cf:
                    headroom -= int(cf.read().strip()) / (1024 ** 3)
            except Exception:
                pass
            headroom = max(0.0, headroom)
            val = headroom if val is None else min(val, headroom)
    except Exception:
        pass
    return val


def usable_cores():
    """Usable CPU count: sched_getaffinity clamped by cgroup-v2 cpu.max. Always ≥ 1."""
    try:
        n = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        n = os.cpu_count() or 1
    try:  # cgroup cpu quota (Docker/CI/WSL) can be below the affinity mask
        with open('/sys/fs/cgroup/cpu.max') as f:
            quota, period = f.read().split()
        if quota != 'max':
            q, p = int(quota), int(period)
            if p > 0:
                n = min(n, max(1, -(-q // p)))  # ceil(quota / period)
    except Exception:
        pass
    return max(1, n)


def auto_worker_count(n_tasks, per_worker_gb=DEFAULT_PER_WORKER_GB):
    """Process-pool worker count from usable cores, free memory, and batch size.

    Uses *all* usable cores (the fork benchmark plateaus at/after core count with no
    penalty), bounded by a memory term (prevents OOM on big images / small RAM), the
    number of tasks, and HARD_CAP.
    """
    if n_tasks <= 1:
        return 1
    cores = usable_cores()
    mem = _available_gb()
    mem_bound = int(mem / per_worker_gb) if mem else cores
    return max(1, min(cores, mem_bound, n_tasks, HARD_CAP))


# --------------------------------------------------------------------------- engine choice
def _engine_override():
    return (os.environ.get('SCAT_PARALLEL_ENGINE') or 'auto').strip().lower()


def process_available():
    """True when a fork process pool is usable (POSIX, non-frozen, fork start method known)."""
    return (hasattr(os, 'fork') and sys.platform != 'win32'
            and not getattr(sys, 'frozen', False)
            and 'fork' in multiprocessing.get_all_start_methods())


def choose_engine(n_tasks, workers):
    """Return 'process' | 'thread' | 'sequential' for the given batch + worker count.

    ``SCAT_PARALLEL_ENGINE`` overrides: ``sequential`` / ``thread`` force those; ``process``
    forces the fork pool where available; ``auto`` (default) uses the pool for batches
    ≥ MIN_BATCH_FOR_PROCESS and falls back to threads otherwise / on unsupported platforms.
    """
    override = _engine_override()
    if override == 'sequential' or n_tasks <= 1 or workers <= 1:
        return 'sequential'
    if override == 'thread':
        return 'thread'
    if override == 'process':
        return 'process' if process_available() else 'thread'
    if n_tasks >= MIN_BATCH_FOR_PROCESS and process_available():
        return 'process'
    return 'thread'


# --------------------------------------------------------------------------- workers
def _fork_worker(path):
    return _WORKER_ANALYZER.analyze_image(path)


def _placeholder(analyzer, path):
    """The empty result used when an image fails (matches the pre-existing thread path)."""
    from .analyzer import AnalysisResult
    return AnalysisResult(filename=Path(path).name, deposits=[], dpi=analyzer.dpi)


def _run_sequential(analyzer, image_paths, progress_callback):
    n = len(image_paths)
    results = []
    for i, path in enumerate(image_paths):
        if progress_callback:
            progress_callback(i + 1, n)  # may raise to cancel the batch
        try:
            results.append(analyzer.analyze_image(path))
        except Exception as e:
            # Per-image failure isolation (matches the thread/process engines): a single
            # unreadable/corrupt image becomes an empty placeholder, the batch continues.
            results.append(_placeholder(analyzer, path))
            print(f"Warning: Failed to analyze {path}: {e}")
    return results


def _run_thread(analyzer, image_paths, workers, progress_callback):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    n = len(image_paths)
    results = [None] * n
    done = 0
    executor = ThreadPoolExecutor(max_workers=workers)
    try:
        fut_to_idx = {executor.submit(analyzer.analyze_image, p): i
                      for i, p in enumerate(image_paths)}
        for fut in as_completed(fut_to_idx):
            idx = fut_to_idx[fut]
            try:
                results[idx] = fut.result()
            except Exception as e:
                results[idx] = _placeholder(analyzer, image_paths[idx])
                print(f"Warning: Failed to analyze {image_paths[idx]}: {e}")
            done += 1
            if progress_callback:
                progress_callback(done, n)  # may raise to cancel the batch
    except BaseException:
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    executor.shutdown(wait=True)
    return results


def _process_safe(analyzer):
    """Fork is unsafe with PyTorch/CUDA — a CUDA context cannot survive fork() (the child's
    handles are invalid, so every worker fails) — and brittle with torch models in general.
    Keep torch-backed analyzers (CNN classifier or a U-Net detector) on the thread engine."""
    if type(getattr(analyzer, 'classifier', None)).__name__ == 'CNNClassifier':
        return False
    if getattr(getattr(analyzer, 'detector', None), 'unet_model_path', None):
        return False
    tmod = sys.modules.get('torch')
    if tmod is not None:
        try:
            if tmod.cuda.is_available() and tmod.cuda.is_initialized():
                return False
        except Exception:
            pass
    return True


def _run_pool_once(analyzer, image_paths, indices, results, workers, progress_callback, counter, total):
    """Run image_paths[i] for i in `indices` through one fresh fork pool, filling results[i]
    and bumping counter[0]/progress. Returns True if the pool broke (some slots left None).
    Cancels (progress_callback raising) propagate out."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from concurrent.futures.process import BrokenProcessPool
    global _WORKER_ANALYZER
    try:  # quiesce native pools before forking (fork-safety + no child oversubscribe)
        import cv2
        cv2.setNumThreads(1)
    except Exception:
        pass
    ctx = multiprocessing.get_context('fork')
    broke = False
    with _fork_lock:
        _WORKER_ANALYZER = analyzer
        try:
            # Forking a process with native (BLAS/cv2) helper threads triggers a py3.12+
            # DeprecationWarning; intra-op threads are pinned and stability is verified, so
            # suppress the advisory noise rather than emit it once per forked worker.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning,
                                        message=r".*fork.*may lead to deadlocks.*")
                executor = ProcessPoolExecutor(max_workers=workers, mp_context=ctx)
                try:
                    fut_to_idx = {executor.submit(_fork_worker, str(image_paths[i])): i
                                  for i in indices}
                    for fut in as_completed(fut_to_idx):
                        idx = fut_to_idx[fut]
                        try:
                            results[idx] = fut.result()
                        except BrokenProcessPool:
                            broke = True
                            break
                        except Exception as e:
                            results[idx] = _placeholder(analyzer, image_paths[idx])
                            print(f"Warning: Failed to analyze {image_paths[idx]}: {e}")
                        counter[0] += 1
                        if progress_callback:
                            progress_callback(counter[0], total)  # may raise to cancel
                except BrokenProcessPool:
                    broke = True
                except BaseException:
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise
                executor.shutdown(wait=False if broke else True, cancel_futures=broke)
        finally:
            _WORKER_ANALYZER = None
    return broke


def _run_process(analyzer, image_paths, workers, progress_callback):
    """Fork process pool with the parent model inherited COW. If the pool breaks (e.g. a
    worker segfaults on a bad image), retry the unfinished slots in a FRESH pool — never in
    the parent, so a crashing input cannot take down the CLI/GUI — then, after the retry
    budget, mark any stragglers as empty placeholders. Safe to retry because analyze_image
    writes nothing (no duplicated side effects)."""
    MAX_POOL_ATTEMPTS = 2
    n = len(image_paths)
    results = [None] * n
    counter = [0]
    remaining = list(range(n))
    for attempt in range(MAX_POOL_ATTEMPTS):
        if not remaining:
            break
        broke = _run_pool_once(analyzer, image_paths, remaining, results,
                               workers, progress_callback, counter, n)
        remaining = [i for i in remaining if results[i] is None]
        if not broke:
            break
        if remaining and attempt + 1 < MAX_POOL_ATTEMPTS:
            print(f"Warning: process pool broke; retrying {len(remaining)} image(s) in a fresh pool")
    for i in remaining:  # still unfilled after the retry budget → placeholder (never re-run in parent)
        results[i] = _placeholder(analyzer, image_paths[i])
        print(f"Warning: giving up on {image_paths[i]} after repeated worker-pool failures")
        counter[0] += 1
        if progress_callback:
            progress_callback(counter[0], n)
    return results


def run_batch(analyzer, image_paths, progress_callback=None, max_workers=0):
    """Analyze ``image_paths`` with ``analyzer.analyze_image``, returning results in input
    order. Chooses the engine automatically (see module docstring). Preserves per-image
    progress, cooperative cancel (``progress_callback`` may raise), per-image failure
    isolation, and input ordering. Returns ``(results, engine_label)``."""
    n = len(image_paths)
    if n == 0:
        return [], 'sequential'
    if max_workers <= 0:
        max_workers = auto_worker_count(n)
    engine = choose_engine(n, max_workers)
    if engine == 'process' and not _process_safe(analyzer):
        engine = 'thread'  # torch/CUDA model can't be forked — fall back to threads
    if engine == 'process':
        return _run_process(analyzer, image_paths, max_workers, progress_callback), \
            f'process(fork,{max_workers})'
    if engine == 'thread':
        workers = max(1, min(max_workers, THREAD_CAP))
        return _run_thread(analyzer, image_paths, workers, progress_callback), f'thread({workers})'
    return _run_sequential(analyzer, image_paths, progress_callback), 'sequential'
