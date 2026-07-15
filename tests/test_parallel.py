"""Tests for the hardware-aware parallel batch engine (scat/parallel.py)."""
import pytest

import scat.parallel as par
from scat.analyzer import Analyzer
from scat.detector import DepositDetector
from scat.classifier import ClassifierConfig
from scat.pipeline import list_images, resolve_model_type
from scat.progress import AnalysisCancelled


def _analyzer():
    mtype, mpath = resolve_model_type(None, None)
    return Analyzer(detector=DepositDetector(),
                    classifier_config=ClassifierConfig(model_type=mtype, model_path=mpath))


# --------------------------------------------------------------- heuristic
def test_auto_worker_count_bounds(monkeypatch):
    monkeypatch.setattr(par, "usable_cores", lambda: 24)
    monkeypatch.setattr(par, "_available_gb", lambda: 60.0)
    assert par.auto_worker_count(1) == 1              # trivial batch
    assert par.auto_worker_count(3) == 3              # bounded by task count
    assert par.auto_worker_count(1000) == 24          # bounded by cores
    # memory-bound: 2 GB free / 0.5 per worker = 4
    monkeypatch.setattr(par, "_available_gb", lambda: 2.0)
    assert par.auto_worker_count(1000) == 4
    # no memory info -> falls back to cores, never crashes
    monkeypatch.setattr(par, "_available_gb", lambda: None)
    assert par.auto_worker_count(1000) == 24


def test_auto_worker_count_hard_cap(monkeypatch):
    monkeypatch.setattr(par, "usable_cores", lambda: 128)
    monkeypatch.setattr(par, "_available_gb", lambda: 999.0)
    assert par.auto_worker_count(1000) == par.HARD_CAP


def test_usable_cores_and_mem_are_sane():
    assert par.usable_cores() >= 1
    gb = par._available_gb()
    assert gb is None or gb > 0


def test_choose_engine(monkeypatch):
    monkeypatch.delenv("SCAT_PARALLEL_ENGINE", raising=False)
    monkeypatch.setattr(par, "process_available", lambda: True)
    assert par.choose_engine(1, 8) == "sequential"              # trivial batch
    assert par.choose_engine(10, 1) == "sequential"             # 1 worker
    assert par.choose_engine(3, 8) == "thread"                  # below MIN_BATCH_FOR_PROCESS
    assert par.choose_engine(10, 8) == "process"               # forkserver pool
    # no forkserver (e.g. Windows) -> thread fallback
    monkeypatch.setattr(par, "process_available", lambda: False)
    assert par.choose_engine(10, 8) == "thread"
    monkeypatch.setenv("SCAT_PARALLEL_ENGINE", "process")       # override still needs support
    assert par.choose_engine(10, 8) == "thread"
    monkeypatch.setattr(par, "process_available", lambda: True)
    assert par.choose_engine(10, 8) == "process"
    monkeypatch.setenv("SCAT_PARALLEL_ENGINE", "sequential")
    assert par.choose_engine(10, 8) == "sequential"
    monkeypatch.setenv("SCAT_PARALLEL_ENGINE", "thread")
    assert par.choose_engine(10, 8) == "thread"


def test_process_available_is_bool():
    assert isinstance(par.process_available(), bool)


def test_process_safe_gates_torch():
    """CNN classifier and U-Net detector (torch/CUDA) must NOT use the fork pool."""
    import types

    class CNNClassifier:  # name is what _process_safe checks
        pass

    rf = types.SimpleNamespace(classifier=object(), detector=types.SimpleNamespace(unet_model_path=None))
    cnn = types.SimpleNamespace(classifier=CNNClassifier(), detector=types.SimpleNamespace(unet_model_path=None))
    unet = types.SimpleNamespace(classifier=object(), detector=types.SimpleNamespace(unet_model_path="/m.pt"))
    assert par._process_safe(rf) is True
    assert par._process_safe(cnn) is False
    assert par._process_safe(unet) is False


def test_process_safe_routes_cnn_to_thread(synth_dir, monkeypatch):
    """run_batch downgrades the process engine to threads for a torch analyzer."""
    imgs = list_images(str(synth_dir))
    az = _analyzer()
    monkeypatch.setattr(par, "_process_safe", lambda a: False)
    _results, engine = par.run_batch(az, imgs, max_workers=4)
    assert engine.startswith("thread")


# --------------------------------------------------------------- execution
def _key(results):
    return [(r.filename, r.n_total, r.n_normal, r.n_rod) for r in results]


def test_engines_equivalent(synth_dir):
    """Sequential, thread, and (when available) forkserver process engines produce identical
    results in identical order — the byte-parity guarantee at the object level."""
    imgs = list_images(str(synth_dir))
    assert len(imgs) >= 4
    az = _analyzer()
    seq = par._run_sequential(az, imgs, None)
    thr = par._run_thread(az, imgs, 4, None)
    assert _key(thr) == _key(seq)
    assert [r.filename for r in seq] == [p.name for p in imgs]  # input order
    if par.process_available():
        proc = par._run_process(az, imgs, 4, None)
        assert _key(proc) == _key(seq)
        assert [r.filename for r in proc] == [p.name for p in imgs]


def test_cancel_stops_batch(synth_dir):
    imgs = list_images(str(synth_dir))
    az = _analyzer()
    seen = {"n": 0}

    def cb(done, total):
        seen["n"] = done
        if done >= 1:
            raise AnalysisCancelled("stop")

    with pytest.raises(AnalysisCancelled):
        par.run_batch(az, imgs, progress_callback=cb, max_workers=4)
    assert seen["n"] >= 1
    assert par._WORKER_ANALYZER is None  # global cleared even on cancel


def test_failure_isolation(synth_dir, tmp_path):
    imgs = list_images(str(synth_dir))
    bad = tmp_path / "corrupt.tif"
    bad.write_text("not an image")
    mixed = imgs[:2] + [bad] + imgs[2:]
    az = _analyzer()
    results, engine = par.run_batch(az, mixed, max_workers=4)
    assert len(results) == len(mixed)
    # the corrupt image becomes an empty placeholder at its input position
    assert results[2].filename == "corrupt.tif"
    assert results[2].n_total == 0


def test_sequential_failure_isolation(synth_dir, tmp_path):
    """The sequential engine must also placeholder a bad image and continue (1-worker batches
    previously went through a thread pool that isolated failures)."""
    imgs = list_images(str(synth_dir))
    bad = tmp_path / "corrupt.tif"
    bad.write_text("not an image")
    mixed = [imgs[0], bad] + imgs[1:]
    az = _analyzer()
    results = par._run_sequential(az, mixed, None)
    assert len(results) == len(mixed)
    assert results[1].filename == "corrupt.tif" and results[1].n_total == 0
