"""The process-global progress + cooperative-cancel channel (scat/progress.py) and its
integration into analyze_folder_service."""
from pathlib import Path

import pytest

from scat import progress
from scat.pipeline import analyze_folder_service


def test_inert_without_active_run():
    # No run open: report is a no-op and cancel never fires, even if requested.
    progress.report_progress(1, 2, "x")   # must not raise
    progress.request_cancel()
    progress.raise_if_cancelled()          # inert — no active run
    assert not progress.is_active()


def test_sink_receives_and_cancel_arms_then_resets():
    seen = []
    with progress.run_progress(lambda c, t, note="": seen.append((c, t, note))):
        assert progress.is_active()
        progress.report_progress(3, 10, "hi")
        assert seen == [(3, 10, "hi")]
        progress.request_cancel()
        with pytest.raises(progress.AnalysisCancelled):
            progress.raise_if_cancelled()
    # After the context, the channel is closed and the flag reset.
    assert not progress.is_active()
    progress.raise_if_cancelled()          # inert again


def test_service_emits_ambient_progress(synth_dir, tmp_path):
    calls = []
    with progress.run_progress(lambda c, t, note="": calls.append((c, t))):
        res = analyze_folder_service(str(synth_dir), output_dir=str(tmp_path / "p"),
                                     annotate=False, parallel=False, ambient_progress=True)
    assert res.n_images == 6
    assert calls and calls[-1] == (6, 6)                 # ends at total
    assert [c for c, _ in calls] == list(range(1, 7))    # ascending, sequential


def test_gui_run_path_ignores_ambient_channel(synth_dir, tmp_path):
    """ambient_progress defaults False, so a service call with a chat sink open (the GUI Run
    path) neither reports into it nor gets cancelled by it — cross-feature isolation (Codex F4)."""
    calls = []
    with progress.run_progress(lambda c, t, note="": calls.append((c, t))):
        progress.request_cancel()   # would cancel an ambient run
        res = analyze_folder_service(str(synth_dir), output_dir=str(tmp_path / "g"),
                                     annotate=False, parallel=False)  # ambient_progress=False
    assert res.n_images == 6        # not cancelled
    assert calls == []             # nothing leaked into the chat sink


def test_service_cancels_sequential_and_writes_nothing(synth_dir, tmp_path):
    out = tmp_path / "cancelled"
    with progress.run_progress(lambda c, t, note="": None):
        progress.request_cancel()
        with pytest.raises(progress.AnalysisCancelled):
            analyze_folder_service(str(synth_dir), output_dir=str(out),
                                   annotate=False, parallel=False, ambient_progress=True)
    assert not out.exists()   # cancel fires before save_all -> no results dir written


def test_parallel_cancel_drops_queued_images(synth_dir, tmp_path, monkeypatch):
    """The dangerous path (Codex F7): with 1 worker + cancel armed, only the first image runs;
    the queued rest are dropped via shutdown(cancel_futures=True) rather than drained."""
    from scat.analyzer import Analyzer
    ran = []
    orig = Analyzer.analyze_image
    monkeypatch.setattr(Analyzer, "analyze_image",
                        lambda self, path: (ran.append(path), orig(self, path))[1])
    out = tmp_path / "pc"
    with progress.run_progress(lambda c, t, note="": None):
        progress.request_cancel()
        with pytest.raises(progress.AnalysisCancelled):
            analyze_folder_service(str(synth_dir), output_dir=str(out), annotate=False,
                                   ambient_progress=True, parallel=True, max_workers=1)
    assert len(ran) < 6, f"cancel must drop queued images; ran {len(ran)}/6"
    assert not out.exists()
