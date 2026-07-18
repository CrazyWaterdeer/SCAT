"""The Assistant's (agent's) analysis results are surfaced in the workspace for review:
chat dock ToolResult -> analysis_ready signal -> AnalysisTab loads the output dir and switches
to the results page (previously the agent produced files but the GUI showed nothing)."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="module")
def app():
    yield QApplication.instance() or QApplication([])


def test_results_dir_from_tool_handles_dict_and_json():
    from scat.agent.chat_widget import _results_dir_from_tool
    # API backend: output is the raw dict
    assert _results_dir_from_tool("analyze_folder", {"output_dir": "/x/out", "n_normal": 5}) == "/x/out"
    # subscription backend: tool results are flattened to a JSON string
    assert _results_dir_from_tool("analyze_folder", json.dumps({"output_dir": "/x/out"})) == "/x/out"
    assert _results_dir_from_tool("combine_results", {"output_dir": "/m"}) == "/m"
    assert _results_dir_from_tool("generate_report", {"report_path": "/x/out/report.html"}) == "/x/out"
    # non-analysis / skipped / unparseable -> nothing to load
    assert _results_dir_from_tool("run_statistics", {"skipped": True}) is None
    assert _results_dir_from_tool("analyze_folder", "not-json") is None


def test_tool_result_emits_analysis_ready(app):
    from scat.agent.chat_widget import ChatDockWidget
    from scat.agent.runner import ToolResult
    w = ChatDockWidget()
    got = []
    w.analysis_ready.connect(got.append)
    w._on_event(ToolResult("tu1", "analyze_folder", {"output_dir": "/tmp/xyz/out"}))
    assert got == ["/tmp/xyz/out"]
    got.clear()
    w._on_event(ToolResult("tu2", "analyze_folder", "boom", is_error=True))   # errored -> no emit
    assert got == []


def test_agent_results_load_into_workspace(app, synth_dir, tmp_path):
    """End-to-end: a real output dir loads into the AnalysisTab results surface and it switches
    to the results page (this is what was missing for the agent flow)."""
    from scat.pipeline import analyze_folder_service
    from scat.main_gui import AnalysisTab
    res = analyze_folder_service(str(synth_dir), output_dir=str(tmp_path / "out"), annotate=False)

    tab = AnalysisTab()
    assert tab.stack.currentWidget() is tab._configure_page   # starts on Configure
    tab.load_results_from_dir(res.output_dir)
    assert tab.results_view.results is not None               # results actually loaded
    assert tab.stack.currentWidget() is tab._results_page      # switched to the results surface
    assert "Assistant analyzed" in tab.results_bar_label.text()
