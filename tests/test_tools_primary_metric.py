import dataclasses

import scat.tools.pipeline_tools as pt


def test_tool_forwards_primary_metric(monkeypatch, synth_dir):
    seen = {}

    # analyze_folder wraps the service return in dataclasses.asdict(...), so the fake
    # must return a dataclass instance (the real return type is AnalyzeResult).
    @dataclasses.dataclass
    class R:
        output_dir: str = "/tmp/x"

    def fake(path, **kw):
        seen.update(kw)
        return R()

    monkeypatch.setattr(pt, "analyze_folder_service", fake)
    pt.analyze_folder(str(synth_dir), primary_metric="mean_area")
    assert seen.get("primary_metric") == "mean_area"
