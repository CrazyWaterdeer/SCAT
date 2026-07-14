from scat.agent import provenance


def test_record_and_read(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance, "_sessions_dir", lambda: tmp_path)
    sid = provenance.start_session(driver="test")
    provenance.record_call("scan_folder", {"path": "/x"}, {"n_images": 3}, 0.01, ok=True)
    rows = provenance.read_session(sid)
    assert len(rows) == 1 and rows[0]["tool"] == "scan_folder" and rows[0]["ok"] is True


def test_summarize_collapses_array():
    import numpy as np
    out = provenance._summarize({"img": np.zeros((4, 4)), "items": list(range(50))})
    assert out["img"]["_kind"] == "array" and out["img"]["shape"] == [4, 4]
    assert len(out["items"]) == 20
