from scat.agent import provenance


def test_record_and_read(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance, "_sessions_dir", lambda: tmp_path)
    sid = provenance.start_session(driver="test")
    provenance.record_call("scan_folder", {"path": "/x"}, {"n_images": 3}, 0.01, ok=True)
    rows = provenance.read_session(sid)
    # first line is the spec §7 session header; then the tagged call record
    assert rows[0]["type"] == "session_header" and rows[0]["driver"] == "test"
    assert "scat_version" in rows[0] and "config" in rows[0]
    calls = [r for r in rows if r.get("type") == "call"]
    assert len(calls) == 1 and calls[0]["tool"] == "scan_folder" and calls[0]["ok"] is True


def test_summarize_collapses_array():
    import numpy as np
    out = provenance._summarize({"img": np.zeros((4, 4)), "items": list(range(50))})
    assert out["img"]["_kind"] == "array" and out["img"]["shape"] == [4, 4]
    assert len(out["items"]) == 20


def test_redact_secrets():
    from scat.agent.provenance import _redact
    out = _redact({"model": "opus", "api_key": "sk-xxx",
                   "nested": {"auth_token": "t", "n": 1}, "list": [{"password": "p"}]})
    assert out["model"] == "opus" and out["api_key"] == "***"
    assert out["nested"]["auth_token"] == "***" and out["nested"]["n"] == 1
    assert out["list"][0]["password"] == "***"
