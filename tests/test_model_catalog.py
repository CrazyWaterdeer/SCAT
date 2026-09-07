"""The model catalog: what the picker offers and how the ``latest`` sentinel resolves."""

import json
import time

import pytest

from scat.agent import model_catalog as mc


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    """Never read or write the real ~/.scat catalog cache from a test."""
    monkeypatch.setattr(mc, "_cache_path", lambda: tmp_path / "model_catalog.json")
    monkeypatch.setattr(mc, "_api_key", lambda explicit=None: (explicit or "").strip())
    yield


def _install_fake_sdk(monkeypatch, models):
    """Stand in for the anthropic SDK so fetch_live_models can be exercised offline."""
    import sys

    class _Info:
        def __init__(self, mid, name):
            self.id, self.display_name = mid, name

    class _Client:
        def __init__(self, **kw):
            self.models = self

        def list(self, limit=100):
            return type("Page", (), {"data": [_Info(mid, name) for mid, name in models]})

    monkeypatch.setitem(sys.modules, "anthropic", type("m", (), {"Anthropic": _Client}))


# ---------------------------------------------------------------- ordering / naming
def test_display_name_and_ordering():
    assert mc.display_name("claude-opus-4-8") == "Opus 4.8"
    assert mc.display_name("claude-opus-5") == "Opus 5"
    assert mc.display_name("gpt-9") == "gpt-9"          # unknown ids pass through unchanged
    ids = ["claude-haiku-4-5", "claude-opus-4-8", "claude-sonnet-5", "claude-opus-5"]
    assert sorted(ids, key=mc._sort_key) == [
        "claude-opus-5", "claude-opus-4-8", "claude-sonnet-5", "claude-haiku-4-5"]


def test_legacy_ids_are_not_offered_but_dated_current_ids_are():
    for legacy in ("claude-3-5-sonnet-20241022", "claude-2", "gpt-5"):
        assert mc._parse(legacy) is None
    # Some current models ship a dated canonical id — dropping those would hide a live model.
    assert mc._parse("claude-haiku-4-5-20251001") == ("haiku", 4, 5, 20251001)
    assert mc.display_name("claude-haiku-4-5-20251001") == "Haiku 4.5"


# ---------------------------------------------------------------- resolution
def test_pinned_model_passes_through_untouched():
    assert mc.resolve_model("claude-haiku-4-5", backend="api") == "claude-haiku-4-5"
    assert mc.resolve_model("claude-haiku-4-5", backend="subscription") == "claude-haiku-4-5"


def test_auto_resolves_to_the_cli_alias_on_the_subscription_path():
    # The `claude` CLI resolves the alias to the newest Opus itself — that is what keeps a
    # subscription user current with no API key and no catalog fetch.
    assert mc.resolve_model(mc.AUTO, backend="subscription") == "opus"


def test_auto_resolves_to_the_newest_catalog_model_on_the_api_path(monkeypatch):
    monkeypatch.setattr(mc, "fetch_live_models",
                        lambda key, timeout=10.0: [("Opus 9.1", "claude-opus-9-1"),
                                                   ("Sonnet 9", "claude-sonnet-9")])
    assert mc.resolve_model(mc.AUTO, backend="api", api_key="sk-x") == "claude-opus-9-1"


def test_auto_falls_back_to_the_static_list_without_a_key_or_network(monkeypatch):
    monkeypatch.setattr(mc, "fetch_live_models",
                        lambda key, timeout=10.0: (_ for _ in ()).throw(RuntimeError("offline")))
    assert mc.resolve_model(mc.AUTO, backend="api", api_key="sk-x") == mc.FALLBACK_MODELS[0][1]
    assert mc.resolve_model(mc.AUTO, backend="api", api_key="") == mc.FALLBACK_MODELS[0][1]


def test_describe_model_marks_an_auto_pick():
    assert mc.describe_model(mc.AUTO, "claude-opus-5") == "claude-opus-5 (latest)"
    assert mc.describe_model("claude-haiku-4-5", "claude-haiku-4-5") == "claude-haiku-4-5"


# ---------------------------------------------------------------- catalog + cache
def test_live_catalog_is_cached_and_reused(monkeypatch):
    calls = []

    def _fetch(key, timeout=10.0):
        calls.append(key)
        return [("Opus 9", "claude-opus-9")]

    monkeypatch.setattr(mc, "fetch_live_models", _fetch)
    assert mc.catalog(api_key="sk-x") == [("Opus 9", "claude-opus-9")]
    assert mc.catalog(api_key="sk-x") == [("Opus 9", "claude-opus-9")]
    assert len(calls) == 1                      # second call served from the disk cache


def test_stale_cache_is_refetched(monkeypatch):
    mc.save_cache([("Opus 8", "claude-opus-8")])
    raw = json.loads(mc._cache_path().read_text())
    raw["fetched_at"] = time.time() - 10 * 24 * 3600      # ten days old
    mc._cache_path().write_text(json.dumps(raw))
    monkeypatch.setattr(mc, "fetch_live_models", lambda key, timeout=10.0: [("Opus 9", "claude-opus-9")])
    assert mc.catalog(api_key="sk-x") == [("Opus 9", "claude-opus-9")]


def test_corrupt_cache_is_ignored(monkeypatch):
    mc._cache_path().write_text("{not json")
    assert mc.catalog(api_key="", allow_fetch=False) == list(mc.FALLBACK_MODELS)


def test_cache_holding_junk_ids_is_ignored():
    mc._cache_path().write_text(json.dumps({"fetched_at": time.time(),
                                            "models": [["Evil", "not-a-claude-model"]]}))
    assert mc.catalog(api_key="", allow_fetch=False) == list(mc.FALLBACK_MODELS)


def test_cache_writes_are_atomic(monkeypatch):
    """A crash mid-write must never leave a truncated file that reads as a valid short catalog."""
    real_replace = mc.os.replace
    monkeypatch.setattr(mc.os, "replace",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("crash")))
    mc.save_cache([("Opus 9", "claude-opus-9")])
    assert not mc._cache_path().exists()                   # nothing half-written left behind
    assert list(mc._cache_path().parent.glob("*.tmp")) == []   # and the temp file is cleaned up
    monkeypatch.setattr(mc.os, "replace", real_replace)
    mc.save_cache([("Opus 9", "claude-opus-9")])
    assert mc.load_cache() == [("Opus 9", "claude-opus-9")]


def test_stale_cache_beats_the_static_fallback_when_the_api_is_down(monkeypatch):
    """A catalog that was true yesterday is better evidence than a constant from release day."""
    mc.save_cache([("Opus 9", "claude-opus-9")])
    raw = json.loads(mc._cache_path().read_text())
    raw["fetched_at"] = time.time() - 10 * 24 * 3600
    mc._cache_path().write_text(json.dumps(raw))
    monkeypatch.setattr(mc, "fetch_live_models",
                        lambda key, timeout=6.0: (_ for _ in ()).throw(RuntimeError("offline")))
    assert mc.catalog(api_key="sk-x") == [("Opus 9", "claude-opus-9")]
    assert mc.resolve_model(mc.AUTO, backend="api", api_key="sk-x") == "claude-opus-9"


def test_auto_never_falls_through_to_another_family(monkeypatch):
    """A catalog without the auto family must not silently promote a pricier tier."""
    monkeypatch.setattr(mc, "fetch_live_models",
                        lambda key, timeout=6.0: [("Fable 9", "claude-fable-9"),
                                                  ("Sonnet 9", "claude-sonnet-9")])
    resolved = mc.resolve_model(mc.AUTO, backend="api", api_key="sk-x")
    assert mc._parse(resolved)[0] == mc.AUTO_FAMILY
    assert resolved in [m for _n, m in mc.FALLBACK_MODELS]


def test_prefetch_is_a_noop_without_a_key(monkeypatch):
    calls = []
    monkeypatch.setattr(mc, "fetch_live_models", lambda key, timeout=6.0: calls.append(key) or [])
    mc.prefetch(api_key="")
    assert calls == []


def test_available_models_puts_auto_first():
    entries = mc.available_models(allow_fetch=False)
    assert entries[0] == (mc.AUTO_LABEL, mc.AUTO)
    assert entries[1:] == list(mc.FALLBACK_MODELS)


def test_fetch_live_models_prefers_the_undated_alias(monkeypatch):
    """Undated and dated ids for one release collapse to a single entry (the undated alias)."""
    _install_fake_sdk(monkeypatch, [("claude-haiku-4-5-20251001", "Claude Haiku 4.5"),
                                    ("claude-haiku-4-5", "Claude Haiku 4.5"),
                                    ("claude-opus-9-20260901", "Claude Opus 9")])
    assert mc.fetch_live_models("sk-x") == [("Opus 9", "claude-opus-9-20260901"),
                                            ("Haiku 4.5", "claude-haiku-4-5")]


def test_fetch_live_models_filters_and_sorts(monkeypatch):
    _install_fake_sdk(monkeypatch, [("claude-3-5-sonnet-20241022", "Claude Sonnet 3.5"),  # dropped
                                    ("claude-sonnet-9", "Claude Sonnet 9"),
                                    ("claude-opus-9-1", "Claude Opus 9.1")])
    assert mc.fetch_live_models("sk-x") == [("Opus 9.1", "claude-opus-9-1"),
                                            ("Sonnet 9", "claude-sonnet-9")]
