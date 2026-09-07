"""Config merge/reset must never mutate the module-global DEFAULT_CONFIG (audit-found bug)."""
import copy

from scat import config as cfgmod


def test_merge_defaults_does_not_mutate_global():
    before = copy.deepcopy(cfgmod.DEFAULT_CONFIG)
    merged = cfgmod.config._merge_defaults({"agent": {"max_loops": 12345}})
    assert merged["agent"]["max_loops"] == 12345                 # nested override applied to the result
    assert cfgmod.DEFAULT_CONFIG["agent"]["max_loops"] == 40     # module global untouched
    assert cfgmod.DEFAULT_CONFIG == before                      # nothing else drifted either


def test_reset_shortcuts_uses_pristine_defaults():
    before = copy.deepcopy(cfgmod.DEFAULT_CONFIG["shortcuts"])
    cfgmod.config._data.setdefault("shortcuts", {})["__probe__"] = "x"
    cfgmod.config.reset_shortcuts()
    assert "__probe__" not in cfgmod.config._data["shortcuts"]
    assert cfgmod.DEFAULT_CONFIG["shortcuts"] == before          # reset didn't alias/mutate the defaults


def test_legacy_pinned_model_migrates_to_auto_latest_once():
    """A config still carrying the old hardcoded default moves to the auto-latest sentinel — but
    only once, so re-pinning that model afterwards sticks."""
    migrated = cfgmod.config._merge_defaults({"agent": {"model": "claude-opus-4-8"}})
    assert migrated["agent"]["model"] == "latest"
    assert migrated["schema_version"] == cfgmod.SCHEMA_VERSION

    already = cfgmod.config._merge_defaults(
        {"schema_version": cfgmod.SCHEMA_VERSION, "agent": {"model": "claude-opus-4-8"}})
    assert already["agent"]["model"] == "claude-opus-4-8"        # deliberate pin survives


def test_other_pinned_models_are_never_migrated():
    merged = cfgmod.config._merge_defaults({"agent": {"model": "claude-haiku-4-5"}})
    assert merged["agent"]["model"] == "claude-haiku-4-5"


def test_invalid_schema_version_does_not_crash_the_load():
    """A hand-edited or garbage schema_version must degrade to 'pre-versioning', not raise."""
    for bad in (None, "one", {"v": 1}, []):
        merged = cfgmod.config._merge_defaults({"schema_version": bad,
                                               "agent": {"model": "claude-opus-4-8"}})
        assert merged["agent"]["model"] == "latest"
        assert merged["schema_version"] == cfgmod.SCHEMA_VERSION


def test_migration_persists_so_it_runs_once(tmp_path, monkeypatch):
    """End-to-end: a legacy config on disk is migrated *and saved*, so re-pinning the old model
    afterwards survives the next launch."""
    import json
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"agent": {"model": "claude-opus-4-8"}}), encoding="utf-8")
    monkeypatch.setattr(cfgmod, "get_config_path", lambda: path)
    original, original_instance = cfgmod.config, cfgmod.Config._instance

    def _fresh():
        cfgmod.Config._instance = None
        return cfgmod.Config()

    try:
        cfg = _fresh()
        assert cfg.get("agent.model") == "latest"
        assert json.loads(path.read_text())["schema_version"] == cfgmod.SCHEMA_VERSION

        cfg.set("agent.model", "claude-opus-4-8")              # deliberate pin, saved
        assert _fresh().get("agent.model") == "claude-opus-4-8"  # not migrated a second time
    finally:
        cfgmod.Config._instance, cfgmod.config = original_instance, original
