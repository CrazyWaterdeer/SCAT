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
