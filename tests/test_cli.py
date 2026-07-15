"""CLI coverage — the scriptable entry point (analyze/chat/train/label/gui) had none.

Drives scat.cli.main() (which parses sys.argv) so the argparse wiring and the headless
`analyze` path are verified, and each subcommand's dispatch to its handler is pinned.
"""
import sys

import pytest

import scat.cli as cli


def _run(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["scat", *argv])
    cli.main()


def test_analyze_end_to_end(synth_dir, tmp_path, monkeypatch):
    """`scat analyze <folder> -m groups.csv --stats --report` runs on the shared services."""
    out = tmp_path / "out"
    _run(monkeypatch, ["analyze", str(synth_dir), "-o", str(out),
                       "-m", str(synth_dir / "groups.csv"), "--stats", "--report"])
    assert (out / "image_summary.csv").exists()
    assert (out / "all_deposits.csv").exists()
    assert (out / "report.html").exists()


def test_cluster_and_propagate_roundtrip(synth_dir, tmp_path, monkeypatch, capsys):
    """`scat cluster` → fill labels → `scat propagate` drives the labeling-assist flow."""
    import pandas as pd
    out = tmp_path / "clus"
    _run(monkeypatch, ["cluster", str(synth_dir), "--output", str(out), "--min-cluster-size", "3"])
    assert (out / "cluster_labels.csv").exists() and (out / "cluster_report.html").exists()
    cl = pd.read_csv(out / "cluster_labels.csv")
    if len(cl) < 2:
        pytest.skip("synth data produced <2 clusters")
    cl["label"] = cl["label"].astype(object)
    cl.loc[0, "label"] = "normal"
    cl.loc[1, "label"] = "rod"
    cl.to_csv(out / "cluster_labels.csv", index=False)
    capsys.readouterr()  # clear
    _run(monkeypatch, ["propagate", str(out)])
    txt = capsys.readouterr().out.lower()
    assert "labeled" in txt and "readiness" in txt


@pytest.mark.parametrize("argv,funcname", [
    (["analyze", "somewhere"], "analyze_command"),
    (["chat"], "chat_command"),
    (["train", "--image-dir", "imgs", "--output", "m.pkl"], "train_command"),
    (["label"], "label_command"),
    (["gui"], "gui_command"),
])
def test_subcommand_dispatch(monkeypatch, argv, funcname):
    """Each subcommand routes to its handler (set_defaults(func=...) resolves the module global)."""
    seen = {}
    monkeypatch.setattr(cli, funcname, lambda args: seen.setdefault("args", args))
    _run(monkeypatch, argv)
    assert "args" in seen, f"{argv} did not dispatch to {funcname}"


def test_no_command_prints_help_and_exits(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["scat"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1
