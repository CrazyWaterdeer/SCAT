"""Significance-bracket policy: which comparisons get drawn is a CHOICE (mode), not hardcoded pairwise."""
import numpy as np
import pandas as pd
import pytest

import scat.visualization as _viz
from scat.visualization import guess_control_group, Visualizer

_viz._load_viz_libs()   # flags are set lazily; force the load so skipif reflects reality
_HAS_VIZ = _viz.HAS_MATPLOTLIB and _viz.HAS_SEABORN


def test_guess_control_group():
    assert guess_control_group(["Treated", "Control", "DoseA"]) == "Control"
    assert guess_control_group(["WT", "mut1", "mut2"]) == "WT"           # wildtype
    assert guess_control_group(["vehicle_1", "drug"]) == "vehicle_1"     # prefix match
    assert guess_control_group(["A", "B", "C"]) is None                  # no reference group


@pytest.mark.skipif(not _HAS_VIZ, reason="viz libs missing")
def test_violin_all_significance_modes_render(tmp_path):
    rng = np.random.RandomState(1)
    df = pd.DataFrame([{"group": g, "n_total": base + rng.randint(0, 4)}
                       for g, base in [("Control", 5), ("DoseA", 10), ("DoseB", 15)]
                       for _ in range(6)])
    viz = Visualizer(tmp_path)
    for mode in ("auto", "vs_control", "adjacent", "pairwise", "none"):
        out = viz.violin_comparison(df, "n_total", "group", control_group="Control",
                                    show_significance=True, significance_mode=mode,
                                    filename=f"v_{mode}.png")
        assert out and (tmp_path / f"v_{mode}.png").exists()


@pytest.mark.skipif(not _HAS_VIZ, reason="viz libs missing")
def test_significance_bracket_counts_by_mode(tmp_path):
    """vs_control draws k-1 brackets; pairwise draws C(k,2); none draws 0 — counted as ax.lines."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.RandomState(2)
    # 4 groups, well separated so every comparison is significant (so counts reflect the MODE, not p)
    df = pd.DataFrame([{"group": g, "y": base + rng.randint(0, 2)}
                       for g, base in [("Control", 0), ("A", 20), ("B", 40), ("C", 60)]
                       for _ in range(6)])
    groups = ["Control", "A", "B", "C"]
    viz = Visualizer(tmp_path)

    def bracket_lines(mode):
        fig, ax = plt.subplots()
        viz._add_significance_annotations(ax, df, "y", "group", groups,
                                          mode=mode, control_group="Control", show_ns=True)
        n = len(ax.lines)
        plt.close(fig)
        return n

    assert bracket_lines("none") == 0
    assert bracket_lines("vs_control") == 3          # each of A/B/C vs Control
    assert bracket_lines("pairwise") == 6            # C(4,2)
    assert bracket_lines("adjacent") == 3            # Control-A, A-B, B-C


def test_order_groups_logical_not_alphabetical():
    from scat.visualization import order_groups
    # ordinal words: control first, then low<mid<high (NOT alphabetical Control/High/Low/Mid)
    assert order_groups(["DoseHigh", "Control", "DoseLow", "DoseMid"]) == \
        ["Control", "DoseLow", "DoseMid", "DoseHigh"]
    # numeric doses sort by value, not lexically (0.5 before 2 before 10)
    assert order_groups(["10uM", "0uM", "2uM", "0.5uM"]) == ["0uM", "0.5uM", "2uM", "10uM"]
    # explicit control_group honored + put first
    assert order_groups(["B", "A", "Ref"], control_group="Ref")[0] == "Ref"
    # no ordinal / no number -> appearance order preserved (not force-alphabetized)
    assert order_groups(["gamma", "alpha", "beta"]) == ["gamma", "alpha", "beta"]


def test_get_palette_no_control_collision():
    from scat.visualization import get_palette, CONTROL_COLOR
    pal = get_palette(["Vehicle", "Drug", "Light"], control_group="Vehicle")
    assert pal["Vehicle"] == CONTROL_COLOR
    assert pal["Drug"] != CONTROL_COLOR and pal["Light"] != CONTROL_COLOR   # no second slate
    assert len({pal["Vehicle"], pal["Drug"], pal["Light"]}) == 3            # all distinct


@pytest.mark.skipif(not _HAS_VIZ, reason="viz libs missing")
def test_draw_condition_matrix_open_closed_circles():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    from scat.visualization import draw_condition_matrix
    groups = ["Veh", "Drug", "Light", "Both"]
    matrix = {"Drug":  {"Veh": False, "Drug": True,  "Light": False, "Both": True},
              "Light": {"Veh": False, "Drug": False, "Light": True,  "Both": True}}
    fig, ax = plt.subplots()
    n = draw_condition_matrix(ax, range(len(groups)), matrix, groups)
    circles = [ln for ln in ax.lines if ln.get_marker() == 'o']
    filled = [ln for ln in circles if to_rgba(ln.get_markerfacecolor()) != to_rgba('white')]
    assert n == 2 and len(circles) == 8          # 2 factors x 4 groups
    assert len(filled) == 4                       # 4 truthy cells filled, 4 open
    plt.close(fig)


@pytest.mark.skipif(not _HAS_VIZ, reason="viz libs missing")
def test_condition_comparison_renders(tmp_path):
    import pandas as pd
    from scat.visualization import Visualizer
    df = pd.DataFrame([{"group": g, "n_total": base}
                       for g, base in [("Vehicle", 8), ("Drug", 12), ("Light", 9), ("Drug+Light", 18)]
                       for _ in range(5)])
    matrix = {"Drug":  {"Vehicle": False, "Drug": True,  "Light": False, "Drug+Light": True},
              "Light": {"Vehicle": False, "Drug": False, "Light": True,  "Drug+Light": True}}
    out = Visualizer(tmp_path).condition_comparison(df, "n_total", "group", matrix,
                                                    control_group="Vehicle", filename="c.png")
    assert out and (tmp_path / "c.png").exists()
