"""Regression tests for SCAT 2.0 Stage-4 report fixes."""
from pathlib import Path

import pandas as pd

from scat.report import generate_report


def test_user_names_are_html_escaped(tmp_path):
    """[11] Filenames and group names are HTML-escaped, not injected raw."""
    film = pd.DataFrame({
        "filename": ["<script>x</script>.tif", "b&c.tif", "d.tif", "e.tif"],
        "group": ["A<b>", "A<b>", "T&T", "T&T"],
        "n_total": [10, 12, 8, 9], "n_normal": [8, 10, 3, 4], "n_rod": [2, 2, 5, 5],
        "n_artifact": [0, 0, 0, 0], "rod_fraction": [0.2, 0.17, 0.62, 0.56],
        "total_iod": [100.0, 110.0, 90.0, 95.0],
        "normal_mean_hue": [210.0, 211.0, 215.0, 214.0],
        "normal_mean_circularity": [0.8, 0.79, 0.75, 0.76],
    })
    html = Path(generate_report(film, output_dir=str(tmp_path), group_by="group",
                                format="html")).read_text(encoding="utf-8")
    # the raw dangerous strings must not appear verbatim in the film table
    assert "<script>x</script>.tif" not in html
    assert "&lt;script&gt;x&lt;/script&gt;.tif" in html      # escaped instead
    assert "b&amp;c.tif" in html                              # & escaped
