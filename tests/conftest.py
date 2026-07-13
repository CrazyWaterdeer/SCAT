"""Shared pytest fixtures for SCAT tests."""
import numpy as np
import pytest
from PIL import Image, ImageDraw


@pytest.fixture(scope="session")
def synth_dir(tmp_path_factory):
    """Generate a small set of synthetic excreta-like images + a groups CSV.

    Light paper background with darker blue/purple blobs (round 'normal' and
    elongated 'ROD' deposits). Deterministic (seeded) so tests are stable.
    """
    d = tmp_path_factory.mktemp("synth")
    rng = np.random.RandomState(0)

    def make(path, n_round, n_oblong, dpi=600):
        w = h = 512
        img = Image.new("RGB", (w, h), (245, 244, 240))
        draw = ImageDraw.Draw(img)
        for _ in range(n_round):
            cx, cy = rng.randint(40, w - 40, 2)
            r = int(rng.randint(8, 16))
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(50, 60, 180))
        for _ in range(n_oblong):
            cx, cy = rng.randint(50, w - 50, 2)
            a = int(rng.randint(18, 30))
            b = int(rng.randint(5, 9))
            draw.ellipse([cx - a, cy - b, cx + a, cy + b], fill=(90, 50, 150))
        img.save(path, dpi=(dpi, dpi))

    for i in range(3):
        make(d / f"ctrl_{i}.tif", n_round=10, n_oblong=3)
    for i in range(3):
        make(d / f"treat_{i}.tif", n_round=5, n_oblong=8)

    rows = ["filename,group"]
    rows += [f"ctrl_{i}.tif,Control" for i in range(3)]
    rows += [f"treat_{i}.tif,Treatment" for i in range(3)]
    (d / "groups.csv").write_text("\n".join(rows) + "\n")
    return d
