"""Spatial point-pattern analysis (scat/spatial.py): Clark-Evans, NND, quadrant, edge guards."""
import numpy as np

from scat.detector import Deposit
from scat.spatial import SpatialAnalyzer


def _dep(cx, cy, label="normal"):
    return Deposit(id=0, contour=np.zeros((1, 2)), x=int(cx), y=int(cy), width=1, height=1,
                   area=1.0, perimeter=1.0, circularity=1.0, aspect_ratio=1.0,
                   centroid=(int(cx), int(cy)), label=label)


def test_spatial_degenerate_under_two_deposits():
    a = SpatialAnalyzer()
    for pts in ([], [_dep(10, 10)]):
        r = a.analyze(pts, (500, 500))
        assert r.clark_evans_r == 1.0 and r.clustering_interpretation == "insufficient_data"
        assert r.quadrant_uniform is True and r.nnd_mean == 0


def test_spatial_clustered_has_R_below_one():
    a = SpatialAnalyzer()
    rng = np.random.RandomState(0)
    clustered = [_dep(rng.randint(0, 20), rng.randint(0, 20)) for _ in range(12)]  # tiny corner
    r = a.analyze(clustered, (500, 500))
    assert r.clark_evans_r < 1.0 and "clustered" in r.clustering_interpretation


def test_spatial_regular_grid_is_dispersed():
    a = SpatialAnalyzer()
    grid = [_dep(x, y) for x in range(50, 450, 100) for y in range(50, 450, 100)]  # 4x4 lattice
    r = a.analyze(grid, (500, 500))
    assert r.clark_evans_r > 1.0 and "dispersed" in r.clustering_interpretation


def test_spatial_nnd_two_points_exact():
    a = SpatialAnalyzer()
    r = a.analyze([_dep(100, 100), _dep(100, 130)], (500, 500))   # 30 px apart
    assert abs(r.nnd_mean - 30.0) < 1e-6 and abs(r.nnd_min - 30.0) < 1e-6


def test_spatial_quadrant_nonuniform_when_all_in_one_quadrant():
    a = SpatialAnalyzer()
    pts = [_dep(x, y) for x in range(10, 60, 10) for y in range(10, 60, 10)]  # 25 pts, one corner
    r = a.analyze(pts, (500, 500))
    assert sum(r.quadrant_counts.values()) == len(pts)
    assert max(r.quadrant_counts.values()) == len(pts)    # all in a single quadrant
    assert not r.quadrant_uniform                         # chi2 rejects uniformity (np.bool_)


def test_spatial_excludes_artifacts():
    a = SpatialAnalyzer()
    pts = [_dep(10, 10), _dep(12, 11), _dep(400, 400, label="artifact")]
    excl = a.analyze(pts, (500, 500), exclude_artifacts=True)
    incl = a.analyze(pts, (500, 500), exclude_artifacts=False)
    assert excl.clark_evans_r != incl.clark_evans_r       # the artifact changes the geometry
