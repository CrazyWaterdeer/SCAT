"""Tests for the unsupervised clustering core (scat/clustering.py) and its pipeline services."""
import numpy as np
import pandas as pd
import pytest

from scat import clustering as C


def _df(n=30, seed=0):
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "filename": [f"img_{i % 3}.tif" for i in range(n)],
        "deposit_id": list(range(n)),
        "area_px": rng.randint(30, 500, n).astype(float),
        "perimeter": rng.randint(20, 200, n).astype(float),
        "circularity": rng.uniform(0.2, 1.0, n),
        "aspect_ratio": rng.uniform(1.0, 3.0, n),
        "mean_hue": rng.uniform(0, 360, n),
        "mean_saturation": rng.uniform(0, 1, n),
        "mean_lightness": rng.uniform(0, 1, n),
        "pigment_density": rng.uniform(0, 1, n),
        "iod": rng.uniform(0, 300, n),
    })


def _blobs():
    rng = np.random.RandomState(1)
    centers = np.array([[0, 0], [8, 8], [-8, 8]], dtype=float)
    return np.vstack([c + rng.randn(40, 2) * 0.3 for c in centers])


# --------------------------------------------------------------- feature matrix
def test_build_feature_matrix_shape_and_hue_encoding():
    df = _df()
    X, names = C.build_feature_matrix(df)
    assert X.shape[0] == len(df)
    assert "mean_hue" not in names
    assert "hue_sin" in names and "hue_cos" in names
    assert np.isfinite(X).all()


def test_build_feature_matrix_median_impute():
    df = _df()
    df.loc[0, "iod"] = np.nan
    X, _ = C.build_feature_matrix(df)
    assert np.isfinite(X).all()


# --------------------------------------------------------------- clustering
def test_cluster_deposits_hdbscan_finds_blobs():
    res = C.cluster_deposits(_blobs(), min_cluster_size=10)
    assert res.method == "hdbscan"
    assert res.n_clusters == 3
    assert res.labels.shape[0] == 120


def test_cluster_deposits_health_flags():
    rng = np.random.RandomState(2)
    X = rng.randn(200, 2) * 0.2
    res = C.cluster_deposits(X, min_cluster_size=10)
    assert any("cluster" in h or "noise" in h for h in res.health)


def test_cluster_deposits_kmeans_labels_everything():
    res = C.cluster_deposits(_blobs(), method="kmeans", k=3)
    assert res.method == "kmeans" and res.n_clusters == 3 and res.n_noise == 0


# --------------------------------------------------------------- representatives
def test_representatives_three_kinds_medoid_closer_than_boundary():
    X = _blobs()
    res = C.cluster_deposits(X, min_cluster_size=10)
    reps = C.representatives(X, res.labels, per_kind=3)
    for cid, kinds in reps.items():
        assert cid != -1
        assert set(kinds) == {"medoid", "random", "boundary"}
        members = np.where(res.labels == cid)[0]
        centroid = X[members].mean(axis=0)
        d_med = np.linalg.norm(X[kinds["medoid"][0]] - centroid)
        d_bnd = np.linalg.norm(X[kinds["boundary"][0]] - centroid)
        assert d_med <= d_bnd


# --------------------------------------------------------------- profile
def test_cluster_profile_accounts_for_all_deposits():
    df = _df(60)
    X, _ = C.build_feature_matrix(df)
    res = C.cluster_deposits(X, min_cluster_size=10)
    prof = C.cluster_profile(df, res.labels)
    assert "cluster_id" in prof.columns and "size" in prof.columns
    assert prof["size"].sum() == len(df)
    assert "area_px" in prof.columns


# --------------------------------------------------------------- CSV + propagation
def test_parse_cluster_labels_csv_validates(tmp_path):
    p = tmp_path / "cl.csv"
    p.write_text("cluster_id,size,label\n0,10, Normal \n1,5,rod\n2,3,\n")
    assert C.parse_cluster_labels_csv(p) == {0: "normal", 1: "rod"}


def test_parse_cluster_labels_csv_rejects_bad_label(tmp_path):
    p = tmp_path / "cl.csv"
    p.write_text("cluster_id,label\n0,banana\n")
    with pytest.raises(ValueError):
        C.parse_cluster_labels_csv(p)


def test_parse_cluster_labels_csv_rejects_duplicate(tmp_path):
    p = tmp_path / "cl.csv"
    p.write_text("cluster_id,label\n0,normal\n0,rod\n")
    with pytest.raises(ValueError):
        C.parse_cluster_labels_csv(p)


def test_propagate_labels_maps_and_defaults_unknown():
    asg = pd.DataFrame({"filename": ["a", "a", "b"], "deposit_id": [0, 1, 0],
                        "cluster_id": [0, -1, 1]})
    labels, summary = C.propagate_labels(asg, {0: "normal", 1: "rod"})
    assert labels[("a", 0)] == "normal"
    assert labels[("a", 1)] == "unknown"
    assert labels[("b", 0)] == "rod"
    assert summary["n_labeled"] == 2 and summary["n_skipped"] == 1


def test_propagate_unknown_cluster_id_defaults_unknown():
    asg = pd.DataFrame({"filename": ["a"], "deposit_id": [0], "cluster_id": [7]})
    labels, summary = C.propagate_labels(asg, {0: "normal"})
    assert labels[("a", 0)] == "unknown" and summary["n_labeled"] == 0


# --------------------------------------------------------------- readiness guard
def test_training_readiness_blocks_one_class():
    rep = C.training_readiness(["normal"] * 10)
    assert rep.verdict == "block" and any("class" in r for r in rep.reasons)


def test_training_readiness_blocks_singleton_class():
    rep = C.training_readiness(["normal"] * 5 + ["rod"])
    assert rep.verdict == "block"


def test_training_readiness_ok():
    rep = C.training_readiness(["normal"] * 20 + ["rod"] * 20 + ["artifact"] * 10)
    assert rep.verdict in ("ok", "warn")
    assert rep.class_counts == {"normal": 20, "rod": 20, "artifact": 10}


def test_all_noise_clustering_profile_handled():
    rng = np.random.RandomState(9)
    X = rng.randn(50, 2) * 5
    res = C.cluster_deposits(X, min_cluster_size=25)
    prof = C.cluster_profile(_df(50), res.labels)
    assert prof["size"].sum() == 50
