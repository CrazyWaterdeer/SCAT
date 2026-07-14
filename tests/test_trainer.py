"""RFTrainer round-trip (scat/trainer.py): train -> save -> load -> predict."""
import numpy as np

from scat.trainer import RFTrainer

_NAMES = ['circularity', 'aspect_ratio', 'area', 'mean_hue', 'mean_saturation', 'mean_lightness', 'iod']


def _feat(rng, **over):
    f = {n: float(rng.rand()) for n in _NAMES}
    f.update(over)
    return f


def test_rf_trainer_train_save_load_predict(tmp_path):
    rng = np.random.RandomState(0)
    feats, labels = [], []
    for _ in range(15):   # round, low-aspect -> 'normal'
        feats.append(_feat(rng, circularity=0.9 + 0.05 * rng.rand(), aspect_ratio=1.0 + 0.1 * rng.rand()))
        labels.append("normal")
    for _ in range(15):   # elongated, low-circularity -> 'rod'
        feats.append(_feat(rng, circularity=0.3 + 0.05 * rng.rand(), aspect_ratio=3.0 + 0.3 * rng.rand()))
        labels.append("rod")

    t = RFTrainer()
    res = t.train(feats, labels, n_estimators=25, cross_validate=False)
    assert res["accuracy"] >= 0.8 and res["train_size"] + res["test_size"] == 30
    assert set(res["feature_importance"]) == set(_NAMES)

    path = tmp_path / "rf.pkl"
    t.save(path)
    t2 = RFTrainer()
    t2.load(path)
    assert t2.feature_names == t.feature_names

    pred, conf = t2.predict([_feat(rng, circularity=0.92, aspect_ratio=1.05)])
    assert pred[0] == "normal" and 0.0 <= conf[0] <= 1.0
