import numpy as np
import pytest

from safeacd.evaluate import run_episodes
from safeacd.ppo import METHODS, DualAdam, TrainConfig, TrainedPolicy, train


def test_dual_adam_projects_and_masks():
    d = DualAdam(4, lr=0.1, init=0.0, mask=np.array([True, True, False, False]))
    for _ in range(5):
        lam = d.step(np.array([1.0, -1.0, 5.0, 5.0]))
    assert lam[0] > 0 and lam[1] == 0.0      # projected onto lambda >= 0
    assert lam[2] == 0.0 and lam[3] == 0.0   # unconstrained channels stay 0


@pytest.mark.parametrize("method", METHODS)
def test_training_smoke(tmp_path, method):
    cfg = TrainConfig(method=method, seed=0, total_steps=2048, n_envs=2, n_steps=512,
                      epochs=1, minibatch=256, hidden=32, out_dir=str(tmp_path))
    out = train(cfg, verbose=False)
    assert (out / "model.pt").exists() and (out / "train_log.csv").exists()
    pol = TrainedPolicy(out)
    rows = run_episodes(pol, "bline", 2)
    assert len(rows) == 2 and "cost_evidence" in rows[0]
