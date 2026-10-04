import numpy as np
import pytest

from safeacd.evaluate import run_episodes
from safeacd.ppo import (METHODS, DualAscent, TrainConfig, TrainedPolicy,
                         make_actor_critic, train)
from safeacd.env import N_ACTIONS, OBS_DIM, SafeACDEnv


def test_dual_ascent_projects_and_masks():
    d = DualAscent(4, lr=0.1, init=0.0, mask=np.array([True, True, False, False]))
    for _ in range(5):
        lam = d.step(np.array([1.0, -1.0, 5.0, 5.0]))
    assert lam[0] > 0 and lam[1] == 0.0      # projected onto lambda >= 0
    assert lam[2] == 0.0 and lam[3] == 0.0   # unconstrained channels stay 0


@pytest.mark.parametrize("arch", ["entity", "mlp"])
def test_actor_critic_shapes(arch):
    import torch
    ac = make_actor_critic(arch, 32)
    env = SafeACDEnv(seed=0)
    o = torch.as_tensor(np.stack([env.reset(seed=i) for i in range(3)]))
    assert ac.logits(o).shape == (3, N_ACTIONS)
    assert ac.values(o).shape == (3, 5)
    assert o.shape[1] == OBS_DIM


def test_entity_logits_follow_action_indexing():
    """Permuting two same-type hosts must permute their action logits."""
    import torch
    from safeacd.env import OBS_PER_HOST, host_action
    torch.manual_seed(0)
    ac = make_actor_critic("entity", 32)
    env = SafeACDEnv(seed=0)
    o = env.reset(seed=1).copy()
    o[1 * OBS_PER_HOST + 0] = 0; o[1 * OBS_PER_HOST + 2] = 1   # User1 privileged ticket
    o2 = o.copy()
    a = slice(1 * OBS_PER_HOST, 2 * OBS_PER_HOST)
    b = slice(2 * OBS_PER_HOST, 3 * OBS_PER_HOST)
    o2[a], o2[b] = o[b].copy(), o[a].copy()                     # swap User1 <-> User2
    l1 = ac.logits(torch.as_tensor(o)[None])[0]
    l2 = ac.logits(torch.as_tensor(o2)[None])[0]
    for k in range(6):
        assert torch.allclose(l1[host_action(k, 1)], l2[host_action(k, 2)], atol=1e-6)


@pytest.mark.parametrize("method", METHODS)
def test_training_smoke(tmp_path, method):
    cfg = TrainConfig(method=method, seed=0, total_steps=2048, n_envs=2, n_steps=512,
                      epochs=1, minibatch=256, hidden=32, out_dir=str(tmp_path))
    assert cfg.arch == "entity"
    out = train(cfg, verbose=False)
    assert (out / "model.pt").exists() and (out / "train_log.csv").exists()
    pol = TrainedPolicy(out)
    rows = run_episodes(pol, "bline", 2)
    assert len(rows) == 2 and "cost_evidence" in rows[0]
