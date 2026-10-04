import numpy as np

from safeacd.env import (A_ISOLATE, A_RESTORE, OP_SERVER, SafeACDEnv,
                         host_action)
from safeacd.shield import Shield


def test_shield_blocks_isolating_opserver():
    env = SafeACDEnv(seed=0)
    env.reset(seed=1)
    assert not Shield().safe_mask(env)[host_action(A_ISOLATE, OP_SERVER)]


def test_shield_blocks_restore_without_image():
    env = SafeACDEnv(seed=0)
    env.reset(seed=1)
    assert not Shield().safe_mask(env)[host_action(A_RESTORE, 0)]


def test_shielded_random_agent_never_violates_critical_rule():
    """H2 is enforced exactly; H1 residuals must be rare (monitor-relative)."""
    shield = Shield()
    rng = np.random.default_rng(0)
    env = SafeACDEnv(seed=0)
    crit = evid = restores = 0
    for ep in range(60):
        env.reset(seed=100 + ep, attacker=("bline", "meander", "stealthy")[ep % 3])
        done = False
        while not done:
            m = env.valid_mask() & shield.safe_mask(env)
            a = int(rng.choice(np.flatnonzero(m)))
            if (a - 1) // 12 == A_RESTORE and a >= 1:
                restores += 1
            _, _, c, done, _ = env.step(a)
            crit += c[3]
            evid += c[2]
    assert crit == 0
    assert restores > 0
    assert evid / restores < 0.05


def test_unshielded_random_agent_does_violate():
    rng = np.random.default_rng(0)
    env = SafeACDEnv(seed=0)
    total = np.zeros(4)
    for ep in range(10):
        env.reset(seed=200 + ep)
        done = False
        while not done:
            a = int(rng.choice(np.flatnonzero(env.valid_mask())))
            _, _, c, done, _ = env.step(a)
            total += c
    assert total[2] > 0 and total[3] > 0
