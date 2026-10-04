import numpy as np

from safeacd.env import (A_FORENSICS, A_ISOLATE, A_RESTORE, BLOCK_BASE,
                         C_CRIT, C_EVID, C_PROD, C_TRAF, LINK_UE, N_ACTIONS,
                         OBS_DIM, OP_SERVER, EnvConfig, SafeACDEnv,
                         host_action)
from safeacd.playbook import RandomAgent


def rollout(env, actions):
    out = []
    for a in actions:
        out.append(env.step(a))
    return out


def test_observation_shape_and_range():
    env = SafeACDEnv(seed=0)
    o = env.reset(seed=1)
    assert o.shape == (OBS_DIM,)
    assert o.min() >= 0.0 and o.max() <= 1.0


def test_determinism_same_seed():
    agent = RandomAgent(0)
    trajs = []
    for _ in range(2):
        env = SafeACDEnv(seed=0)
        env.reset(seed=42, attacker="bline")
        agent.rng = np.random.default_rng(0)
        tr = []
        done = False
        while not done:
            o, r, c, done, _ = env.step(agent.act(env))
            tr.append((r, tuple(c)))
        trajs.append(tr)
    assert trajs[0] == trajs[1]


def test_sleep_has_zero_collateral_cost():
    env = SafeACDEnv(seed=0)
    env.reset(seed=3, attacker="bline")
    total = np.zeros(4)
    done = False
    while not done:
        _, _, c, done, _ = env.step(0)
        total += c
    assert total.sum() == 0.0


def test_restore_without_image_destroys_evidence():
    env = SafeACDEnv(seed=0)
    env.reset(seed=5, attacker="bline")
    u = int(np.flatnonzero(env.tainted)[0])  # initial (hidden) foothold
    _, _, c, _, _ = env.step(host_action(A_RESTORE, u))
    assert c[C_EVID] == 1.0
    assert c[C_PROD] > 0  # host offline while reimaging


def test_image_then_restore_preserves_evidence():
    env = SafeACDEnv(seed=0)
    env.reset(seed=5, attacker="bline")
    u = int(np.flatnonzero(env.tainted)[0])
    env.step(host_action(A_FORENSICS, u))
    _, _, c, _, _ = env.step(host_action(A_RESTORE, u))
    assert c[C_EVID] == 0.0


def test_restore_of_clean_host_is_not_an_evidence_violation():
    env = SafeACDEnv(seed=0)
    env.reset(seed=5, attacker="bline")
    _, _, c, _, _ = env.step(host_action(A_RESTORE, OP_SERVER))
    assert c[C_EVID] == 0.0


def test_isolating_opserver_is_critical_violation():
    env = SafeACDEnv(seed=0)
    env.reset(seed=5)
    _, _, c, _, _ = env.step(host_action(A_ISOLATE, OP_SERVER))
    assert c[C_CRIT] == 1.0
    _, _, c, _, _ = env.step(0)
    assert c[C_CRIT] == 1.0  # counted every step it stays severed


def test_blocking_link_blocks_legitimate_traffic():
    env = SafeACDEnv(seed=0)
    env.reset(seed=5)
    tot = 0.0
    env.step(BLOCK_BASE + LINK_UE)
    for _ in range(20):
        _, _, c, _, _ = env.step(0)
        tot += c[C_TRAF]
    assert tot > 0


def test_valid_mask_shape():
    env = SafeACDEnv(seed=0)
    m = env.valid_mask()
    assert m.shape == (N_ACTIONS,) and m[0]


def test_undefended_bline_reaches_impact():
    env = SafeACDEnv(EnvConfig(), seed=0)
    env.reset(seed=11, attacker="bline")
    done = False
    while not done:
        _, _, _, done, _ = env.step(0)
    assert env.impact_steps > 0
