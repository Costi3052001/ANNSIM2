"""Evaluation harness: common-random-number episodes and per-episode records."""
from __future__ import annotations

import numpy as np

from .env import COST_NAMES, EnvConfig, SafeACDEnv

EVAL_SEED_BASE = 1_000_000  # disjoint from training seeds


def run_episodes(agent, attacker: str, n_episodes: int, env_cfg: EnvConfig | None = None,
                 seed_base: int = EVAL_SEED_BASE, shield=None):
    """Roll out ``agent`` for ``n_episodes`` against ``attacker``.

    Episode *i* always uses environment seed ``seed_base + i`` so every policy
    faces the same red/green/detection random streams (common random numbers),
    which makes paired comparisons between methods far less noisy.

    If ``shield`` is given, the shield's audit monitor counts how often the
    agent *chose* an action the shield would have blocked (for unshielded
    agents this is the would-be intervention rate).
    """
    env = SafeACDEnv(env_cfg or EnvConfig(), seed=seed_base)
    rows = []
    for i in range(n_episodes):
        env.reset(seed=seed_base + i, attacker=attacker)
        agent.reset()
        ret, cost, done, flagged = 0.0, np.zeros(len(COST_NAMES)), False, 0
        while not done:
            a = agent.act(env)
            if shield is not None and not shield.safe_mask(env)[a]:
                flagged += 1
            _, r, c, done, info = env.step(a)
            ret += r
            cost += c
        row = {"episode": i, "attacker": attacker, "return": ret,
               "impact_steps": env.impact_steps,
               "first_impact": env.first_impact,
               "final_compromised": int((env.level > 0).sum()),
               "shield_flags": flagged}
        row.update({f"cost_{k}": float(v) for k, v in zip(COST_NAMES, cost)})
        rows.append(row)
    return rows
