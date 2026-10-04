"""Rule-based SOC playbook (non-learning reference policy).

Encodes common incident-response practice (NIST SP 800-61 / SP 800-86):
triage alerts, evict user-level footholds, and image a host before reimaging
it.  It never isolates the OpServer and never blocks zone links, so it is
compliant with both hard rules by construction.  Its mean collateral cost
under the training distribution is used to set the operational budgets
("do no more harm than the incumbent playbook").
"""
from __future__ import annotations

import numpy as np

from .env import (A_ANALYSE, A_FORENSICS, A_REMOVE, A_RESTORE, A_SLEEP,
                  N_HOSTS, SEC_VALUE, host_action)

HUNT_ORDER = (8, 7, 5, 6, 9, 10, 11, 0, 1, 2, 3, 4)


class PlaybookAgent:
    def __init__(self):
        self._hunt = 0

    def reset(self):
        self._hunt = 0

    def act(self, env) -> int:
        valid = env.valid_mask()
        order = np.argsort(-SEC_VALUE, kind="stable")
        fresh = env.image_valid()
        # 1. Privileged compromise suspected: image, then reimage.
        for h in order:
            if env.ticket[h] == 2 and env.restore_timer[h] == 0:
                a = host_action(A_RESTORE if fresh[h] else A_FORENSICS, h)
                if valid[a]:
                    return a
        # 2. User-level foothold suspected: evict.
        for h in order:
            if env.ticket[h] == 1 and env.restore_timer[h] == 0:
                a = host_action(A_REMOVE, h)
                if valid[a]:
                    return a
        # 3. Proactive threat hunting (round-robin analysis).
        for _ in range(N_HOSTS):
            h = HUNT_ORDER[self._hunt % N_HOSTS]
            self._hunt += 1
            a = host_action(A_ANALYSE, h)
            if valid[a]:
                return a
        return A_SLEEP


class RandomAgent:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        pass

    def act(self, env) -> int:
        idx = np.flatnonzero(env.valid_mask())
        return int(self.rng.choice(idx))


class SleepAgent:
    def reset(self):
        pass

    def act(self, env) -> int:
        return A_SLEEP
