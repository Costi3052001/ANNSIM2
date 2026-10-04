"""Runtime safety shield for the hard (zero-tolerance) constraints.

The shield is *preemptive* in the sense of Alshiekh et al. (AAAI 2018): before
the agent samples an action, every action that may violate a hard rule is
removed from its choice set.  The shield only uses information that a real
SOC would have (open alerts, forensic-image timestamps, isolation state) -- it
never reads the simulator's ground-truth compromise state.

Rules
-----
H1 (evidence)  Restore(h) is allowed only if a forensic image of h exists that
               was taken after the host's currently open alert ticket was
               raised and after its last reimage ("image before reimage").
H2 (critical)  Isolate(OpServer) is never allowed (the safety-critical OT
               controller may not be severed by an autonomous agent).

Because H1 depends on *observed* alerts, the guarantee is relative to the
monitor: a compromise that starts after the image and is never detected can
still have its artefacts destroyed.  We report these residual violations
explicitly instead of claiming a zero-violation guarantee.
"""
from __future__ import annotations

import numpy as np

from .env import (A_ISOLATE, A_RESTORE, N_ACTIONS, N_HOSTS, OP_SERVER,
                  host_action)

RULES = ("H1_image_before_reimage", "H2_never_isolate_opserver")


class Shield:
    """Preemptive shield.

    ``max_image_age`` (steps) optionally tightens H1 with a freshness window:
    the image must also be at most that old.  This narrows the window in which
    an *undetected* new compromise can slip in between imaging and reimaging
    (a deployment-time variant evaluated in the RQ4 discussion).
    """

    def __init__(self, rules=RULES, max_image_age: int | None = None):
        self.rules = set(rules)
        self.max_image_age = max_image_age

    def _h1_allowed(self, env) -> np.ndarray:
        ok = env.image_valid()
        if self.max_image_age is not None:
            age_half_steps = 2 * env.t - env.image_time
            ok = ok & (age_half_steps <= 2 * self.max_image_age)
        return ok

    def safe_mask(self, env) -> np.ndarray:
        m = np.ones(N_ACTIONS, dtype=bool)
        if "H1_image_before_reimage" in self.rules:
            start = host_action(A_RESTORE, 0)
            m[start:start + N_HOSTS] = self._h1_allowed(env)
        if "H2_never_isolate_opserver" in self.rules:
            m[host_action(A_ISOLATE, OP_SERVER)] = False
        return m

    def explain(self, env, action: int) -> str | None:
        """Human-readable reason an action is blocked (for audit logs)."""
        from .env import decode_action
        name, h = decode_action(action)
        if name == "Restore" and "H1_image_before_reimage" in self.rules and not self._h1_allowed(env)[h]:
            return f"H1: Restore({h}) blocked - no forensic image after ticket opened"
        if name == "Isolate" and h == OP_SERVER and "H2_never_isolate_opserver" in self.rules:
            return "H2: Isolate(OpServer) blocked - safety-critical asset"
        return None
