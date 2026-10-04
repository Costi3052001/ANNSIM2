"""Scripted red (attacker) agents.

* ``bline``   -- goal-directed: shortest path User -> Ent -> Ent2 -> OpServer -> impact
                (analogue of CAGE-2 B_lineAgent).
* ``meander`` -- breadth-first: escalates everywhere and expands subnet by subnet
                (analogue of CAGE-2 RedMeanderAgent).
* ``stealthy``-- held-out test attacker: B-line with low-and-slow behaviour
                (dwells 30% of steps) and 60% lower detection probability.
"""
from __future__ import annotations

from .env import (ENT0, ENT1, ENT2, OP_HOSTS, OP_SERVER, SUBNET, USERS)


class Attacker:
    detect_scale = 1.0

    def __init__(self, cfg):
        self.cfg = cfg

    def act(self, env, rng):  # pragma: no cover - interface
        raise NotImplementedError

    @staticmethod
    def _can_impact(env):
        return env.level[OP_SERVER] == 2 and env.online()[OP_SERVER]

    @staticmethod
    def _escalatable(env, hosts):
        on = env.online()
        return [h for h in hosts if env.level[h] == 1 and on[h]]


class BLine(Attacker):
    """Goal-directed attacker following the shortest path to the OpServer."""

    def act(self, env, rng):
        if self._can_impact(env):
            return ("impact",)
        if self._escalatable(env, [OP_SERVER]):
            return ("escalate", OP_SERVER)
        ex = env.feasible_exploits(dst_filter=(OP_SERVER,))
        if ex:
            return ("exploit",) + ex[rng.integers(len(ex))]
        if self._escalatable(env, [ENT2]):
            return ("escalate", ENT2)
        ex = env.feasible_exploits(dst_filter=(ENT2,))
        if ex:
            return ("exploit",) + ex[rng.integers(len(ex))]
        ex = env.feasible_exploits(dst_filter=(ENT0, ENT1))
        if ex:
            return ("exploit",) + ex[rng.integers(len(ex))]
        esc = self._escalatable(env, USERS)
        if esc:
            return ("escalate", esc[rng.integers(len(esc))])
        # Path blocked: try to widen the foothold anywhere (fallback).
        esc = self._escalatable(env, range(len(SUBNET)))
        if esc:
            return ("escalate", esc[rng.integers(len(esc))])
        ex = env.feasible_exploits()
        if ex:
            return ("exploit",) + ex[rng.integers(len(ex))]
        return ("wait",)


class Meander(Attacker):
    """Breadth-first attacker: escalate everything, expand lowest subnet first."""

    def act(self, env, rng):
        if self._can_impact(env):
            return ("impact",)
        esc = self._escalatable(env, range(len(SUBNET)))
        if esc:
            return ("escalate", esc[rng.integers(len(esc))])
        ex = env.feasible_exploits()
        if ex:
            lowest = min(SUBNET[d] for (_, d) in ex)
            ex = [(s, d) for (s, d) in ex if SUBNET[d] == lowest]
            return ("exploit",) + ex[rng.integers(len(ex))]
        return ("wait",)


class Stealthy(BLine):
    """Low-and-slow variant of B-line used only for out-of-distribution testing."""

    detect_scale = 0.4
    p_dwell = 0.3

    def act(self, env, rng):
        if rng.random() < self.p_dwell:
            return ("wait",)
        return super().act(env, rng)


ATTACKERS = {"bline": BLine, "meander": Meander, "stealthy": Stealthy}


def make_attacker(name: str, cfg):
    return ATTACKERS[name](cfg)


# Re-exported for convenience in tests.
__all__ = ["ATTACKERS", "make_attacker", "BLine", "Meander", "Stealthy", "OP_HOSTS"]
