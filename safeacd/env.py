"""SafeACD-Sim: a discrete-time cyber-defence simulation with typed safety costs.

The model is a stochastic, partially observable, discrete-time simulation of a
12-host enterprise network whose topology mirrors the CybORG CAGE-2 scenario
(user, enterprise and operational subnets).  A scripted red agent progresses
along an attack graph, green (benign) traffic flows between hosts, and a blue
(defender) agent chooses one remediation action per step.

The key modelling choice is that *security* and *collateral harm* are emitted
as separate signals:

* ``reward``   -- security outcome only (compromise level and mission impact),
* ``cost``     -- a vector with one channel per kind of defender-induced harm:

    0  PROD   production downtime (host-steps offline, criticality weighted)
    1  TRAF   legitimate traffic blocked (weighted active flows not served)
    2  EVID   forensic evidence destroyed (reimage of a tainted, un-imaged host)
    3  CRIT   safety-critical asset severed (OpServer isolated by the defender)

PROD and TRAF are *budget* constraints (some collateral harm is tolerated up to
an operational budget); EVID and CRIT are *hard* constraints (zero tolerance:
evidence destruction is irreversible, OT safety rules are absolute).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# --------------------------------------------------------------------------- #
# Static network description
# --------------------------------------------------------------------------- #
HOST_NAMES = (
    "User0", "User1", "User2", "User3", "User4",
    "Ent0", "Ent1", "Ent2",
    "OpServer", "OpHost0", "OpHost1", "OpHost2",
)
N_HOSTS = len(HOST_NAMES)
USERS = tuple(range(0, 5))
ENT0, ENT1, ENT2 = 5, 6, 7
OP_SERVER = 8
OP_HOSTS = (9, 10, 11)
SUBNET = np.array([0] * 5 + [1] * 3 + [2] * 4)

# Security value of each host (CAGE-2 convention: users/op-hosts 0.1, servers 1).
SEC_VALUE = np.array([0.1] * 5 + [1.0] * 3 + [1.0] + [0.1] * 3)
# Business criticality used for the production-downtime cost channel.
PROD_WEIGHT = np.array([0.1] * 5 + [1.0] * 3 + [2.0] + [1.0] * 3)
IMPACT_PENALTY = 10.0

# Zone links that the defender can block.
LINK_UE, LINK_EO = 0, 1
N_LINKS = 2


def _build_attack_edges():
    """Directed edges (src, dst, link) of the attack graph; link=-1 if intra-zone."""
    edges = []
    for a in USERS:
        for b in USERS:
            if a != b:
                edges.append((a, b, -1))
        for b in (ENT0, ENT1):
            edges.append((a, b, LINK_UE))
    for a in (ENT0, ENT1, ENT2):
        for b in (ENT0, ENT1, ENT2):
            if a != b:
                edges.append((a, b, -1))
    for b in (OP_SERVER,) + OP_HOSTS:
        edges.append((ENT2, b, LINK_EO))
    ops = (OP_SERVER,) + OP_HOSTS
    for a in ops:
        for b in ops:
            if a != b:
                edges.append((a, b, -1))
    return edges


ATTACK_EDGES = _build_attack_edges()
# in_edges[dst] -> list of (src, link)
IN_EDGES = {h: [(s, l) for (s, d, l) in ATTACK_EDGES if d == h] for h in range(N_HOSTS)}


def _build_flows():
    """Legitimate (green) flows: (src, dst, weight, p_active, link)."""
    flows = []
    for u in USERS:
        flows.append((u, ENT0, 1.0, 0.8, LINK_UE))   # web / business app
        flows.append((u, ENT1, 0.5, 0.5, LINK_UE))   # file shares
    flows.append((ENT0, ENT2, 1.0, 1.0, -1))         # authentication
    flows.append((ENT1, ENT2, 1.0, 1.0, -1))
    flows.append((ENT2, OP_SERVER, 2.0, 1.0, LINK_EO))  # historian / telemetry
    for o in OP_HOSTS:
        flows.append((o, OP_SERVER, 2.0, 1.0, -1))   # control loop
    return flows


FLOWS = _build_flows()
FLOW_SRC = np.array([f[0] for f in FLOWS])
FLOW_DST = np.array([f[1] for f in FLOWS])
FLOW_W = np.array([f[2] for f in FLOWS])
FLOW_P = np.array([f[3] for f in FLOWS])
FLOW_LINK = np.array([f[4] for f in FLOWS])

# --------------------------------------------------------------------------- #
# Action space
# --------------------------------------------------------------------------- #
HOST_ACTIONS = ("Analyse", "Forensics", "Remove", "Restore", "Isolate", "Reconnect")
A_SLEEP = 0
A_ANALYSE, A_FORENSICS, A_REMOVE, A_RESTORE, A_ISOLATE, A_RECONNECT = range(6)
N_HOST_ACTION_TYPES = len(HOST_ACTIONS)
BLOCK_BASE = 1 + N_HOST_ACTION_TYPES * N_HOSTS      # Block(link)
UNBLOCK_BASE = BLOCK_BASE + N_LINKS                 # Unblock(link)
N_ACTIONS = UNBLOCK_BASE + N_LINKS


def host_action(kind: int, host: int) -> int:
    return 1 + kind * N_HOSTS + host


def decode_action(a: int):
    """Return (name, host_or_link) for an action index."""
    if a == A_SLEEP:
        return "Sleep", None
    if a < BLOCK_BASE:
        k, h = divmod(a - 1, N_HOSTS)
        return HOST_ACTIONS[k], h
    if a < UNBLOCK_BASE:
        return "Block", a - BLOCK_BASE
    return "Unblock", a - UNBLOCK_BASE


# Cost channels
COST_NAMES = ("prod", "traffic", "evidence", "critical")
C_PROD, C_TRAF, C_EVID, C_CRIT = range(4)
N_COSTS = len(COST_NAMES)
BUDGET_CHANNELS = (C_PROD, C_TRAF)
HARD_CHANNELS = (C_EVID, C_CRIT)

OBS_PER_HOST = 7
OBS_DIM = N_HOSTS * OBS_PER_HOST + N_LINKS + 1


@dataclass
class EnvConfig:
    horizon: int = 100
    restore_steps: int = 2          # host is offline this many steps when reimaged
    p_exploit: float = 0.75
    p_escalate: float = 0.8
    p_detect_exploit: float = 0.85
    p_detect_escalate: float = 0.7
    p_detect_impact: float = 1.0
    p_false_alert: float = 0.01
    p_phish: float = 0.1            # re-entry probability when red has no foothold
    attackers: tuple = ("bline", "meander")  # sampled uniformly per episode
    attacker_params: dict = field(default_factory=dict)


class SafeACDEnv:
    """Single-environment simulator with a gym-like API.

    ``step`` returns ``(obs, reward, cost_vector, done, info)``.
    """

    def __init__(self, config: EnvConfig | None = None, seed: int | None = None):
        from .red import make_attacker  # local import avoids a cycle

        self.cfg = config or EnvConfig()
        self._make_attacker = make_attacker
        self._seed_seq = np.random.SeedSequence(seed)
        self.t = 0
        self.attacker_name = None
        self.reset(seed=seed)

    # ------------------------------------------------------------------ #
    def reset(self, seed: int | None = None, attacker: str | None = None):
        if seed is not None:
            self._seed_seq = np.random.SeedSequence(seed)
        # Independent streams (common random numbers across policies).
        ss_ep = self._seed_seq.spawn(1)[0]
        s_red, s_green, s_det, s_misc = ss_ep.spawn(4)
        self.rng_red = np.random.default_rng(s_red)
        self.rng_green = np.random.default_rng(s_green)
        self.rng_det = np.random.default_rng(s_det)
        self.rng_misc = np.random.default_rng(s_misc)

        name = attacker or self.cfg.attackers[self.rng_misc.integers(len(self.cfg.attackers))]
        self.attacker_name = name
        self.red = self._make_attacker(name, self.cfg)

        self.t = 0
        self.level = np.zeros(N_HOSTS, dtype=np.int8)       # 0 clean, 1 user, 2 privileged
        self.isolated = np.zeros(N_HOSTS, dtype=bool)
        self.restore_timer = np.zeros(N_HOSTS, dtype=np.int8)
        self.tainted = np.zeros(N_HOSTS, dtype=bool)         # unpreserved-evidence-relevant
        self.preserved = np.zeros(N_HOSTS, dtype=bool)
        # Event clocks in half-steps: blue acts at 2t, red/alerts at 2t+1.
        self.image_time = np.full(N_HOSTS, -1, dtype=np.int32)       # last forensic image
        self.ticket_open_time = np.full(N_HOSTS, -1, dtype=np.int32)
        self.ticket = np.zeros(N_HOSTS, dtype=np.int8)       # defender's open ticket severity
        self.alert = np.zeros(N_HOSTS, dtype=np.int8)        # alerts raised this step
        self.blocked = np.zeros(N_LINKS, dtype=bool)
        self.impact_steps = 0
        self.first_impact = -1

        # Initial foothold: a random user host, undetected (CAGE-2 starts on a
        # fixed User0, which agents learn to exploit by quarantining that host).
        u0 = int(self.rng_red.integers(len(USERS)))
        self.level[u0] = 1
        self.tainted[u0] = True
        return self._obs()

    # ------------------------------------------------------------------ #
    def online(self) -> np.ndarray:
        return (~self.isolated) & (self.restore_timer == 0)

    def can_exploit(self, src: int, dst: int, link: int, on=None) -> bool:
        on = self.online() if on is None else on
        if not (on[src] and on[dst]) or self.level[src] < 1 or self.level[dst] > 0:
            return False
        if link >= 0 and (self.blocked[link] or self.level[src] < 2):
            return False
        return True

    def feasible_exploits(self, dst_filter=None):
        """All (src, dst) exploits red could attempt now (in ATTACK_EDGES order)."""
        on = self.online()
        lvl = self.level
        ok_src = on & (lvl >= 1)
        ok_dst = on & (lvl == 0)
        dsts = range(N_HOSTS) if dst_filter is None else sorted(dst_filter)
        if dst_filter is not None and not ok_dst[list(dsts)].any():
            return []
        out = []
        for (s, d, l) in ATTACK_EDGES:
            if not (ok_src[s] and ok_dst[d]):
                continue
            if dst_filter is not None and d not in dst_filter:
                continue
            if l >= 0 and (self.blocked[l] or lvl[s] < 2):
                continue
            out.append((s, d))
        return out

    # ------------------------------------------------------------------ #
    def valid_mask(self) -> np.ndarray:
        """Actions that are syntactically meaningful (applied to all methods)."""
        m = np.empty(N_ACTIONS, dtype=bool)
        m[A_SLEEP] = True
        ok = self.restore_timer == 0
        hm = m[1:BLOCK_BASE].reshape(N_HOST_ACTION_TYPES, N_HOSTS)
        hm[A_ANALYSE] = ok
        hm[A_FORENSICS] = ok
        hm[A_REMOVE] = ok
        hm[A_RESTORE] = ok
        hm[A_ISOLATE] = ok & ~self.isolated
        hm[A_RECONNECT] = self.isolated
        m[BLOCK_BASE:UNBLOCK_BASE] = ~self.blocked
        m[UNBLOCK_BASE:] = self.blocked
        return m

    def image_valid(self) -> np.ndarray:
        """Defender-observable: an image exists that post-dates the open ticket.

        Uses only SOC-visible information (image timestamps, ticket timestamps),
        never the ground-truth compromise state.
        """
        return (self.image_time >= 0) & (self.image_time > self.ticket_open_time)

    def _set_ticket(self, h: int, sev: int, clock: int):
        if sev > 0 and self.ticket[h] == 0:
            self.ticket_open_time[h] = clock
        if sev == 0:
            self.ticket_open_time[h] = -1
        self.ticket[h] = sev

    # ------------------------------------------------------------------ #
    def _taint(self, h: int):
        """Attacker activity leaves artefacts on h.

        Evidence is incident-scoped: the first compromise of a clean host opens
        a new incident whose artefacts are not in any earlier image; further
        activity on an already-tainted host belongs to the same incident.
        """
        if not self.tainted[h]:
            self.tainted[h] = True
            self.preserved[h] = False

    def _raise_alert(self, h: int, severity: int, p: float):
        if self.rng_det.random() < p:
            self.alert[h] = max(self.alert[h], severity)

    def _apply_blue(self, action: int, cost: np.ndarray):
        name, x = decode_action(action)
        if name == "Sleep":
            return
        if name == "Block":
            self.blocked[x] = True
            return
        if name == "Unblock":
            self.blocked[x] = False
            return
        h = x
        clk = 2 * self.t
        if self.restore_timer[h] > 0:
            return  # host is being reimaged; action has no effect
        if name == "Analyse":
            self._set_ticket(h, int(self.level[h]), clk)
        elif name == "Forensics":
            self.preserved[h] = self.tainted[h] or self.preserved[h]
            self.image_time[h] = clk
        elif name == "Remove":
            if self.level[h] == 1:
                self.level[h] = 0
                self._set_ticket(h, 0, clk)
            elif self.level[h] == 2:
                self._set_ticket(h, 2, clk)  # EDR reports privileged persistence
            else:
                self._set_ticket(h, 0, clk)
        elif name == "Restore":
            if self.tainted[h] and not self.preserved[h]:
                cost[C_EVID] += 1.0
            self.level[h] = 0
            self.tainted[h] = False
            self.preserved[h] = False
            self._set_ticket(h, 0, clk)
            self.restore_timer[h] = self.cfg.restore_steps
            self.image_time[h] = -1
        elif name == "Isolate":
            self.isolated[h] = True
        elif name == "Reconnect":
            self.isolated[h] = False

    def _apply_red(self):
        cfg = self.cfg
        act = self.red.act(self, self.rng_red)
        det_scale = self.red.detect_scale
        kind = act[0]
        impact = False
        if kind == "exploit":
            _, src, dst = act
            if self.rng_red.random() < cfg.p_exploit:
                self.level[dst] = 1
                self._taint(dst)
                self._raise_alert(dst, 1, cfg.p_detect_exploit * det_scale)
            else:
                self._raise_alert(dst, 1, 0.5 * cfg.p_detect_exploit * det_scale)
        elif kind == "escalate":
            _, h = act
            if self.rng_red.random() < cfg.p_escalate:
                self.level[h] = 2
                self._taint(h)
                self._raise_alert(h, 2, cfg.p_detect_escalate * det_scale)
            else:
                self._raise_alert(h, 1, 0.5 * cfg.p_detect_escalate * det_scale)
        elif kind == "impact":
            impact = True
            self._taint(OP_SERVER)
            self._raise_alert(OP_SERVER, 2, cfg.p_detect_impact)
        # Phishing re-entry if red has no *usable* foothold (all evicted or
        # quarantined): a new user host is compromised with probability p_phish.
        on = self.online()
        if not ((self.level > 0) & on).any() and self.rng_red.random() < cfg.p_phish:
            cand = [u for u in USERS if on[u] and self.level[u] == 0]
            if cand:
                u = cand[int(self.rng_red.integers(len(cand)))]
                self.level[u] = 1
                self._taint(u)
                self._raise_alert(u, 1, cfg.p_detect_exploit * det_scale)
        return impact

    # ------------------------------------------------------------------ #
    def step(self, action: int):
        cfg = self.cfg
        cost = np.zeros(N_COSTS)
        self.alert[:] = 0

        self._apply_blue(int(action), cost)
        impact = self._apply_red()

        # False-positive alerts from benign activity.
        fp = self.rng_det.random(N_HOSTS) < cfg.p_false_alert
        self.alert = np.maximum(self.alert, fp.astype(np.int8))
        for h in np.flatnonzero(self.alert > self.ticket):
            self._set_ticket(h, int(self.alert[h]), 2 * self.t + 1)

        # Collateral-harm costs (defender-induced).
        offline = ~self.online()
        cost[C_PROD] += float(PROD_WEIGHT[offline].sum())
        active = self.rng_green.random(len(FLOWS)) < FLOW_P
        link_ok = np.where(FLOW_LINK >= 0, ~self.blocked[np.maximum(FLOW_LINK, 0)], True)
        served = (~offline[FLOW_SRC]) & (~offline[FLOW_DST]) & link_ok
        cost[C_TRAF] += float((FLOW_W * (active & ~served)).sum())
        if self.isolated[OP_SERVER]:
            cost[C_CRIT] += 1.0

        # Security reward (attacker presence and mission impact only).
        comp = np.where(self.level == 2, 1.0, np.where(self.level == 1, 0.5, 0.0))
        reward = -float((SEC_VALUE * comp).sum())
        if impact:
            reward -= IMPACT_PENALTY
            self.impact_steps += 1
            if self.first_impact < 0:
                self.first_impact = self.t

        # Advance clocks.
        self.restore_timer = np.maximum(self.restore_timer - 1, 0).astype(np.int8)
        self.t += 1
        done = self.t >= cfg.horizon
        info = {
            "impact": impact,
            "n_privileged": int((self.level == 2).sum()),
            "n_compromised": int((self.level > 0).sum()),
            "attacker": self.attacker_name,
        }
        return self._obs(), reward, cost, done, info

    # ------------------------------------------------------------------ #
    def _obs(self) -> np.ndarray:
        o = np.zeros((N_HOSTS, OBS_PER_HOST), dtype=np.float32)
        o[np.arange(N_HOSTS), self.ticket] = 1.0                  # ticket one-hot (3)
        o[:, 3] = self.alert / 2.0
        o[:, 4] = self.isolated
        o[:, 5] = self.restore_timer > 0
        o[:, 6] = self.image_valid()
        g = np.concatenate([self.blocked.astype(np.float32), [self.t / self.cfg.horizon]])
        return np.concatenate([o.ravel(), g]).astype(np.float32)
