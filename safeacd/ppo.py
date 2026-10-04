"""PPO family used in the study.

One implementation covers every learning condition so that the *only*
difference between conditions is how safety is handled:

=============  ==================  ================  ==================
method         reward used         Lagrangian on     shield (training)
=============  ==================  ================  ==================
ppo            security            --                no
shaped         security - b*u.c    --                no
lag            security            all 4 channels    no
shield         security            --                yes (H1, H2)
typed          security            budget channels   yes (H1, H2)
=============  ==================  ================  ==================

``typed`` is the proposed constraint-type-aware design: hard, irreversible
constraints are enforced by the shield; budget constraints by Lagrangian
relaxation of the CMDP (PPO-Lagrangian, Ray et al. 2019).
"""
from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .env import (BUDGET_CHANNELS, COST_NAMES, ENT2, N_ACTIONS, N_COSTS,
                  N_HOST_ACTION_TYPES, N_HOSTS, N_LINKS, OBS_DIM, OBS_PER_HOST,
                  OP_HOSTS, OP_SERVER, PROD_WEIGHT, SEC_VALUE, USERS,
                  EnvConfig, SafeACDEnv)
from .shield import Shield

METHODS = ("ppo", "shaped", "lag", "shield", "typed")


@dataclass
class TrainConfig:
    method: str = "typed"
    seed: int = 0
    total_steps: int = 1_000_000
    n_envs: int = 8
    n_steps: int = 256
    epochs: int = 10
    minibatch: int = 256
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: int = 64
    arch: str = "entity"           # "entity" (host-shared) or "mlp"
    reward_scale: float = 0.1
    # Constraint thresholds per episode: prod, traffic, evidence, critical.
    budgets: tuple = (10.0, 50.0, 0.0, 0.0)
    lagrange_lr: float = 0.05      # dual step size (per PPO update)
    lagrange_init: float = 0.0
    # Reward-shaping baseline: r' = r - beta * sum_k u_k c_k.
    shaping_beta: float = 1.0
    shaping_u: tuple = (1.0, 1.0, 10.0, 10.0)
    attackers: tuple = ("bline", "meander")
    out_dir: str = "results/runs"
    log_every: int = 1
    torch_threads: int = 1
    env: dict = field(default_factory=dict)

    @property
    def run_name(self) -> str:
        tag = self.method
        if self.method == "shaped":
            tag += f"-b{self.shaping_beta:g}"
        return f"{tag}_s{self.seed}"


def method_flags(method: str):
    """Return (use_shield, lagrangian_channels, use_shaping)."""
    if method == "ppo":
        return False, (), False
    if method == "shaped":
        return False, (), True
    if method == "lag":
        return False, tuple(range(N_COSTS)), False
    if method == "shield":
        return True, (), False
    if method == "typed":
        return True, BUDGET_CHANNELS, False
    raise ValueError(method)


# --------------------------------------------------------------------------- #
def _mlp(inp, out, hidden, out_gain):
    layers = [nn.Linear(inp, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
              nn.Linear(hidden, out)]
    for i, l in enumerate(x for x in layers if isinstance(x, nn.Linear)):
        gain = out_gain if i == 2 else np.sqrt(2)
        nn.init.orthogonal_(l.weight, gain)
        nn.init.zeros_(l.bias)
    return nn.Sequential(*layers)


class MLPActorCritic(nn.Module):
    """Flat baseline: separate actor and multi-head critic MLPs."""

    def __init__(self, hidden=256):
        super().__init__()
        self.actor = _mlp(OBS_DIM, N_ACTIONS, hidden, 0.01)
        self.critic = _mlp(OBS_DIM, 1 + N_COSTS, hidden, 1.0)

    def logits(self, obs):
        return self.actor(obs)

    def values(self, obs):
        return self.critic(obs)


def _host_static_features():
    f = np.zeros((N_HOSTS, 7), dtype=np.float32)
    for h in range(N_HOSTS):
        f[h, 0] = h in USERS
        f[h, 1] = h in (5, 6)
        f[h, 2] = h == ENT2
        f[h, 3] = h == OP_SERVER
        f[h, 4] = h in OP_HOSTS
    f[:, 5] = SEC_VALUE
    f[:, 6] = PROD_WEIGHT / 2.0
    return f


def _lin(i, o, gain=np.sqrt(2)):
    l = nn.Linear(i, o)
    nn.init.orthogonal_(l.weight, gain)
    nn.init.zeros_(l.bias)
    return l


class _EntityTrunk(nn.Module):
    """Shared per-host encoder + permutation-invariant global context."""

    def __init__(self, hidden):
        super().__init__()
        self.register_buffer("static", torch.as_tensor(_host_static_features()))
        d_in = OBS_PER_HOST + self.static.shape[1]
        self.enc = nn.Sequential(_lin(d_in, hidden), nn.Tanh(), _lin(hidden, hidden), nn.Tanh())
        self.ctx = nn.Sequential(_lin(2 * hidden + N_LINKS + 1, hidden), nn.Tanh())

    def forward(self, obs):
        b = obs.shape[0]
        hosts = obs[:, :N_HOSTS * OBS_PER_HOST].reshape(b, N_HOSTS, OBS_PER_HOST)
        glob = obs[:, N_HOSTS * OBS_PER_HOST:]
        x = torch.cat([hosts, self.static.expand(b, -1, -1)], dim=-1)
        e = self.enc(x)                                            # (B, H, d)
        c = self.ctx(torch.cat([e.mean(1), e.max(1).values, glob], dim=-1))  # (B, d)
        return e, c


class EntityActorCritic(nn.Module):
    """Host-shared ("entity-based") actor-critic.

    Per-host action logits come from one network applied to every host, so a
    rule such as "privileged ticket and valid image -> Restore" is learned once
    rather than twelve times (cf. Symes Thompson et al. 2024).  Global actions
    (Sleep, Block/Unblock) are read from the pooled context.
    """

    def __init__(self, hidden=128):
        super().__init__()
        self.a_trunk = _EntityTrunk(hidden)
        self.a_host = nn.Sequential(_lin(2 * hidden, hidden), nn.Tanh(),
                                    _lin(hidden, N_HOST_ACTION_TYPES, 0.01))
        self.a_glob = nn.Sequential(_lin(hidden, hidden), nn.Tanh(),
                                    _lin(hidden, 1 + 2 * N_LINKS, 0.01))
        self.c_trunk = _EntityTrunk(hidden)
        self.c_head = nn.Sequential(_lin(hidden, hidden), nn.Tanh(), _lin(hidden, 1 + N_COSTS, 1.0))

    def logits(self, obs):
        e, c = self.a_trunk(obs)
        h = self.a_host(torch.cat([e, c[:, None, :].expand_as(e)], dim=-1))  # (B, H, K)
        host_logits = h.transpose(1, 2).reshape(obs.shape[0], -1)          # kind-major
        g = self.a_glob(c)                                                 # Sleep, Block.., Unblock..
        return torch.cat([g[:, :1], host_logits, g[:, 1:]], dim=-1)

    def values(self, obs):
        _, c = self.c_trunk(obs)
        return self.c_head(c)


def make_actor_critic(arch: str, hidden: int) -> nn.Module:
    if arch == "entity":
        return EntityActorCritic(hidden)
    if arch == "mlp":
        return MLPActorCritic(hidden)
    raise ValueError(arch)


def masked_dist(ac, obs, mask):
    logits = ac.logits(obs)
    logits = torch.where(mask, logits, torch.full_like(logits, -1e9))
    return torch.distributions.Categorical(logits=logits)


# --------------------------------------------------------------------------- #
class DualAscent:
    """Projected dual ascent on the Lagrange multipliers.

    lambda_k <- max(0, lambda_k + lr * clip((J_k - d_k) / max(d_k, 1), -1, 1)).

    Normalising by the budget makes channels comparable; clipping bounds the
    per-update change so that a large early violation cannot drive lambda far
    above its equilibrium (the overshoot that PID-Lagrangian methods address,
    Stooke et al. 2020), and lambda decays as fast as it grows once the
    constraint is slack.
    """

    def __init__(self, n, lr, init, mask, clip=1.0):
        self.lr, self.clip, self.mask = lr, clip, mask
        self.lam = np.where(mask, init, 0.0).astype(float)

    def step(self, norm_violation):
        g = np.clip(norm_violation, -self.clip, self.clip)
        self.lam = np.where(self.mask, np.maximum(0.0, self.lam + self.lr * g), 0.0)
        return self.lam


# --------------------------------------------------------------------------- #
class VecEnv:
    """Minimal synchronous vector env with auto-reset and episode accounting."""

    def __init__(self, n, env_cfg: EnvConfig, seed: int, shield: Shield | None):
        self.envs = [SafeACDEnv(env_cfg, seed=seed * 1000 + i) for i in range(n)]
        self.shield = shield
        self.ep_ret = np.zeros(n)
        self.ep_cost = np.zeros((n, N_COSTS))
        self.obs = np.stack([e._obs() for e in self.envs])

    def masks(self):
        ms = []
        for e in self.envs:
            m = e.valid_mask()
            if self.shield is not None:
                m = m & self.shield.safe_mask(e)
            ms.append(m)
        return np.stack(ms)

    def step(self, actions):
        n = len(self.envs)
        rew = np.zeros(n)
        cost = np.zeros((n, N_COSTS))
        done = np.zeros(n, dtype=bool)
        finished = []
        for i, (e, a) in enumerate(zip(self.envs, actions)):
            o, r, c, d, info = e.step(int(a))
            rew[i], cost[i], done[i] = r, c, d
            self.ep_ret[i] += r
            self.ep_cost[i] += c
            if d:
                finished.append({"return": self.ep_ret[i], "cost": self.ep_cost[i].copy(),
                                 "impact": e.impact_steps, "attacker": e.attacker_name})
                self.ep_ret[i] = 0.0
                self.ep_cost[i] = 0.0
                o = e.reset()
            self.obs[i] = o
        return self.obs.copy(), rew, cost, done, finished


# --------------------------------------------------------------------------- #
def train(cfg: TrainConfig, verbose: bool = True) -> Path:
    torch.set_num_threads(cfg.torch_threads)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    use_shield, lag_ch, use_shaping = method_flags(cfg.method)
    env_cfg = EnvConfig(attackers=tuple(cfg.attackers), **cfg.env)
    shield = Shield() if use_shield else None
    venv = VecEnv(cfg.n_envs, env_cfg, cfg.seed, shield)
    ac = make_actor_critic(cfg.arch, cfg.hidden)
    opt = torch.optim.Adam(ac.parameters(), lr=cfg.lr, eps=1e-5)

    budgets = np.asarray(cfg.budgets, dtype=float)
    lam_mask = np.zeros(N_COSTS, dtype=bool)
    lam_mask[list(lag_ch)] = True
    dual = DualAscent(N_COSTS, cfg.lagrange_lr, cfg.lagrange_init, lam_mask)
    lam = dual.lam
    shaping_u = np.asarray(cfg.shaping_u, dtype=float)

    out = Path(cfg.out_dir) / cfg.run_name
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    log_f = open(out / "train_log.csv", "w", newline="")
    log = csv.writer(log_f)
    log.writerow(["update", "steps", "episodes", "ep_return"]
                 + [f"ep_{k}" for k in COST_NAMES]
                 + [f"lambda_{k}" for k in COST_NAMES]
                 + ["cum_" + k for k in COST_NAMES] + ["entropy", "wall_s"])

    T, N = cfg.n_steps, cfg.n_envs
    obs_buf = np.zeros((T, N, OBS_DIM), dtype=np.float32)
    mask_buf = np.zeros((T, N, N_ACTIONS), dtype=bool)
    act_buf = np.zeros((T, N), dtype=np.int64)
    logp_buf = np.zeros((T, N), dtype=np.float32)
    rew_buf = np.zeros((T, N), dtype=np.float32)
    cost_buf = np.zeros((T, N, N_COSTS), dtype=np.float32)
    done_buf = np.zeros((T, N), dtype=np.float32)
    val_buf = np.zeros((T, N, 1 + N_COSTS), dtype=np.float32)

    obs = venv.obs.copy()
    n_updates = cfg.total_steps // (T * N)
    cum_cost = np.zeros(N_COSTS)   # training-time violations (safe exploration)
    t0 = time.time()
    steps = 0
    for upd in range(1, n_updates + 1):
        finished = []
        for t in range(T):
            mask = venv.masks()
            with torch.no_grad():
                ot = torch.as_tensor(obs)
                d = masked_dist(ac, ot, torch.as_tensor(mask))
                a = d.sample()
                logp = d.log_prob(a)
                v = ac.values(ot)
            nobs, r, c, done, fin = venv.step(a.numpy())
            finished += fin
            obs_buf[t], mask_buf[t], act_buf[t] = obs, mask, a.numpy()
            logp_buf[t], val_buf[t] = logp.numpy(), v.numpy()
            r_used = r - (cfg.shaping_beta * (c @ shaping_u) if use_shaping else 0.0)
            rew_buf[t] = r_used * cfg.reward_scale
            cost_buf[t] = c * cfg.reward_scale
            done_buf[t] = done
            cum_cost += c.sum(0)
            obs = nobs
        steps += T * N

        # ---- GAE for reward and every cost channel ------------------------ #
        with torch.no_grad():
            last_v = ac.values(torch.as_tensor(obs)).numpy()
        sig = np.concatenate([rew_buf[..., None], cost_buf], axis=-1)  # (T,N,1+K)
        adv = np.zeros_like(sig)
        gae = np.zeros((N, 1 + N_COSTS), dtype=np.float32)
        for t in reversed(range(T)):
            nv = last_v if t == T - 1 else val_buf[t + 1]
            nonterm = (1.0 - done_buf[t])[:, None]
            delta = sig[t] + cfg.gamma * nv * nonterm - val_buf[t]
            gae = delta + cfg.gamma * cfg.gae_lambda * nonterm * gae
            adv[t] = gae
        ret = adv + val_buf

        # ---- Lagrange multiplier update (projected dual ascent) ------------- #
        if finished:
            J = np.stack([f["cost"] for f in finished]).mean(0)
            lam = dual.step((J - budgets) / np.maximum(budgets, 1.0))

        # Combined advantage: (A_r - sum_k lam_k A_ck) / (1 + sum_k lam_k).
        a_r = adv[..., 0]
        a_c = adv[..., 1:]
        a_comb = (a_r - (a_c * lam[None, None, :]).sum(-1)) / (1.0 + lam.sum())

        b_obs = torch.as_tensor(obs_buf.reshape(T * N, -1))
        b_mask = torch.as_tensor(mask_buf.reshape(T * N, -1))
        b_act = torch.as_tensor(act_buf.reshape(-1))
        b_logp = torch.as_tensor(logp_buf.reshape(-1))
        b_adv = torch.as_tensor(a_comb.reshape(-1), dtype=torch.float32)
        b_ret = torch.as_tensor(ret.reshape(T * N, -1))

        idx = np.arange(T * N)
        ent_acc = []
        for _ in range(cfg.epochs):
            np.random.shuffle(idx)
            for s in range(0, T * N, cfg.minibatch):
                mb = torch.as_tensor(idx[s:s + cfg.minibatch])
                d = masked_dist(ac, b_obs[mb], b_mask[mb])
                logp = d.log_prob(b_act[mb])
                ratio = torch.exp(logp - b_logp[mb])
                adv_mb = b_adv[mb]
                adv_mb = (adv_mb - adv_mb.mean()) / (adv_mb.std() + 1e-8)
                pg = -torch.min(ratio * adv_mb,
                                torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv_mb).mean()
                v_loss = ((ac.values(b_obs[mb]) - b_ret[mb]) ** 2).mean()
                ent = d.entropy().mean()
                loss = pg + cfg.vf_coef * v_loss - cfg.ent_coef * ent
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), cfg.max_grad_norm)
                opt.step()
                ent_acc.append(ent.item())

        if finished and upd % cfg.log_every == 0:
            ep_ret = float(np.mean([f["return"] for f in finished]))
            ep_c = np.stack([f["cost"] for f in finished]).mean(0)
            log.writerow([upd, steps, len(finished), round(ep_ret, 3)]
                         + [round(x, 4) for x in ep_c] + [round(x, 5) for x in lam]
                         + [round(x, 2) for x in cum_cost]
                         + [round(float(np.mean(ent_acc)), 4), round(time.time() - t0, 1)])
            log_f.flush()
            if verbose and (upd % 20 == 0 or upd == n_updates):
                print(f"[{cfg.run_name}] upd {upd}/{n_updates} steps {steps} R {ep_ret:8.1f} "
                      + " ".join(f"{k[:4]} {x:6.1f}" for k, x in zip(COST_NAMES, ep_c))
                      + " lam " + ",".join(f"{x:.2f}" for x in lam)
                      + f" ({time.time() - t0:.0f}s)", flush=True)

    log_f.close()
    torch.save({"model": ac.state_dict(), "config": asdict(cfg), "lambda": lam.tolist()},
               out / "model.pt")
    return out


# --------------------------------------------------------------------------- #
class TrainedPolicy:
    """Wraps a trained network as an evaluation agent."""

    def __init__(self, path, use_shield: bool | None = None, greedy: bool = False, seed: int = 0):
        ck = torch.load(Path(path) / "model.pt", weights_only=False)
        self.cfg = ck["config"]
        self.ac = make_actor_critic(self.cfg.get("arch", "mlp"), self.cfg["hidden"])
        self.ac.load_state_dict(ck["model"])
        self.ac.eval()
        trained_shield, _, _ = method_flags(self.cfg["method"])
        self.use_shield = trained_shield if use_shield is None else use_shield
        self.shield = Shield() if self.use_shield else None
        self.greedy = greedy
        self.gen = torch.Generator().manual_seed(seed)

    def reset(self):
        pass

    def act(self, env) -> int:
        m = env.valid_mask()
        if self.shield is not None:
            m = m & self.shield.safe_mask(env)
        with torch.no_grad():
            d = masked_dist(self.ac, torch.as_tensor(env._obs())[None], torch.as_tensor(m)[None])
            if self.greedy:
                return int(d.probs.argmax())
            return int(torch.multinomial(d.probs, 1, generator=self.gen).item())
