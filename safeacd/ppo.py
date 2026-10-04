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

from .env import (BUDGET_CHANNELS, COST_NAMES, HARD_CHANNELS, N_ACTIONS,
                  N_COSTS, OBS_DIM, EnvConfig, SafeACDEnv)
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
    hidden: int = 256
    reward_scale: float = 0.1
    # Constraint thresholds per episode: prod, traffic, evidence, critical.
    budgets: tuple = (10.0, 50.0, 0.0, 0.0)
    lagrange_lr: float = 0.05      # Adam step size for the dual variables
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


class ActorCritic(nn.Module):
    """Separate actor and multi-head critic (1 reward head + one per cost)."""

    def __init__(self, hidden=256):
        super().__init__()
        self.actor = _mlp(OBS_DIM, N_ACTIONS, hidden, 0.01)
        self.critic = _mlp(OBS_DIM, 1 + N_COSTS, hidden, 1.0)

    def dist(self, obs, mask):
        logits = self.actor(obs)
        logits = torch.where(mask, logits, torch.full_like(logits, -1e9))
        return torch.distributions.Categorical(logits=logits)

    def values(self, obs):
        return self.critic(obs)


# --------------------------------------------------------------------------- #
class DualAdam:
    """Projected dual ascent on the Lagrange multipliers with Adam step sizes.

    lambda_k <- max(0, lambda_k + lr * Adam(J_k - d_k)), as in OmniSafe's
    PPO-Lagrangian.  Adam bounds the per-update change of lambda, avoiding the
    overshoot of plain gradient ascent when early costs are far above budget.
    """

    def __init__(self, n, lr, init, mask, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.mask = mask
        self.lam = np.where(mask, init, 0.0).astype(float)
        self.m = np.zeros(n)
        self.v = np.zeros(n)
        self.t = 0

    def step(self, grad):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * grad ** 2
        m_hat = self.m / (1 - self.b1 ** self.t)
        v_hat = self.v / (1 - self.b2 ** self.t)
        self.lam = np.where(self.mask, np.maximum(0.0, self.lam + self.lr * m_hat / (np.sqrt(v_hat) + self.eps)), 0.0)
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
    ac = ActorCritic(cfg.hidden)
    opt = torch.optim.Adam(ac.parameters(), lr=cfg.lr, eps=1e-5)

    budgets = np.asarray(cfg.budgets, dtype=float)
    lam_mask = np.zeros(N_COSTS, dtype=bool)
    lam_mask[list(lag_ch)] = True
    dual = DualAdam(N_COSTS, cfg.lagrange_lr, cfg.lagrange_init, lam_mask)
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
                d = ac.dist(ot, torch.as_tensor(mask))
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
                d = ac.dist(b_obs[mb], b_mask[mb])
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
        self.ac = ActorCritic(self.cfg["hidden"])
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
            d = self.ac.dist(torch.as_tensor(env._obs())[None], torch.as_tensor(m)[None])
            if self.greedy:
                return int(d.probs.argmax())
            return int(torch.multinomial(d.probs, 1, generator=self.gen).item())
