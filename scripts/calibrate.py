#!/usr/bin/env python
"""Operational validation: run reference policies against every attacker (methodology §3.3)."""
import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from safeacd.env import SafeACDEnv, EnvConfig, COST_NAMES, host_action, A_ISOLATE, BLOCK_BASE, OP_SERVER
from safeacd.playbook import PlaybookAgent, RandomAgent, SleepAgent

class IsolateOS:
    """Reward-hacking reference: sever the OpServer and block user->enterprise."""
    def reset(self): self.t=0
    def act(self, env):
        m = env.valid_mask()
        a = host_action(A_ISOLATE, OP_SERVER)
        if m[a]: return a
        if m[BLOCK_BASE+0]: return BLOCK_BASE+0
        return 0

def run(agent, attacker, n=100):
    env = SafeACDEnv(EnvConfig(), seed=0)
    R=[];C=[];imp=[];fi=[]
    t0=time.time()
    for i in range(n):
        env.reset(seed=1_000_000+i, attacker=attacker); agent.reset()
        r=0; c=np.zeros(4); done=False
        while not done:
            a = agent.act(env)
            _, rew, cost, done, info = env.step(a)
            r+=rew; c+=cost
        R.append(r); C.append(c); imp.append(env.impact_steps); fi.append(env.first_impact)
    C=np.array(C)
    dt=time.time()-t0
    print(f"{type(agent).__name__:12s} {attacker:9s} R={np.mean(R):8.1f}±{np.std(R):6.1f} impact={np.mean(imp):5.1f} firstimp={np.mean([x for x in fi if x>=0]) if any(x>=0 for x in fi) else -1:5.1f} " +
          " ".join(f"{k}={C[:,j].mean():6.2f}" for j,k in enumerate(COST_NAMES)) + f"  ({n*100/dt:.0f} steps/s)")

if __name__ == "__main__":
    for att in ["bline","meander","stealthy"]:
        for ag in [SleepAgent(), RandomAgent(0), PlaybookAgent(), IsolateOS()]:
            run(ag, att)
