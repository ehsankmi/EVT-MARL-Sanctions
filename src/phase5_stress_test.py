"""
phase5_stress_test.py
---------------------
Phase 5: Stress-testing and fragility/antifragility analysis.

Three-part analysis:
    1. Replay frozen (trained) policies through real historical crisis
       windows: 2018 sanctions reimposition and 2025 conflict/snapback.
    2. Crossover analysis: scale crisis-window shocks from 0.5x to 4x
       to find the intensity at which risk-aware overtakes risk-neutral.
    3. Fragility index: sweep bootstrap-sampled shocks at increasing
       intensity and compute the second derivative of loss (fragility).

Key finding (see working paper Section 5.3–5.4):
    - Risk-aware advantage is CONDITIONAL on shock type: present for
      sudden discrete shocks (2018 episode), absent for gradual trends.
    - Risk-aware fragility SD = 0.07 vs risk-neutral SD = 1.60
      across 5 random seeds (rebuilt, lower-variance estimator).

Usage:
    python phase5_stress_test.py
    (trains agents internally; no saved model files required)

Outputs:
    results/figures/phase5_crisis_replay.png
    results/figures/phase5_crossover.png
    results/figures/phase5_fragility.png
    results/phase5_fragility_summary.csv
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.agents import FirmAgent, ACTION_PROFILES, N_ACTIONS
from src.evt_utils import fit_gpd, cvar_from_gpd, compute_fragility_index, classify_fragility

# ── Setup ──────────────────────────────────────────────────────────────────────
SEED = 42
rng = np.random.default_rng(SEED)

fx = pd.read_csv("data/usd_irr_daily.csv", parse_dates=["date"])
fx["log_ret"] = np.log(fx["usd_sell"]).diff()
fx = fx.dropna(subset=["log_ret"]).reset_index(drop=True)
shock_pool = fx["log_ret"].values
pool_std = shock_pool.std()

losses = -shock_pool
gpd_params = fit_gpd(losses)
CVaR_shock = cvar_from_gpd(gpd_params)

N_AGENTS = 10
EPISODE_LENGTH, N_EPISODES = 250, 300
BANKRUPTCY_THRESHOLD = 0.25
EPS_START, EPS_END, EPS_DECAY_STEPS = 1.0, 0.05, 45000
LAMBDA1, LAMBDA2 = 0.4, 0.3

ACTION_NAMES = [ACTION_PROFILES[a]["name"] for a in range(N_ACTIONS)]

def epsilon_at(step):
    frac = min(step / EPS_DECAY_STEPS, 1.0)
    return EPS_START + frac * (EPS_END - EPS_START)

# ── Train (shared helper) ──────────────────────────────────────────────────────
def train(risk_aware: bool, seed: int = SEED):
    local_rng = np.random.default_rng(seed)
    agents = [FirmAgent(i, local_rng) for i in range(N_AGENTS)]
    global_step = 0
    for ep in range(N_EPISODES):
        for agent in agents:
            agent.cash_ratio, agent.alive = 1.0, True
        last_shock = 0.0
        for _ in range(EPISODE_LENGTH):
            shock = local_rng.choice(shock_pool)
            sb = FirmAgent.shock_bucket(last_shock, pool_std)
            eps = epsilon_at(global_step)
            step_data = []
            for agent in agents:
                if not agent.alive: continue
                s = FirmAgent.state_index(FirmAgent.cash_bucket(agent.cash_ratio), sb)
                a = agent.act(s, eps)
                pr = ACTION_PROFILES[a]["base_return"] + ACTION_PROFILES[a]["shock_beta"] * shock
                step_data.append((agent, s, a, pr, ACTION_PROFILES[a]["shock_beta"]))
            L_sys = np.mean([-pr for (_, _, _, pr, _) in step_data]) if step_data else 0.0
            for agent, s, a, pr, beta in step_data:
                agent.cash_ratio = max(agent.cash_ratio * (1 + pr), 0.0)
                reward = (pr - LAMBDA1 * beta * CVaR_shock - LAMBDA2 * max(L_sys,0)*2.5
                          if risk_aware else pr)
                if agent.cash_ratio <= BANKRUPTCY_THRESHOLD:
                    reward -= 1.0; agent.alive = False
                s_next = FirmAgent.state_index(FirmAgent.cash_bucket(agent.cash_ratio),
                                               FirmAgent.shock_bucket(shock, pool_std))
                agent.update(s, a, reward, s_next)
            last_shock = shock
            global_step += 1
    return agents

print("Training risk-neutral agents...")
agents_neutral = train(risk_aware=False)
print("Training risk-aware agents...")
agents_aware = train(risk_aware=True)

# ── Part 1: Crisis window replay ───────────────────────────────────────────────
def replay(agents, shocks, intensity=1.0):
    finals, paths = [], []
    for agent in agents:
        cash, last_shock = 1.0, 0.0
        path = [cash]
        for shock in shocks:
            sb = FirmAgent.shock_bucket(last_shock, pool_std)
            s = FirmAgent.state_index(FirmAgent.cash_bucket(cash), sb)
            a = int(np.argmax(agent.Q[s]))
            pr = ACTION_PROFILES[a]["base_return"] + ACTION_PROFILES[a]["shock_beta"] * shock * intensity
            cash = max(cash * (1 + pr), 0.0)
            if cash <= BANKRUPTCY_THRESHOLD:
                break
            path.append(cash)
            last_shock = shock * intensity
        finals.append(cash); paths.append(path)
    return np.mean(finals), paths

windows = {
    "2018 sanctions reimposition\n(May–Nov 2018)":
        fx[(fx["date"] >= "2018-05-01") & (fx["date"] <= "2018-11-30")]["log_ret"].values,
    "2025 conflict + snapback\n(Jun–Sep 2025)":
        fx[(fx["date"] >= "2025-06-01") & (fx["date"] <= "2025-09-30")]["log_ret"].values,
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
for idx, (name, shocks) in enumerate(windows.items()):
    if not len(shocks): continue
    _, paths_n = replay(agents_neutral, shocks)
    _, paths_a = replay(agents_aware, shocks)
    mean_n = np.mean([p + [p[-1]]*(len(shocks)-len(p)+1) for p in paths_n], axis=0)
    mean_a = np.mean([p + [p[-1]]*(len(shocks)-len(p)+1) for p in paths_a], axis=0)
    ax = axes[idx]
    ax.plot(mean_n[:len(shocks)+1], label="Risk-neutral", color="#c0392b")
    ax.plot(mean_a[:len(shocks)+1], label="Risk-aware",   color="#27ae60")
    ax.axhline(BANKRUPTCY_THRESHOLD, color="gray", linestyle="--", linewidth=1, label="Bankruptcy line")
    ax.set_title(f"Mean Cash Trajectory\n{name}")
    ax.set_xlabel("Trading day"); ax.set_ylabel("Cash ratio"); ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("results/figures/phase5_crisis_replay.png", dpi=150)

# ── Part 2: Crossover analysis ─────────────────────────────────────────────────
intensities = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for idx, (name, shocks) in enumerate(windows.items()):
    if not len(shocks): continue
    vals_n = [replay(agents_neutral, shocks, m)[0] for m in intensities]
    vals_a = [replay(agents_aware,   shocks, m)[0] for m in intensities]
    crossover = next((m for m, nv, av in zip(intensities, vals_n, vals_a) if av >= nv), None)
    ax = axes[idx]
    ax.plot(intensities, vals_n, "o-", label="Risk-neutral", color="#c0392b")
    ax.plot(intensities, vals_a, "o-", label="Risk-aware",   color="#27ae60")
    if crossover:
        ax.axvline(crossover, color="gray", linestyle="--", label=f"Crossover ≈ {crossover}x")
    ax.set_title(f"Crossover Analysis\n{name.splitlines()[0]}")
    ax.set_xlabel("Shock intensity multiplier"); ax.set_ylabel("Mean final cash ratio")
    ax.legend(fontsize=8)
    print(f"{name.splitlines()[0]}: crossover = {crossover}x (None = no crossover found)")
plt.tight_layout()
plt.savefig("results/figures/phase5_crossover.png", dpi=150)

# ── Part 3: Fragility index (population-averaged, bootstrap CI) ───────────────
fragility_intensities = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

def make_loss_fn(agents, intensity):
    """Average loss across all agents for a given shock intensity."""
    total = []
    for agent in agents:
        cash, last_shock = 1.0, 0.0
        for _ in range(EPISODE_LENGTH):
            shock = rng.choice(shock_pool) * intensity
            sb = FirmAgent.shock_bucket(last_shock, pool_std)
            s = FirmAgent.state_index(FirmAgent.cash_bucket(cash), sb)
            a = int(np.argmax(agent.Q[s]))
            cash = max(cash * (1 + ACTION_PROFILES[a]["base_return"]
                               + ACTION_PROFILES[a]["shock_beta"] * shock), 0.0)
            last_shock = shock
            if cash <= 0.01: break
        total.append(1 - cash)
    return np.mean(total)

frag_n, ci_n_lo, ci_n_hi = compute_fragility_index(
    lambda m: make_loss_fn(agents_neutral, m), fragility_intensities, n_trials=20)
frag_a, ci_a_lo, ci_a_hi = compute_fragility_index(
    lambda m: make_loss_fn(agents_aware,   m), fragility_intensities, n_trials=20)

print(f"\nFragility index:")
print(f"  Risk-neutral: {frag_n:.4f} [{ci_n_lo:.3f}, {ci_n_hi:.3f}] → {classify_fragility(frag_n)}")
print(f"  Risk-aware:   {frag_a:.4f} [{ci_a_lo:.3f}, {ci_a_hi:.3f}] → {classify_fragility(frag_a)}")

pd.DataFrame([
    {"type": "risk_neutral", "fragility": frag_n, "ci_low": ci_n_lo, "ci_high": ci_n_hi,
     "classification": classify_fragility(frag_n)},
    {"type": "risk_aware",   "fragility": frag_a, "ci_low": ci_a_lo, "ci_high": ci_a_hi,
     "classification": classify_fragility(frag_a)},
]).to_csv("results/phase5_fragility_summary.csv", index=False)

print("\nSaved all Phase 5 results to results/")
