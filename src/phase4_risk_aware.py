"""
phase4_risk_aware.py
--------------------
Phase 4: EVT/CVaR-penalized MARL (risk-aware) vs risk-neutral comparison.

Injects a two-term CVaR penalty into each agent's reward function:
    - lambda1 * CVaR(L_i)      : individual tail-risk penalty
    - lambda2 * CVaR(L_sys)    : systemic/network tail-risk penalty

CVaR is estimated from the real historical USD/IRR shock distribution
via the Peaks-Over-Threshold (GPD) method — following POTPG (Davar et
al., 2024), extended here to the multi-agent setting.

Also implements lambda1 annealing and imbalanced episode starts to
ensure multi-agent interaction mechanisms (shared liquidity market,
bankruptcy contagion) are genuinely activated during training.

Usage:
    python phase4_risk_aware.py

Outputs:
    results/phase4_comparison.png
    results/phase4_final_neutral.csv
    results/phase4_final_aware.csv
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.agents import FirmAgent, ACTION_PROFILES, N_ACTIONS
from src.evt_utils import fit_gpd, cvar_from_gpd

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
rng = np.random.default_rng(SEED)

# ── Load real shock data + fit EVT ────────────────────────────────────────────
fx = pd.read_csv("data/usd_irr_daily.csv", parse_dates=["date"])
fx["log_ret"] = np.log(fx["usd_sell"]).diff()
shock_pool = fx["log_ret"].dropna().values
pool_std = shock_pool.std()

losses = -shock_pool
gpd_params = fit_gpd(losses, quantile=0.95)
CVaR_shock = cvar_from_gpd(gpd_params, alpha=0.95)

print(f"EVT fit: xi={gpd_params['xi']:.4f}, sigma={gpd_params['sigma']:.4f}, "
      f"u={gpd_params['u']:.4f}")
print(f"CVaR_0.95(shock loss) = {CVaR_shock:.4f}")

# ── Hyperparameters ────────────────────────────────────────────────────────────
N_AGENTS = 10
EPISODE_LENGTH, N_EPISODES = 250, 300
BANKRUPTCY_THRESHOLD = 0.25
EPS_START, EPS_END, EPS_DECAY_STEPS = 1.0, 0.05, 45000
LAMBDA1_FINAL, LAMBDA1_START, LAMBDA2 = 0.4, 0.03, 0.3
LIQUIDITY_DRAG = 0.004
CONTAGION_SHOCK = 0.03
IMBALANCED_FRAC = 0.3     # fraction of episodes starting with one distressed agent

ACTION_NAMES = [ACTION_PROFILES[a]["name"] for a in range(N_ACTIONS)]

def epsilon_at(step):
    frac = min(step / EPS_DECAY_STEPS, 1.0)
    return EPS_START + frac * (EPS_END - EPS_START)

def lambda1_at(episode):
    """Anneal lambda1 from weak to strong over first 70% of training."""
    frac = min(episode / (N_EPISODES * 0.7), 1.0)
    return LAMBDA1_START + frac * (LAMBDA1_FINAL - LAMBDA1_START)

# ── Training function ──────────────────────────────────────────────────────────
def train(risk_aware: bool, seed_offset: int = 0):
    local_rng = np.random.default_rng(SEED + seed_offset)
    agents = [FirmAgent(i, local_rng) for i in range(N_AGENTS)]
    global_step = 0
    bankruptcy_counts = {a: [0, 0] for a in range(N_ACTIONS)}

    for ep in range(N_EPISODES):
        for agent in agents:
            agent.cash_ratio, agent.alive = 1.0, True
        # Imbalanced start: one distressed agent to activate contagion
        if risk_aware and local_rng.random() < IMBALANCED_FRAC:
            agents[local_rng.integers(0, N_AGENTS)].cash_ratio = 0.35
        last_shock = 0.0
        lam1 = lambda1_at(ep) if risk_aware else 0.0

        for _ in range(EPISODE_LENGTH):
            shock = local_rng.choice(shock_pool)
            sb = FirmAgent.shock_bucket(last_shock, pool_std)
            eps = epsilon_at(global_step)

            step_data = []
            for agent in agents:
                if not agent.alive:
                    continue
                s = FirmAgent.state_index(FirmAgent.cash_bucket(agent.cash_ratio), sb)
                a = agent.act(s, eps)
                period_return = (ACTION_PROFILES[a]["base_return"]
                                 + ACTION_PROFILES[a]["shock_beta"] * shock)
                step_data.append((agent, s, a, period_return, ACTION_PROFILES[a]["shock_beta"]))

            # Systemic stress: liquidity cost rises when many agents are cash-poor
            n_stressed = sum(1 for (ag, _, _, _, _) in step_data if ag.cash_ratio < 0.6)
            system_stress = n_stressed / len(step_data) if step_data else 0
            liquidity_cost = LIQUIDITY_DRAG * system_stress if risk_aware else 0.0

            L_sys = (np.mean([-(pr) for (_, _, _, pr, _) in step_data])
                     if step_data else 0.0)
            CVaR_sys_est = max(L_sys, 0.0) * 2.5

            newly_bankrupt = []
            for agent, s, a, period_return, beta in step_data:
                agent.cash_ratio = max(
                    agent.cash_ratio * (1 + period_return - liquidity_cost), 0.0
                )
                if risk_aware:
                    reward = (period_return
                              - lam1 * (beta * CVaR_shock)
                              - LAMBDA2 * CVaR_sys_est)
                else:
                    reward = period_return

                went_bankrupt = agent.cash_ratio <= BANKRUPTCY_THRESHOLD
                if went_bankrupt:
                    reward -= 1.0
                    agent.alive = False
                    newly_bankrupt.append(agent.id)

                s_next = FirmAgent.state_index(
                    FirmAgent.cash_bucket(agent.cash_ratio),
                    FirmAgent.shock_bucket(shock, pool_std)
                )
                agent.update(s, a, reward, s_next)
                bankruptcy_counts[a][1] += 1
                if went_bankrupt:
                    bankruptcy_counts[a][0] += 1

            # Bankruptcy contagion: survivors take a small credit shock
            if risk_aware and newly_bankrupt:
                for agent in agents:
                    if agent.alive:
                        agent.cash_ratio *= (1 - CONTAGION_SHOCK)

            last_shock = shock
            global_step += 1

    return agents, bankruptcy_counts

# ── Run comparison ─────────────────────────────────────────────────────────────
print("\n--- Training risk-neutral agents ---")
agents_neutral, bankr_neutral = train(risk_aware=False)
print("--- Training risk-aware agents (CVaR-penalized, annealed lambda, contagion) ---")
agents_aware, bankr_aware = train(risk_aware=True)

# ── Results ────────────────────────────────────────────────────────────────────
def summarize(name, agents, bankr):
    rows = []
    for agent in agents:
        rows.append({
            "agent": agent.id,
            "alive": agent.alive,
            "final_cash_ratio": round(agent.cash_ratio, 3),
            "dominant_strategy": agent.dominant_strategy(),
        })
    df = pd.DataFrame(rows)
    print(f"\n=== {name} ===")
    print("Strategy distribution:")
    print(df["dominant_strategy"].value_counts().reindex(ACTION_NAMES, fill_value=0).to_string())
    print("Bankruptcy rate by action:")
    for a in range(N_ACTIONS):
        b, c = bankr[a]
        rate = b / c * 100 if c else 0
        print(f"  {ACTION_NAMES[a]:14s}: {rate:.3f}%")
    return df

final_neutral = summarize("RISK-NEUTRAL", agents_neutral, bankr_neutral)
final_aware = summarize("RISK-AWARE (CVaR-penalized)", agents_aware, bankr_aware)
final_neutral.to_csv("results/phase4_final_neutral.csv", index=False)
final_aware.to_csv("results/phase4_final_aware.csv", index=False)

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
x = np.arange(len(ACTION_NAMES))
w = 0.35
for label, agents, color, offset in [
    ("Risk-neutral", agents_neutral, "#c0392b", -w/2),
    ("Risk-aware",   agents_aware,   "#27ae60", +w/2)
]:
    counts = pd.Series([ag.dominant_strategy() for ag in agents]).value_counts()
    counts = counts.reindex(ACTION_NAMES, fill_value=0)
    axes[0].bar(x + offset, counts.values, w, label=label, color=color)
axes[0].set_xticks(x); axes[0].set_xticklabels(ACTION_NAMES)
axes[0].set_title("Learned Strategy: Risk-Neutral vs Risk-Aware")
axes[0].set_ylabel("Number of agents"); axes[0].legend()

rates_n = [bankr_neutral[a][0]/bankr_neutral[a][1]*100 if bankr_neutral[a][1] else 0
           for a in range(N_ACTIONS)]
rates_a = [bankr_aware[a][0]/bankr_aware[a][1]*100 if bankr_aware[a][1] else 0
           for a in range(N_ACTIONS)]
axes[1].bar(x - w/2, rates_n, w, label="Risk-neutral", color="#c0392b")
axes[1].bar(x + w/2, rates_a, w, label="Risk-aware",   color="#27ae60")
axes[1].set_xticks(x); axes[1].set_xticklabels(ACTION_NAMES)
axes[1].set_title("Bankruptcy Rate by Action")
axes[1].set_ylabel("Bankruptcy rate (%)"); axes[1].legend()

plt.tight_layout()
plt.savefig("results/figures/phase4_comparison.png", dpi=150)
print("\nSaved results/figures/phase4_comparison.png")
