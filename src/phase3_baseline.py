"""
phase3_baseline.py
------------------
Phase 3: Risk-neutral MARL baseline.

10 firm-agents with independent tabular Q-learning, trained in an
episodic environment where shocks are bootstrapped from real historical
USD/IRR exchange rate data (2012-2026).

This is the risk-NEUTRAL baseline (no CVaR penalty). Run this first
to establish benchmark performance before Phase 4 (risk-aware).

Usage:
    python phase3_baseline.py

Outputs:
    phase3_agent_outcomes.csv   — final strategy per agent
    phase3_baseline_results.png — learning curve and strategy distribution
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.agents import FirmAgent, ACTION_PROFILES, N_ACTIONS, ACTION_NAMES

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
rng = np.random.default_rng(SEED)

# ── Load real shock data ───────────────────────────────────────────────────────
fx = pd.read_csv("data/usd_irr_daily.csv", parse_dates=["date"])
fx["log_ret"] = np.log(fx["usd_sell"]).diff()
shock_pool = fx["log_ret"].dropna().values
pool_std = shock_pool.std()

print(f"Shock pool: {len(shock_pool)} daily log-returns (USD/IRR, 2012-2026)")
print(f"Pool std: {pool_std:.4f}  |  Excess kurtosis: {pd.Series(shock_pool).kurtosis():.2f}")

# ── Hyperparameters ────────────────────────────────────────────────────────────
N_AGENTS = 10
EPISODE_LENGTH = 250        # ~1 trading year
N_EPISODES = 300
BANKRUPTCY_THRESHOLD = 0.25 # firm insolvent below 25% of starting capital
EPS_START, EPS_END, EPS_DECAY_STEPS = 1.0, 0.05, 45000

ACTION_NAMES = [ACTION_PROFILES[a]["name"] for a in range(N_ACTIONS)]

def epsilon_at(step):
    frac = min(step / EPS_DECAY_STEPS, 1.0)
    return EPS_START + frac * (EPS_END - EPS_START)

# ── Training loop ──────────────────────────────────────────────────────────────
agents = [FirmAgent(i, rng) for i in range(N_AGENTS)]
global_step = 0
episode_log = []
action_bankruptcy = {a: [0, 0] for a in range(N_ACTIONS)}   # [bankruptcies, chosen]

for ep in range(N_EPISODES):
    for agent in agents:
        agent.cash_ratio, agent.alive = 1.0, True
    last_shock = 0.0

    for _ in range(EPISODE_LENGTH):
        shock = rng.choice(shock_pool)
        sb = FirmAgent.shock_bucket(last_shock, pool_std)
        eps = epsilon_at(global_step)

        step_data = []
        for agent in agents:
            if not agent.alive:
                continue
            s = FirmAgent.state_index(FirmAgent.cash_bucket(agent.cash_ratio), sb)
            a = agent.act(s, eps)
            profile = ACTION_PROFILES[a]
            period_return = profile["base_return"] + profile["shock_beta"] * shock
            step_data.append((agent, s, a, period_return))

        for agent, s, a, period_return in step_data:
            agent.cash_ratio = max(agent.cash_ratio * (1 + period_return), 0.0)
            reward = period_return                          # risk-neutral: raw return
            went_bankrupt = agent.cash_ratio <= BANKRUPTCY_THRESHOLD
            if went_bankrupt:
                reward -= 1.0
                agent.alive = False
            s_next = FirmAgent.state_index(
                FirmAgent.cash_bucket(agent.cash_ratio),
                FirmAgent.shock_bucket(shock, pool_std)
            )
            agent.update(s, a, reward, s_next)
            action_bankruptcy[a][1] += 1
            if went_bankrupt:
                action_bankruptcy[a][0] += 1

        last_shock = shock
        global_step += 1

    if ep % 10 == 0:
        n_alive = sum(ag.alive for ag in agents)
        avg_cash = np.mean([ag.cash_ratio for ag in agents if ag.alive]) if n_alive else 0
        episode_log.append({"episode": ep, "alive_end_of_ep": n_alive,
                             "avg_cash_ratio": round(avg_cash, 3),
                             "epsilon": round(epsilon_at(global_step), 3)})

# ── Results ────────────────────────────────────────────────────────────────────
log_df = pd.DataFrame(episode_log)

print("\n=== Bankruptcy rate by action (risk-neutral baseline) ===")
for a in range(N_ACTIONS):
    b, c = action_bankruptcy[a]
    rate = b / c * 100 if c else 0
    print(f"  {ACTION_NAMES[a]:14s}: chosen {c:6d} times, bankruptcy rate = {rate:.2f}%")

final_rows = []
for agent in agents:
    final_rows.append({
        "agent": agent.id,
        "alive": agent.alive,
        "final_cash_ratio": round(agent.cash_ratio, 3),
        "dominant_strategy": agent.dominant_strategy(),
    })
final_df = pd.DataFrame(final_rows)
print("\n=== Final agent outcomes ===")
print(final_df.to_string(index=False))
final_df.to_csv("results/phase3_agent_outcomes.csv", index=False)

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(log_df["episode"], log_df["alive_end_of_ep"], color="#2c3e50")
axes[0].set_title("Firms Surviving to End of Episode\n(Risk-Neutral Baseline)")
axes[0].set_xlabel("Training episode")
axes[0].set_ylabel("Firms alive out of 10")

strategy_counts = final_df["dominant_strategy"].value_counts().reindex(ACTION_NAMES, fill_value=0)
axes[1].bar(strategy_counts.index, strategy_counts.values,
            color=["#27ae60", "#f39c12", "#c0392b"])
axes[1].set_title("Final Learned Strategy\n(Risk-Neutral Baseline)")
axes[1].set_ylabel("Number of agents")

plt.tight_layout()
plt.savefig("results/figures/phase3_baseline_results.png", dpi=150)
print("\nSaved results/phase3_agent_outcomes.csv and results/figures/phase3_baseline_results.png")
