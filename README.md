# EVT-MARL: Risk-Averse Multi-Agent Reinforcement Learning via Extreme Value Theory
*Modeling Firm Decision-Making under Heavy-Tailed Political-Economic Uncertainty*

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![NetworkX](https://img.shields.io/badge/NetworkX-Graph_Simulation-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Interactive_Dashboard-red)
![Status](https://img.shields.io/badge/Status-Working_Paper_2026-yellow)

> *Working Paper — Ehsan Karimi (2026)*

---

## 📌 Overview

This repository provides an **Extreme Value Theory (EVT) informed Multi-Agent Reinforcement Learning (MARL)** framework. It is designed to model firm decision-making under severe, heavy-tailed political-economic uncertainty (e.g., sanctions-driven exchange rate shocks).

Standard MARL often assumes well-behaved environmental stochasticity, leaving multi-agent networks vulnerable to catastrophic failure. This framework extends the single-agent POTPG approach (Davar, Godin & Garrido, 2024) to a **multi-agent, networked setting**. Each agent's (firm's) reward function is penalized by two Conditional Value-at-Risk (CVaR) terms:

1. **Individual Tail Risk:** Estimated from real historical exchange rate data using a Generalized Pareto Distribution (GPD).
2. **Systemic/Network Risk:** Capturing the correlated nature of macro-shocks that impact the entire network simultaneously.

### 🔑 Key Finding
The EVT-aware framework produces substantially **lower-variance and more predictable fragility behavior** under severe shocks compared to a risk-neutral baseline (fragility SD: 0.07 vs 1.60 across seeds). This advantage is highly concentrated during **sudden, discrete extreme events** rather than gradual trends—a boundary condition identified via crossover analysis on empirical sanctions data (2018–2025).

---

## 📊 Empirical Data & Tail Risk

The simulation is grounded in empirical financial and political data:

| Dataset | Coverage | Description | Tail Behavior |
|---------|----------|-------------|---------------|
| **USD/IRR Free-Market Rate** | 2012–2026 (5,052 obs.) | Real exchange rate shocks | Fat-tailed (Excess kurtosis: 17.4, GPD ξ ≈ +0.11) |
| **Sanctions Events** | 2012–2025 (21 events) | OFAC, Iran Primer, EU Consilium | Discrete extreme political shocks |
| **TEDPIX (Tehran Stock Exchange)** | 2008–2026 (4,260 obs.) | Market response index | Bounded tail (ξ ≈ −0.18) due to regulatory daily price bands |

---

## 🧠 Framework & Mathematics

### Reward Function
Standard RL maximizes expected cumulative reward. Our risk-aware MARL penalizes extreme tail losses using EVT:

```math
R_{MARL}(s, a_i, a_{-i}) = R(s, a_i) - \lambda_1 \cdot CVaR_\alpha(L_i) - \lambda_2 \cdot CVaR_\alpha(L_{sys})
