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

Fragility_i = \frac{\partial^2 L_i}{\partial \theta^2}

EVT-MARL-Sanctions/
├── data/
│   └── sanctions_events.csv     # 21 key sanctions/political events
├── notebooks/
│   ├── 01_data_analysis.ipynb   # EVT analysis on USD/IRR and TEDPIX
│   ├── 02_baseline_marl.ipynb   # Risk-neutral baseline
│   ├── 03_risk_aware_marl.ipynb # CVaR-penalized MARL (Individual + Systemic)
│   └── 04_stress_testing.ipynb  # Crisis scenario replay & fragility analysis
├── src/
│   ├── agents.py                # FirmAgent class with Q-learning
│   ├── environment.py           # Simulation environment & NetworkX dynamics
│   ├── evt_utils.py             # EVT / CVaR estimation utilities
│   └── fragility.py             # Fragility index computation
├── app/
│   └── main.py                  # Streamlit Interactive Dashboard
└── requirements.txt

git clone https://github.com/ehsankmi/EVT-MARL-Sanctions.git
cd EVT-MARL-Sanctions

pip install -r requirements.txt

streamlit run app/main.py

📚 Related Work

    Davar, A., Godin, F., & Garrido, J. (2024). Catastrophic-risk-aware reinforcement learning with extreme-value-theory-based policy gradients. arXiv:2406.15612

    NS et al. (2023). Extreme Risk Mitigation in Reinforcement Learning using Extreme Value Theory. arXiv:2308.13011

🚀 Project Status

    [x] Empirical data collection and EVT analysis

    [x] Risk-neutral MARL baseline implementation

    [x] CVaR-penalized MARL (modeling individual + systemic risk)

    [x] Multi-seed robustness testing (fragility/antifragility analysis)

    [x] Interactive Network Simulation Dashboard

    [ ] Working paper (in preparation)

    [ ] Deep MARL extension (MAPPO architecture)

📬 Contact

Ehsan Karimi

Independent Researcher | Ph.D. in IT, MBA

📧 e.karimi@ut.ac.ir | ehsankarimi61@gmail.com
