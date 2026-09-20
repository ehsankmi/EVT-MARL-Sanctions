[repo_README.md](https://github.com/user-attachments/files/32430155/repo_README.md)# EVT-MARL-Sanctions

**Heavy-Tailed Political-Economic Uncertainty and Firm Decision-Making under Sanctions: An Extreme Value–Multi-Agent Reinforcement Learning Framework**

> *Working paper — Ehsan Karimi (2026)*

---

## Overview

This repository contains the full implementation of an **Extreme Value Theory (EVT) informed Multi-Agent Reinforcement Learning (MARL)** framework for modeling firm decision-making under sanctions-driven political-economic uncertainty.

The framework extends the single-agent POTPG approach (Davar, Godin & Garrido, 2024) to a **multi-agent, networked setting**, where each firm-agent's reward function is penalized by two CVaR terms:

1. **Individual tail risk** — estimated from real historical exchange rate data via Generalized Pareto Distribution (GPD)
2. **Systemic/network risk** — capturing the correlated nature of political shocks that hit all firms simultaneously

### Key Finding

> The risk-aware framework produces substantially **lower-variance and more predictable fragility behavior** under severe shocks than a risk-neutral baseline (fragility SD: 0.07 vs 1.60 across seeds), with the advantage concentrated in **sudden, discrete shocks** rather than gradual trends — a boundary condition identified via crossover analysis on the 2018 and 2025 Iran sanctions episodes.

---

## Data

| Dataset | Source | Coverage |
|---|---|---|
| USD/IRR free-market rate | [bonbast.com archive](https://github.com/SamadiPour/rial-exchange-rates-archive) | 2012–2026 (5,052 daily obs.) |
| Sanctions events | OFAC, Iran Primer, CRS, EU Consilium | 2012–2025 (21 events) |
| TEDPIX (Tehran Stock Exchange) | TSETMC via `pytse-client` | 2008–2026 (4,260 daily obs.) |

**Note:** The USD/IRR series exhibits excess kurtosis of 17.4 and GPD tail index ξ ≈ +0.11 (fat-tailed). TEDPIX shows a bounded tail (ξ ≈ −0.18) due to regulatory daily price bands — a finding discussed in the working paper.

---

## Framework

### Reward Function

Standard RL maximizes expected cumulative reward. The risk-aware extension penalizes tail losses:

**Single-agent (baseline, following POTPG):**
```
R_risk(s,a) = R(s,a) - λ · CVaR_α(L)
```

**Multi-agent extension (this work):**
```
R_MARL(s, a_i, a_{-i}) = R(s, a_i) - λ₁ · CVaR_α(L_i) - λ₂ · CVaR_α(L_sys)
```

where `L_sys = Σ w_j L_j` is the network-wide aggregate loss.

### Fragility Index
```
Fragility_i = ∂²L_i / ∂θ²
```
- Positive → Fragile
- Near zero → Robust
- Negative → Antifragile

---

## Structure

```
EVT-MARL-Sanctions/
├── data/
│   └── sanctions_events.csv     # 21 key sanctions/political events with severity scores
├── notebooks/
│   ├── 01_data_analysis.ipynb   # EVT analysis on USD/IRR and TEDPIX
│   ├── 02_baseline_marl.ipynb   # Risk-neutral baseline (Phase 3)
│   ├── 03_risk_aware_marl.ipynb # CVaR-penalized MARL (Phase 4)
│   └── 04_stress_testing.ipynb  # Crisis scenario replay + fragility analysis (Phase 5)
├── src/
│   ├── agents.py                # FirmAgent class with Q-learning
│   ├── environment.py           # Simulation environment
│   ├── evt_utils.py             # EVT/CVaR estimation utilities
│   └── fragility.py             # Fragility index computation
└── results/
    └── figures/                 # Generated plots
```

---

## Reproducing Results

```bash
git clone https://github.com/[your-username]/EVT-MARL-Sanctions
cd EVT-MARL-Sanctions
pip install -r requirements.txt

# Download exchange rate data
python src/download_data.py

# Run Phase 3–5 notebooks in order
jupyter notebook notebooks/
```

---

## Requirements

```
numpy>=1.24
pandas>=2.0
scipy>=1.10
matplotlib>=3.7
```

---

## Related Work

- Davar, A., Godin, F., & Garrido, J. (2024). *Catastrophic-risk-aware reinforcement learning with extreme-value-theory-based policy gradients.* arXiv:2406.15612
- NS et al. (2023). *Extreme Risk Mitigation in Reinforcement Learning using Extreme Value Theory.* arXiv:2308.13011

---

## Status

- [x] Data collection and EVT analysis
- [x] Risk-neutral MARL baseline
- [x] CVaR-penalized MARL (individual + systemic risk)
- [x] Multi-seed robustness testing (5 seeds, improved estimator)
- [x] Stress-testing and fragility/antifragility analysis
- [ ] Working paper (in preparation)
- [ ] Deep RL extension (MAPPO)

---

## Contact

**Ehsan Karimi**
e.karimi@ut.ac.ir

# EVT-MARL-Sanctions
