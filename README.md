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
```
*Where `L_sys = Σ w_j L_j` represents the network-wide aggregate loss.*

### Fragility Index
We measure the robustness of the multi-agent topology using a fragility index:

```math
Fragility_i = \frac{\partial^2 L_i}{\partial \theta^2}
```
*(Positive → Fragile | Near zero → Robust | Negative → Antifragile)*

---

## 🚀 Interactive Demo & Simulation
*(Screenshot of the Streamlit Dashboard / NetworkX Graph)*
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 420" width="100%" height="100%">
  <rect width="1200" height="420" fill="#0d1117" rx="10"/>
  
  <!-- Panel 1: NetworkX Topology -->
  <rect x="20" y="20" width="370" height="380" fill="#161b22" stroke="#30363d" rx="8"/>
  <text x="205" y="50" fill="#58a6ff" font-family="Segoe UI, Arial, sans-serif" font-size="14" font-weight="bold" text-anchor="middle">Multi-Agent Firm Topology (EVT-MARL)</text>
  <text x="205" y="70" fill="#8b949e" font-family="Segoe UI, Arial, sans-serif" font-size="11" text-anchor="middle">Node Color: Second-Order Fragility (∂²Lᵢ / ∂θ²)</text>
  <line x1="287.8" y1="205.5" x2="308.6" y2="236.8" stroke="#484f58" stroke-width="1.2" opacity="0.55"/>
<line x1="287.8" y1="205.5" x2="333.3" y2="288.3" stroke="#484f58" stroke-width="1.2" opacity="0.55"/>
<line x1="287.8" y1="205.5" x2="270.4" y2="255.9" stroke="#484f58" stroke-width="1.2" opacity="0.55"/>
<line x1="287.8" y1="205.5" x2="258.7" y2="293.8" stroke="#484f58" stroke-width="1.2" opacity="0.55"/>
<line x1="287.8" y1="205.5" x2="253.6" y2="158.2" stroke="#484f58" stroke-width="1.2" opacity="0.55"/>
<line x1="287.8" y1="205.5" x2="290.8" y2="143.2" stroke="#484f58" stroke-width="1.2" opacity="0.55"/>
<line x1="287.8" y1="205.5" x2="332.6" y2="136.2" stroke="#484f58" stroke-width="1.2" opacity="0.55"/>
<line x1="287.8" y1="205.5" x2="276.9" y2="192.5" stroke="#484f58" stroke-width="1.2" opacity="0.55"/>
<line x1="308.6" y1="236.8" x2="333.3" y2="288.3" stroke="#484f58" stroke-width="1.2" opacity="0.55"/>
<line x1="308.6" y1="236.8" x2="270.4" y2="255.9" stroke="#484f58" stroke-width="1.2" opacity="0.55"/>
<line x1="308.6" y1="236.8" x2="258.7" y2="293.8" stroke="#484f58" stroke-width="1.2" opacity="0.55"/>
<line x1="308.6" y1="236.8" x2="276.9" y2="192.5" stroke="#484f58" stroke-width="1.2" opacity="0.55"/>
<line x1="333.3" y1="288.3" x2="270.4" y2="255.9" stroke="#484f58" stroke-width="1.2" opacity="0.55"/>
<line x1="333.3" y1="288.3" x2="258.7" y2="293.8" stroke="#484f58" stroke-width="1.2" opacity="0.55"/>
<line x1="270.4" y1="255.9" x2="258.7" y2="293.8" stroke="#484f58" stroke-width="1.2" opacity="0.55"/>
<line x1="270.4" y1="255.9" x2="232.7" y2="342.8" stroke="#484f58" stroke-width="1.2" opacity="0.55"/>
<line x1="270.4" y1="255.9" x2="201.3" y2="273.3" stroke="#484f58" stroke-width="1.2" opacity="0.55"/>
<line x1="270.4" y1="255.9" x2="276.9" y2="192.5" stroke="#484f58" stroke-width="1.2" opacity="0.55"/>
<line x1="258.7" y1="293.8" x2="232.7" y2="342.8" stroke="#484f58" stroke-width="1.2" opacity="0.55"/>
<line x1="258.7" y1="293.8" x2="201.3" y2="273.3" stroke="#484f58" stroke-width="1.2" opacity="0.55"/>
<line x1="258.7" y1="293.8" x2="182.7" y2="304.1" stroke="#484f58" stroke-width="1.2" opacity="0.55"/>
<line x1="232.7" y1="342.8" x2="201.3" y2="273.3" stroke="#484f58" stroke-width="1.2" opacity="0.55"/>
<line x1="232.7" y1="342.8" x2="182.7" y2="304.1" stroke="#484f58" stroke-width="1.2" opacity="0.55"/>
<line x1="201.3" y1="273.3" x2="182.7" y2="304.1" stroke="#484f58" stroke-width="1.2" opacity="0.55"/>
<line x1="201.3" y1="273.3" x2="129.9" y2="314.3" stroke="#484f58" stroke-width="1.2" opacity="0.55"/>
<line x1="201.3" y1="273.3" x2="171.5" y2="240.3" stroke="#484f58" stroke-width="1.2" opacity="0.55"/>
<line x1="201.3" y1="273.3" x2="133.0" y2="246.3" stroke="#484f58" stroke-width="1.2" opacity="0.55"/>
<line x1="201.3" y1="273.3" x2="166.8" y2="193.4" stroke="#484f58" stroke-width="1.2" opacity="0.55"/>
<line x1="182.7" y1="304.1" x2="129.9" y2="314.3" stroke="#484f58" stroke-width="1.2" opacity="0.55"/>
<line x1="182.7" y1="304.1" x2="171.5" y2="240.3" stroke="#484f58" stroke-width="1.2" opacity="0.55"/>
<line x1="182.7" y1="304.1" x2="133.0" y2="246.3" stroke="#484f58" stroke-width="1.2" opacity="0.55"/>
<line x1="129.9" y1="314.3" x2="171.5" y2="240.3" stroke="#484f58" stroke-width="1.2" opacity="0.55"/>
<line x1="129.9" y1="314.3" x2="133.0" y2="246.3" stroke="#484f58" stroke-width="1.2" opacity="0.55"/>
<line x1="171.5" y1="240.3" x2="133.0" y2="246.3" stroke="#484f58" stroke-width="1.2" opacity="0.55"/>
<line x1="171.5" y1="240.3" x2="87.8" y2="208.1" stroke="#484f58" stroke-width="1.2" opacity="0.55"/>
<line x1="171.5" y1="240.3" x2="166.8" y2="193.4" stroke="#484f58" stroke-width="1.2" opacity="0.55"/>
<line x1="171.5" y1="240.3" x2="199.1" y2="160.5" stroke="#484f58" stroke-width="1.2" opacity="0.55"/>
<line x1="133.0" y1="246.3" x2="87.8" y2="208.1" stroke="#484f58" stroke-width="1.2" opacity="0.55"/>
<line x1="133.0" y1="246.3" x2="166.8" y2="193.4" stroke="#484f58" stroke-width="1.2" opacity="0.55"/>
<line x1="133.0" y1="246.3" x2="130.3" y2="154.5" stroke="#484f58" stroke-width="1.2" opacity="0.55"/>
<line x1="87.8" y1="208.1" x2="166.8" y2="193.4" stroke="#484f58" stroke-width="1.2" opacity="0.55"/>
<line x1="87.8" y1="208.1" x2="130.3" y2="154.5" stroke="#484f58" stroke-width="1.2" opacity="0.55"/>
<line x1="166.8" y1="193.4" x2="130.3" y2="154.5" stroke="#484f58" stroke-width="1.2" opacity="0.55"/>
<line x1="166.8" y1="193.4" x2="142.5" y2="119.6" stroke="#484f58" stroke-width="1.2" opacity="0.55"/>
<line x1="166.8" y1="193.4" x2="199.1" y2="160.5" stroke="#484f58" stroke-width="1.2" opacity="0.55"/>
<line x1="166.8" y1="193.4" x2="206.9" y2="128.4" stroke="#484f58" stroke-width="1.2" opacity="0.55"/>
<line x1="166.8" y1="193.4" x2="253.6" y2="158.2" stroke="#484f58" stroke-width="1.2" opacity="0.55"/>
<line x1="130.3" y1="154.5" x2="142.5" y2="119.6" stroke="#484f58" stroke-width="1.2" opacity="0.55"/>
<line x1="130.3" y1="154.5" x2="199.1" y2="160.5" stroke="#484f58" stroke-width="1.2" opacity="0.55"/>
<line x1="130.3" y1="154.5" x2="206.9" y2="128.4" stroke="#484f58" stroke-width="1.2" opacity="0.55"/>
<line x1="142.5" y1="119.6" x2="199.1" y2="160.5" stroke="#484f58" stroke-width="1.2" opacity="0.55"/>
<line x1="142.5" y1="119.6" x2="206.9" y2="128.4" stroke="#484f58" stroke-width="1.2" opacity="0.55"/>
<line x1="199.1" y1="160.5" x2="206.9" y2="128.4" stroke="#484f58" stroke-width="1.2" opacity="0.55"/>
<line x1="199.1" y1="160.5" x2="235.9" y2="88.4" stroke="#484f58" stroke-width="1.2" opacity="0.55"/>
<line x1="199.1" y1="160.5" x2="253.6" y2="158.2" stroke="#484f58" stroke-width="1.2" opacity="0.55"/>
<line x1="199.1" y1="160.5" x2="290.8" y2="143.2" stroke="#484f58" stroke-width="1.2" opacity="0.55"/>
<line x1="199.1" y1="160.5" x2="276.9" y2="192.5" stroke="#484f58" stroke-width="1.2" opacity="0.55"/>
<line x1="206.9" y1="128.4" x2="235.9" y2="88.4" stroke="#484f58" stroke-width="1.2" opacity="0.55"/>
<line x1="206.9" y1="128.4" x2="253.6" y2="158.2" stroke="#484f58" stroke-width="1.2" opacity="0.55"/>
<line x1="206.9" y1="128.4" x2="290.8" y2="143.2" stroke="#484f58" stroke-width="1.2" opacity="0.55"/>
<line x1="206.9" y1="128.4" x2="276.9" y2="192.5" stroke="#484f58" stroke-width="1.2" opacity="0.55"/>
<line x1="235.9" y1="88.4" x2="253.6" y2="158.2" stroke="#484f58" stroke-width="1.2" opacity="0.55"/>
<line x1="235.9" y1="88.4" x2="290.8" y2="143.2" stroke="#484f58" stroke-width="1.2" opacity="0.55"/>
<line x1="253.6" y1="158.2" x2="290.8" y2="143.2" stroke="#484f58" stroke-width="1.2" opacity="0.55"/>
<line x1="253.6" y1="158.2" x2="332.6" y2="136.2" stroke="#484f58" stroke-width="1.2" opacity="0.55"/>
<line x1="253.6" y1="158.2" x2="276.9" y2="192.5" stroke="#484f58" stroke-width="1.2" opacity="0.55"/>
<line x1="290.8" y1="143.2" x2="332.6" y2="136.2" stroke="#484f58" stroke-width="1.2" opacity="0.55"/>
<line x1="290.8" y1="143.2" x2="276.9" y2="192.5" stroke="#484f58" stroke-width="1.2" opacity="0.55"/>
<line x1="332.6" y1="136.2" x2="276.9" y2="192.5" stroke="#484f58" stroke-width="1.2" opacity="0.55"/>

  <circle cx="287.8" cy="205.5" r="6.0" fill="#3fb950" stroke="#ffffff" stroke-width="1"/>
<circle cx="308.6" cy="236.8" r="8.5" fill="#3fb950" stroke="#ffffff" stroke-width="1"/>
<circle cx="333.3" cy="288.3" r="11.0" fill="#3fb950" stroke="#ffffff" stroke-width="1"/>
<circle cx="270.4" cy="255.9" r="13.5" fill="#2ea043" stroke="#ffffff" stroke-width="1"/>
<circle cx="258.7" cy="293.8" r="6.0" fill="#d29922" stroke="#ffffff" stroke-width="1"/>
<circle cx="232.7" cy="342.8" r="8.5" fill="#ff7b72" stroke="#ffffff" stroke-width="1"/>
<circle cx="201.3" cy="273.3" r="11.0" fill="#3fb950" stroke="#ffffff" stroke-width="1"/>
<circle cx="182.7" cy="304.1" r="13.5" fill="#3fb950" stroke="#ffffff" stroke-width="1"/>
<circle cx="129.9" cy="314.3" r="6.0" fill="#3fb950" stroke="#ffffff" stroke-width="1"/>
<circle cx="171.5" cy="240.3" r="8.5" fill="#2ea043" stroke="#ffffff" stroke-width="1"/>
<circle cx="133.0" cy="246.3" r="11.0" fill="#d29922" stroke="#ffffff" stroke-width="1"/>
<circle cx="87.8" cy="208.1" r="13.5" fill="#ff7b72" stroke="#ffffff" stroke-width="1"/>
<circle cx="166.8" cy="193.4" r="6.0" fill="#3fb950" stroke="#ffffff" stroke-width="1"/>
<circle cx="130.3" cy="154.5" r="8.5" fill="#3fb950" stroke="#ffffff" stroke-width="1"/>
<circle cx="142.5" cy="119.6" r="11.0" fill="#3fb950" stroke="#ffffff" stroke-width="1"/>
<circle cx="199.1" cy="160.5" r="13.5" fill="#2ea043" stroke="#ffffff" stroke-width="1"/>
<circle cx="206.9" cy="128.4" r="6.0" fill="#d29922" stroke="#ffffff" stroke-width="1"/>
<circle cx="235.9" cy="88.4" r="8.5" fill="#ff7b72" stroke="#ffffff" stroke-width="1"/>
<circle cx="253.6" cy="158.2" r="11.0" fill="#3fb950" stroke="#ffffff" stroke-width="1"/>
<circle cx="290.8" cy="143.2" r="13.5" fill="#3fb950" stroke="#ffffff" stroke-width="1"/>
<circle cx="332.6" cy="136.2" r="6.0" fill="#3fb950" stroke="#ffffff" stroke-width="1"/>
<circle cx="276.9" cy="192.5" r="8.5" fill="#2ea043" stroke="#ffffff" stroke-width="1"/>

  <circle cx="70" cy="365" r="6" fill="#3fb950"/><text x="82" y="369" fill="#c9d1d9" font-family="Segoe UI, Arial, sans-serif" font-size="11">Robust (Near 0)</text>
  <circle cx="185" cy="365" r="6" fill="#d29922"/><text x="197" y="369" fill="#c9d1d9" font-family="Segoe UI, Arial, sans-serif" font-size="11">Moderate</text>
  <circle cx="275" cy="365" r="6" fill="#ff7b72"/><text x="287" y="369" fill="#c9d1d9" font-family="Segoe UI, Arial, sans-serif" font-size="11">Fragile (&gt;0)</text>

  <!-- Panel 2: Multi-Seed Fragility Stability -->
  <rect x="410" y="20" width="380" height="380" fill="#161b22" stroke="#30363d" rx="8"/>
  <text x="600" y="50" fill="#58a6ff" font-family="Segoe UI, Arial, sans-serif" font-size="14" font-weight="bold" text-anchor="middle">Multi-Seed Fragility Stability</text>
  <text x="600" y="70" fill="#8b949e" font-family="Segoe UI, Arial, sans-serif" font-size="11" text-anchor="middle">Sudden Discrete Shocks (2018–2025)</text>
  
  <!-- Axes -->
  <line x1="450" y1="330" x2="765" y2="330" stroke="#484f58" stroke-width="1.5"/>
  <line x1="450" y1="100" x2="450" y2="330" stroke="#484f58" stroke-width="1.5"/>
  
  <!-- Shock Vertical Lines -->
  <line x1="520" y1="100" x2="520" y2="330" stroke="#d29922" stroke-dasharray="4,4" stroke-width="1.2"/>
  <line x1="610" y1="100" x2="610" y2="330" stroke="#d29922" stroke-dasharray="4,4" stroke-width="1.2"/>
  <line x1="700" y1="100" x2="700" y2="330" stroke="#d29922" stroke-dasharray="4,4" stroke-width="1.2"/>
  
  <!-- Baseline Variance Band & Curve (Red) -->
  <polygon points="450,315 500,310 520,120 545,295 590,305 610,105 635,290 680,305 700,130 730,300 760,310 760,328 700,240 610,230 520,245 450,325" fill="#ff7b72" opacity="0.22"/>
  <polyline points="450,320 500,315 520,165 545,305 590,312 610,150 635,300 680,312 700,175 730,308 760,318" fill="none" stroke="#ff7b72" stroke-width="2.2"/>
  
  <!-- EVT-MARL Curve (Green - Low Variance) -->
  <polyline points="450,322 500,321 520,298 545,320 590,321 610,295 635,320 680,321 700,300 730,320 760,322" fill="none" stroke="#3fb950" stroke-width="3"/>
  
  <!-- Legend -->
  <line x1="465" y1="358" x2="485" y2="358" stroke="#ff7b72" stroke-width="2.5"/><text x="492" y="362" fill="#c9d1d9" font-family="Segoe UI, Arial, sans-serif" font-size="11">Risk-Neutral Baseline (SD: 1.60)</text>
  <line x1="465" y1="378" x2="485" y2="378" stroke="#3fb950" stroke-width="2.5"/><text x="492" y="382" fill="#c9d1d9" font-family="Segoe UI, Arial, sans-serif" font-size="11">EVT-Aware MARL (SD: 0.07)</text>

  <!-- Panel 3: Empirical Tail Risk (GPD Fit) -->
  <rect x="810" y="20" width="370" height="380" fill="#161b22" stroke="#30363d" rx="8"/>
  <text x="995" y="50" fill="#58a6ff" font-family="Segoe UI, Arial, sans-serif" font-size="14" font-weight="bold" text-anchor="middle">Empirical Tail Behavior (GPD Fit)</text>
  <text x="995" y="70" fill="#8b949e" font-family="Segoe UI, Arial, sans-serif" font-size="11" text-anchor="middle">Individual &amp; Systemic CVaR_α Estimation</text>
  
  <!-- Axes -->
  <line x1="850" y1="330" x2="1155" y2="330" stroke="#484f58" stroke-width="1.5"/>
  <line x1="850" y1="100" x2="850" y2="330" stroke="#484f58" stroke-width="1.5"/>
  
  <!-- Shaded Tail Area -->
  <polygon points="990,208 1050,240 1110,265 1145,278 1145,330 990,330" fill="#f0883e" opacity="0.28"/>
  
  <!-- Fat Tail Curve (USD/IRR) -->
  <path d="M 850,115 Q 920,175 990,208 T 1145,278" fill="none" stroke="#f0883e" stroke-width="2.5"/>
  
  <!-- Bounded Tail Curve (TEDPIX) -->
  <path d="M 850,115 Q 910,200 965,330" fill="none" stroke="#a371f7" stroke-width="2.2" stroke-dasharray="6,3"/>
  
  <!-- POT Threshold Line -->
  <line x1="990" y1="100" x2="990" y2="330" stroke="#ff7b72" stroke-dasharray="4,4" stroke-width="1.5"/>
  <text x="996" y="135" fill="#ff7b72" font-family="Segoe UI, Arial, sans-serif" font-size="10">POT Threshold (CVaR_α)</text>
  
  <!-- Legend -->
  <line x1="860" y1="358" x2="880" y2="358" stroke="#f0883e" stroke-width="2.5"/><text x="887" y="362" fill="#c9d1d9" font-family="Segoe UI, Arial, sans-serif" font-size="11">USD/IRR Fat Tail (ξ ≈ +0.11, Kurt: 17.4)</text>
  <line x1="860" y1="378" x2="880" y2="378" stroke="#a371f7" stroke-width="2.5" stroke-dasharray="4,2"/><text x="887" y="382" fill="#c9d1d9" font-family="Segoe UI, Arial, sans-serif" font-size="11">TEDPIX Bounded Tail (ξ ≈ −0.18)</text>
</svg>
<img width="300" height="105" alt="evt_marl_dashboard" src="https://github.com/user-attachments/assets/d2427447-0042-46ca-9f3c-42a16f2e955c" />

---

## 📁 Repository Structure

```text
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
```

---

## ⚙️ Installation & Usage

1. **Clone the repository:**
```bash
git clone [https://github.com/ehsankmi/EVT-MARL-Sanctions.git](https://github.com/ehsankmi/EVT-MARL-Sanctions.git)
cd EVT-MARL-Sanctions
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the Notebooks or the Dashboard:**
To explore the data and models, run the Jupyter notebooks in order. To launch the interactive dashboard:
```bash
streamlit run app/main.py
```

---

## 📚 Related Work
- Davar, A., Godin, F., & Garrido, J. (2024). *Catastrophic-risk-aware reinforcement learning with extreme-value-theory-based policy gradients.* arXiv:2406.15612
- NS et al. (2023). *Extreme Risk Mitigation in Reinforcement Learning using Extreme Value Theory.* arXiv:2308.13011

---

## 🚀 Project Status
- [x] Empirical data collection and EVT analysis
- [x] Risk-neutral MARL baseline implementation
- [x] CVaR-penalized MARL (modeling individual + systemic risk)
- [x] Multi-seed robustness testing (fragility/antifragility analysis)
- [x] Interactive Network Simulation Dashboard
- [x] Working paper (in preparation)
- [ ] Deep MARL extension (MAPPO architecture)

---

## 📬 Contact
**Ehsan Karimi**  
Independent Researcher | Ph.D. in IT, MBA  
📧 e.karimi@ut.ac.ir | ehsankarimi61@gmail.com
