[data_README.md](https://github.com/user-attachments/files/32430281/data_README.md)
# Data

## USD/IRR Free-Market Exchange Rate

**Source:** [rial-exchange-rates-archive](https://github.com/SamadiPour/rial-exchange-rates-archive) (bonbast.com, updated daily via GitHub Actions)

**Coverage:** 2012-10-09 to present (~5,000+ daily observations)

**How to download:**
```python
import requests, json, pandas as pd

url = "https://raw.githubusercontent.com/SamadiPour/rial-exchange-rates-archive/data/gregorian_imp.min.json"
raw = requests.get(url).json()

rows = []
for date_str, vals in raw.items():
    usd = vals.get("usd", {}).get("sell")
    if usd:
        rows.append({"date": date_str, "usd_sell": usd})

df = pd.DataFrame(rows)
df["date"] = pd.to_datetime(df["date"], format="%Y/%m/%d")
df = df.sort_values("date").reset_index(drop=True)
df.to_csv("usd_irr_daily.csv", index=False)
```

**EVT findings:** Excess kurtosis = 17.4; GPD tail index ξ ≈ +0.11 (fat-tailed, Pareto-type).

---

## Tehran Stock Exchange Index (TEDPIX)

**Source:** TSETMC official data via `pytse-client` Python package

**Coverage:** 2008-12-06 to present (~4,260 trading days)

**How to download:**
```python
pip install pytse-client

import pytse_client as tse
index = tse.FinancialIndex(symbol="شاخص کل")
df = index.history
df.to_csv("tedpix_full.csv", index=False)
```

**EVT findings:** Excess kurtosis = 2.8; GPD tail index ξ ≈ −0.18 (bounded tail, due to regulatory ±5% daily price band). See paper Section 4 for discussion.

---

## Sanctions Events Dataset

**File:** `sanctions_events.csv`

**Coverage:** 21 key political-economic events, 2012–2025

**Columns:**
- `date` — event date (YYYY-MM-DD)
- `event` — event description
- `type` — `escalation` or `relief`
- `severity_1to5` — subjective severity score
- `source` — OFAC, Iran Primer, Congressional Research Service, EU Consilium

**Sources:**
- [OFAC Iran Sanctions](https://ofac.treasury.gov/sanctions-programs-and-country-information/iran-sanctions)
- [Iran Primer (USIP)](https://iranprimer.usip.org/resource/sanctions)
- [Congressional Research Service](https://crsreports.congress.gov)
- [EU Council Consilium](https://www.consilium.europa.eu)
