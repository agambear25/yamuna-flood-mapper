# 🌊 Yamuna Flood Map — Delhi

**An interactive flood-risk map for Delhi, built from free satellite data.** It shows where the record **July 2023 Yamuna flood** actually hit, shades the city by how flood-prone each area is, ranks the **most at-risk roads**, and estimates **how many people live in the danger zones**.

![Delhi flood risk map](data/outputs/dashboard_risk_map.png)

---

## What it does

- 🛰️ **Maps the real July 2023 flood** from satellite radar (which sees through monsoon cloud).
- 🗺️ **Shades the whole city by flood risk** and flags the **most flood-prone roads** — useful for commuters and emergency teams.
- 👥 **Estimates ~600,000 people** living in high-risk zones, broken down by district.
- 📊 **Wraps it all in an interactive dashboard** anyone can open in a browser.

## Why it stands out

- Turns free, public satellite data into a **clear, decision-ready map** — no expensive software.
- The risk map is **checked against the real flood**: it correctly separates flood-prone from safe ground **~84% of the time** (a machine-learning version reaches **~92%**).
- Tells a **complete story** end to end: detect the flood → map the risk → identify who's affected → ship a dashboard.

## How it works, in plain English

1. **Find** — satellite radar spots where water sat during the flood.
2. **Learn** — a model studies the terrain at those flooded spots (low? flat? near the river?).
3. **Map** — it shades the whole city by risk, ranks the riskiest roads, and counts people at risk.

## Honest about its limits

This is a **planning and awareness tool, not an official flood warning.** It detects river flooding well, but **can't see shallow street-flooding between buildings**. The risk map is *relative* and based on this one flood event. Being upfront about this is part of doing it properly.

## Built with

`Python` · `Google Earth Engine` (free satellite data) · `machine learning` · `OpenStreetMap`

*This grew into a multi-city engine: [urban-flood-ml](https://github.com/agambear25/urban-flood-ml) — the same approach packaged to run Delhi, Mumbai, Bengaluru, and Chandigarh from one tool.*

---

<details>
<summary><b>🔧 Technical details</b> (for engineers — click to expand)</summary>

Built as documented Jupyter notebooks in two phases:

**Phase 1 — `notebooks/`** (SAR detection + transparent risk model)
1. `01_get_sar_data` — pull pre/post Sentinel-1 VV composites from Google Earth Engine.
2. `02_flood_mask` — speckle filter, change-detect, threshold → binary flood mask (~29 km²).
3. `03_context_layers` — DEM, slope, distance-to-river, built-up, aligned to the flood grid.
4. `04_risk_map` — weighted-overlay risk (**AUC 0.84**), road ranking, publication maps.

**Phase 2 — `phase2/`** (machine learning)
5–8. Feature engineering (HAND, curvature) → **XGBoost with spatial cross-validation (AUC 0.92)** → SHAP explainability → city-wide susceptibility → **WorldPop people-at-risk by district (~613k)**.

`build_dashboard.py` assembles everything into `index.html`.

**Reproduce**
```bash
conda env create -f environment.yml && conda activate yamuna-flood
jupyter lab                  # run notebooks/01–04, then phase2/05–08
python build_dashboard.py    # regenerate the dashboard
```
Needs a free Google Earth Engine account; set `EE_PROJECT` in `01_get_sar_data.ipynb`.

**Data:** ESA Sentinel-1, Copernicus DEM, ESA WorldCover, WorldPop (via Google Earth Engine), OpenStreetMap.

**Honest caveats (detail):** SAR sees riverine flooding, not in-street waterlogging; `built-up` anti-correlates with this riverine flood (it sits on higher ground); risk is relative and validated on a single event.

</details>
