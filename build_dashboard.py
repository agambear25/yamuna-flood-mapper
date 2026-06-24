"""
Build a static HTML dashboard for the Yamuna flood project.

Reads the outputs produced by the notebooks (data/outputs/), regenerates
publication-styled maps (scale bar + north arrow + clean typography), computes
the headline stats, and writes a self-contained `index.html` at the repo root.

Run:  python build_dashboard.py
"""
from pathlib import Path
import csv

import numpy as np
import rasterio
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, ListedColormap

REPO = Path(__file__).resolve().parent
OUT = REPO / "data" / "outputs"
W, S, E, N = 77.18, 28.50, 77.38, 28.78
LAT_MID = (S + N) / 2


def load(name):
    with rasterio.open(OUT / name) as src:
        return src.read(1).astype("float32"), src.transform


# ----------------------------------------------------------------------------- stats
flood, transform = load("flood_mask.tif")
risk, _ = load("flood_susceptibility.tif")
zones, _ = load("risk_zones.tif")
dem, _ = load("dem.tif")
valid = (flood != 255) & np.isfinite(dem)

px_km2 = (abs(transform.a) * 111.32 * np.cos(np.deg2rad(LAT_MID))) * (abs(transform.e) * 111.32)
flood_km2 = float((flood == 1).sum() * px_km2)
mapped_km2 = float(valid.sum() * px_km2)
zone_km2 = {z: float((zones == z).sum() * px_km2) for z in (1, 2, 3, 4)}


def auc(score):
    f = score[valid & (flood == 1)]
    nf = score[valid & (flood == 0)]
    rng = np.random.default_rng(0)
    f = rng.choice(f, min(50000, len(f)), replace=False)
    nf = rng.choice(nf, min(50000, len(nf)), replace=False)
    a = np.concatenate([f, nf])
    r = a.argsort().argsort() + 1
    return (r[: len(f)].sum() - len(f) * (len(f) + 1) / 2) / (len(f) * len(nf))


model_auc = float(auc(np.nan_to_num(risk)))

top_roads = []
with open(OUT / "top_risk_roads.csv") as fh:
    for row in csv.DictReader(fh):
        top_roads.append((row["name"], float(row["mean_risk"])))
top_roads = top_roads[:10]

# --- Phase 2 stats ---
SPATIAL_AUC = 0.92   # XGBoost spatial-block CV (notebook 6)
districts = []
people_at_risk = 0
dist_csv = OUT / "district_risk_scores.csv"
if dist_csv.exists():
    with open(dist_csv) as fh:
        for row in csv.DictReader(fh):
            districts.append((row["district"], int(row["population"]),
                              int(row["pop_at_risk"]), float(row["pct_at_risk"])))
    people_at_risk = sum(d[2] for d in districts)
has_phase2 = bool(districts) and (OUT / "shap_summary.png").exists()


# ----------------------------------------------------------------------------- maps
ls = LightSource(azdeg=315, altdeg=45)
px_x = abs(transform.a) * 111_320 * np.cos(np.deg2rad(LAT_MID))
px_y = abs(transform.e) * 111_320
hill = ls.hillshade(np.nan_to_num(dem, nan=float(np.nanmean(dem))), vert_exag=5, dx=px_x, dy=px_y)
extent = [W, E, S, N]
rivers = gpd.read_file(OUT / "rivers.geojson")
roads = gpd.read_file(OUT / "roads.geojson")


def add_scalebar(ax, km=5):
    deg = km / (111.32 * np.cos(np.deg2rad(LAT_MID)))
    x0, y0 = W + 0.012, S + 0.015
    ax.plot([x0, x0 + deg], [y0, y0], color="#1a1a1a", lw=3, solid_capstyle="butt")
    ax.text(x0 + deg / 2, y0 + 0.005, f"{km} km", ha="center", va="bottom",
            fontsize=9, color="#1a1a1a")


def add_north(ax):
    x, y = E - 0.018, N - 0.012
    ax.annotate("N", xy=(x, y), xytext=(x, y - 0.028),
                arrowprops=dict(arrowstyle="-|>", color="#1a1a1a", lw=1.6),
                ha="center", va="center", fontsize=13, fontweight="bold", color="#1a1a1a")


def style_axes(ax, title, subtitle):
    ax.set_xlim(W, E); ax.set_ylim(S, N)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#cccccc")
    ax.set_title(f"{title}\n", fontsize=15, fontweight="600", loc="left", color="#1a2b3c")
    ax.text(0.0, 1.005, subtitle, transform=ax.transAxes, fontsize=9.5, color="#5a6b7c")


# --- risk map ---
fig, ax = plt.subplots(figsize=(8.2, 9.6))
ax.imshow(hill, cmap="gray", extent=extent, alpha=0.55)
rim = ax.imshow(np.ma.masked_invalid(risk), cmap="YlOrRd", alpha=0.68, extent=extent, vmin=0, vmax=1)
rivers.plot(ax=ax, color="#1f5fa8", linewidth=1.3)
roads.plot(ax=ax, color="#222222", linewidth=0.28, alpha=0.45)
add_scalebar(ax); add_north(ax)
style_axes(ax, "Flood-risk surface",
           "Weighted-overlay model · validated against Sentinel-1 SAR (AUC %.2f)" % model_auc)
cb = fig.colorbar(rim, ax=ax, shrink=0.42, pad=0.02)
cb.set_label("Relative flood risk", fontsize=9); cb.ax.tick_params(labelsize=8)
plt.tight_layout()
plt.savefig(OUT / "dashboard_risk_map.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close()

# --- flood extent map ---
fig, ax = plt.subplots(figsize=(8.2, 9.6))
ax.imshow(hill, cmap="gray", extent=extent, alpha=0.55)
ax.imshow(np.ma.masked_where(flood != 1, flood), cmap=ListedColormap(["#1c7ed6"]),
          extent=extent, alpha=0.85)
rivers.plot(ax=ax, color="#0b3d70", linewidth=1.3)
add_scalebar(ax); add_north(ax)
style_axes(ax, "Observed flood extent",
           "Sentinel-1 SAR change detection · Yamuna, Delhi · July 2023 · %.0f km²" % flood_km2)
plt.tight_layout()
plt.savefig(OUT / "dashboard_flood_extent.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("Maps written.")


# ----------------------------------------------------------------------------- html
def kpi(value, label, sub=""):
    return f'<div class="kpi"><div class="kpi-v">{value}</div><div class="kpi-l">{label}</div><div class="kpi-s">{sub}</div></div>'


zone_total = sum(zone_km2.values()) or 1
zone_colors = {1: "#fed976", 2: "#fd8d3c", 3: "#e31a1c", 4: "#800026"}
zone_names = {1: "Low", 2: "Medium", 3: "High", 4: "Very High"}
zone_bars = "".join(
    f'<div class="zrow"><span class="zname">{zone_names[z]}</span>'
    f'<span class="zbar"><span style="width:{100*zone_km2[z]/zone_total:.0f}%;background:{zone_colors[z]}"></span></span>'
    f'<span class="zval">{zone_km2[z]:.0f} km²</span></div>'
    for z in (4, 3, 2, 1)
)

road_rows = "".join(
    f'<tr><td>{i+1}</td><td>{name}</td>'
    f'<td><span class="rbar"><span style="width:{r*100:.0f}%"></span></span></td>'
    f'<td class="rnum">{r:.2f}</td></tr>'
    for i, (name, r) in enumerate(top_roads)
)

district_rows = "".join(
    f'<tr><td>{n}</td><td class="rnum">{p:,}</td><td class="rnum">{r:,}</td>'
    f'<td class="rnum">{pct:.1f}</td></tr>'
    for (n, p, r, pct) in districts[:8]
)

phase2_html = f"""
  <h2>Phase 2 &mdash; machine-learning model</h2>
  <p class="lead">An XGBoost model trained on the SAR flood with seven terrain features, validated by
  <b>spatial</b> cross-validation (AUC {SPATIAL_AUC:.2f}, vs. {model_auc:.2f} for the weighted overlay).
  Ordinary random CV would report 0.97 &mdash; that gap is spatial leakage, which spatial CV removes.</p>
  <div class="grid2">
    <div class="card"><img src="data/outputs/compare_overlay_vs_ml.png" alt="overlay vs ML"/>
      <div class="cap">Observed flood vs. Phase 1 overlay vs. Phase 2 ML &mdash; the ML map resolves finer floodplain structure.</div></div>
    <div class="card"><img src="data/outputs/shap_summary.png" alt="SHAP feature importance"/>
      <div class="cap">SHAP: distance-to-river and HAND drive risk; built-up pushes toward &ldquo;dry&rdquo; (a SAR blind-spot, not safety).</div></div>
  </div>

  <h2>People at risk by district</h2>
  <div class="grid2">
    <div class="card"><img src="data/outputs/people_at_risk_by_district.png" alt="people at risk by district"/></div>
    <div class="card"><table>
      <tr><th>District</th><th class="rnum">Population</th><th class="rnum">At risk</th><th class="rnum">%</th></tr>
      {district_rows}
    </table><div class="cap">Residents where modelled susceptibility &ge; 0.5 (WorldPop 2020). ~{people_at_risk/1000:.0f}k people total.</div></div>
  </div>
""" if has_phase2 else ""

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Yamuna Urban Flood Risk · Delhi</title>
<style>
  :root {{ --ink:#1a2b3c; --muted:#5a6b7c; --line:#e3e8ee; --bg:#f6f8fa; --card:#ffffff; --accent:#1c7ed6; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
          color:var(--ink); background:var(--bg); line-height:1.55; }}
  .wrap {{ max-width:1100px; margin:0 auto; padding:0 22px 60px; }}
  header {{ background:linear-gradient(135deg,#0b3d70,#1c7ed6); color:#fff; padding:42px 0 34px; }}
  header .wrap {{ padding-bottom:0; }}
  h1 {{ margin:0 0 6px; font-size:30px; font-weight:650; letter-spacing:-0.2px; }}
  .tag {{ font-size:15px; opacity:.92; max-width:720px; }}
  .meta {{ margin-top:14px; font-size:12.5px; opacity:.8; }}
  .kpis {{ display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin:-26px 0 30px; }}
  .kpi {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:18px 16px;
          box-shadow:0 1px 3px rgba(20,40,70,.06); }}
  .kpi-v {{ font-size:26px; font-weight:680; color:var(--ink); }}
  .kpi-l {{ font-size:13px; font-weight:600; margin-top:2px; }}
  .kpi-s {{ font-size:11.5px; color:var(--muted); margin-top:2px; }}
  h2 {{ font-size:19px; margin:34px 0 12px; font-weight:620; }}
  .lead {{ font-size:13.5px; color:var(--muted); margin:-4px 0 14px; max-width:760px; }}
  .grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:22px; }}
  .card {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:16px;
           box-shadow:0 1px 3px rgba(20,40,70,.06); }}
  .card img {{ width:100%; border-radius:8px; display:block; }}
  .cap {{ font-size:12.5px; color:var(--muted); margin-top:8px; }}
  table {{ width:100%; border-collapse:collapse; font-size:13.5px; }}
  th, td {{ text-align:left; padding:8px 10px; border-bottom:1px solid var(--line); }}
  th {{ font-size:11.5px; text-transform:uppercase; letter-spacing:.4px; color:var(--muted); }}
  td.rnum {{ font-variant-numeric:tabular-nums; font-weight:600; text-align:right; width:54px; }}
  .rbar {{ display:inline-block; width:120px; height:8px; background:#eef1f4; border-radius:4px; overflow:hidden; vertical-align:middle; }}
  .rbar span {{ display:block; height:100%; background:linear-gradient(90deg,#fd8d3c,#800026); }}
  .zrow {{ display:flex; align-items:center; gap:10px; margin:9px 0; font-size:13.5px; }}
  .zname {{ width:78px; }}
  .zbar {{ flex:1; height:11px; background:#eef1f4; border-radius:6px; overflow:hidden; }}
  .zbar span {{ display:block; height:100%; }}
  .zval {{ width:64px; text-align:right; font-variant-numeric:tabular-nums; color:var(--muted); }}
  iframe {{ width:100%; height:520px; border:1px solid var(--line); border-radius:10px; }}
  .note {{ background:#fff8e6; border:1px solid #ffe39e; border-radius:10px; padding:14px 16px; font-size:13px; }}
  .note b {{ color:#9a6b00; }}
  footer {{ margin-top:40px; padding-top:18px; border-top:1px solid var(--line); font-size:12.5px; color:var(--muted); }}
  a {{ color:var(--accent); text-decoration:none; }}
  @media (max-width:760px) {{ .kpis{{grid-template-columns:repeat(2,1fr);}} .grid2{{grid-template-columns:1fr;}} }}
</style>
</head>
<body>
<header><div class="wrap">
  <h1>Yamuna Urban Flood Risk · Delhi</h1>
  <div class="tag">Sentinel-1 SAR detection of the July 2023 flood, and a flood-risk model
  validated against it — built end-to-end in Python.</div>
  <div class="meta">Area of interest: Yamuna corridor, Delhi (28.50–28.78°N, 77.18–77.38°E) · 10 m resolution</div>
</div></header>

<div class="wrap">
  <div class="kpis">
    {kpi(f"{flood_km2:.0f} km²", "Observed flood", "SAR, July 2023")}
    {kpi(f"{model_auc:.2f} &rarr; {SPATIAL_AUC:.2f}", "Model AUC", "overlay &rarr; ML")}
    {kpi(f"{people_at_risk/1000:.0f}k", "People at risk", "in high-risk zones")}
    {kpi(f"{mapped_km2:.0f} km²", "Area mapped", "10 m grid")}
  </div>

  <div class="grid2">
    <div class="card"><img src="data/outputs/dashboard_risk_map.png" alt="Flood risk map"/>
      <div class="cap">Relative flood risk from a weighted overlay of elevation, distance-to-river, slope and built-up surface.</div></div>
    <div class="card"><img src="data/outputs/dashboard_flood_extent.png" alt="Observed flood extent"/>
      <div class="cap">Flood extent detected from Sentinel-1 radar change detection (June vs. July 2023).</div></div>
  </div>

  <h2>Risk-zone breakdown</h2>
  <div class="card">{zone_bars}</div>

  <h2>Highest-risk roads &mdash; motorist watch-list</h2>
  <div class="card"><table>
    <tr><th>#</th><th>Road</th><th>Risk</th><th class="rnum">Score</th></tr>
    {road_rows}
  </table>
  <div class="cap">Mean modelled flood risk along each major road. A starting point for driver advisories.</div></div>

  <h2>Explore the interactive map</h2>
  <div class="card"><iframe src="data/outputs/yamuna_flood_risk_interactive.html" title="Interactive flood risk map"></iframe></div>
{phase2_html}
  <h2>Method &amp; honest limitations</h2>
  <div class="note">
    <b>What this shows:</b> riverine floodplain risk, validated against a real event.
    <b>What it doesn't:</b> SAR is largely blind to shallow street-waterlogging between buildings,
    so the road ranking reflects floodplain exposure, not live street flooding. Risk is
    <i>relative</i> within this area and validated on a single event. Built-up area anti-correlates
    with this riverine flood (it sits on higher ground) and is weighted low. A machine-learning
    Phase&nbsp;2 extends this with spatial cross-validation and population exposure.
  </div>

  <footer>
    Data: ESA Sentinel-1 &amp; Copernicus DEM and ESA WorldCover (via Google Earth Engine),
    OpenStreetMap. Built with rasterio, geopandas, matplotlib, folium.
  </footer>
</div>
</body>
</html>"""

(REPO / "index.html").write_text(html, encoding="utf-8")
print("Wrote index.html")
print(f"flood={flood_km2:.0f} km2 | AUC={model_auc:.2f} | mapped={mapped_km2:.0f} km2 | veryhigh={zone_km2[4]:.0f} km2")
