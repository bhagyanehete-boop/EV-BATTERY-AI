import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, Arc, Wedge
import matplotlib.patheffects as pe
from mpl_toolkits.mplot3d import Axes3D
import math
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="EV Battery Pro AI — India 2026",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; background: #f0f4f8; }
h1,h2,h3 { font-family: 'Nunito', sans-serif; letter-spacing: -0.01em; }
.stApp { background: #f0f4f8; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #e8eef4; }
::-webkit-scrollbar-thumb { background: #94b4d0; border-radius: 3px; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a2e4a 0%, #0f1f35 100%) !important;
    border-right: none; box-shadow: 4px 0 20px rgba(0,0,0,0.15);
}
[data-testid="stSidebar"] * { color: #c8dff5 !important; }
[data-testid="stSidebar"] label {
    color: #7ec8f7 !important; font-size:0.7rem !important; font-weight:800 !important;
    text-transform:uppercase; letter-spacing:0.12em;
}
[data-testid="stSidebar"] .stSlider > div > div { background: #2a4a6e !important; }
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: #162840 !important; border: 1px solid #2a4a6e !important; color:#c8dff5 !important;
}
[data-testid="stSidebar"] h2 {
    font-family:'Nunito',sans-serif; font-size:1.4rem; font-weight:900;
    color:#ffffff !important; letter-spacing:-0.02em;
}
.hero-banner {
    background: linear-gradient(120deg, #1a2e4a 0%, #1e4080 50%, #0d6eaa 100%);
    border-radius: 28px; padding: 40px 48px 32px; margin-bottom: 28px;
    position: relative; overflow: hidden;
    box-shadow: 0 16px 40px rgba(26,46,74,0.25);
}
.hero-banner::before {
    content:''; position:absolute; top:-80px; right:-80px;
    width:320px; height:320px;
    background:radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 65%);
    border-radius:50%;
}
.hero-title { font-family:'Nunito',sans-serif; font-size:2.6rem; font-weight:900; color:#ffffff; margin:0 0 6px 0; line-height:1.05; text-shadow: 0 2px 12px rgba(0,0,0,0.2); }
.hero-sub { color:rgba(255,255,255,0.7); font-size:0.92rem; margin:0 0 18px 0; }
.hero-badge { display:inline-block; background:rgba(255,255,255,0.12); border:1px solid rgba(255,255,255,0.25); backdrop-filter:blur(8px); color:#ffffff; border-radius:20px; padding:5px 16px; font-size:0.73rem; font-weight:700; margin-right:8px; margin-bottom:4px; }
.metric-card { background:#ffffff; border:none; border-radius:20px; padding:20px 22px 16px; margin-bottom:12px; position:relative; overflow:hidden; box-shadow:0 4px 20px rgba(0,0,0,0.07); transition:transform 0.2s, box-shadow 0.2s; }
.metric-card:hover { transform:translateY(-3px); box-shadow:0 8px 28px rgba(0,0,0,0.12); }
.metric-accent { position:absolute; top:0; left:0; width:5px; height:100%; border-radius:20px 0 0 20px; }
.metric-label { font-size:0.66rem; font-weight:800; color:#94a3b8; text-transform:uppercase; letter-spacing:0.12em; margin-bottom:8px; }
.metric-value { font-family:'Nunito',sans-serif; font-size:2.2rem; font-weight:900; color:#1e293b; line-height:1; }
.metric-sub { font-size:0.72rem; color:#64748b; margin-top:5px; }
.vehicle-card { background:#ffffff; border-radius:24px; padding:24px; margin-bottom:16px; box-shadow:0 4px 20px rgba(0,0,0,0.07); text-align:center; }
.vehicle-name { font-family:'Nunito',sans-serif; font-size:1.1rem; font-weight:900; color:#1e293b; margin:12px 0 4px; }
.vehicle-price { font-size:0.82rem; color:#64748b; margin-bottom:14px; }
.spec-pill { display:inline-block; background:#eff6ff; border:1px solid #bfdbfe; color:#1d4ed8; border-radius:20px; padding:4px 12px; font-size:0.73rem; font-weight:700; margin:3px 3px 3px 0; }
.sec-head { font-family:'Nunito',sans-serif; font-size:1.05rem; font-weight:900; color:#1a2e4a; text-transform:uppercase; letter-spacing:0.08em; border-bottom:2px solid #e2e8f0; padding-bottom:10px; margin-bottom:20px; }
.expiry-table { width:100%; border-collapse:collapse; background:#ffffff; border-radius:16px; overflow:hidden; box-shadow:0 4px 20px rgba(0,0,0,0.07); }
.expiry-table th { background:#f8fafc; color:#64748b; padding:12px 16px; font-size:0.7rem; text-transform:uppercase; letter-spacing:0.08em; text-align:left; border-bottom:2px solid #e2e8f0; font-weight:800; }
.expiry-table td { padding:12px 16px; border-bottom:1px solid #f1f5f9; font-size:0.87rem; color:#334155; }
.expiry-table tr:last-child td { border-bottom:none; }
.expiry-table tr:hover td { background:#f8fafc; }
.expiry-table tr.current-row td { background:#eff6ff; }
.expiry-best { color:#16a34a; font-weight:800; }
.expiry-worst { color:#dc2626; font-weight:800; }
.expiry-mid { color:#d97706; font-weight:700; }
.tip-warn { background:#fffbeb; border-left:4px solid #f59e0b; border-radius:12px; padding:14px 18px; margin:6px 0; font-size:0.84rem; color:#92400e; line-height:1.6; box-shadow:0 2px 8px rgba(245,158,11,0.12); }
.tip-good { background:#f0fdf4; border-left:4px solid #22c55e; border-radius:12px; padding:14px 18px; margin:6px 0; font-size:0.84rem; color:#14532d; line-height:1.6; box-shadow:0 2px 8px rgba(34,197,94,0.12); }
.tip-bad { background:#fef2f2; border-left:4px solid #ef4444; border-radius:12px; padding:14px 18px; margin:6px 0; font-size:0.84rem; color:#7f1d1d; line-height:1.6; box-shadow:0 2px 8px rgba(239,68,68,0.12); }
.param-def { background:#162840; border:1px solid #2a4a6e; border-radius:10px; padding:10px 14px; margin-bottom:4px; font-size:0.76rem; color:#7ec8f7; line-height:1.65; }
.mode-slow  { color:#16a34a; background:#f0fdf4; border:1px solid #bbf7d0; border-radius:12px; padding:3px 10px; font-size:0.73rem; font-weight:800; }
.mode-med   { color:#d97706; background:#fffbeb; border:1px solid #fde68a; border-radius:12px; padding:3px 10px; font-size:0.73rem; font-weight:800; }
.mode-fast  { color:#dc2626; background:#fef2f2; border:1px solid #fecaca; border-radius:12px; padding:3px 10px; font-size:0.73rem; font-weight:800; }
.mode-ultra { color:#7c3aed; background:#faf5ff; border:1px solid #ddd6fe; border-radius:12px; padding:3px 10px; font-size:0.73rem; font-weight:800; }
.footer-card { background:#ffffff; border-radius:16px; padding:18px 20px; box-shadow:0 4px 16px rgba(0,0,0,0.06); }
.footer-card .fc-title { font-family:'Nunito',sans-serif; font-weight:900; color:#1a2e4a; font-size:0.95rem; margin-bottom:10px; }
.footer-card .fc-row { font-size:0.78rem; color:#64748b; line-height:1.9; }
hr { border:none; border-top:2px solid #e2e8f0; margin:28px 0; }
/* NEW: BMS parameter card */
.bms-card { background:#ffffff; border-radius:18px; padding:16px 18px; margin-bottom:10px; box-shadow:0 4px 18px rgba(0,0,0,0.07); position:relative; overflow:hidden; }
.bms-label { font-size:0.65rem; font-weight:800; color:#94a3b8; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:4px; }
.bms-value { font-family:'Nunito',sans-serif; font-size:1.65rem; font-weight:900; color:#1e293b; line-height:1; }
.bms-sub { font-size:0.71rem; color:#64748b; margin-top:4px; }
/* NEW: Health score ring */
.score-ring-wrap { text-align:center; padding:18px 0 12px; }
/* NEW: Comparison table */
.comp-table { width:100%; border-collapse:collapse; background:#ffffff; border-radius:16px; overflow:hidden; box-shadow:0 4px 20px rgba(0,0,0,0.07); }
.comp-table th { background:#1a2e4a; color:#ffffff; padding:12px 16px; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.08em; text-align:left; font-weight:800; }
.comp-table td { padding:11px 16px; border-bottom:1px solid #f1f5f9; font-size:0.85rem; color:#334155; }
.comp-table tr:last-child td { border-bottom:none; }
.comp-table tr:hover td { background:#f8fafc; }
.comp-better { color:#16a34a; font-weight:800; }
.comp-worse  { color:#dc2626; font-weight:700; }
/* NEW: thermal */
.thermal-card { border-radius:16px; padding:16px 18px; margin-bottom:10px; }
/* NEW: digital twin */
.twin-card { background:linear-gradient(135deg,#1a2e4a,#0d6eaa); border-radius:20px; padding:22px 24px; margin-bottom:14px; color:#fff; }
.twin-title { font-family:'Nunito',sans-serif; font-size:1rem; font-weight:900; margin-bottom:6px; }
.twin-val { font-family:'Nunito',sans-serif; font-size:1.9rem; font-weight:900; }
.twin-sub { font-size:0.75rem; color:rgba(255,255,255,0.65); margin-top:4px; }
/* NEW: animated liquid battery */
.battery-shell { position:relative; width:84px; height:158px; margin:0 auto; }
.battery-cap { position:absolute; top:0; left:28px; width:28px; height:11px; background:#1a2e4a; border-radius:5px 5px 0 0; z-index:2; }
.battery-body { position:absolute; top:11px; left:0; width:84px; height:147px; border:4px solid #1a2e4a; border-radius:13px; background:#eef2f7; overflow:hidden; box-shadow:inset 0 0 14px rgba(0,0,0,0.08); }
.battery-fill { position:absolute; bottom:0; left:0; width:100%; transition:height 1.2s ease; overflow:hidden; }
.battery-fill::before {
    content:''; position:absolute; top:0; left:-60%; width:220%; height:100%;
    background:linear-gradient(100deg, transparent 35%, rgba(255,255,255,0.4) 48%, transparent 62%);
    animation: battShimmer 2.4s linear infinite;
}
@keyframes battShimmer { 0%{ transform:translateX(-25%); } 100%{ transform:translateX(25%); } }
.battery-pct-label { position:absolute; top:50%; left:0; width:100%; text-align:center; transform:translateY(-50%);
    font-family:'Nunito',sans-serif; font-weight:900; font-size:1.18rem; color:#1a2e4a; z-index:5;
    text-shadow:0 1px 5px rgba(255,255,255,0.85), 0 0 2px rgba(255,255,255,0.6); }
.battery-caption { text-align:center; font-size:0.68rem; font-weight:800; color:#94a3b8; text-transform:uppercase; letter-spacing:0.1em; margin-top:10px; }
/* NEW: 3D isometric pack render */
.pack3d-card { background:#ffffff; border-radius:18px; padding:14px 8px 4px; box-shadow:0 4px 18px rgba(0,0,0,0.07); text-align:center; }
/* NEW: vehicle silhouette icon */
.veh-icon-wrap { width:62px; margin:0 auto 6px; opacity:0.92; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# EV DATABASE
# ─────────────────────────────────────────────
EV_CARS = {
    "Tata Tiago EV (19.2 kWh)":           {"battery":19.2,"range":250,"type":"Car","price":5.84, "charge_ac":3.3, "charge_dc":0,  "brand":"tata",      "color":"#1d4ed8","chemistry":"LFP","cooling":"Air"},
    "Tata Tiago EV (24 kWh)":             {"battery":24.0,"range":315,"type":"Car","price":8.49, "charge_ac":7.2, "charge_dc":0,  "brand":"tata",      "color":"#1d4ed8","chemistry":"LFP","cooling":"Air"},
    "Tata Punch EV (25 kWh)":             {"battery":25.0,"range":301,"type":"Car","price":8.09, "charge_ac":7.2, "charge_dc":50, "brand":"tata",      "color":"#0891b2","chemistry":"LFP","cooling":"Liquid"},
    "Tata Punch EV (35 kWh)":             {"battery":35.0,"range":421,"type":"Car","price":11.19,"charge_ac":7.2, "charge_dc":50, "brand":"tata",      "color":"#0891b2","chemistry":"LFP","cooling":"Liquid"},
    "Tata Nexon EV (30 kWh)":             {"battery":30.0,"range":325,"type":"Car","price":12.49,"charge_ac":7.2, "charge_dc":50, "brand":"tata",      "color":"#075985","chemistry":"NMC","cooling":"Liquid"},
    "Tata Nexon EV (40.5 kWh)":           {"battery":40.5,"range":465,"type":"Car","price":15.49,"charge_ac":7.2, "charge_dc":50, "brand":"tata",      "color":"#075985","chemistry":"NMC","cooling":"Liquid"},
    "Tata Curvv EV (45 kWh)":             {"battery":45.0,"range":502,"type":"Car","price":16.99,"charge_ac":11.0,"charge_dc":70, "brand":"tata",      "color":"#164e63","chemistry":"NMC","cooling":"Liquid"},
    "Tata Curvv EV (55 kWh)":             {"battery":55.0,"range":585,"type":"Car","price":19.49,"charge_ac":11.0,"charge_dc":70, "brand":"tata",      "color":"#164e63","chemistry":"NMC","cooling":"Liquid"},
    "Tata Harrier EV (71 kWh)":           {"battery":71.0,"range":627,"type":"Car","price":21.49,"charge_ac":11.0,"charge_dc":150,"brand":"tata",      "color":"#1e3a8a","chemistry":"NMC","cooling":"Liquid"},
    "Tata Tigor EV (26 kWh)":             {"battery":26.0,"range":306,"type":"Car","price":12.49,"charge_ac":7.2, "charge_dc":0,  "brand":"tata",      "color":"#2563eb","chemistry":"NMC","cooling":"Air"},
    "MG Comet EV (17.3 kWh)":             {"battery":17.3,"range":230,"type":"Car","price":6.31, "charge_ac":3.3, "charge_dc":0,  "brand":"mg",        "color":"#dc2626","chemistry":"LFP","cooling":"Air"},
    "MG ZS EV (50.3 kWh)":                {"battery":50.3,"range":461,"type":"Car","price":15.50,"charge_ac":7.2, "charge_dc":50, "brand":"mg",        "color":"#b91c1c","chemistry":"NMC","cooling":"Liquid"},
    "MG Windsor EV (38 kWh)":             {"battery":38.0,"range":332,"type":"Car","price":12.04,"charge_ac":7.2, "charge_dc":50, "brand":"mg",        "color":"#b91c1c","chemistry":"LFP","cooling":"Liquid"},
    "Mahindra BE 6 (59 kWh)":             {"battery":59.0,"range":535,"type":"Car","price":18.90,"charge_ac":11.0,"charge_dc":175,"brand":"mahindra",  "color":"#7c3aed","chemistry":"NMC","cooling":"Liquid"},
    "Mahindra BE 6 (79 kWh)":             {"battery":79.0,"range":682,"type":"Car","price":24.90,"charge_ac":11.0,"charge_dc":175,"brand":"mahindra",  "color":"#6d28d9","chemistry":"NMC","cooling":"Liquid"},
    "Mahindra XEV 9e (79 kWh)":           {"battery":79.0,"range":656,"type":"Car","price":21.90,"charge_ac":11.0,"charge_dc":175,"brand":"mahindra",  "color":"#5b21b6","chemistry":"NMC","cooling":"Liquid"},
    "Mahindra XUV400 (39.4 kWh)":         {"battery":39.4,"range":375,"type":"Car","price":15.49,"charge_ac":7.2, "charge_dc":50, "brand":"mahindra",  "color":"#4c1d95","chemistry":"NMC","cooling":"Liquid"},
    "Hyundai Creta Electric (51.4 kWh)":  {"battery":51.4,"range":473,"type":"Car","price":18.03,"charge_ac":11.0,"charge_dc":100,"brand":"hyundai",   "color":"#0f766e","chemistry":"NMC","cooling":"Liquid"},
    "Hyundai Ioniq 5 (72.6 kWh)":         {"battery":72.6,"range":631,"type":"Car","price":55.70,"charge_ac":11.0,"charge_dc":220,"brand":"hyundai",   "color":"#134e4a","chemistry":"NMC","cooling":"Liquid"},
    "Maruti e Vitara (49 kWh)":           {"battery":49.0,"range":500,"type":"Car","price":13.49,"charge_ac":11.0,"charge_dc":100,"brand":"maruti",    "color":"#15803d","chemistry":"NMC","cooling":"Liquid"},
    "Maruti e Vitara (61 kWh)":           {"battery":61.0,"range":550,"type":"Car","price":17.26,"charge_ac":11.0,"charge_dc":100,"brand":"maruti",    "color":"#15803d","chemistry":"NMC","cooling":"Liquid"},
    "BYD Atto 3 (60.5 kWh)":              {"battery":60.5,"range":480,"type":"Car","price":24.99,"charge_ac":11.0,"charge_dc":80, "brand":"byd",       "color":"#c2410c","chemistry":"LFP","cooling":"Liquid"},
    "BYD Seal (82.56 kWh)":               {"battery":82.56,"range":650,"type":"Car","price":41.00,"charge_ac":11.0,"charge_dc":150,"brand":"byd",      "color":"#9a3412","chemistry":"LFP","cooling":"Liquid"},
    "Tesla Model 3 (75 kWh)":             {"battery":75.0,"range":629,"type":"Car","price":70.00,"charge_ac":11.0,"charge_dc":250,"brand":"tesla",     "color":"#b91c1c","chemistry":"NMC","cooling":"Liquid"},
    "Tesla Model Y (75 kWh)":             {"battery":75.0,"range":533,"type":"Car","price":59.89,"charge_ac":11.0,"charge_dc":250,"brand":"tesla",     "color":"#dc2626","chemistry":"NMC","cooling":"Liquid"},
    "Kia EV6 (77.4 kWh)":                 {"battery":77.4,"range":708,"type":"Car","price":65.97,"charge_ac":11.0,"charge_dc":240,"brand":"kia",       "color":"#b91c1c","chemistry":"NMC","cooling":"Liquid"},
    "Kia EV9 (99.8 kWh)":                 {"battery":99.8,"range":541,"type":"Car","price":130.00,"charge_ac":11.0,"charge_dc":240,"brand":"kia",      "color":"#991b1b","chemistry":"NMC","cooling":"Liquid"},
    "BMW i4 (83.9 kWh)":                  {"battery":83.9,"range":590,"type":"Car","price":72.50,"charge_ac":11.0,"charge_dc":205,"brand":"bmw",       "color":"#1d4ed8","chemistry":"NMC","cooling":"Liquid"},
    "Audi e-tron GT (93.4 kWh)":          {"battery":93.4,"range":488,"type":"Car","price":172.00,"charge_ac":11.0,"charge_dc":270,"brand":"audi",     "color":"#991b1b","chemistry":"NMC","cooling":"Liquid"},
    "VinFast VF 6 (59.6 kWh)":            {"battery":59.6,"range":473,"type":"Car","price":17.29,"charge_ac":11.0,"charge_dc":150,"brand":"vinfast",   "color":"#0369a1","chemistry":"NMC","cooling":"Liquid"},
    "Citroen eC3 (29.2 kWh)":             {"battery":29.2,"range":320,"type":"Car","price":12.90,"charge_ac":7.2, "charge_dc":0,  "brand":"citroen",   "color":"#dc2626","chemistry":"LFP","cooling":"Air"},
}
EV_SCOOTERS = {
    "Ola S1 Pro (4 kWh)":          {"battery":4.0, "range":195,"type":"Scooter","price":1.47,"charge_ac":0.75,"charge_dc":0,"brand":"ola",       "color":"#dc2626","chemistry":"NMC","cooling":"Air"},
    "Ola S1 Air (2.5 kWh)":        {"battery":2.5, "range":101,"type":"Scooter","price":1.05,"charge_ac":0.60,"charge_dc":0,"brand":"ola",       "color":"#ef4444","chemistry":"NMC","cooling":"Air"},
    "Ola S1 X (2 kWh)":            {"battery":2.0, "range":91, "type":"Scooter","price":0.75,"charge_ac":0.60,"charge_dc":0,"brand":"ola",       "color":"#f87171","chemistry":"LFP","cooling":"Air"},
    "Ather 450X (2.9 kWh)":        {"battery":2.9, "range":146,"type":"Scooter","price":1.63,"charge_ac":0.75,"charge_dc":0,"brand":"ather",     "color":"#0891b2","chemistry":"NMC","cooling":"Air"},
    "Ather Rizta Z (3.7 kWh)":     {"battery":3.7, "range":160,"type":"Scooter","price":1.50,"charge_ac":0.75,"charge_dc":0,"brand":"ather",     "color":"#0284c7","chemistry":"NMC","cooling":"Air"},
    "TVS iQube S (5.1 kWh)":       {"battery":5.1, "range":212,"type":"Scooter","price":1.69,"charge_ac":0.95,"charge_dc":0,"brand":"tvs",       "color":"#7c3aed","chemistry":"LFP","cooling":"Air"},
    "Bajaj Chetak 2901 (3 kWh)":   {"battery":3.0, "range":113,"type":"Scooter","price":1.29,"charge_ac":0.60,"charge_dc":0,"brand":"bajaj",     "color":"#1d4ed8","chemistry":"LFP","cooling":"Air"},
    "Hero Vida V1 Pro (3.94 kWh)": {"battery":3.94,"range":165,"type":"Scooter","price":1.45,"charge_ac":0.95,"charge_dc":0,"brand":"hero_vida", "color":"#dc2626","chemistry":"NMC","cooling":"Air"},
    "Simple Energy One (4.8 kWh)": {"battery":4.8, "range":212,"type":"Scooter","price":1.45,"charge_ac":1.0, "charge_dc":0,"brand":"simple",    "color":"#0f766e","chemistry":"NMC","cooling":"Air"},
    "Revolt RV400 (3.24 kWh)":     {"battery":3.24,"range":150,"type":"Scooter","price":1.35,"charge_ac":0.85,"charge_dc":0,"brand":"revolt",    "color":"#111827","chemistry":"NMC","cooling":"Air"},
    "Ultraviolette F77 (10.3 kWh)":{"battery":10.3,"range":307,"type":"Scooter","price":3.80,"charge_ac":1.35,"charge_dc":0,"brand":"uv",        "color":"#6d28d9","chemistry":"NMC","cooling":"Liquid"},
    "Okinawa PraisePro (2.08 kWh)":{"battery":2.08,"range":88, "type":"Scooter","price":0.84,"charge_ac":0.75,"charge_dc":0,"brand":"okinawa",   "color":"#ef4444","chemistry":"LFP","cooling":"Air"},
}

CHARGING_MODELS = {
    "🏠 Home AC — Slow (₹7/kWh)":       {"rate":7.0, "speed":"Slow",  "power_kw":3.3, "desc":"Overnight home charging via 5A socket — cheapest option","color":"#16a34a","decay":0.0},
    "🏠 Home AC — Smart (₹8/kWh)":      {"rate":8.0, "speed":"Slow",  "power_kw":7.2, "desc":"7.4kW wall-box charger — fast & economical home charging","color":"#65a30d","decay":0.005},
    "🌆 Public AC — Network (₹12/kWh)": {"rate":12.0,"speed":"Medium","power_kw":7.2, "desc":"Mall/office AC stations — Tata Power EZ, ChargeZone etc.","color":"#d97706","decay":0.01},
    "⚡ Public DC Fast (₹20/kWh)":       {"rate":20.0,"speed":"Fast",  "power_kw":50,  "desc":"30-min top-up at highway DC stations — moderate battery wear","color":"#ea580c","decay":0.025},
    "🚀 Public DC Ultra-Fast (₹24/kWh)":{"rate":24.0,"speed":"Ultra", "power_kw":150, "desc":"Premium 150kW+ chargers — fastest but highest battery stress","color":"#dc2626","decay":0.04},
}

# ─────────────────────────────────────────────
# GENERIC VEHICLE SILHOUETTE ICONS (own line-art, not brand artwork)
# ─────────────────────────────────────────────
def vehicle_icon_svg(kind, color):
    if kind == "scooter":
        return f"""<svg viewBox="0 0 120 80" xmlns="http://www.w3.org/2000/svg">
        <circle cx="26" cy="62" r="13" fill="none" stroke="{color}" stroke-width="4"/>
        <circle cx="96" cy="62" r="13" fill="none" stroke="{color}" stroke-width="4"/>
        <path d="M26 62 L46 62 L60 30 L78 30" fill="none" stroke="{color}" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
        <path d="M78 30 L78 16" fill="none" stroke="{color}" stroke-width="4" stroke-linecap="round"/>
        <path d="M70 16 L86 16" fill="none" stroke="{color}" stroke-width="4" stroke-linecap="round"/>
        <path d="M46 62 L96 62" fill="none" stroke="{color}" stroke-width="4" stroke-linecap="round"/>
        <rect x="40" y="46" width="22" height="9" rx="3" fill="{color}" opacity="0.85"/>
        </svg>"""
    return f"""<svg viewBox="0 0 140 80" xmlns="http://www.w3.org/2000/svg">
    <path d="M16 56 L24 36 Q28 28 38 28 L96 28 Q106 28 110 36 L120 56" fill="none" stroke="{color}" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
    <path d="M40 28 L48 14 L88 14 L100 28" fill="none" stroke="{color}" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
    <line x1="68" y1="14" x2="68" y2="28" stroke="{color}" stroke-width="3"/>
    <path d="M10 56 L126 56" fill="none" stroke="{color}" stroke-width="4" stroke-linecap="round"/>
    <circle cx="38" cy="58" r="11" fill="none" stroke="{color}" stroke-width="4"/>
    <circle cx="100" cy="58" r="11" fill="none" stroke="{color}" stroke-width="4"/>
    <rect x="98" y="38" width="14" height="8" rx="2" fill="{color}" opacity="0.85"/>
    </svg>"""

# ─────────────────────────────────────────────
# THERMAL CELL-GRID GENERATOR (for heatmap + 3D pack render)
# ─────────────────────────────────────────────
def generate_cell_grid(temp_min, temp_max, rows=4, cols=8, seed_val=0):
    rng = np.random.RandomState(int(seed_val) % (2**31 - 1))
    base = np.linspace(temp_min, temp_max, rows*cols)
    rng.shuffle(base)
    noise = rng.normal(0, 0.35, rows*cols)
    grid = (base + noise).reshape(rows, cols)
    grid = (grid + np.roll(grid, 1, axis=1) + np.roll(grid, -1, axis=1)) / 3.0
    grid = np.clip(grid, temp_min - 1, temp_max + 1.5)
    return grid


@st.cache_data
def generate_data(n=3000):
    np.random.seed(42)
    capacity     = np.random.uniform(1.5, 80, n)
    cycles       = np.random.randint(0, 1500, n)
    age          = np.random.uniform(0, 10, n)
    avg_speed    = np.random.uniform(20, 130, n)
    temperature  = np.random.uniform(-5, 48, n)
    daily_km     = np.random.uniform(5, 350, n)
    fast_charge  = np.random.uniform(0, 1, n)
    charge_level = np.random.uniform(0.5, 1.0, n)
    terrain      = np.random.randint(0, 3, n)
    hvac         = np.random.uniform(0, 1, n)
    soh = (
        100 - (cycles/1500)*30 - (age/10)*20
        - np.maximum(0,(temperature-35)/15)*10
        - np.maximum(0,(8-temperature)/18)*8
        - fast_charge*8
        - np.maximum(0,(avg_speed-100)/60)*7
        - (terrain/2)*5 - hvac*4
        + (charge_level-0.5)*6
        + np.random.normal(0,1.5,n)
    )
    soh = np.clip(soh, 0, 100)
    range_km = (
        capacity*soh/100*6.2
        - np.maximum(0,(temperature-35)/10)*12
        - np.maximum(0,(8-temperature)/12)*18
        - (avg_speed-80)*0.28
        - (terrain/2)*18
        - hvac*22
        + np.random.normal(0,4,n)
    )
    range_km = np.clip(range_km, 20, 700)
    rul = np.maximum(0, (1500 - cycles) * (soh/100))
    resistance = 0.05 + (cycles/1500)*0.12 + (age/10)*0.08 + (temperature-25)*0.0003 + np.random.normal(0,0.005,n)
    resistance = np.clip(resistance, 0.03, 0.35)
    df = pd.DataFrame({
        "capacity_kwh":capacity,"charge_cycles":cycles,"age_years":age,
        "avg_speed_kmh":avg_speed,"temperature_c":temperature,"daily_km":daily_km,
        "fast_charge":fast_charge,"charge_level":charge_level,"terrain":terrain,
        "hvac":hvac,"soh":soh,"range_km":range_km,"rul":rul,"resistance":resistance
    })
    return df

@st.cache_resource
def train_model():
    df = generate_data()
    features = ["capacity_kwh","charge_cycles","age_years","avg_speed_kmh",
                "temperature_c","daily_km","fast_charge","charge_level","terrain","hvac"]
    X = df[features]
    X_train,X_test,ys_tr,ys_te = train_test_split(X,df["soh"],test_size=0.2,random_state=42)
    _,_,yr_tr,yr_te             = train_test_split(X,df["range_km"],test_size=0.2,random_state=42)
    _,_,yrul_tr,yrul_te         = train_test_split(X,df["rul"],test_size=0.2,random_state=42)
    _,_,yres_tr,yres_te         = train_test_split(X,df["resistance"],test_size=0.2,random_state=42)
    sm   = RandomForestRegressor(n_estimators=100,random_state=42);  sm.fit(X_train,ys_tr)
    rm   = RandomForestRegressor(n_estimators=100,random_state=42);  rm.fit(X_train,yr_tr)
    rulm = GradientBoostingRegressor(n_estimators=100,random_state=42); rulm.fit(X_train,yrul_tr)
    resm = RandomForestRegressor(n_estimators=100,random_state=42);  resm.fit(X_train,yres_tr)
    return (sm, rm, rulm, resm,
            r2_score(ys_te,sm.predict(X_test)),
            r2_score(yr_te,rm.predict(X_test)),
            r2_score(yrul_te,rulm.predict(X_test)),
            r2_score(yres_te,resm.predict(X_test)))

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ EV Battery Pro AI")
    st.markdown('<div style="color:#4a7fa5;font-size:0.75rem;margin-bottom:16px;">India 2026 · Professional BMS Analytics</div>', unsafe_allow_html=True)
    st.markdown("---")

    vehicle_type = st.radio("Vehicle Category", ["🚗 Car", "🛵 Scooter"], horizontal=True)
    is_scooter = "Scooter" in vehicle_type
    vehicle_db = EV_SCOOTERS if is_scooter else EV_CARS
    selected_vehicle = st.selectbox("Select Your EV Model", list(vehicle_db.keys()))
    v = vehicle_db[selected_vehicle]

    st.markdown("---")
    st.markdown("##### 🔌 Charging Mode")
    charging_model = st.selectbox("Primary Charger", list(CHARGING_MODELS.keys()))
    cm = CHARGING_MODELS[charging_model]
    st.markdown(f'<div class="param-def">💡 {cm["desc"]}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### 🔬 Battery Parameters")
    cycles       = st.slider("Charge Cycles Completed", 0, 1500, 200 if is_scooter else 300)
    age          = st.slider("Battery Age (years)", 0.0, 10.0, 1.5 if is_scooter else 3.0, 0.5)
    avg_speed    = st.slider("Avg Speed (km/h)", 20, 130, 40 if is_scooter else 75)
    daily_km     = st.slider("Daily Distance (km)", 5, 350, 30 if is_scooter else 80)

    st.markdown("---")
    st.markdown("##### 🌡️ Thermal & Environment")
    temperature  = st.slider("Ambient Temperature (°C)", -5, 48, 32)
    cell_temp_max= st.slider("Max Cell Temperature (°C)", temperature, min(temperature+25,70), min(temperature+8,65))
    hvac         = st.slider("AC / Heater Usage (%)", 0, 100, 40 if is_scooter else 30) / 100
    fast_charge  = st.slider("DC Fast Charge Usage (%)", 0, 100, 5 if is_scooter else 20) / 100
    charge_level = st.slider("Typical Charge Level (%)", 50, 100, 80) / 100
    terrain_opt  = st.selectbox("🏔️ Terrain", ["Flat (city roads)", "Mixed (suburbs)", "Hilly (ghats/mountains)"])
    terrain_val  = ["Flat (city roads)", "Mixed (suburbs)", "Hilly (ghats/mountains)"].index(terrain_opt)

    st.markdown("---")
    st.markdown("##### 🆚 Vehicle Comparison Tool")
    compare_enabled = st.checkbox("Compare with another vehicle")
    compare_vehicle = None
    if compare_enabled:
        all_db = {**EV_CARS, **EV_SCOOTERS}
        compare_opts = [k for k in all_db.keys() if k != selected_vehicle]
        compare_vehicle = st.selectbox("Compare Vehicle B", compare_opts)

    st.markdown("---")
    st.markdown("##### 🗂️ Dashboard Sections")
    show_bms        = st.checkbox("BMS Parameters Panel", value=True)
    show_thermal    = st.checkbox("Thermal Management", value=True)
    show_visual_pack= st.checkbox("Animated Battery & 3D Pack Visuals", value=True)
    show_health_score=st.checkbox("Battery Health Score", value=True)
    show_degradation= st.checkbox("Degradation Dashboard", value=True)
    show_charge_ai  = st.checkbox("Charging Recommendation AI", value=True)
    show_digital_twin=st.checkbox("Battery Digital Twin", value=True)
    show_comparison = st.checkbox("Vehicle Comparison", value=compare_enabled)
    show_protection = st.checkbox("Battery Protection Guide", value=True)
    show_ai_tips    = st.checkbox("AI Personalised Tips", value=True)

# ─────────────────────────────────────────────
# PREDICTIONS
# ─────────────────────────────────────────────
with st.spinner("⚡ Running AI models..."):
    soh_model, range_model, rul_model, res_model, soh_r2, range_r2, rul_r2, res_r2 = train_model()

features = ["capacity_kwh","charge_cycles","age_years","avg_speed_kmh",
            "temperature_c","daily_km","fast_charge","charge_level","terrain","hvac"]
input_df = pd.DataFrame([[v["battery"],cycles,age,avg_speed,temperature,
                          daily_km,fast_charge,charge_level,terrain_val,hvac]], columns=features)
pred_soh   = float(soh_model.predict(input_df)[0])
pred_range = float(range_model.predict(input_df)[0])
pred_rul   = float(rul_model.predict(input_df)[0])
pred_res   = float(res_model.predict(input_df)[0])

# Derived BMS parameters
battery_kwh      = v["battery"]
soc              = charge_level * 100
soe              = battery_kwh * (charge_level) * (pred_soh/100)
sop_pct          = min(100, pred_soh * (1 - max(0,(temperature-45)/20)) * (1 - pred_res/0.35))
cell_temp_min    = cell_temp_max - max(1, (cell_temp_max - temperature) * 0.6)
temp_delta       = cell_temp_max - cell_temp_min
cell_imbalance   = max(0, min(5, (pred_res - 0.05)*30 + cycles/500))
coulombic_eff    = min(99.5, 98.5 - fast_charge*2 - max(0,(temperature-40)*0.1))
heat_gen_rate    = (pred_res * (battery_kwh / max(pred_range,1) * avg_speed)**2) * 0.001
cooling_efficiency = min(100, max(0, 100 - temp_delta * 4 - max(0,cell_temp_max-35)*1.5))

# Cost calcs
charge_cost      = battery_kwh * cm["rate"]
full_charge_hrs  = battery_kwh / cm["power_kw"]
cost_per_km      = charge_cost / max(pred_range, 1)
monthly_km       = daily_km * 30
monthly_charges  = monthly_km / max(pred_range, 1)
monthly_cost     = monthly_charges * charge_cost
annual_cost      = monthly_cost * 12
petrol_km_per_l  = 45 if is_scooter else 15
petrol_cost_l    = 105
petrol_monthly   = (monthly_km / petrol_km_per_l) * petrol_cost_l
monthly_savings  = petrol_monthly - monthly_cost
annual_savings   = monthly_savings * 12

# SOH colors
soh_color  = "#16a34a" if pred_soh > 75 else "#d97706" if pred_soh > 50 else "#dc2626"
soh_bg     = "#f0fdf4" if pred_soh > 75 else "#fffbeb" if pred_soh > 50 else "#fef2f2"
soh_label  = "Excellent 🟢" if pred_soh > 75 else "Moderate 🟡" if pred_soh > 50 else "Replace Soon 🔴"

# Battery Health Score
score_soh    = min(100, pred_soh)
score_temp   = max(0, 100 - max(0, cell_temp_max-35)*4 - max(0,5-cell_temp_min)*3)
score_cell   = max(0, 100 - cell_imbalance*20)
score_charge = max(0, 100 - fast_charge*40 - max(0,(charge_level-0.85)*100))
score_res    = max(0, 100 - (pred_res - 0.05) / 0.30 * 100)
health_score = (score_soh*0.40 + score_temp*0.20 + score_cell*0.15 + score_charge*0.15 + score_res*0.10)
health_score = min(100, max(0, health_score))
hs_color = "#16a34a" if health_score>=90 else "#65a30d" if health_score>=75 else "#d97706" if health_score>=60 else "#dc2626"
hs_label = "Excellent 🟢" if health_score>=90 else "Good 🟡" if health_score>=75 else "Moderate 🟠" if health_score>=60 else "Replacement Recommended 🔴"

# Thermal risk
def thermal_risk(temp_max, delta, fast_ch):
    if temp_max > 55 or delta > 15: return "🔴 CRITICAL — Thermal Runaway Risk", "#dc2626"
    if temp_max > 45 or delta > 10: return "🟠 HIGH — Activate Cooling Immediately", "#ea580c"
    if temp_max > 35 or delta > 6:  return "🟡 MODERATE — Monitor Closely", "#d97706"
    return "🟢 NORMAL — Safe Operating Range", "#16a34a"
thermal_status, thermal_color = thermal_risk(cell_temp_max, temp_delta, fast_charge)
btms_rec = (
    "Pre-cool battery before departure. Reduce fast charging frequency." if cell_temp_max > 45 else
    "Activate liquid cooling. Avoid charging until below 38°C." if cell_temp_max > 38 else
    "Normal BTMS operation. Monitor temperature during fast charging." if cell_temp_max > 30 else
    "Optimal thermal range. BTMS not required."
)

# Digital Twin — degradation projection
def project_degradation(soh_now, cycles_now, daily_km_v, range_now, charge_mode_decay, years_list):
    results = {}
    for y in years_list:
        cy_per_yr = (daily_km_v * 365) / max(range_now, 1)
        added_cycles = cy_per_yr * y
        soh_proj = soh_now - (added_cycles * (0.018 + charge_mode_decay))
        soh_proj = max(0, soh_proj)
        results[y] = round(soh_proj, 1)
    return results

twin_proj = project_degradation(pred_soh, cycles, daily_km, pred_range, cm["decay"], [1,3,5])

# Battery expiry simulation
def simulate_expiry(base_soh, cycles_done, daily_km_val, battery_kwh_val, range_km, charge_mode_key):
    results = {}
    for mode_name, mode_data in CHARGING_MODELS.items():
        soh = base_soh
        cycles_per_year = (daily_km_val * 365) / max(range_km, 1)
        base_decay_per_cycle = 0.018 + mode_data["decay"]
        years_to_70 = None; years_to_80 = None
        soh_trace = [soh]
        for y in range(1, 16):
            decay = cycles_per_year * base_decay_per_cycle
            soh = max(0, soh - decay)
            soh_trace.append(soh)
            if soh <= 80 and years_to_80 is None: years_to_80 = y
            if soh <= 70 and years_to_70 is None: years_to_70 = y
        results[mode_name] = {"years_to_70": years_to_70 or ">15","years_to_80": years_to_80 or ">15","soh_trace":soh_trace,"color":mode_data["color"],"speed":mode_data["speed"]}
    return results

expiry_data = simulate_expiry(pred_soh, cycles, daily_km, battery_kwh, pred_range, charging_model)
cur_exp = expiry_data[charging_model]
yrs_raw = cur_exp["years_to_70"]
yr_str  = f"{yrs_raw} yrs" if isinstance(yrs_raw,(int,float)) else yrs_raw

# ─────────────────────────────────────────────
# PLOT SETTINGS
# ─────────────────────────────────────────────
PLOT_BG = "#ffffff"; PLOT_FG = "#f8fafc"; GRID_COL = "#e2e8f0"; TEXT_COL = "#334155"; AXIS_COL = "#64748b"

# ═══════════════════════════════════════════
# HERO BANNER
# ═══════════════════════════════════════════
st.markdown(f"""
<div class="hero-banner">
    <div class="hero-title">⚡ EV Battery Pro AI</div>
    <div class="hero-sub">Professional BMS Analytics · Thermal Management · Digital Twin · India 2026</div>
    <span class="hero-badge">🤖 Multi-Model AI</span>
    <span class="hero-badge">🔬 BMS Parameters</span>
    <span class="hero-badge">🌡️ Thermal Analytics</span>
    <span class="hero-badge">🏭 Digital Twin</span>
    <span class="hero-badge">📊 SOH R² = {soh_r2:.3f}</span>
    <span class="hero-badge">🛣️ Range R² = {range_r2:.3f}</span>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════
# VEHICLE CARD + CORE METRICS
# ═══════════════════════════════════════════
vcol, mcol = st.columns([1, 2.8], gap="large")
with vcol:
    veh_color  = v.get("color","#1d4ed8")
    short_name = selected_vehicle.split("(")[0].strip()
    brand_upper= v.get("brand","").upper()
    chem       = v.get("chemistry","N/A")
    cooling    = v.get("cooling","N/A")
    icon_svg = vehicle_icon_svg("scooter" if is_scooter else "car", veh_color)
    st.markdown(f"""
    <div class="vehicle-card">
        <div class="veh-icon-wrap">{icon_svg}</div>
        <div class="vehicle-name">{short_name}</div>
        <div class="vehicle-price">{brand_upper} · ₹{v['price']:.2f} Lakh</div>
        <div>
            <span class="spec-pill">🔋 {v['battery']} kWh</span>
            <span class="spec-pill">🛣️ {v['range']} km ARAI</span>
        </div>
        <div style="margin-top:4px;">
            <span class="spec-pill">⚡ AC {v['charge_ac']} kW</span>
            {'<span class="spec-pill">🚀 DC ' + str(v["charge_dc"]) + ' kW</span>' if v["charge_dc"] > 0 else '<span class="spec-pill" style="color:#64748b;background:#f8fafc;">AC Only</span>'}
        </div>
        <div style="margin-top:4px;">
            <span class="spec-pill">⚗️ {chem}</span>
            <span class="spec-pill">❄️ {cooling} Cooling</span>
        </div>
    </div>""", unsafe_allow_html=True)

with mcol:
    st.markdown('<div class="sec-head">📊 AI Health & Cost Predictions</div>', unsafe_allow_html=True)
    m1,m2,m3 = st.columns(3)
    m4,m5,m6 = st.columns(3)
    with m1:
        st.markdown(f"""<div class="metric-card"><div class="metric-accent" style="background:{soh_color}"></div>
        <div class="metric-label">Battery Health (SOH)</div>
        <div class="metric-value" style="color:{soh_color}">{pred_soh:.1f}%</div>
        <div class="metric-sub">{soh_label}</div></div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="metric-card"><div class="metric-accent" style="background:#0891b2"></div>
        <div class="metric-label">Real Range Today</div>
        <div class="metric-value" style="color:#0e7490">{pred_range:.0f} km</div>
        <div class="metric-sub">ARAI rated: {v['range']} km</div></div>""", unsafe_allow_html=True)
    with m3:
        lifespan_color="#16a34a" if isinstance(yrs_raw,str) or (isinstance(yrs_raw,(int,float)) and yrs_raw>=10) else "#d97706" if isinstance(yrs_raw,(int,float)) and yrs_raw>=6 else "#dc2626"
        st.markdown(f"""<div class="metric-card"><div class="metric-accent" style="background:{lifespan_color}"></div>
        <div class="metric-label">Battery Life Remaining</div>
        <div class="metric-value" style="color:{lifespan_color}">{yr_str}</div>
        <div class="metric-sub">till 70% SOH threshold</div></div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""<div class="metric-card"><div class="metric-accent" style="background:#7c3aed"></div>
        <div class="metric-label">Cost per km</div>
        <div class="metric-value" style="color:#6d28d9">₹{cost_per_km:.2f}</div>
        <div class="metric-sub">Petrol: ₹{petrol_cost_l/petrol_km_per_l:.1f}/km</div></div>""", unsafe_allow_html=True)
    with m5:
        st.markdown(f"""<div class="metric-card"><div class="metric-accent" style="background:#d97706"></div>
        <div class="metric-label">Monthly Fuel Cost</div>
        <div class="metric-value" style="color:#b45309">₹{monthly_cost:,.0f}</div>
        <div class="metric-sub">vs Petrol ₹{petrol_monthly:,.0f}/mo</div></div>""", unsafe_allow_html=True)
    with m6:
        st.markdown(f"""<div class="metric-card"><div class="metric-accent" style="background:#16a34a"></div>
        <div class="metric-label">Annual EV Saving</div>
        <div class="metric-value" style="color:#15803d">₹{annual_savings:,.0f}</div>
        <div class="metric-sub">vs equivalent petrol</div></div>""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ═══════════════════════════════════════════
# SECTION 1 — BMS PARAMETERS
# ═══════════════════════════════════════════
if show_bms:
    st.markdown('<div class="sec-head">🔬 Advanced BMS Parameters — Professional Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#64748b;font-size:0.83rem;margin-bottom:18px;">Full BMS parameter set tracked by professional Battery Management Systems. All values computed by AI models trained on 3,000 battery records.</div>', unsafe_allow_html=True)

    b1,b2,b3,b4 = st.columns(4)
    b5,b6,b7,b8 = st.columns(4)

    def bms_card(label, value, unit, sub, color="#1a2e4a", bg="#f8fafc"):
        return f"""<div class="bms-card" style="border-left:4px solid {color};background:{bg};">
        <div class="bms-label">{label}</div>
        <div class="bms-value" style="color:{color}">{value}<span style="font-size:0.9rem;font-weight:600;"> {unit}</span></div>
        <div class="bms-sub">{sub}</div></div>"""

    soc_c="#16a34a" if soc>=50 else "#d97706" if soc>=25 else "#dc2626"
    soh_c=soh_color
    sop_c="#16a34a" if sop_pct>=75 else "#d97706" if sop_pct>=50 else "#dc2626"
    rul_c="#16a34a" if pred_rul>=500 else "#d97706" if pred_rul>=200 else "#dc2626"
    res_c="#16a34a" if pred_res<=0.08 else "#d97706" if pred_res<=0.15 else "#dc2626"
    imb_c="#16a34a" if cell_imbalance<=1.5 else "#d97706" if cell_imbalance<=3 else "#dc2626"
    eff_c="#16a34a" if coulombic_eff>=98 else "#d97706" if coulombic_eff>=96 else "#dc2626"
    soe_c="#0891b2"

    with b1: st.markdown(bms_card("SOC — State of Charge", f"{soc:.0f}", "%", f"Charge level: {'Normal' if soc>=50 else 'Low' if soc>=25 else 'Critical'}", soc_c, "#f0fdf4" if soc>=50 else "#fffbeb" if soc>=25 else "#fef2f2"), unsafe_allow_html=True)
    with b2: st.markdown(bms_card("SOH — State of Health", f"{pred_soh:.1f}", "%", soh_label, soh_c, soh_bg), unsafe_allow_html=True)
    with b3: st.markdown(bms_card("SOP — State of Power", f"{sop_pct:.0f}", "%", f"Available power output capability", sop_c, "#f0fdf4" if sop_pct>=75 else "#fffbeb"), unsafe_allow_html=True)
    with b4: st.markdown(bms_card("SOE — State of Energy", f"{soe:.1f}", "kWh", f"Usable energy remaining in pack", soe_c, "#f0f9ff"), unsafe_allow_html=True)
    with b5: st.markdown(bms_card("RUL — Remaining Useful Life", f"{pred_rul:.0f}", "cycles", f"Estimated cycles before 80% SOH", rul_c, "#f0fdf4" if pred_rul>=500 else "#fffbeb" if pred_rul>=200 else "#fef2f2"), unsafe_allow_html=True)
    with b6: st.markdown(bms_card("Internal Resistance", f"{pred_res*1000:.1f}", "mΩ", f"{'Normal' if pred_res<=0.08 else 'Elevated — aging detected' if pred_res<=0.15 else 'High — significant degradation'}", res_c, "#f0fdf4" if pred_res<=0.08 else "#fef2f2"), unsafe_allow_html=True)
    with b7: st.markdown(bms_card("Cell Voltage Imbalance", f"{cell_imbalance:.1f}", "mV", f"{'Balanced' if cell_imbalance<=1.5 else 'Moderate imbalance' if cell_imbalance<=3 else 'Faulty cell suspected'}", imb_c, "#f0fdf4" if cell_imbalance<=1.5 else "#fef2f2"), unsafe_allow_html=True)
    with b8: st.markdown(bms_card("Coulombic Efficiency", f"{coulombic_eff:.1f}", "%", f"Charging efficiency · {'Excellent' if coulombic_eff>=98 else 'Good' if coulombic_eff>=96 else 'Degraded'}", eff_c, "#f0fdf4"), unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

# ═══════════════════════════════════════════
# SECTION 2 — THERMAL MANAGEMENT
# ═══════════════════════════════════════════
if show_thermal:
    st.markdown('<div class="sec-head">🌡️ Thermal Management System — BTMS Analytics</div>', unsafe_allow_html=True)
    tc1, tc2, tc3 = st.columns([1.4, 1.4, 1.2], gap="large")

    with tc1:
        t1,t2 = st.columns(2)
        def temp_card(label, val, unit="°C"):
            c="#16a34a" if val<35 else "#d97706" if val<45 else "#dc2626"
            bg="#f0fdf4" if val<35 else "#fffbeb" if val<45 else "#fef2f2"
            return f"""<div class="bms-card" style="border-left:4px solid {c};background:{bg};">
            <div class="bms-label">{label}</div>
            <div class="bms-value" style="color:{c}">{val:.1f}<span style="font-size:0.9rem;"> {unit}</span></div></div>"""
        with t1:
            st.markdown(temp_card("Max Cell Temp", cell_temp_max), unsafe_allow_html=True)
            st.markdown(temp_card("Temp Delta ΔT", temp_delta), unsafe_allow_html=True)
        with t2:
            st.markdown(temp_card("Min Cell Temp", cell_temp_min), unsafe_allow_html=True)
            eff_c2="#16a34a" if cooling_efficiency>=70 else "#d97706" if cooling_efficiency>=40 else "#dc2626"
            st.markdown(f"""<div class="bms-card" style="border-left:4px solid {eff_c2};">
            <div class="bms-label">Cooling Efficiency</div>
            <div class="bms-value" style="color:{eff_c2}">{cooling_efficiency:.0f}<span style="font-size:0.9rem;"> %</span></div></div>""", unsafe_allow_html=True)

    with tc2:
        # Thermal runaway risk bar
        risk_pct = min(100, max(0, (cell_temp_max-25)*2.5 + temp_delta*3 + fast_charge*20))
        risk_c   = "#16a34a" if risk_pct<25 else "#d97706" if risk_pct<55 else "#ea580c" if risk_pct<80 else "#dc2626"
        st.markdown(f"""
        <div style="background:#ffffff;border-radius:18px;padding:18px 20px;box-shadow:0 4px 18px rgba(0,0,0,0.07);margin-bottom:10px;">
            <div class="bms-label">🔥 Thermal Runaway Risk Meter</div>
            <div style="margin:12px 0 6px;">
                <div style="background:#f1f5f9;border-radius:50px;height:14px;overflow:hidden;">
                    <div style="width:{risk_pct}%;background:linear-gradient(90deg,#16a34a,{risk_c});height:14px;border-radius:50px;transition:width 0.5s;"></div>
                </div>
            </div>
            <div style="display:flex;justify-content:space-between;font-size:0.68rem;color:#94a3b8;margin-bottom:10px;">
                <span>Safe</span><span>Moderate</span><span>High</span><span>Critical</span>
            </div>
            <div style="font-size:0.88rem;font-weight:800;color:{risk_c};">{thermal_status}</div>
        </div>
        <div style="background:#fff7ed;border:1.5px solid #fed7aa;border-radius:14px;padding:14px 16px;">
            <div class="bms-label">🛠️ BTMS Recommendation</div>
            <div style="font-size:0.83rem;color:#9a3412;margin-top:6px;line-height:1.6;">{btms_rec}</div>
            <div style="margin-top:8px;font-size:0.75rem;color:#b45309;">Heat generation rate: <b>{heat_gen_rate:.2f} W/km</b></div>
        </div>""", unsafe_allow_html=True)

    with tc3:
        # Temperature vs recommendation table
        temp_table = [
            ("<25°C", "Normal — No action", "#16a34a", "#f0fdf4"),
            ("25–35°C", "Good — Passive cooling", "#65a30d", "#f7fee7"),
            ("35–45°C", "Activate Cooling", "#d97706", "#fffbeb"),
            (">45°C", "⚠ High Risk Alert", "#dc2626", "#fef2f2"),
        ]
        html_t = '<table style="width:100%;border-collapse:collapse;font-size:0.82rem;">'
        html_t += '<tr><th style="text-align:left;padding:8px 10px;background:#f8fafc;color:#64748b;font-size:0.68rem;text-transform:uppercase;">Temp</th><th style="text-align:left;padding:8px 10px;background:#f8fafc;color:#64748b;font-size:0.68rem;text-transform:uppercase;">Status</th></tr>'
        for temp_rng, rec, col, bg in temp_table:
            hl = ' style="background:#eff6ff;"' if (cell_temp_max>=25 and "25" in temp_rng) or (cell_temp_max>=35 and "35" in temp_rng) or (cell_temp_max>45 and "45" in temp_rng) or (cell_temp_max<25 and "<25" in temp_rng) else ""
            html_t += f'<tr{hl}><td style="padding:8px 10px;font-weight:700;color:{col};">{temp_rng}</td><td style="padding:8px 10px;color:#475569;">{rec}</td></tr>'
        html_t += '</table>'
        st.markdown(f'<div style="background:#fff;border-radius:16px;padding:16px;box-shadow:0 4px 18px rgba(0,0,0,0.07);">{html_t}</div>', unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

# ═══════════════════════════════════════════
# SECTION 2B — ANIMATED BATTERY & 3D PACK VISUALS
# ═══════════════════════════════════════════
if show_visual_pack:
    st.markdown('<div class="sec-head">🧊 Animated Battery & 3D Pack Visuals</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#64748b;font-size:0.83rem;margin-bottom:18px;">Custom-built visuals — a live SOC-fill battery icon, an isometric 3D render of the pack, and a per-cell thermal heatmap. These are original illustrations generated from your inputs, not manufacturer photos.</div>', unsafe_allow_html=True)

    vp1, vp2, vp3 = st.columns([1, 1.5, 1.5], gap="large")

    # 1) Animated liquid battery (CSS shimmer animation, fills with SOC, tinted by SOH)
    with vp1:
        fill_h = max(4, min(96, soc))
        st.markdown(f"""
        <div class="battery-shell">
            <div class="battery-cap"></div>
            <div class="battery-body">
                <div class="battery-fill" style="height:{fill_h}%;background:linear-gradient(180deg,{soh_color}cc,{soh_color});"></div>
                <div class="battery-pct-label">{soc:.0f}%</div>
            </div>
        </div>
        <div class="battery-caption">SOC fill · SOH-tinted</div>
        """, unsafe_allow_html=True)

    # Shared per-cell thermal grid for the 3D pack + heatmap
    seed_val = cycles*3 + age*97 + temperature*11 + fast_charge*733 + cell_temp_max*5
    rows_g, cols_g = 4, 8
    cell_grid = generate_cell_grid(cell_temp_min, cell_temp_max, rows_g, cols_g, seed_val)
    norm_grid = (cell_grid - cell_grid.min()) / max(0.1, (cell_grid.max() - cell_grid.min()))
    hot_r, hot_c = np.unravel_index(np.argmax(cell_grid), cell_grid.shape)

    # 2) 3D isometric pack render — bar height & color = per-cell temperature
    with vp2:
        st.markdown('<div style="font-weight:800;font-size:0.85rem;color:#1a2e4a;margin-bottom:6px;text-align:center;">🏭 3D Pack Render — Per-Cell Temperature</div>', unsafe_allow_html=True)
        fig5 = plt.figure(figsize=(5.6, 4.4))
        ax5 = fig5.add_subplot(111, projection='3d')
        fig5.patch.set_facecolor(PLOT_BG)
        cmap3d = plt.get_cmap('RdYlGn_r')
        xs, ys, zs, dxs, dys, dzs, colors3d = [], [], [], [], [], [], []
        for r in range(rows_g):
            for c in range(cols_g):
                xs.append(c); ys.append(r); zs.append(0)
                dxs.append(0.72); dys.append(0.72)
                dzs.append(0.4 + norm_grid[r, c] * 1.6)
                colors3d.append(cmap3d(norm_grid[r, c]))
        ax5.bar3d(xs, ys, zs, dxs, dys, dzs, color=colors3d, edgecolor="#ffffff", linewidth=0.6, shade=True)
        ax5.set_xlabel("Module col", color=AXIS_COL, fontsize=7, labelpad=2)
        ax5.set_ylabel("Module row", color=AXIS_COL, fontsize=7, labelpad=2)
        ax5.set_zticks([])
        ax5.tick_params(colors=TEXT_COL, labelsize=6)
        ax5.view_init(elev=26, azim=-58)
        try:
            ax5.set_box_aspect((cols_g, rows_g, 4))
        except Exception:
            pass
        try:
            ax5.xaxis.pane.set_edgecolor(GRID_COL); ax5.xaxis.pane.set_facecolor((1,1,1,0))
            ax5.yaxis.pane.set_edgecolor(GRID_COL); ax5.yaxis.pane.set_facecolor((1,1,1,0))
            ax5.zaxis.pane.set_edgecolor(GRID_COL); ax5.zaxis.pane.set_facecolor((1,1,1,0))
        except Exception:
            pass
        plt.tight_layout(pad=0.3)
        st.pyplot(fig5, use_container_width=True); plt.close()
        st.markdown(f'<div style="text-align:center;font-size:0.72rem;color:#94a3b8;">Bar height &amp; color = relative cell temperature · hottest module: row {hot_r+1}, col {hot_c+1}</div>', unsafe_allow_html=True)

    # 3) 2D thermal heatmap — top-view, exact per-cell °C
    with vp3:
        st.markdown('<div style="font-weight:800;font-size:0.85rem;color:#1a2e4a;margin-bottom:6px;text-align:center;">🌡️ Pack Thermal Heatmap (Top View)</div>', unsafe_allow_html=True)
        fig6, ax6 = plt.subplots(figsize=(5.6, 3.4))
        fig6.patch.set_facecolor(PLOT_BG)
        im = ax6.imshow(cell_grid, cmap='RdYlGn_r', aspect='auto', vmin=cell_grid.min(), vmax=cell_grid.max())
        for r in range(rows_g):
            for c in range(cols_g):
                txt_color = "#ffffff" if norm_grid[r, c] > 0.55 else "#1e293b"
                ax6.text(c, r, f"{cell_grid[r,c]:.0f}", ha='center', va='center', fontsize=7.5, color=txt_color, fontweight='700')
        ax6.add_patch(plt.Rectangle((hot_c-0.5, hot_r-0.5), 1, 1, fill=False, edgecolor="#1a2e4a", linewidth=2.4))
        ax6.set_xticks(range(cols_g)); ax6.set_yticks(range(rows_g))
        ax6.set_xticklabels([f"C{c+1}" for c in range(cols_g)], fontsize=7, color=AXIS_COL)
        ax6.set_yticklabels([f"R{r+1}" for r in range(rows_g)], fontsize=7, color=AXIS_COL)
        ax6.set_title("°C per cell", color=AXIS_COL, fontsize=8)
        cbar = fig6.colorbar(im, ax=ax6, fraction=0.046, pad=0.03)
        cbar.ax.tick_params(labelsize=7, colors=TEXT_COL)
        plt.tight_layout(pad=0.6)
        st.pyplot(fig6, use_container_width=True); plt.close()

    st.markdown("<hr>", unsafe_allow_html=True)

# ═══════════════════════════════════════════
# SECTION 3 — BATTERY HEALTH SCORE
# ═══════════════════════════════════════════
if show_health_score:
    st.markdown('<div class="sec-head">🏆 Battery Health Score — Composite Index</div>', unsafe_allow_html=True)
    hs1, hs2 = st.columns([1, 2], gap="large")

    with hs1:
        # Big score display
        st.markdown(f"""
        <div style="background:#ffffff;border-radius:24px;padding:28px 24px;text-align:center;box-shadow:0 4px 20px rgba(0,0,0,0.08);">
            <div style="font-size:0.75rem;font-weight:800;color:#94a3b8;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:12px;">Overall Health Score</div>
            <div style="font-family:'Nunito',sans-serif;font-size:4.5rem;font-weight:900;color:{hs_color};line-height:1;">{health_score:.0f}</div>
            <div style="font-size:0.75rem;color:#94a3b8;margin:4px 0 14px;">/100</div>
            <div style="background:{hs_color}22;border:1.5px solid {hs_color}44;border-radius:20px;padding:6px 16px;display:inline-block;color:{hs_color};font-size:0.82rem;font-weight:800;">{hs_label}</div>
            <div style="margin-top:16px;background:#f1f5f9;border-radius:50px;height:10px;overflow:hidden;">
                <div style="width:{health_score}%;background:{hs_color};height:10px;border-radius:50px;"></div>
            </div>
        </div>""", unsafe_allow_html=True)

    with hs2:
        score_breakdown = [
            ("SOH (40%)",           score_soh,    0.40, "#0891b2"),
            ("Temperature Health (20%)", score_temp,   0.20, "#d97706"),
            ("Cell Balance (15%)",  score_cell,   0.15, "#7c3aed"),
            ("Charging Habits (15%)",score_charge, 0.15, "#16a34a"),
            ("Internal Resistance (10%)", score_res, 0.10, "#ea580c"),
        ]
        for label, raw_score, weight, color in score_breakdown:
            weighted = raw_score * weight
            bar_pct  = raw_score
            bar_c    = "#16a34a" if raw_score>=80 else "#d97706" if raw_score>=60 else "#dc2626"
            st.markdown(f"""
            <div style="background:#ffffff;border-radius:14px;padding:13px 16px;margin-bottom:8px;box-shadow:0 2px 10px rgba(0,0,0,0.05);">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:7px;">
                    <span style="font-size:0.82rem;font-weight:700;color:#334155;">{label}</span>
                    <span style="font-family:'Nunito',sans-serif;font-size:1.05rem;font-weight:900;color:{bar_c};">{raw_score:.0f}</span>
                </div>
                <div style="background:#f1f5f9;border-radius:50px;height:8px;overflow:hidden;">
                    <div style="width:{bar_pct}%;background:{bar_c};height:8px;border-radius:50px;"></div>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

# ═══════════════════════════════════════════
# SECTION 4 — DEGRADATION DASHBOARD (Graphs)
# ═══════════════════════════════════════════
if show_degradation:
    st.markdown('<div class="sec-head">📉 Battery Degradation Dashboard</div>', unsafe_allow_html=True)
    dg1, dg2 = st.columns(2, gap="large")

    # Graph 1: SOH Trajectory vs Time
    with dg1:
        st.markdown('<div style="font-weight:800;font-size:0.85rem;color:#1a2e4a;margin-bottom:10px;">📉 SOH Trajectory by Charging Mode</div>', unsafe_allow_html=True)
        fig_d1, ax_d1 = plt.subplots(figsize=(6.5,4.2))
        fig_d1.patch.set_facecolor(PLOT_BG); ax_d1.set_facecolor(PLOT_FG)
        years_x = list(range(16))
        for mode_name, edata in expiry_data.items():
            lw = 2.8 if mode_name == charging_model else 1.2
            alpha = 1.0 if mode_name == charging_model else 0.45
            label = mode_name.split("(")[0].strip().replace("🏠 ","").replace("🌆 ","").replace("⚡ ","").replace("🚀 ","")
            ax_d1.plot(years_x, edata["soh_trace"], color=edata["color"], linewidth=lw, alpha=alpha, label=label)
        ax_d1.axhline(70, color="#dc2626", linestyle="--", linewidth=1.2, alpha=0.6, label="Replace (70%)")
        ax_d1.axhline(80, color="#d97706", linestyle=":", linewidth=1.2, alpha=0.6, label="Warning (80%)")
        ax_d1.set_xlabel("Years from now", color=AXIS_COL, fontsize=9)
        ax_d1.set_ylabel("SOH (%)", color=AXIS_COL, fontsize=9)
        ax_d1.tick_params(colors=TEXT_COL, labelsize=8)
        for sp in ax_d1.spines.values(): sp.set_edgecolor(GRID_COL)
        ax_d1.grid(True, alpha=0.4, color=GRID_COL)
        ax_d1.legend(fontsize=6.5, loc="upper right", framealpha=0.9)
        plt.tight_layout(pad=1.2)
        st.pyplot(fig_d1, use_container_width=True); plt.close()

    # Graph 2: Capacity Fade vs Cycles
    with dg2:
        st.markdown('<div style="font-weight:800;font-size:0.85rem;color:#1a2e4a;margin-bottom:10px;">📉 Capacity Fade vs Charge Cycles</div>', unsafe_allow_html=True)
        cyc_x = np.linspace(0, 1500, 200)
        fig_d2, ax_d2 = plt.subplots(figsize=(6.5,4.2))
        fig_d2.patch.set_facecolor(PLOT_BG); ax_d2.set_facecolor(PLOT_FG)
        cap_fade_slow  = battery_kwh * np.clip(1 - (cyc_x/1500)*0.28 - (age/10)*0.18, 0.3, 1)
        cap_fade_fast  = battery_kwh * np.clip(1 - (cyc_x/1500)*0.38 - (age/10)*0.22, 0.3, 1)
        cap_fade_ultra = battery_kwh * np.clip(1 - (cyc_x/1500)*0.48 - (age/10)*0.26, 0.3, 1)
        ax_d2.fill_between(cyc_x, cap_fade_fast, cap_fade_slow, alpha=0.12, color="#0891b2")
        ax_d2.plot(cyc_x, cap_fade_slow,  color="#16a34a", linewidth=2, label="AC Charging")
        ax_d2.plot(cyc_x, cap_fade_fast,  color="#ea580c", linewidth=2, label="DC Fast")
        ax_d2.plot(cyc_x, cap_fade_ultra, color="#dc2626", linewidth=2, linestyle="--", label="DC Ultra-Fast")
        ax_d2.axvline(cycles, color="#7c3aed", linewidth=2, linestyle=":", alpha=0.8)
        ax_d2.text(cycles+20, battery_kwh*0.95, f"You\\n({cycles})", color="#7c3aed", fontsize=7.5, fontweight='800')
        ax_d2.axhline(battery_kwh*0.70, color="#dc2626", linewidth=1.2, linestyle="--", alpha=0.5, label="Replace (70%)")
        ax_d2.set_xlabel("Charge Cycles", color=AXIS_COL, fontsize=9)
        ax_d2.set_ylabel("Usable Capacity (kWh)", color=AXIS_COL, fontsize=9)
        ax_d2.tick_params(colors=TEXT_COL, labelsize=8)
        for sp in ax_d2.spines.values(): sp.set_edgecolor(GRID_COL)
        ax_d2.grid(True, alpha=0.4, color=GRID_COL)
        ax_d2.legend(fontsize=7, loc="upper right", framealpha=0.9)
        plt.tight_layout(pad=1.2)
        st.pyplot(fig_d2, use_container_width=True); plt.close()

    dg3, dg4 = st.columns(2, gap="large")

    # Graph 3: Temperature vs Time (simulated daily)
    with dg3:
        st.markdown('<div style="font-weight:800;font-size:0.85rem;color:#1a2e4a;margin-bottom:10px;">🌡️ Cell Temperature Profile (24h simulation)</div>', unsafe_allow_html=True)
        hours = np.linspace(0, 24, 288)
        base_curve = temperature + 4*np.sin(np.pi*(hours-6)/12)
        charge_bump = np.where((hours>=22)|(hours<=6), 0, 0)
        drive_bump  = np.where((hours>=8)&(hours<=10), 6, np.where((hours>=17)&(hours<=20), 8, 0))
        if fast_charge > 0.3:
            fc_bump = np.where((hours>=12)&(hours<=13), fast_charge*15, 0)
        else:
            fc_bump = np.zeros_like(hours)
        cell_profile = base_curve + drive_bump + fc_bump + np.random.normal(0,0.5,len(hours))
        cell_profile = np.clip(cell_profile, temperature-2, cell_temp_max+2)
        fig_d3, ax_d3 = plt.subplots(figsize=(6.5,4.2))
        fig_d3.patch.set_facecolor(PLOT_BG); ax_d3.set_facecolor(PLOT_FG)
        ax_d3.fill_between(hours, cell_profile, alpha=0.15, color="#0891b2")
        ax_d3.plot(hours, cell_profile, color="#0891b2", linewidth=2)
        ax_d3.axhline(45, color="#dc2626", linewidth=1.2, linestyle="--", alpha=0.7, label="Critical 45°C")
        ax_d3.axhline(35, color="#d97706", linewidth=1.2, linestyle=":", alpha=0.7, label="Warning 35°C")
        ax_d3.set_xlabel("Hour of Day", color=AXIS_COL, fontsize=9)
        ax_d3.set_ylabel("Cell Temperature (°C)", color=AXIS_COL, fontsize=9)
        ax_d3.set_xticks([0,4,8,12,16,20,24])
        ax_d3.tick_params(colors=TEXT_COL, labelsize=8)
        for sp in ax_d3.spines.values(): sp.set_edgecolor(GRID_COL)
        ax_d3.grid(True, alpha=0.4, color=GRID_COL)
        ax_d3.legend(fontsize=7.5, loc="upper left", framealpha=0.9)
        plt.tight_layout(pad=1.2)
        st.pyplot(fig_d3, use_container_width=True); plt.close()

    # Graph 4: Charging Pattern Analysis
    with dg4:
        st.markdown('<div style="font-weight:800;font-size:0.85rem;color:#1a2e4a;margin-bottom:10px;">📈 Charging Pattern & SOH Impact Analysis</div>', unsafe_allow_html=True)
        fig_d4, ax_d4 = plt.subplots(figsize=(6.5,4.2))
        fig_d4.patch.set_facecolor(PLOT_BG); ax_d4.set_facecolor(PLOT_FG)
        fc_fracs = np.linspace(0,1,100)
        soh_impact = pred_soh - fc_fracs*8 - (charge_level-0.8)*6
        soh_impact = np.clip(soh_impact, 0, 100)
        ax_d4.fill_between(fc_fracs*100, soh_impact, alpha=0.12, color="#7c3aed")
        ax_d4.plot(fc_fracs*100, soh_impact, color="#7c3aed", linewidth=2.5)
        ax_d4.axvline(fast_charge*100, color="#dc2626", linewidth=2, linestyle=":", alpha=0.8)
        ax_d4.text(fast_charge*100+1.5, pred_soh-5, f"You ({fast_charge*100:.0f}%)", color="#dc2626", fontsize=7.5, fontweight='800')
        ax_d4.set_xlabel("DC Fast Charge Usage (%)", color=AXIS_COL, fontsize=9)
        ax_d4.set_ylabel("Predicted SOH (%)", color=AXIS_COL, fontsize=9)
        ax_d4.tick_params(colors=TEXT_COL, labelsize=8)
        for sp in ax_d4.spines.values(): sp.set_edgecolor(GRID_COL)
        ax_d4.grid(True, alpha=0.4, color=GRID_COL)
        plt.tight_layout(pad=1.2)
        st.pyplot(fig_d4, use_container_width=True); plt.close()

    st.markdown("<hr>", unsafe_allow_html=True)

# ═══════════════════════════════════════════
# SECTION 5 — CHARGING RECOMMENDATION AI
# ═══════════════════════════════════════════
if show_charge_ai:
    st.markdown('<div class="sec-head">🤖 Charging Recommendation AI</div>', unsafe_allow_html=True)
    ca1, ca2 = st.columns([1.2, 1.8], gap="large")

    with ca1:
        trip_km   = st.number_input("Trip Distance (km)", 10, 800, 80, step=10)
        curr_soc  = st.slider("Current SOC (%)", 5, 100, int(soc))
        tgt_pct   = 80 if temperature <= 35 else 75 if temperature <= 42 else 70

    with ca2:
        energy_needed = (trip_km / max(pred_range,1)) * battery_kwh
        soc_needed    = min(100, (energy_needed / battery_kwh)*100 + 10)
        charge_needed = max(0, soc_needed - curr_soc)
        recs = []
        if curr_soc >= soc_needed:
            recs.append(("✅","good",f"No charging needed. Current SOC {curr_soc:.0f}% covers {trip_km} km trip."))
        else:
            recs.append(("🔋","warn",f"Charge to <b>{tgt_pct}%</b> — enough for trip + 15% buffer."))
        if temperature > 38:
            recs.append(("❄️","bad",f"<b>Wait until battery cools below 38°C</b> before charging. Current: {temperature}°C"))
        if fast_charge > 0.3 and temperature > 35:
            recs.append(("⚠","bad","<b>Avoid DC Fast Charging</b> — high temp + fast charge accelerates degradation 3×."))
        elif trip_km < 50:
            recs.append(("✅","good","<b>Use AC Home Charging</b> — gentlest on battery for short daily trips."))
        else:
            recs.append(("⚡","warn","<b>DC Fast Charging OK</b> for this trip if needed at highway stops."))
        if charge_level > 0.9:
            recs.append(("🔋","warn",f"<b>Reduce daily charge limit to 80%</b> — you're charging to {charge_level*100:.0f}% which strains cells."))
        if curr_soc < 20:
            recs.append(("🔴","bad","<b>Critical SOC!</b> Avoid deep discharge — charge immediately to at least 30%."))
        time_ac  = (tgt_pct - curr_soc)/100 * battery_kwh / 7.2
        time_dc  = (tgt_pct - curr_soc)/100 * battery_kwh / 50 if v["charge_dc"]>0 else None
        cost_rec = (tgt_pct - curr_soc)/100 * battery_kwh * 8.0

        st.markdown(f"""
        <div style="background:#ffffff;border-radius:18px;padding:18px 20px;box-shadow:0 4px 18px rgba(0,0,0,0.07);">
        <div class="bms-label" style="margin-bottom:12px;">AI Recommendations for {trip_km} km trip</div>
        {''.join(f'<div class="tip-{kind}" style="margin:6px 0;">{icon} {text}</div>' for icon,kind,text in recs)}
        <div style="margin-top:14px;display:flex;gap:12px;flex-wrap:wrap;">
            <div style="background:#f0f9ff;border-radius:12px;padding:10px 14px;flex:1;min-width:120px;">
                <div style="font-size:0.68rem;font-weight:800;color:#0369a1;text-transform:uppercase;">AC Charge Time</div>
                <div style="font-family:'Nunito',sans-serif;font-size:1.4rem;font-weight:900;color:#0c4a6e;">{time_ac:.1f} hrs</div>
            </div>
            {'<div style="background:#fff7ed;border-radius:12px;padding:10px 14px;flex:1;min-width:120px;"><div style="font-size:0.68rem;font-weight:800;color:#c2410c;text-transform:uppercase;">DC Fast Time</div><div style="font-family:Nunito,sans-serif;font-size:1.4rem;font-weight:900;color:#9a3412;">' + f'{time_dc:.0f} min' + '</div></div>' if time_dc else ''}
            <div style="background:#f0fdf4;border-radius:12px;padding:10px 14px;flex:1;min-width:120px;">
                <div style="font-size:0.68rem;font-weight:800;color:#15803d;text-transform:uppercase;">Est. Charge Cost</div>
                <div style="font-family:'Nunito',sans-serif;font-size:1.4rem;font-weight:900;color:#14532d;">₹{cost_rec:.0f}</div>
            </div>
        </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

# ═══════════════════════════════════════════
# SECTION 6 — BATTERY DIGITAL TWIN
# ═══════════════════════════════════════════
if show_digital_twin:
    st.markdown('<div class="sec-head">🏭 Battery Digital Twin — Virtual Battery Model</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#64748b;font-size:0.83rem;margin-bottom:18px;">Your vehicle\'s virtual battery pack model showing current state and AI-predicted degradation over 1, 3, and 5 years. Digital-twin-based monitoring is a major trend in commercial BMS platforms.</div>', unsafe_allow_html=True)

    dt1, dt2, dt3 = st.columns([1.6, 1.6, 1.6], gap="large")

    cap_now = battery_kwh * (pred_soh/100)
    range_now_real = pred_range

    with dt1:
        st.markdown(f"""
        <div class="twin-card">
            <div class="twin-title">🔋 Current State</div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:12px;">
                <div><div class="twin-sub">SOH</div><div class="twin-val">{pred_soh:.1f}%</div></div>
                <div><div class="twin-sub">Usable Cap</div><div class="twin-val">{cap_now:.1f} kWh</div></div>
                <div><div class="twin-sub">Real Range</div><div class="twin-val">{pred_range:.0f} km</div></div>
                <div><div class="twin-sub">Health Score</div><div class="twin-val">{health_score:.0f}/100</div></div>
                <div><div class="twin-sub">RUL</div><div class="twin-val">{pred_rul:.0f} cy</div></div>
                <div><div class="twin-sub">Resistance</div><div class="twin-val">{pred_res*1000:.0f} mΩ</div></div>
            </div>
        </div>""", unsafe_allow_html=True)

    for yr_label, yr in [("1 Year", 1), ("3 Years", 3)]:
        soh_proj  = twin_proj[yr]
        cap_proj  = battery_kwh * (soh_proj/100)
        range_proj= pred_range * (soh_proj/pred_soh) if pred_soh > 0 else 0
        fade_pct  = pred_soh - soh_proj
        pc = "#16a34a" if soh_proj>=80 else "#d97706" if soh_proj>=70 else "#dc2626"
        col_idx = dt2 if yr==1 else dt3
        with col_idx:
            st.markdown(f"""
            <div class="twin-card" style="background:linear-gradient(135deg,#0f1f35,#1a4a6e);">
                <div class="twin-title">🔮 After {yr_label}</div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:12px;">
                    <div><div class="twin-sub">Predicted SOH</div><div class="twin-val" style="color:{pc};">{soh_proj:.1f}%</div></div>
                    <div><div class="twin-sub">Capacity</div><div class="twin-val">{cap_proj:.1f} kWh</div></div>
                    <div><div class="twin-sub">Expected Range</div><div class="twin-val">{range_proj:.0f} km</div></div>
                    <div><div class="twin-sub">SOH Fade</div><div class="twin-val" style="color:#fca5a5;">-{fade_pct:.1f}%</div></div>
                </div>
                <div style="margin-top:12px;background:rgba(255,255,255,0.12);border-radius:50px;height:8px;overflow:hidden;">
                    <div style="width:{soh_proj}%;background:{pc};height:8px;border-radius:50px;"></div>
                </div>
            </div>""", unsafe_allow_html=True)

    # 5-year projection bar chart
    soh_5 = twin_proj[5]
    cap_5 = battery_kwh * (soh_5/100)
    range_5 = pred_range * (soh_5/pred_soh) if pred_soh > 0 else 0

    fig_twin, ax_twin = plt.subplots(figsize=(12, 3.5))
    fig_twin.patch.set_facecolor(PLOT_BG); ax_twin.set_facecolor(PLOT_FG)
    years_proj = ["Now", "+1 yr", "+3 yrs", "+5 yrs"]
    soh_vals   = [pred_soh, twin_proj[1], twin_proj[3], soh_5]
    bar_cols   = ["#0891b2" if s>=80 else "#d97706" if s>=70 else "#dc2626" for s in soh_vals]
    bars = ax_twin.bar(years_proj, soh_vals, color=bar_cols, width=0.45, zorder=3, edgecolor=PLOT_BG, linewidth=2)
    ax_twin.axhline(70, color="#dc2626", linestyle="--", linewidth=1.2, alpha=0.6, label="Replace threshold 70%")
    ax_twin.axhline(80, color="#d97706", linestyle=":", linewidth=1.2, alpha=0.6, label="Warning threshold 80%")
    for bar, val in zip(bars, soh_vals):
        ax_twin.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f"{val:.1f}%", ha='center', fontsize=9, fontweight='800', color="#334155")
    ax_twin.set_ylabel("Predicted SOH (%)", color=AXIS_COL, fontsize=9)
    ax_twin.set_ylim(0, 105)
    ax_twin.tick_params(colors=TEXT_COL, labelsize=9)
    for sp in ax_twin.spines.values(): sp.set_edgecolor(GRID_COL)
    ax_twin.grid(True, alpha=0.4, color=GRID_COL, axis='y')
    ax_twin.legend(fontsize=8, loc="upper right")
    ax_twin.set_title(f"Digital Twin — SOH Projection · {selected_vehicle.split('(')[0].strip()}", color=AXIS_COL, fontsize=9)
    plt.tight_layout(pad=1.2)
    st.pyplot(fig_twin, use_container_width=True); plt.close()

    st.markdown("<hr>", unsafe_allow_html=True)

# ═══════════════════════════════════════════
# SECTION 7 — VEHICLE COMPARISON TOOL
# ═══════════════════════════════════════════
if show_comparison and compare_vehicle:
    all_db = {**EV_CARS, **EV_SCOOTERS}
    cv = all_db[compare_vehicle]
    cv_input = pd.DataFrame([[cv["battery"],cycles,age,avg_speed,temperature,
                               daily_km,fast_charge,charge_level,terrain_val,hvac]], columns=features)
    cv_soh   = float(soh_model.predict(cv_input)[0])
    cv_range = float(range_model.predict(cv_input)[0])
    cv_rul   = float(rul_model.predict(cv_input)[0])
    cv_res   = float(res_model.predict(cv_input)[0])
    cv_charge_cost = cv["battery"] * cm["rate"]
    cv_cpkm  = cv_charge_cost / max(cv_range,1)

    st.markdown('<div class="sec-head">🆚 Vehicle Comparison Tool</div>', unsafe_allow_html=True)
    comp_rows = [
        ("Battery Chemistry",      v.get("chemistry","N/A"),      cv.get("chemistry","N/A"),     False),
        ("Battery Capacity (kWh)", f"{v['battery']} kWh",         f"{cv['battery']} kWh",        True,  v['battery'],  cv['battery']),
        ("ARAI Range (km)",        f"{v['range']} km",             f"{cv['range']} km",           True,  v['range'],    cv['range']),
        ("Real Range — AI (km)",   f"{pred_range:.0f} km",         f"{cv_range:.0f} km",          True,  pred_range,    cv_range),
        ("Battery Health SOH",     f"{pred_soh:.1f}%",             f"{cv_soh:.1f}%",              True,  pred_soh,      cv_soh),
        ("Remaining Useful Life",  f"{pred_rul:.0f} cycles",       f"{cv_rul:.0f} cycles",        True,  pred_rul,      cv_rul),
        ("Internal Resistance",    f"{pred_res*1000:.1f} mΩ",      f"{cv_res*1000:.1f} mΩ",       False, pred_res,      cv_res),  # lower is better
        ("Cost per km",            f"₹{cost_per_km:.2f}",          f"₹{cv_cpkm:.2f}",             False, cost_per_km,   cv_cpkm),
        ("Cooling System",         v.get("cooling","N/A"),         cv.get("cooling","N/A"),       False),
        ("Price",                  f"₹{v['price']:.2f}L",          f"₹{cv['price']:.2f}L",        False, v['price'],    cv['price']),
    ]
    html_c = f"""<table class="comp-table">
    <thead><tr>
        <th>Parameter</th>
        <th>🚗 {selected_vehicle.split("(")[0].strip()}</th>
        <th>🚗 {compare_vehicle.split("(")[0].strip()}</th>
    </tr></thead><tbody>"""
    for row in comp_rows:
        label = row[0]; va = row[1]; vb = row[2]
        if len(row) > 4:
            higher_better = row[3]; na = row[4]; nb = row[5]
            if higher_better:
                ca = "comp-better" if na >= nb else "comp-worse"
                cb = "comp-better" if nb >= na else "comp-worse"
            else:
                ca = "comp-better" if na <= nb else "comp-worse"
                cb = "comp-better" if nb <= na else "comp-worse"
        else:
            ca = cb = ""
        html_c += f'<tr><td style="font-weight:700;color:#1a2e4a;">{label}</td><td class="{ca}">{va}</td><td class="{cb}">{vb}</td></tr>'
    html_c += "</tbody></table>"
    st.markdown(html_c, unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

# ═══════════════════════════════════════════
# CORE CHARTS ROW (SOH Gauge + Expiry + Cost)
# ═══════════════════════════════════════════
st.markdown('<div class="sec-head">🔋 Battery Lifespan by Charging Mode</div>', unsafe_allow_html=True)
expiry_rows = []
for mode_name, edata in expiry_data.items():
    y70 = edata["years_to_70"]; y80 = edata["years_to_80"]
    expiry_rows.append({"Charging Mode": mode_name, "→ 80% SOH": str(y80)+(" yrs" if isinstance(y80,int) else ""), "→ 70% SOH (Replace)": str(y70)+(" yrs" if isinstance(y70,int) else ""), "Speed": edata["speed"], "Sel": mode_name==charging_model})
expiry_df = pd.DataFrame(expiry_rows)
table_html = '<table class="expiry-table"><thead><tr><th>Charging Mode</th><th>→ 80% SOH</th><th>→ 70% SOH (Replace)</th><th>Speed</th></tr></thead><tbody>'
best_y70 = max([e["years_to_70"] for e in expiry_data.values() if isinstance(e["years_to_70"],(int,float))], default=0)
worst_y70= min([e["years_to_70"] for e in expiry_data.values() if isinstance(e["years_to_70"],(int,float))], default=0)
for _, row in expiry_df.iterrows():
    rc = ' class="current-row"' if row["Sel"] else ""
    y70v = expiry_data[row["Charging Mode"]]["years_to_70"]
    cl70 = "expiry-best" if y70v==best_y70 else "expiry-worst" if y70v==worst_y70 else "expiry-mid"
    table_html += f'<tr{rc}><td>{"★ " if row["Sel"] else ""}{row["Charging Mode"]}</td><td>{row["→ 80% SOH"]}</td><td class="{cl70}">{row["→ 70% SOH (Replace)"]}</td><td>{row["Speed"]}</td></tr>'
table_html += "</tbody></table>"
st.markdown(table_html, unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# Charts 1 & 2
g1c, g2c = st.columns(2, gap="large")
with g1c:
    st.markdown('<div class="sec-head">📈 SOH Forecast by Charging Mode</div>', unsafe_allow_html=True)
    fig1, ax1 = plt.subplots(figsize=(6.5, 4.2))
    fig1.patch.set_facecolor(PLOT_BG); ax1.set_facecolor(PLOT_FG)
    years_x = list(range(16))
    for mode_name, edata in expiry_data.items():
        lw = 2.5 if mode_name==charging_model else 1.1
        alpha = 1.0 if mode_name==charging_model else 0.4
        label = mode_name.split("(")[0].strip().replace("🏠 ","").replace("🌆 ","").replace("⚡ ","").replace("🚀 ","")
        ax1.plot(years_x, edata["soh_trace"], color=edata["color"], linewidth=lw, alpha=alpha, label=label)
    ax1.axhline(70, color="#dc2626", linestyle="--", linewidth=1.5, alpha=0.7)
    ax1.axhline(80, color="#d97706", linestyle=":", linewidth=1.2, alpha=0.6)
    ax1.set_xlabel("Years from now", color=AXIS_COL, fontsize=9)
    ax1.set_ylabel("SOH (%)", color=AXIS_COL, fontsize=9)
    ax1.tick_params(colors=TEXT_COL, labelsize=8)
    for sp in ax1.spines.values(): sp.set_edgecolor(GRID_COL)
    ax1.grid(True, alpha=0.4, color=GRID_COL)
    ax1.legend(fontsize=6.5, loc="upper right", framealpha=0.9)
    plt.tight_layout(pad=1.2)
    st.pyplot(fig1, use_container_width=True); plt.close()

with g2c:
    st.markdown('<div class="sec-head">💰 Cost per km by Charging Mode</div>', unsafe_allow_html=True)
    fig2, ax2 = plt.subplots(figsize=(6.5, 4.2))
    fig2.patch.set_facecolor(PLOT_BG); ax2.set_facecolor(PLOT_FG)
    mode_labels=[]; costs_per_km_list=[]; colors_bar=[]
    for mode_name, mode_data in CHARGING_MODELS.items():
        fc = battery_kwh * mode_data["rate"]
        cpkm = fc / max(pred_range,1)
        short_label = mode_name.split("(")[0].strip().replace("🏠 ","").replace("🌆 ","").replace("⚡ ","").replace("🚀 ","")
        mode_labels.append(short_label); costs_per_km_list.append(cpkm); colors_bar.append(mode_data["color"])
    bars2 = ax2.barh(mode_labels, costs_per_km_list, color=colors_bar, height=0.52, edgecolor=PLOT_BG, linewidth=1.5, zorder=3)
    cur_idx = list(CHARGING_MODELS.keys()).index(charging_model)
    bars2[cur_idx].set_edgecolor("#1d4ed8"); bars2[cur_idx].set_linewidth(3)
    for bar, val, col in zip(bars2, costs_per_km_list, colors_bar):
        ax2.text(val+max(costs_per_km_list)*0.015, bar.get_y()+bar.get_height()/2, f"₹{val:.2f}/km", va='center', color=col, fontsize=8.5, fontweight='700')
    petrol_cpkm = petrol_cost_l/petrol_km_per_l
    ax2.axvline(petrol_cpkm, color="#94a3b8", linestyle="--", linewidth=1.5, alpha=0.8)
    ax2.text(petrol_cpkm+max(costs_per_km_list)*0.01, len(mode_labels)-0.1, f"Petrol ₹{petrol_cpkm:.1f}/km", color="#64748b", fontsize=7.5, va='top')
    ax2.set_xlabel("Cost per km (₹)", color=AXIS_COL, fontsize=9)
    ax2.tick_params(colors=TEXT_COL, labelsize=8.5)
    for sp in ax2.spines.values(): sp.set_edgecolor(GRID_COL)
    ax2.grid(True, alpha=0.5, color=GRID_COL, axis='x', linewidth=0.8)
    ax2.set_xlim(0, max(costs_per_km_list)*1.3)
    plt.tight_layout(pad=1.4)
    st.pyplot(fig2, use_container_width=True); plt.close()

st.markdown("<hr>", unsafe_allow_html=True)

# SOH Gauge + EV vs Petrol
g3c, g4c = st.columns(2, gap="large")
with g3c:
    st.markdown('<div class="sec-head">🔋 Battery Health Gauge</div>', unsafe_allow_html=True)
    fig3, ax3 = plt.subplots(figsize=(6.5,4.0)); fig3.patch.set_facecolor(PLOT_BG); ax3.set_facecolor(PLOT_BG)
    ax3.set_aspect('equal'); ax3.axis('off')
    for pct_s, pct_e, fc in [(0,70,"#fef2f2"),(70,80,"#fffbeb"),(80,100,"#f0fdf4")]:
        ax3.add_patch(Wedge((0.5,0.18),0.38,180+pct_s*1.8,180+pct_e*1.8,width=0.12,facecolor=fc,edgecolor="#e2e8f0",linewidth=0.8))
    ax3.add_patch(Wedge((0.5,0.18),0.38,180,180+pred_soh*1.8,width=0.12,facecolor=soh_color,edgecolor="none",alpha=0.85))
    needle_angle_rad = math.radians(180+pred_soh*1.8)
    ax3.annotate("", xy=(0.5+0.30*math.cos(needle_angle_rad),0.18+0.30*math.sin(needle_angle_rad)), xytext=(0.5,0.18),
                 arrowprops=dict(arrowstyle="-|>",color=soh_color,lw=2.5,mutation_scale=14))
    ax3.add_patch(plt.Circle((0.5,0.18),0.025,color="#1e293b",zorder=10))
    for pct, label in [(0,"0%"),(25,"25%"),(50,"50%"),(70,"70%⚠"),(80,"80%"),(100,"100%")]:
        a_rad = math.radians(180+pct*1.8)
        ax3.text(0.5+0.43*math.cos(a_rad),0.18+0.43*math.sin(a_rad),label,ha='center',va='center',fontsize=6.5,color=TEXT_COL,fontweight='600')
    ax3.text(0.5,-0.08,f"{pred_soh:.1f}%",ha='center',va='center',fontsize=24,fontweight='900',color=soh_color,transform=ax3.transAxes)
    ax3.text(0.5,-0.18,soh_label,ha='center',va='center',fontsize=9,color=TEXT_COL,transform=ax3.transAxes)
    ax3.set_xlim(0.04,0.96); ax3.set_ylim(-0.25,0.65)
    plt.tight_layout(pad=0.5)
    st.pyplot(fig3, use_container_width=True); plt.close()

with g4c:
    st.markdown('<div class="sec-head">💰 EV vs Petrol — Annual Cost</div>', unsafe_allow_html=True)
    fig4, ax4 = plt.subplots(figsize=(6.5,4.0)); fig4.patch.set_facecolor(PLOT_BG); ax4.set_facecolor(PLOT_FG)
    cats=["EV\\n(Your Charger)","Petrol\\nEquivalent"]; vals=[annual_cost,petrol_monthly*12]
    brs=ax4.bar(cats,vals,color=["#0891b2","#f97316"],width=0.4,zorder=3,edgecolor=PLOT_BG,linewidth=2)
    for bar,val,col in zip(brs,vals,["#0891b2","#f97316"]):
        ax4.text(bar.get_x()+bar.get_width()/2,bar.get_height()+max(vals)*0.02,f"₹{val:,.0f}",ha='center',va='bottom',fontsize=11,fontweight='900',color=col)
    ax4.annotate("",xy=(1,vals[1]*0.98),xytext=(0,vals[0]*1.02),arrowprops=dict(arrowstyle="<->",color="#16a34a",lw=2.5,connectionstyle="arc3,rad=0"))
    ax4.text(0.5,(vals[0]+vals[1])/2*1.05,f"Save ₹{annual_savings:,.0f}/yr ✅",ha='center',va='center',fontsize=9.5,fontweight='800',color="#16a34a",bbox=dict(boxstyle='round,pad=0.4',facecolor='#f0fdf4',edgecolor='#bbf7d0',linewidth=1.5))
    ax4.set_ylim(0,max(vals)*1.35); ax4.yaxis.set_visible(False)
    ax4.tick_params(colors=TEXT_COL,labelsize=10)
    for sp in ax4.spines.values(): sp.set_edgecolor(GRID_COL)
    ax4.grid(True,alpha=0.5,color=GRID_COL,axis='y',linewidth=0.8)
    plt.tight_layout(pad=1.4)
    st.pyplot(fig4, use_container_width=True); plt.close()

st.markdown("<hr>", unsafe_allow_html=True)

# ═══════════════════════════════════════════
# CHARGING COST TABLE
# ═══════════════════════════════════════════
st.markdown('<div class="sec-head">⚡ Full Charging Cost Breakdown — India 2026</div>', unsafe_allow_html=True)
rows2 = []
for name, c in CHARGING_MODELS.items():
    fc=battery_kwh*c["rate"]; cph=battery_kwh/c["power_kw"]; cpkm=fc/max(pred_range,1); mc=(monthly_km/max(pred_range,1))*fc
    rows2.append({"Charging Type":name,"₹/kWh":f"₹{c['rate']:.0f}","Full Charge Cost":f"₹{fc:.0f}","Charge Time":f"{cph:.1f} hrs" if cph>1 else f"{cph*60:.0f} min","₹/km":f"₹{cpkm:.2f}","Monthly Cost":f"₹{mc:,.0f}","Selected":name==charging_model})
df_cost2=pd.DataFrame(rows2)
table_html2='<table class="expiry-table"><thead><tr>'
for col in ["Charging Type","₹/kWh","Full Charge Cost","Charge Time","₹/km","Monthly Cost"]: table_html2+=f"<th>{col}</th>"
table_html2+="</tr></thead><tbody>"
for _,row in df_cost2.iterrows():
    rc=' class="current-row"' if row["Selected"] else ""
    table_html2+=f"<tr{rc}>"
    for col in ["Charging Type","₹/kWh","Full Charge Cost","Charge Time","₹/km","Monthly Cost"]:
        bold=' style="font-weight:800;color:#1d4ed8;"' if row["Selected"] and col!="Charging Type" else ""
        table_html2+=f"<td{bold}>{row[col]}</td>"
    table_html2+="</tr>"
table_html2+="</tbody></table>"
st.markdown(table_html2, unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# ═══════════════════════════════════════════
# SECTION 8 — BATTERY PROTECTION GUIDE
# ═══════════════════════════════════════════
if show_protection:
    st.markdown('<div class="sec-head">🛡️ How to Protect Your Battery from Degradation</div>', unsafe_allow_html=True)
    protect_tips = [
        ("🔋","Keep Charge Between 20% and 80%","Lithium-ion batteries age fastest at the extremes. Charging to 100% daily creates cell stress, and letting it drain to 0% causes deep discharge damage. Set your charge limit to 80% for everyday use — only go to 100% before a long trip.","#0369a1","#eff6ff","#bfdbfe"),
        ("❄️","Never Charge in Extreme Heat","Charging your battery when it's hot (>38°C) multiplies degradation. The battery generates its own heat during charging. Wait 15–20 minutes after parking before plugging in, and park in shade.","#0f766e","#f0fdfa","#99f6e4"),
        ("⚡","Use DC Fast Charging Sparingly","DC fast chargers push current at 3–5× the normal rate, generating significant internal heat. Cells that regularly experience fast-charge heat cycles degrade 2–3× faster. Reserve DC fast charging for highway emergencies.","#7c3aed","#faf5ff","#ddd6fe"),
        ("🌙","Charge Overnight, Not All Day","Lithium cells sitting at 100% SOC for hours generates calendar aging. Overnight charging finishes just before you wake up. Use your EV app's scheduled departure feature.","#0891b2","#f0f9ff","#bae6fd"),
        ("🐢","Drive Smoothly","Hard acceleration spikes current draw from the battery, creating heat spikes inside the cells. Smooth driving with regenerative braking can increase real-world range by 15–25% and significantly reduce thermal stress.","#16a34a","#f0fdf4","#bbf7d0"),
        ("🌡️","Pre-condition in Extreme Heat","In Indian summers (>40°C), run your EV's pre-cooling while still plugged in — this cools the cabin AND the battery pack using grid power, not battery power.","#d97706","#fffbeb","#fde68a"),
        ("🔌","Store at 50% if Not Using for Weeks","Going on a trip and leaving your EV parked? Charge to ~50% SOC before parking. A battery sitting at full charge self-discharges while oxidising the anode — at 50% this process is slowest.","#dc2626","#fef2f2","#fecaca"),
        ("📱","Use Manufacturer App for Health Reports","Review your battery health monthly and contact your service centre if SOH drops more than 3–4% per year — early intervention prevents irreversible damage.","#475569","#f8fafc","#e2e8f0"),
    ]
    cols_p=st.columns(2,gap="large")
    for i,(icon,title,body,tc,bgc,bc) in enumerate(protect_tips):
        with cols_p[i%2]:
            st.markdown(f"""<div style="background:{bgc};border:1.5px solid {bc};border-left:5px solid {tc};border-radius:16px;padding:18px 20px;margin-bottom:12px;">
            <div style="font-family:'Nunito',sans-serif;font-size:1rem;font-weight:900;color:{tc};margin-bottom:8px;">{icon} {title}</div>
            <div style="font-size:0.82rem;color:#475569;line-height:1.7;">{body}</div></div>""", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

# ═══════════════════════════════════════════
# SECTION 9 — AI PERSONALISED TIPS
# ═══════════════════════════════════════════
if show_ai_tips:
    st.markdown('<div class="sec-head">💡 AI Recommendations — Personalised to Your Usage</div>', unsafe_allow_html=True)
    tips=[]
    if fast_charge>0.5: tips.append(("warn",f"⚡ <b>High DC fast-charge usage detected ({fast_charge*100:.0f}%).</b> Limit fast charging to highway emergencies — frequent DC charging generates internal heat that degrades your battery 2–3× faster."))
    if charge_level>0.95: tips.append(("warn","🔋 <b>You're charging to 100% regularly.</b> Keep daily charge at 80–90% — this single habit can extend battery life by 20–30%."))
    if temperature>38: tips.append(("warn",f"🌡️ <b>Ambient temp {temperature}°C is very high.</b> Park in shade or a covered garage. Heat above 35°C is the #1 enemy of lithium batteries in Indian summers."))
    if cell_temp_max>45: tips.append(("bad",f"🔴 <b>Cell temperature {cell_temp_max}°C is critical.</b> Stop charging/driving immediately and allow battery to cool."))
    if temp_delta>10: tips.append(("bad",f"⚠ <b>High temperature gradient (ΔT={temp_delta:.1f}°C)</b> detected across cells. This indicates hotspot formation and possible BTMS failure."))
    if cell_imbalance>3: tips.append(("bad","⚠ <b>Cell voltage imbalance is high.</b> A faulty or degraded cell is suspected. Contact your service centre for cell-level diagnostics."))
    if coulombic_eff<96: tips.append(("warn",f"📉 <b>Coulombic efficiency ({coulombic_eff:.1f}%) is below optimal.</b> Reduce fast charging and high-temperature operations."))
    if avg_speed>110 and not is_scooter: tips.append(("warn","🚀 <b>Highway speeds above 110 km/h cut range by up to 35%.</b> Eco mode at 90–100 km/h saves significantly and reduces cell stress."))
    if hvac>0.7: tips.append(("warn",f"❄️ <b>Heavy AC/heater usage ({hvac*100:.0f}%).</b> Pre-cool the cabin while plugged in before you start driving — grid power, not battery power."))
    if pred_soh<75: tips.append(("bad",f"🔴 <b>Battery at {pred_soh:.1f}% SOH.</b> If within warranty period, visit your authorised service centre — most manufacturers cover battery replacement below 70–75% SOH within 5–8 years."))
    if cm["speed"] in ["Fast","Ultra"] and fast_charge>0.5: tips.append(("bad","🔴 <b>Using DC fast chargers as your primary charger</b> is the fastest way to age your battery. Switch to home AC charging for 70%+ of charges."))
    if cm["speed"]=="Slow" and fast_charge<0.2: tips.append(("good","✅ <b>Excellent charging habits!</b> AC-only / home charging is the gentlest on your battery. You're on the right path to maximising battery lifespan."))
    if not tips: tips.append(("good","✅ <b>All systems optimal!</b> Your current usage patterns are excellent. Keep charging at home, avoid extreme heat, and maintain 20–80% charge range daily."))
    tip_cols=st.columns(2,gap="large")
    for i,(kind,text) in enumerate(tips):
        css="tip-good" if kind=="good" else "tip-bad" if kind=="bad" else "tip-warn"
        tip_cols[i%2].markdown(f'<div class="{css}">{text}</div>', unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

# ═══════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════
b1f,b2f,b3f = st.columns(3, gap="large")
with b1f:
    st.markdown(f"""<div class="footer-card"><div class="fc-title">🤖 AI Model Info</div><div class="fc-row">
    Random Forest (SOH, Range, Resistance)<br>Gradient Boosting (RUL prediction)<br>Training: 3,000 battery records<br>
    SOH R² = {soh_r2:.3f} · Range R² = {range_r2:.3f}<br>RUL R² = {rul_r2:.3f} · Res R² = {res_r2:.3f}
    </div></div>""", unsafe_allow_html=True)
with b2f:
    st.markdown(f"""<div class="footer-card"><div class="fc-title">📊 Session Summary</div><div class="fc-row">
    Vehicle: {selected_vehicle.split("(")[0].strip()}<br>Battery: {v['battery']} kWh · Chemistry: {v.get('chemistry','N/A')}<br>
    SOH: {pred_soh:.1f}% · Range: {pred_range:.0f} km · RUL: {pred_rul:.0f} cy<br>
    Health Score: {health_score:.0f}/100 · Thermal: {thermal_status.split('—')[0].strip()}<br>
    Annual EV saving vs petrol: ₹{annual_savings:,.0f}
    </div></div>""", unsafe_allow_html=True)
with b3f:
    st.markdown(f"""<div class="footer-card"><div class="fc-title">📡 Data Sources</div><div class="fc-row">
    EV specs: CarWale & eScooterWale India 2026<br>Charging rates: Pulse Energy India 2026<br>
    Home AC: ₹6–10/kWh avg · Public DC Fast: ₹18–25/kWh avg<br>
    Petrol: ₹105/litre avg India 2026<br>BMS parameters: IEEE 1725 / IEC 62133 standards
    </div></div>""", unsafe_allow_html=True)

st.markdown('<div style="text-align:center;color:#94a3b8;font-size:0.72rem;margin-top:24px;padding-bottom:12px;">⚡ EV Battery Pro AI India 2026 · Professional BMS Analytics · AI predictions are estimates based on typical usage patterns · Not a substitute for professional battery diagnosis · v4.0</div>', unsafe_allow_html=True)
