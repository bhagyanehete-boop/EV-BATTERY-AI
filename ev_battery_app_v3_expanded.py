import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, Arc, Wedge
import matplotlib.patheffects as pe
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="EV Battery AI — India 2026",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# WHITE THEME & CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; background: #f0f4f8; }
h1,h2,h3 { font-family: 'Nunito', sans-serif; letter-spacing: -0.01em; }

.stApp { background: #f0f4f8; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #e8eef4; }
::-webkit-scrollbar-thumb { background: #94b4d0; border-radius: 3px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a2e4a 0%, #0f1f35 100%) !important;
    border-right: none;
    box-shadow: 4px 0 20px rgba(0,0,0,0.15);
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

/* ── Hero ── */
.hero-banner {
    background: linear-gradient(120deg, #1a2e4a 0%, #1e4080 50%, #0d6eaa 100%);
    border-radius: 28px;
    padding: 40px 48px 32px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 16px 40px rgba(26,46,74,0.25);
}
.hero-banner::before {
    content: '';
    position: absolute; top:-80px; right:-80px;
    width:320px; height:320px;
    background: radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 65%);
    border-radius:50%;
}
.hero-banner::after {
    content: '';
    position: absolute; bottom:-40px; right:60px;
    width:200px; height:200px;
    background: radial-gradient(circle, rgba(100,200,255,0.10) 0%, transparent 70%);
    border-radius:50%;
}
.hero-title {
    font-family:'Nunito',sans-serif; font-size:2.6rem; font-weight:900;
    color:#ffffff; margin:0 0 6px 0; line-height:1.05;
    text-shadow: 0 2px 12px rgba(0,0,0,0.2);
}
.hero-sub { color:rgba(255,255,255,0.7); font-size:0.92rem; margin:0 0 18px 0; }
.hero-badge {
    display:inline-block; background:rgba(255,255,255,0.12);
    border:1px solid rgba(255,255,255,0.25); backdrop-filter:blur(8px);
    color:#ffffff; border-radius:20px; padding:5px 16px;
    font-size:0.73rem; font-weight:700; margin-right:8px; margin-bottom:4px;
}

/* ── Metric cards ── */
.metric-card {
    background: #ffffff;
    border: none;
    border-radius: 20px;
    padding: 20px 22px 16px;
    margin-bottom: 12px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 20px rgba(0,0,0,0.07);
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover { transform: translateY(-3px); box-shadow: 0 8px 28px rgba(0,0,0,0.12); }
.metric-accent { position:absolute; top:0; left:0; width:5px; height:100%; border-radius:20px 0 0 20px; }
.metric-label { font-size:0.66rem; font-weight:800; color:#94a3b8; text-transform:uppercase; letter-spacing:0.12em; margin-bottom:8px; }
.metric-value { font-family:'Nunito',sans-serif; font-size:2.2rem; font-weight:900; color:#1e293b; line-height:1; }
.metric-sub { font-size:0.72rem; color:#64748b; margin-top:5px; }

/* ── Vehicle card ── */
.vehicle-card {
    background: #ffffff;
    border-radius: 24px;
    padding: 24px;
    margin-bottom: 16px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.07);
    text-align: center;
}
.vehicle-name {
    font-family:'Nunito',sans-serif; font-size:1.1rem; font-weight:900;
    color:#1e293b; margin: 12px 0 4px;
}
.vehicle-price { font-size:0.82rem; color:#64748b; margin-bottom:14px; }
.spec-pill {
    display:inline-block;
    background:#eff6ff; border:1px solid #bfdbfe;
    color:#1d4ed8; border-radius:20px;
    padding:4px 12px; font-size:0.73rem; font-weight:700; margin:3px 3px 3px 0;
}

/* ── Section headers ── */
.sec-head {
    font-family:'Nunito',sans-serif; font-size:1.05rem; font-weight:900;
    color:#1a2e4a; text-transform:uppercase; letter-spacing:0.08em;
    border-bottom:2px solid #e2e8f0; padding-bottom:10px; margin-bottom:20px;
}

/* ── Battery lifespan table ── */
.expiry-table { width:100%; border-collapse:collapse; background:#ffffff; border-radius:16px; overflow:hidden; box-shadow:0 4px 20px rgba(0,0,0,0.07); }
.expiry-table th { background:#f8fafc; color:#64748b; padding:12px 16px; font-size:0.7rem; text-transform:uppercase; letter-spacing:0.08em; text-align:left; border-bottom:2px solid #e2e8f0; font-weight:800; }
.expiry-table td { padding:12px 16px; border-bottom:1px solid #f1f5f9; font-size:0.87rem; color:#334155; }
.expiry-table tr:last-child td { border-bottom: none; }
.expiry-table tr:hover td { background:#f8fafc; }
.expiry-table tr.current-row td { background:#eff6ff; }
.expiry-best { color:#16a34a; font-weight:800; }
.expiry-worst { color:#dc2626; font-weight:800; }
.expiry-mid   { color:#d97706; font-weight:700; }

/* ── Tip boxes ── */
.tip-warn {
    background:#fffbeb; border-left:4px solid #f59e0b;
    border-radius:12px; padding:14px 18px; margin:6px 0;
    font-size:0.84rem; color:#92400e; line-height:1.6;
    box-shadow: 0 2px 8px rgba(245,158,11,0.12);
}
.tip-good {
    background:#f0fdf4; border-left:4px solid #22c55e;
    border-radius:12px; padding:14px 18px; margin:6px 0;
    font-size:0.84rem; color:#14532d; line-height:1.6;
    box-shadow: 0 2px 8px rgba(34,197,94,0.12);
}
.tip-bad {
    background:#fef2f2; border-left:4px solid #ef4444;
    border-radius:12px; padding:14px 18px; margin:6px 0;
    font-size:0.84rem; color:#7f1d1d; line-height:1.6;
    box-shadow: 0 2px 8px rgba(239,68,68,0.12);
}

/* ── Param def boxes in sidebar ── */
.param-def { background:#162840; border:1px solid #2a4a6e; border-radius:10px; padding:10px 14px; margin-bottom:4px; font-size:0.76rem; color:#7ec8f7; line-height:1.65; }

/* ── Charging mode badges ── */
.mode-slow  { color:#16a34a; background:#f0fdf4; border:1px solid #bbf7d0; border-radius:12px; padding:3px 10px; font-size:0.73rem; font-weight:800; }
.mode-med   { color:#d97706; background:#fffbeb; border:1px solid #fde68a; border-radius:12px; padding:3px 10px; font-size:0.73rem; font-weight:800; }
.mode-fast  { color:#dc2626; background:#fef2f2; border:1px solid #fecaca; border-radius:12px; padding:3px 10px; font-size:0.73rem; font-weight:800; }
.mode-ultra { color:#7c3aed; background:#faf5ff; border:1px solid #ddd6fe; border-radius:12px; padding:3px 10px; font-size:0.73rem; font-weight:800; }

/* ── Protection tips special card ── */
.protect-card {
    background: linear-gradient(135deg, #ffffff 0%, #f0f9ff 100%);
    border: 1px solid #bae6fd;
    border-radius: 20px; padding: 22px 24px; margin-bottom:10px;
    box-shadow: 0 4px 16px rgba(14,165,233,0.1);
}
.protect-card h4 {
    font-family:'Nunito',sans-serif; font-size:0.95rem; font-weight:900;
    color:#0369a1; margin:0 0 8px 0;
}
.protect-card p { font-size:0.82rem; color:#475569; margin:0; line-height:1.65; }

/* ── Footer cards ── */
.footer-card {
    background: #ffffff; border-radius: 16px; padding:18px 20px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.06);
}
.footer-card .fc-title { font-family:'Nunito',sans-serif; font-weight:900; color:#1a2e4a; font-size:0.95rem; margin-bottom:10px; }
.footer-card .fc-row { font-size:0.78rem; color:#64748b; line-height:1.9; }

/* ── Divider ── */
hr { border: none; border-top: 2px solid #e2e8f0; margin: 28px 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# EV Database (unchanged from original)
# ─────────────────────────────────────────────
EV_CARS = {
    # ── TATA ──
    "Tata Tiago EV (19.2 kWh)":           {"battery":19.2, "range":250,  "type":"Car","price":5.84,  "charge_ac":3.3,  "charge_dc":0,   "brand":"tata",        "color":"#1d4ed8"},
    "Tata Tiago EV (24 kWh)":             {"battery":24.0, "range":315,  "type":"Car","price":8.49,  "charge_ac":7.2,  "charge_dc":0,   "brand":"tata",        "color":"#1d4ed8"},
    "Tata Punch EV (25 kWh)":             {"battery":25.0, "range":301,  "type":"Car","price":8.09,  "charge_ac":7.2,  "charge_dc":50,  "brand":"tata",        "color":"#0891b2"},
    "Tata Punch EV (35 kWh)":             {"battery":35.0, "range":421,  "type":"Car","price":11.19, "charge_ac":7.2,  "charge_dc":50,  "brand":"tata",        "color":"#0891b2"},
    "Tata Nexon EV (30 kWh)":             {"battery":30.0, "range":325,  "type":"Car","price":12.49, "charge_ac":7.2,  "charge_dc":50,  "brand":"tata",        "color":"#075985"},
    "Tata Nexon EV (40.5 kWh)":           {"battery":40.5, "range":465,  "type":"Car","price":15.49, "charge_ac":7.2,  "charge_dc":50,  "brand":"tata",        "color":"#075985"},
    "Tata Curvv EV (45 kWh)":             {"battery":45.0, "range":502,  "type":"Car","price":16.99, "charge_ac":11.0, "charge_dc":70,  "brand":"tata",        "color":"#164e63"},
    "Tata Curvv EV (55 kWh)":             {"battery":55.0, "range":585,  "type":"Car","price":19.49, "charge_ac":11.0, "charge_dc":70,  "brand":"tata",        "color":"#164e63"},
    "Tata Harrier EV (71 kWh)":           {"battery":71.0, "range":627,  "type":"Car","price":21.49, "charge_ac":11.0, "charge_dc":150, "brand":"tata",        "color":"#1e3a8a"},
    "Tata Tigor EV (26 kWh)":             {"battery":26.0, "range":306,  "type":"Car","price":12.49, "charge_ac":7.2,  "charge_dc":0,   "brand":"tata",        "color":"#2563eb"},
    # ── MG ──
    "MG Comet EV (17.3 kWh)":             {"battery":17.3, "range":230,  "type":"Car","price":6.31,  "charge_ac":3.3,  "charge_dc":0,   "brand":"mg",          "color":"#dc2626"},
    "MG ZS EV (50.3 kWh)":                {"battery":50.3, "range":461,  "type":"Car","price":15.50, "charge_ac":7.2,  "charge_dc":50,  "brand":"mg",          "color":"#b91c1c"},
    "MG Windsor EV (38 kWh)":             {"battery":38.0, "range":332,  "type":"Car","price":12.04, "charge_ac":7.2,  "charge_dc":50,  "brand":"mg",          "color":"#b91c1c"},
    "MG Cyberster (77 kWh)":              {"battery":77.0, "range":480,  "type":"Car","price":77.50, "charge_ac":11.0, "charge_dc":135, "brand":"mg",          "color":"#991b1b"},
    "MG M9 (90 kWh)":                     {"battery":90.0, "range":520,  "type":"Car","price":75.90, "charge_ac":11.0, "charge_dc":150, "brand":"mg",          "color":"#7f1d1d"},
    # ── MAHINDRA ──
    "Mahindra BE 6 (59 kWh)":             {"battery":59.0, "range":535,  "type":"Car","price":18.90, "charge_ac":11.0, "charge_dc":175, "brand":"mahindra",    "color":"#7c3aed"},
    "Mahindra BE 6 (79 kWh)":             {"battery":79.0, "range":682,  "type":"Car","price":24.90, "charge_ac":11.0, "charge_dc":175, "brand":"mahindra",    "color":"#6d28d9"},
    "Mahindra BE 6 FE (79 kWh)":          {"battery":79.0, "range":682,  "type":"Car","price":23.69, "charge_ac":11.0, "charge_dc":175, "brand":"mahindra",    "color":"#6d28d9"},
    "Mahindra XEV 9e (79 kWh)":           {"battery":79.0, "range":656,  "type":"Car","price":21.90, "charge_ac":11.0, "charge_dc":175, "brand":"mahindra",    "color":"#5b21b6"},
    "Mahindra XEV 9S (79 kWh)":           {"battery":79.0, "range":656,  "type":"Car","price":19.95, "charge_ac":11.0, "charge_dc":175, "brand":"mahindra",    "color":"#5b21b6"},
    "Mahindra XUV400 (39.4 kWh)":         {"battery":39.4, "range":375,  "type":"Car","price":15.49, "charge_ac":7.2,  "charge_dc":50,  "brand":"mahindra",    "color":"#4c1d95"},
    # ── HYUNDAI ──
    "Hyundai Creta Electric (51.4 kWh)":  {"battery":51.4, "range":473,  "type":"Car","price":18.03, "charge_ac":11.0, "charge_dc":100, "brand":"hyundai",     "color":"#0f766e"},
    "Hyundai Ioniq 5 (72.6 kWh)":         {"battery":72.6, "range":631,  "type":"Car","price":55.70, "charge_ac":11.0, "charge_dc":220, "brand":"hyundai",     "color":"#134e4a"},
    # ── MARUTI ──
    "Maruti e Vitara (49 kWh)":           {"battery":49.0, "range":500,  "type":"Car","price":13.49, "charge_ac":11.0, "charge_dc":100, "brand":"maruti",      "color":"#15803d"},
    "Maruti e Vitara (61 kWh)":           {"battery":61.0, "range":550,  "type":"Car","price":17.26, "charge_ac":11.0, "charge_dc":100, "brand":"maruti",      "color":"#15803d"},
    # ── BYD ──
    "BYD Atto 3 (60.5 kWh)":              {"battery":60.5, "range":480,  "type":"Car","price":24.99, "charge_ac":11.0, "charge_dc":80,  "brand":"byd",         "color":"#c2410c"},
    "BYD Seal (82.56 kWh)":               {"battery":82.56,"range":650,  "type":"Car","price":41.00, "charge_ac":11.0, "charge_dc":150, "brand":"byd",         "color":"#9a3412"},
    "BYD Sealion 7 (82.56 kWh)":          {"battery":82.56,"range":567,  "type":"Car","price":49.40, "charge_ac":11.0, "charge_dc":150, "brand":"byd",         "color":"#7c2d12"},
    "BYD eMax 7 (71.8 kWh)":              {"battery":71.8, "range":530,  "type":"Car","price":26.90, "charge_ac":11.0, "charge_dc":100, "brand":"byd",         "color":"#ea580c"},
    # ── VINFAST ──
    "VinFast VF 6 (59.6 kWh)":            {"battery":59.6, "range":473,  "type":"Car","price":17.29, "charge_ac":11.0, "charge_dc":150, "brand":"vinfast",     "color":"#0369a1"},
    "VinFast VF 7 (75.3 kWh)":            {"battery":75.3, "range":450,  "type":"Car","price":21.89, "charge_ac":11.0, "charge_dc":150, "brand":"vinfast",     "color":"#0284c7"},
    "VinFast VF MPV 7 (94 kWh)":          {"battery":94.0, "range":580,  "type":"Car","price":24.49, "charge_ac":11.0, "charge_dc":150, "brand":"vinfast",     "color":"#0ea5e9"},
    # ── KIA ──
    "Kia Carens Clavis EV (51.4 kWh)":    {"battery":51.4, "range":473,  "type":"Car","price":17.99, "charge_ac":11.0, "charge_dc":100, "brand":"kia",         "color":"#dc2626"},
    "Kia EV6 (77.4 kWh)":                 {"battery":77.4, "range":708,  "type":"Car","price":65.97, "charge_ac":11.0, "charge_dc":240, "brand":"kia",         "color":"#b91c1c"},
    "Kia EV9 (99.8 kWh)":                 {"battery":99.8, "range":541,  "type":"Car","price":130.00,"charge_ac":11.0, "charge_dc":240, "brand":"kia",         "color":"#991b1b"},
    # ── CITROEN ──
    "Citroen eC3 (29.2 kWh)":             {"battery":29.2, "range":320,  "type":"Car","price":12.90, "charge_ac":7.2,  "charge_dc":0,   "brand":"citroen",     "color":"#dc2626"},
    # ── MERCEDES-BENZ ──
    "Mercedes-Benz CLA EV (85 kWh)":      {"battery":85.0, "range":750,  "type":"Car","price":55.00, "charge_ac":11.0, "charge_dc":200, "brand":"mercedes",    "color":"#374151"},
    "Mercedes-Benz EQS (107.8 kWh)":      {"battery":107.8,"range":857,  "type":"Car","price":130.00,"charge_ac":11.0, "charge_dc":200, "brand":"mercedes",    "color":"#1f2937"},
    "Mercedes-Benz EQS SUV (108.4 kWh)":  {"battery":108.4,"range":675,  "type":"Car","price":134.00,"charge_ac":11.0, "charge_dc":200, "brand":"mercedes",    "color":"#111827"},
    "Mercedes-Benz EQE SUV (90.6 kWh)":   {"battery":90.6, "range":590,  "type":"Car","price":141.00,"charge_ac":11.0, "charge_dc":170, "brand":"mercedes",    "color":"#1f2937"},
    "Mercedes-Benz AMG EQS (107.8 kWh)":  {"battery":107.8,"range":600,  "type":"Car","price":245.00,"charge_ac":11.0, "charge_dc":200, "brand":"mercedes",    "color":"#111827"},
    "Mercedes-Benz G580 EQ (116 kWh)":    {"battery":116.0,"range":473,  "type":"Car","price":310.00,"charge_ac":11.0, "charge_dc":200, "brand":"mercedes",    "color":"#374151"},
    "Mercedes-Benz Maybach EQS SUV (108 kWh)":{"battery":108.0,"range":600,"type":"Car","price":240.00,"charge_ac":11.0,"charge_dc":200,"brand":"mercedes",   "color":"#111827"},
    # ── BMW ──
    "BMW iX1 LWB (64.7 kWh)":             {"battery":64.7, "range":440,  "type":"Car","price":51.40, "charge_ac":11.0, "charge_dc":130, "brand":"bmw",         "color":"#1e40af"},
    "BMW iX1 (64.7 kWh)":                 {"battery":64.7, "range":438,  "type":"Car","price":66.90, "charge_ac":11.0, "charge_dc":130, "brand":"bmw",         "color":"#1e40af"},
    "BMW i4 (83.9 kWh)":                  {"battery":83.9, "range":590,  "type":"Car","price":72.50, "charge_ac":11.0, "charge_dc":205, "brand":"bmw",         "color":"#1d4ed8"},
    "BMW i5 (84 kWh)":                    {"battery":84.0, "range":582,  "type":"Car","price":120.00,"charge_ac":11.0, "charge_dc":205, "brand":"bmw",         "color":"#1e3a8a"},
    "BMW i7 (101.7 kWh)":                 {"battery":101.7,"range":625,  "type":"Car","price":205.00,"charge_ac":11.0, "charge_dc":195, "brand":"bmw",         "color":"#172554"},
    "BMW iX (111.5 kWh)":                 {"battery":111.5,"range":630,  "type":"Car","price":121.00,"charge_ac":11.0, "charge_dc":200, "brand":"bmw",         "color":"#1e3a8a"},
    # ── TESLA ──
    "Tesla Model Y (75 kWh)":             {"battery":75.0, "range":533,  "type":"Car","price":59.89, "charge_ac":11.0, "charge_dc":250, "brand":"tesla",       "color":"#dc2626"},
    # ── VOLVO ──
    "Volvo EX30 (69 kWh)":                {"battery":69.0, "range":476,  "type":"Car","price":41.00, "charge_ac":11.0, "charge_dc":153, "brand":"volvo",       "color":"#0f766e"},
    "Volvo EX40 (82 kWh)":                {"battery":82.0, "range":530,  "type":"Car","price":50.10, "charge_ac":11.0, "charge_dc":150, "brand":"volvo",       "color":"#115e59"},
    "Volvo EC40 (82 kWh)":                {"battery":82.0, "range":530,  "type":"Car","price":59.00, "charge_ac":11.0, "charge_dc":150, "brand":"volvo",       "color":"#134e4a"},
    # ── AUDI ──
    "Audi e-tron GT (93.4 kWh)":          {"battery":93.4, "range":488,  "type":"Car","price":172.00,"charge_ac":11.0, "charge_dc":270, "brand":"audi",        "color":"#991b1b"},
    "Audi Q8 e-tron (114 kWh)":           {"battery":114.0,"range":582,  "type":"Car","price":115.00,"charge_ac":11.0, "charge_dc":170, "brand":"audi",        "color":"#7f1d1d"},
    "Audi Q8 Sportback e-tron (114 kWh)": {"battery":114.0,"range":600,  "type":"Car","price":119.00,"charge_ac":11.0, "charge_dc":170, "brand":"audi",        "color":"#7f1d1d"},
    "Audi e-tron (95 kWh)":               {"battery":95.0, "range":484,  "type":"Car","price":102.00,"charge_ac":11.0, "charge_dc":150, "brand":"audi",        "color":"#991b1b"},
    "Audi e-tron Sportback (95 kWh)":     {"battery":95.0, "range":484,  "type":"Car","price":120.00,"charge_ac":11.0, "charge_dc":150, "brand":"audi",        "color":"#7f1d1d"},
    # ── PORSCHE ──
    "Porsche Taycan (93.4 kWh)":          {"battery":93.4, "range":484,  "type":"Car","price":170.00,"charge_ac":11.0, "charge_dc":270, "brand":"porsche",     "color":"#374151"},
    "Porsche Macan Turbo EV (100 kWh)":   {"battery":100.0,"range":516,  "type":"Car","price":122.00,"charge_ac":11.0, "charge_dc":270, "brand":"porsche",     "color":"#1f2937"},
    "Porsche Cayenne EV (130 kWh)":       {"battery":130.0,"range":600,  "type":"Car","price":176.00,"charge_ac":11.0, "charge_dc":270, "brand":"porsche",     "color":"#111827"},
    # ── MINI ──
    "Mini Cooper SE (54.2 kWh)":          {"battery":54.2, "range":402,  "type":"Car","price":53.50, "charge_ac":11.0, "charge_dc":95,  "brand":"mini",        "color":"#374151"},
    "Mini Countryman Electric (64.7 kWh)":{"battery":64.7, "range":462,  "type":"Car","price":55.65, "charge_ac":11.0, "charge_dc":130, "brand":"mini",        "color":"#1f2937"},
    # ── LEXUS ──
    "Lexus ES 300h (8.8 kWh PHEV)":       {"battery":8.8,  "range":54,   "type":"Car","price":89.99, "charge_ac":3.3,  "charge_dc":0,   "brand":"lexus",       "color":"#1e3a8a"},
    # ── ROLLS-ROYCE ──
    "Rolls-Royce Spectre (102 kWh)":      {"battery":102.0,"range":520,  "type":"Car","price":750.00,"charge_ac":11.0, "charge_dc":195, "brand":"rolls_royce", "color":"#374151"},
    # ── LOTUS ──
    "Lotus Emeya (102 kWh)":              {"battery":102.0,"range":610,  "type":"Car","price":234.00,"charge_ac":11.0, "charge_dc":350, "brand":"lotus",       "color":"#15803d"},
    "Lotus Eletre (112 kWh)":             {"battery":112.0,"range":600,  "type":"Car","price":255.00,"charge_ac":11.0, "charge_dc":350, "brand":"lotus",       "color":"#166534"},
    # -- Additional CarWale listed/upcoming global EVs --
    "Toyota Urban Cruiser Ebella (61 kWh)": {"battery":61.0, "range":500, "type":"Car","price":23.60, "charge_ac":11.0, "charge_dc":100, "brand":"toyota",      "color":"#dc2626"},
    "Skoda Enyaq (82 kWh)":              {"battery":82.0, "range":565, "type":"Car","price":50.00, "charge_ac":11.0, "charge_dc":175, "brand":"skoda",       "color":"#15803d"},
    "Skoda Epiq (55 kWh)":               {"battery":55.0, "range":400, "type":"Car","price":21.00, "charge_ac":11.0, "charge_dc":125, "brand":"skoda",       "color":"#16a34a"},
    "Tesla Model 3 (75 kWh)":            {"battery":75.0, "range":629, "type":"Car","price":70.00, "charge_ac":11.0, "charge_dc":250, "brand":"tesla",       "color":"#b91c1c"},
    "Tata Avinya (80 kWh)":              {"battery":80.0, "range":500, "type":"Car","price":45.00, "charge_ac":11.0, "charge_dc":150, "brand":"tata",        "color":"#1d4ed8"},
    "Tata Altroz EV (30 kWh)":           {"battery":30.0, "range":350, "type":"Car","price":12.00, "charge_ac":7.2,  "charge_dc":50,  "brand":"tata",        "color":"#2563eb"},
    "Tata New Tigor EV (26 kWh)":        {"battery":26.0, "range":315, "type":"Car","price":12.49, "charge_ac":7.2,  "charge_dc":0,   "brand":"tata",        "color":"#1e40af"},
    "Tata Sierra EV (60 kWh)":           {"battery":60.0, "range":500, "type":"Car","price":20.00, "charge_ac":11.0, "charge_dc":100, "brand":"tata",        "color":"#075985"},
    "MG IM6 (100 kWh)":                  {"battery":100.0,"range":650, "type":"Car","price":60.00, "charge_ac":11.0, "charge_dc":200, "brand":"mg",          "color":"#991b1b"},
    "Hyundai Ioniq 9 (110 kWh)":         {"battery":110.0,"range":620, "type":"Car","price":120.00,"charge_ac":11.0, "charge_dc":240, "brand":"hyundai",     "color":"#0f766e"},
    "Honda e:Ny1 (68.8 kWh)":            {"battery":68.8, "range":412, "type":"Car","price":40.00, "charge_ac":11.0, "charge_dc":100, "brand":"honda",       "color":"#dc2626"},
    "Renault 5 E-Tech (52 kWh)":         {"battery":52.0, "range":400, "type":"Car","price":25.00, "charge_ac":11.0, "charge_dc":100, "brand":"renault",     "color":"#f59e0b"},
    "Volkswagen ID.4 (82 kWh)":          {"battery":82.0, "range":520, "type":"Car","price":55.00, "charge_ac":11.0, "charge_dc":175, "brand":"volkswagen",  "color":"#1d4ed8"},
    "Nissan Ariya (87 kWh)":             {"battery":87.0, "range":533, "type":"Car","price":45.00, "charge_ac":11.0, "charge_dc":130, "brand":"nissan",      "color":"#475569"},
    "Jeep Avenger Electric (54 kWh)":    {"battery":54.0, "range":400, "type":"Car","price":50.00, "charge_ac":11.0, "charge_dc":100, "brand":"jeep",        "color":"#15803d"},
    "Land Rover Range Rover Electric (118 kWh)": {"battery":118.0,"range":500,"type":"Car","price":250.00,"charge_ac":11.0,"charge_dc":200,"brand":"land_rover","color":"#166534"},
    "Jaguar I-Pace (90 kWh)":            {"battery":90.0, "range":470, "type":"Car","price":125.00,"charge_ac":11.0, "charge_dc":100, "brand":"jaguar",      "color":"#111827"},
    "Maserati GranTurismo Folgore (92.5 kWh)": {"battery":92.5,"range":450,"type":"Car","price":250.00,"charge_ac":11.0,"charge_dc":270,"brand":"maserati","color":"#1e3a8a"},
    "Bentley EXP 12 Speed 6e (100 kWh)": {"battery":100.0,"range":500, "type":"Car","price":350.00,"charge_ac":11.0, "charge_dc":200, "brand":"bentley",     "color":"#166534"},
}

EV_SCOOTERS = {
    # ── OLA ──
    "Ola S1 Air (2 kWh)":            {"battery":2.0,  "range":101,"type":"Scooter","price":0.80, "charge_ac":0.75,"charge_dc":0,"brand":"ola",       "color":"#c2410c"},
    "Ola S1 X (3 kWh)":              {"battery":3.0,  "range":151,"type":"Scooter","price":1.10, "charge_ac":0.75,"charge_dc":0,"brand":"ola",       "color":"#ea580c"},
    "Ola S1 X+ (4 kWh)":             {"battery":4.0,  "range":190,"type":"Scooter","price":1.20, "charge_ac":0.75,"charge_dc":0,"brand":"ola",       "color":"#f97316"},
    "Ola S1 Pro (4 kWh)":            {"battery":4.0,  "range":195,"type":"Scooter","price":1.35, "charge_ac":0.75,"charge_dc":0,"brand":"ola",       "color":"#f97316"},
    "Ola S1 Pro Gen 2 (4 kWh)":      {"battery":4.0,  "range":181,"type":"Scooter","price":1.47, "charge_ac":0.90,"charge_dc":0,"brand":"ola",       "color":"#dc2626"},
    # ── ATHER ──
    "Ather Rizta S (2.9 kWh)":       {"battery":2.9,  "range":123,"type":"Scooter","price":1.10, "charge_ac":0.9, "charge_dc":0,"brand":"ather",     "color":"#0891b2"},
    "Ather Rizta Z (3.7 kWh)":       {"battery":3.7,  "range":160,"type":"Scooter","price":1.30, "charge_ac":0.9, "charge_dc":0,"brand":"ather",     "color":"#0284c7"},
    "Ather 450S (3.7 kWh)":          {"battery":3.7,  "range":135,"type":"Scooter","price":1.30, "charge_ac":0.9, "charge_dc":0,"brand":"ather",     "color":"#0369a1"},
    "Ather 450X Gen 3 (3.7 kWh)":    {"battery":3.7,  "range":150,"type":"Scooter","price":1.43, "charge_ac":0.9, "charge_dc":0,"brand":"ather",     "color":"#0369a1"},
    # ── TVS ──
    "TVS iQube S (3.04 kWh)":        {"battery":3.04, "range":145,"type":"Scooter","price":1.09, "charge_ac":0.95,"charge_dc":0,"brand":"tvs",       "color":"#dc2626"},
    "TVS iQube ST (5.1 kWh)":        {"battery":5.1,  "range":228,"type":"Scooter","price":1.38, "charge_ac":0.95,"charge_dc":0,"brand":"tvs",       "color":"#b91c1c"},
    "TVS iQube Electric (2.25 kWh)": {"battery":2.25, "range":100,"type":"Scooter","price":0.99, "charge_ac":0.90,"charge_dc":0,"brand":"tvs",       "color":"#ef4444"},
    # ── BAJAJ ──
    "Bajaj Chetak Premium (3.2 kWh)":{"battery":3.2,  "range":126,"type":"Scooter","price":1.15, "charge_ac":0.75,"charge_dc":0,"brand":"bajaj",     "color":"#16a34a"},
    "Bajaj Chetak Urbane (3.2 kWh)": {"battery":3.2,  "range":113,"type":"Scooter","price":0.99, "charge_ac":0.75,"charge_dc":0,"brand":"bajaj",     "color":"#15803d"},
    # ── HONDA ──
    "Honda Activa e: (3 kWh)":       {"battery":3.0,  "range":102,"type":"Scooter","price":1.17, "charge_ac":0.9, "charge_dc":0,"brand":"honda",     "color":"#7c3aed"},
    "Honda QC1 (1.5 kWh)":           {"battery":1.5,  "range":80, "type":"Scooter","price":0.90, "charge_ac":0.5, "charge_dc":0,"brand":"honda",     "color":"#6d28d9"},
    # ── SIMPLE ENERGY ──
    "Simple Energy One (4.8 kWh)":   {"battery":4.8,  "range":212,"type":"Scooter","price":1.45, "charge_ac":1.0, "charge_dc":0,"brand":"simple",    "color":"#0f766e"},
    # ── GREAVES / AMPERE ──
    "Ampere Magnus EX (3.9 kWh)":    {"battery":3.9,  "range":121,"type":"Scooter","price":1.10, "charge_ac":0.75,"charge_dc":0,"brand":"ampere",    "color":"#0891b2"},
    "Ampere Primus (3.5 kWh)":       {"battery":3.5,  "range":108,"type":"Scooter","price":0.95, "charge_ac":0.75,"charge_dc":0,"brand":"ampere",    "color":"#0284c7"},
    "Ampere Nexus (3.9 kWh)":        {"battery":3.9,  "range":136,"type":"Scooter","price":1.30, "charge_ac":0.90,"charge_dc":0,"brand":"ampere",    "color":"#0369a1"},
    # ── HERO ELECTRIC ──
    "Hero Electric Optima CX (3.2 kWh)":{"battery":3.2,"range":82,"type":"Scooter","price":0.72,"charge_ac":0.65,"charge_dc":0,"brand":"hero_elec", "color":"#16a34a"},
    "Hero Vida V1 Pro (3.94 kWh)":   {"battery":3.94, "range":165,"type":"Scooter","price":1.45, "charge_ac":0.95,"charge_dc":0,"brand":"hero_vida", "color":"#dc2626"},
    "Hero Vida V2 Lite (3.94 kWh)":  {"battery":3.94, "range":126,"type":"Scooter","price":1.10, "charge_ac":0.90,"charge_dc":0,"brand":"hero_vida", "color":"#b91c1c"},
    # ── SUZUKI ──
    "Suzuki Access Electric (3 kWh)":{"battery":3.0,  "range":130,"type":"Scooter","price":1.10, "charge_ac":0.80,"charge_dc":0,"brand":"suzuki",    "color":"#1d4ed8"},
    # ── BOUNCE ──
    "Bounce Infinity E1 (2 kWh)":    {"battery":2.0,  "range":85, "type":"Scooter","price":0.75, "charge_ac":0.75,"charge_dc":0,"brand":"bounce",    "color":"#7c3aed"},
    # ── PURE EV ──
    "Pure EV ePluto 7G (3.8 kWh)":   {"battery":3.8,  "range":171,"type":"Scooter","price":1.10, "charge_ac":0.75,"charge_dc":0,"brand":"pure_ev",  "color":"#0891b2"},
    "Pure EV ETrance Neo (4.5 kWh)": {"battery":4.5,  "range":165,"type":"Scooter","price":1.25, "charge_ac":0.90,"charge_dc":0,"brand":"pure_ev",  "color":"#0284c7"},
    # ── KABIRA ──
    "Kabira KM3000 (3.5 kWh)":       {"battery":3.5,  "range":120,"type":"Scooter","price":1.05, "charge_ac":0.80,"charge_dc":0,"brand":"kabira",   "color":"#0f766e"},
    # ── YULU ──
    "Yulu Wynn (0.6 kWh)":           {"battery":0.6,  "range":40, "type":"Scooter","price":0.45, "charge_ac":0.25,"charge_dc":0,"brand":"yulu",     "color":"#22c55e"},
    # -- Additional ScooterWale listed two-wheel EVs --
    "Okinawa PraisePro (2.08 kWh)":      {"battery":2.08, "range":88, "type":"Scooter","price":0.84, "charge_ac":0.75,"charge_dc":0,"brand":"okinawa",      "color":"#ef4444"},
    "Okinawa iPraise+ (3.3 kWh)":        {"battery":3.3,  "range":139,"type":"Scooter","price":1.46, "charge_ac":0.85,"charge_dc":0,"brand":"okinawa",      "color":"#dc2626"},
    "Okinawa Ridge+ (1.74 kWh)":         {"battery":1.74, "range":84, "type":"Scooter","price":0.70, "charge_ac":0.65,"charge_dc":0,"brand":"okinawa",      "color":"#f97316"},
    "Revolt RV400 (3.24 kWh)":           {"battery":3.24, "range":150,"type":"Scooter","price":1.35, "charge_ac":0.85,"charge_dc":0,"brand":"revolt",       "color":"#111827"},
    "Ultraviolette F77 (10.3 kWh)":      {"battery":10.3, "range":307,"type":"Scooter","price":3.80, "charge_ac":1.35,"charge_dc":0,"brand":"ultraviolette","color":"#6d28d9"},
    "Tork Kratos R (4 kWh)":             {"battery":4.0,  "range":180,"type":"Scooter","price":1.50, "charge_ac":1.0, "charge_dc":0,"brand":"tork",         "color":"#dc2626"},
    "Oben Rorr (4.4 kWh)":               {"battery":4.4,  "range":187,"type":"Scooter","price":1.50, "charge_ac":1.0, "charge_dc":0,"brand":"oben",         "color":"#f97316"},
    "River Indie (4 kWh)":               {"battery":4.0,  "range":161,"type":"Scooter","price":1.38, "charge_ac":0.9, "charge_dc":0,"brand":"river",        "color":"#0891b2"},
    "Komaki SE (3 kWh)":                 {"battery":3.0,  "range":140,"type":"Scooter","price":0.97, "charge_ac":0.8, "charge_dc":0,"brand":"komaki",       "color":"#7c3aed"},
    "Komaki TN95 (3.5 kWh)":             {"battery":3.5,  "range":180,"type":"Scooter","price":1.31, "charge_ac":0.8, "charge_dc":0,"brand":"komaki",       "color":"#6d28d9"},
    "Joy e-bike Mihos (2.96 kWh)":       {"battery":2.96, "range":130,"type":"Scooter","price":1.35, "charge_ac":0.75,"charge_dc":0,"brand":"joy",          "color":"#0f766e"},
    "Joy e-bike Wolf+ (2.2 kWh)":        {"battery":2.2,  "range":100,"type":"Scooter","price":0.95, "charge_ac":0.65,"charge_dc":0,"brand":"joy",          "color":"#14b8a6"},
    "Benling Aura (2.88 kWh)":           {"battery":2.88, "range":120,"type":"Scooter","price":1.22, "charge_ac":0.75,"charge_dc":0,"brand":"benling",      "color":"#2563eb"},
    "BGauss C12i EX (2 kWh)":            {"battery":2.0,  "range":85, "type":"Scooter","price":1.00, "charge_ac":0.65,"charge_dc":0,"brand":"bgauss",       "color":"#1d4ed8"},
    "BGauss RUV 350 (3 kWh)":            {"battery":3.0,  "range":145,"type":"Scooter","price":1.10, "charge_ac":0.75,"charge_dc":0,"brand":"bgauss",       "color":"#1e40af"},
    "Lectrix LXS G3.0 (3 kWh)":          {"battery":3.0,  "range":120,"type":"Scooter","price":1.00, "charge_ac":0.75,"charge_dc":0,"brand":"lectrix",      "color":"#16a34a"},
    "Lectrix SX25 (1.4 kWh)":            {"battery":1.4,  "range":60, "type":"Scooter","price":0.55, "charge_ac":0.5, "charge_dc":0,"brand":"lectrix",      "color":"#22c55e"},
    "Evolet Derby (1.8 kWh)":            {"battery":1.8,  "range":90, "type":"Scooter","price":0.75, "charge_ac":0.6, "charge_dc":0,"brand":"evolet",       "color":"#0284c7"},
    "Gemopai Astrid Lite (1.7 kWh)":     {"battery":1.7,  "range":90, "type":"Scooter","price":0.92, "charge_ac":0.6, "charge_dc":0,"brand":"gemopai",      "color":"#7c3aed"},
    "Lohia Oma Star Li (1.5 kWh)":       {"battery":1.5,  "range":70, "type":"Scooter","price":0.52, "charge_ac":0.5, "charge_dc":0,"brand":"hero_elec",    "color":"#65a30d"},
    "Tunwal Storm ZX (1.56 kWh)":        {"battery":1.56, "range":75, "type":"Scooter","price":0.90, "charge_ac":0.55,"charge_dc":0,"brand":"komaki",       "color":"#9333ea"},
    "Odysse Hawk Plus (2.96 kWh)":       {"battery":2.96, "range":170,"type":"Scooter","price":1.18, "charge_ac":0.75,"charge_dc":0,"brand":"joy",          "color":"#0d9488"},
    "Okaya Faast F4 (4.4 kWh)":          {"battery":4.4,  "range":160,"type":"Scooter","price":1.32, "charge_ac":0.9, "charge_dc":0,"brand":"okinawa",      "color":"#ea580c"},
}

CHARGING_MODELS = {
    "🏠 Home AC — Slow (₹7/kWh)":         {"rate":7.0,  "speed":"Slow",   "power_kw":3.3,  "desc":"Overnight home charging via 5A socket — cheapest option","color":"#16a34a","decay":0.0},
    "🏠 Home AC — Smart (₹8/kWh)":        {"rate":8.0,  "speed":"Slow",   "power_kw":7.2,  "desc":"7.4kW wall-box charger — fast & economical home charging","color":"#65a30d","decay":0.005},
    "🌆 Public AC — Network (₹12/kWh)":   {"rate":12.0, "speed":"Medium", "power_kw":7.2,  "desc":"Mall/office AC stations — Tata Power EZ, ChargeZone etc.","color":"#d97706","decay":0.01},
    "⚡ Public DC Fast (₹20/kWh)":         {"rate":20.0, "speed":"Fast",   "power_kw":50,   "desc":"30-min top-up at highway DC stations — moderate battery wear","color":"#ea580c","decay":0.025},
    "🚀 Public DC Ultra-Fast (₹24/kWh)":  {"rate":24.0, "speed":"Ultra",  "power_kw":150,  "desc":"Premium 150kW+ chargers — fastest but highest battery stress","color":"#dc2626","decay":0.04},
}

# ─────────────────────────────────────────────
# ML Model
# ─────────────────────────────────────────────
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
    df = pd.DataFrame({
        "capacity_kwh":capacity,"charge_cycles":cycles,"age_years":age,
        "avg_speed_kmh":avg_speed,"temperature_c":temperature,"daily_km":daily_km,
        "fast_charge":fast_charge,"charge_level":charge_level,"terrain":terrain,
        "hvac":hvac,"soh":soh,"range_km":range_km
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
    sm = RandomForestRegressor(100,random_state=42); sm.fit(X_train,ys_tr)
    rm = RandomForestRegressor(100,random_state=42); rm.fit(X_train,yr_tr)
    return sm,rm,r2_score(ys_te,sm.predict(X_test)),r2_score(yr_te,rm.predict(X_test))

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ EV Battery AI")
    st.markdown('<div style="color:#4a7fa5;font-size:0.75rem;margin-bottom:16px;">India 2026 · AI-Powered Predictor</div>', unsafe_allow_html=True)
    st.markdown("---")

    vehicle_type = st.radio("Vehicle Category", ["🚗 Car", "🛵 Scooter"], horizontal=True)
    is_scooter = "Scooter" in vehicle_type

    vehicle_db = EV_SCOOTERS if is_scooter else EV_CARS
    selected_vehicle = st.selectbox("Select Your EV Model", list(vehicle_db.keys()))
    v = vehicle_db[selected_vehicle]

    st.markdown("---")
    st.markdown("##### 🔌 Charging Mode")
    charging_model = st.selectbox("Your Primary Charger", list(CHARGING_MODELS.keys()))
    cm = CHARGING_MODELS[charging_model]
    st.markdown(f'<div class="param-def">💡 {cm["desc"]}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### 🚦 Usage Conditions")

    st.markdown('<div class="param-def">🔄 <b>Charge Cycles</b><br>Each full charge+discharge = 1 cycle. Higher cycles = more wear on the battery cells.</div>', unsafe_allow_html=True)
    cycles = st.slider("Charge Cycles Completed", 0, 1500, 200 if is_scooter else 300)

    st.markdown('<div class="param-def">📅 <b>Battery Age</b><br>How old your EV battery is. Older batteries naturally lose capacity even without heavy use.</div>', unsafe_allow_html=True)
    age = st.slider("Battery Age (years)", 0.0, 10.0, 1.5 if is_scooter else 3.0, 0.5)

    st.markdown('<div class="param-def">🏎️ <b>Average Speed</b><br>Your typical driving speed. Higher speeds = more energy draw = more heat = faster wear.</div>', unsafe_allow_html=True)
    avg_speed = st.slider("Avg Speed (km/h)", 20, 130, 40 if is_scooter else 75)

    st.markdown('<div class="param-def">📍 <b>Daily Distance</b><br>Total km you ride/drive each day. Decides how many charges you need per week.</div>', unsafe_allow_html=True)
    daily_km = st.slider("Daily Distance (km)", 5, 350, 30 if is_scooter else 80)

    st.markdown("---")
    st.markdown("##### 🌡️ Environment & Habits")

    st.markdown('<div class="param-def">🌡️ <b>Ambient Temperature</b><br>Outside temperature where you use the EV. Extreme heat (>35°C) or cold (<5°C) damages cells.</div>', unsafe_allow_html=True)
    temperature = st.slider("Temperature (°C)", -5, 48, 32)

    st.markdown('<div class="param-def">❄️ <b>AC / Heater Usage</b><br>How heavily you use AC (summer) or heater (winter). Heavy usage can drain 25–30% extra battery.</div>', unsafe_allow_html=True)
    hvac = st.slider("AC / Heater Usage (%)", 0, 100, 40 if is_scooter else 30) / 100

    st.markdown('<div class="param-def">⚡ <b>Fast Charge Usage</b><br>% of your charges done at DC fast chargers. Fast charging = heat inside cells = accelerated aging.</div>', unsafe_allow_html=True)
    fast_charge = st.slider("DC Fast Charge Usage (%)", 0, 100, 5 if is_scooter else 20) / 100

    st.markdown('<div class="param-def">🔋 <b>Charge Level</b><br>% you usually charge to. Daily 100% charges stress the battery. 80% is optimal for longevity.</div>', unsafe_allow_html=True)
    charge_level = st.slider("Typical Charge Level (%)", 50, 100, 80) / 100

    terrain_opt = st.selectbox("🏔️ Terrain Type", ["Flat (city roads)", "Mixed (suburbs)", "Hilly (ghats/mountains)"])
    terrain_val = ["Flat (city roads)", "Mixed (suburbs)", "Hilly (ghats/mountains)"].index(terrain_opt)

# ─────────────────────────────────────────────
# PREDICTIONS
# ─────────────────────────────────────────────
with st.spinner("⚡ Running AI model..."):
    soh_model, range_model, soh_r2, range_r2 = train_model()

features = ["capacity_kwh","charge_cycles","age_years","avg_speed_kmh",
            "temperature_c","daily_km","fast_charge","charge_level","terrain","hvac"]
input_df = pd.DataFrame([[v["battery"],cycles,age,avg_speed,temperature,
                          daily_km,fast_charge,charge_level,terrain_val,hvac]], columns=features)
pred_soh   = soh_model.predict(input_df)[0]
pred_range = range_model.predict(input_df)[0]

battery_kwh     = v["battery"]
charge_cost     = battery_kwh * cm["rate"]
full_charge_hrs = battery_kwh / cm["power_kw"]
cost_per_km     = charge_cost / max(pred_range, 1)
monthly_km      = daily_km * 30
monthly_charges = monthly_km / max(pred_range, 1)
monthly_cost    = monthly_charges * charge_cost
annual_cost     = monthly_cost * 12
years_left      = max(0, (pred_soh - 70) / 8)

petrol_km_per_l = 45 if is_scooter else 15
petrol_cost_l   = 105
petrol_monthly  = (monthly_km / petrol_km_per_l) * petrol_cost_l
monthly_savings = petrol_monthly - monthly_cost
annual_savings  = monthly_savings * 12

soh_color   = "#16a34a" if pred_soh > 75 else "#d97706" if pred_soh > 50 else "#dc2626"
soh_bg      = "#f0fdf4" if pred_soh > 75 else "#fffbeb" if pred_soh > 50 else "#fef2f2"
soh_label   = "Excellent 🟢" if pred_soh > 75 else "Moderate 🟡" if pred_soh > 50 else "Replace Soon 🔴"

# ─────────────────────────────────────────────
# BATTERY EXPIRY SIMULATION
# ─────────────────────────────────────────────
def simulate_expiry(base_soh, cycles_done, daily_km_val, battery_kwh_val, range_km, charge_mode_key):
    results = {}
    for mode_name, mode_data in CHARGING_MODELS.items():
        soh = base_soh
        cycles_per_year = (daily_km_val * 365) / max(range_km, 1)
        base_decay_per_cycle = 0.018 + mode_data["decay"]
        years_to_70 = None
        years_to_80 = None
        soh_trace = [soh]
        for y in range(1, 16):
            decay = cycles_per_year * base_decay_per_cycle
            soh = max(0, soh - decay)
            soh_trace.append(soh)
            if soh <= 80 and years_to_80 is None:
                years_to_80 = y
            if soh <= 70 and years_to_70 is None:
                years_to_70 = y
        results[mode_name] = {
            "years_to_70": years_to_70 or ">15",
            "years_to_80": years_to_80 or ">15",
            "soh_trace": soh_trace,
            "color": mode_data["color"],
            "speed": mode_data["speed"],
        }
    return results

expiry_data = simulate_expiry(pred_soh, cycles, daily_km, battery_kwh, pred_range, charging_model)

# ─────────────────────────────────────────────
# HERO BANNER
# ─────────────────────────────────────────────
st.markdown(f"""
<div class="hero-banner">
    <div class="hero-title">⚡ EV Battery AI Predictor</div>
    <div class="hero-sub">Real Indian EV models · AI-powered health, range & lifespan analysis · India 2026</div>
    <span class="hero-badge">🤖 Random Forest AI</span>
    <span class="hero-badge">🇮🇳 India Charging Rates</span>
    <span class="hero-badge">📊 SOH R² = {soh_r2:.3f}</span>
    <span class="hero-badge">🛣️ Range R² = {range_r2:.3f}</span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# VEHICLE CARD + METRICS
# ─────────────────────────────────────────────
vcol, mcol = st.columns([1, 2.8], gap="large")

with vcol:
    veh_color = v.get("color", "#1d4ed8")
    short_name = selected_vehicle.split("(")[0].strip()
    brand_upper = v.get("brand","").upper()
    st.markdown(f"""
    <div class="vehicle-card">
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
    </div>
    """, unsafe_allow_html=True)

with mcol:
    st.markdown('<div class="sec-head">📊 AI Health & Cost Predictions</div>', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m4, m5, m6 = st.columns(3)

    cur_exp = expiry_data[charging_model]
    yrs_raw = cur_exp["years_to_70"]
    yr_str  = f"{yrs_raw} yrs" if isinstance(yrs_raw, (int,float)) else yrs_raw

    with m1:
        st.markdown(f"""<div class="metric-card">
        <div class="metric-accent" style="background:{soh_color}"></div>
        <div class="metric-label">Battery Health (SOH)</div>
        <div class="metric-value" style="color:{soh_color}">{pred_soh:.1f}%</div>
        <div class="metric-sub">{soh_label}</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="metric-card">
        <div class="metric-accent" style="background:#0891b2"></div>
        <div class="metric-label">Real Range Today</div>
        <div class="metric-value" style="color:#0e7490">{pred_range:.0f} km</div>
        <div class="metric-sub">ARAI rated: {v['range']} km</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class="metric-card">
        <div class="metric-accent" style="background:#7c3aed"></div>
        <div class="metric-label">Cost per km</div>
        <div class="metric-value" style="color:#6d28d9">₹{cost_per_km:.2f}</div>
        <div class="metric-sub">Petrol: ₹{petrol_cost_l/petrol_km_per_l:.1f}/km</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""<div class="metric-card">
        <div class="metric-accent" style="background:#d97706"></div>
        <div class="metric-label">Monthly Fuel Cost</div>
        <div class="metric-value" style="color:#b45309">₹{monthly_cost:,.0f}</div>
        <div class="metric-sub">vs Petrol ₹{petrol_monthly:,.0f}/mo</div>
        </div>""", unsafe_allow_html=True)
    with m5:
        st.markdown(f"""<div class="metric-card">
        <div class="metric-accent" style="background:#16a34a"></div>
        <div class="metric-label">Annual EV Saving</div>
        <div class="metric-value" style="color:#15803d">₹{annual_savings:,.0f}</div>
        <div class="metric-sub">vs equivalent petrol vehicle</div>
        </div>""", unsafe_allow_html=True)
    with m6:
        lifespan_color = "#16a34a" if isinstance(yrs_raw, str) or (isinstance(yrs_raw,(int,float)) and yrs_raw >= 10) else "#d97706" if isinstance(yrs_raw,(int,float)) and yrs_raw >= 6 else "#dc2626"
        st.markdown(f"""<div class="metric-card">
        <div class="metric-accent" style="background:{lifespan_color}"></div>
        <div class="metric-label">Battery Life Remaining</div>
        <div class="metric-value" style="color:{lifespan_color}">{yr_str}</div>
        <div class="metric-sub">till 70% SOH (replace threshold)</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# BATTERY EXPIRY TABLE
# ─────────────────────────────────────────────
st.markdown('<div class="sec-head">🔋 Battery Lifespan by Charging Mode</div>', unsafe_allow_html=True)
st.markdown('<div style="color:#64748b;font-size:0.83rem;margin-bottom:16px;">How long will your battery last depending on which charger you use most? <b>80% SOH</b> = noticeable range loss. <b>70% SOH</b> = replacement recommended. Your current charger is highlighted in blue.</div>', unsafe_allow_html=True)

rows_html = ""
for mode_name, edata in expiry_data.items():
    color = edata["color"]
    y70 = edata["years_to_70"]
    y80 = edata["years_to_80"]
    is_current = (mode_name == charging_model)
    row_cls = ' class="current-row"' if is_current else ""
    cur_tag  = ' &nbsp;<b style="color:#1d4ed8;font-size:0.72rem">← Your Charger</b>' if is_current else ""
    # Color-code years to 70
    if isinstance(y70, str):
        y70_cls = "expiry-best"
    elif y70 >= 10:
        y70_cls = "expiry-best"
    elif y70 >= 6:
        y70_cls = "expiry-mid"
    else:
        y70_cls = "expiry-worst"
    y70_str = f"{y70} yrs" if isinstance(y70,(int,float)) else y70
    y80_str = f"{y80} yrs" if isinstance(y80,(int,float)) else y80
    spd = edata["speed"]
    spd_badge = f'<span class="mode-slow">{spd}</span>' if spd=="Slow" else f'<span class="mode-med">{spd}</span>' if spd=="Medium" else f'<span class="mode-fast">{spd}</span>' if spd=="Fast" else f'<span class="mode-ultra">{spd}</span>'
    rows_html += f"""<tr{row_cls}>
        <td>{mode_name}{cur_tag}</td>
        <td>{spd_badge}</td>
        <td>{y80_str}</td>
        <td class="{y70_cls}">{y70_str}</td>
    </tr>"""

st.markdown(f"""
<table class="expiry-table">
<thead><tr>
    <th>Charging Mode</th><th>Speed</th>
    <th>Drops to 80% SOH</th>
    <th>Drops to 70% SOH (Replacement)</th>
</tr></thead>
<tbody>{rows_html}</tbody>
</table>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Roadmap section removed per user request.

# GRAPHS — Charging Cost per km bar chart
# ─────────────────────────────────────────────
PLOT_BG   = "#ffffff"
PLOT_FG   = "#f8fafc"
AXIS_COL  = "#94a3b8"
GRID_COL  = "#e2e8f0"
TEXT_COL  = "#475569"
TITLE_COL = "#1a2e4a"

st.markdown('<div class="sec-head">⚡ Charging Cost per km Comparison</div>', unsafe_allow_html=True)
st.markdown('<div style="color:#64748b;font-size:0.8rem;margin-bottom:12px;">How much each type of charger costs you per km driven with <b>{}</b>. Your active charger has a bold border. The green bar = cheapest; red = most expensive.</div>'.format(selected_vehicle.split("(")[0].strip()), unsafe_allow_html=True)

fig2, ax2 = plt.subplots(figsize=(9, 3.5))
fig2.patch.set_facecolor(PLOT_BG)
ax2.set_facecolor(PLOT_FG)

mode_labels = []
costs_per_km = []
colors_bar = []
for mode_name, mode_data in CHARGING_MODELS.items():
    fc = battery_kwh * mode_data["rate"]
    cpkm = fc / max(pred_range, 1)
    short_label = (mode_name.split("(")[0].strip()
                   .replace("🏠 ","").replace("🌆 ","").replace("⚡ ","").replace("🚀 ",""))
    mode_labels.append(short_label)
    costs_per_km.append(cpkm)
    colors_bar.append(mode_data["color"])

bars = ax2.barh(mode_labels, costs_per_km, color=colors_bar, height=0.52,
                edgecolor=PLOT_BG, linewidth=1.5, zorder=3)

cur_idx = list(CHARGING_MODELS.keys()).index(charging_model)
bars[cur_idx].set_edgecolor("#1d4ed8")
bars[cur_idx].set_linewidth(3)

for bar, val, col in zip(bars, costs_per_km, colors_bar):
    ax2.text(val + max(costs_per_km)*0.015,
             bar.get_y() + bar.get_height()/2,
             f"₹{val:.2f}/km", va='center', color=col,
             fontsize=8.5, fontweight='700')

# Petrol reference
petrol_cpkm = petrol_cost_l / petrol_km_per_l
ax2.axvline(petrol_cpkm, color="#94a3b8", linestyle="--", linewidth=1.5, alpha=0.8, zorder=4)
ax2.text(petrol_cpkm + max(costs_per_km)*0.01, len(mode_labels)-0.1,
         f"Petrol ₹{petrol_cpkm:.1f}/km", color="#64748b", fontsize=7.5, va='top')

ax2.set_xlabel("Cost per km (₹)", color=AXIS_COL, fontsize=9)
ax2.tick_params(colors=TEXT_COL, labelsize=8.5)
for spine in ax2.spines.values():
    spine.set_edgecolor(GRID_COL)
ax2.grid(True, alpha=0.5, color=GRID_COL, axis='x', linewidth=0.8)
ax2.set_xlim(0, max(costs_per_km) * 1.3)
ax2.set_title("Your active charger is shown with a blue border", color=AXIS_COL, fontsize=8, pad=6)
plt.tight_layout(pad=1.4)
st.pyplot(fig2, use_container_width=True)
plt.close()

# ─────────────────────────────────────────────
# GRAPHS — Row 2
# ─────────────────────────────────────────────
g3, g4 = st.columns(2, gap="large")

# ── Graph 3: SOH Gauge (semi-circle) ──
with g3:
    st.markdown('<div class="sec-head">🔋 Battery Health Gauge</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#64748b;font-size:0.8rem;margin-bottom:12px;">A speedometer-style view of your current battery SOH. The needle position shows where your battery sits between fully healthy (100%) and the replacement zone (70%). Green = good, Yellow = watch it, Red = replace soon.</div>', unsafe_allow_html=True)

    fig3, ax3 = plt.subplots(figsize=(6.5, 4.0))
    fig3.patch.set_facecolor(PLOT_BG)
    ax3.set_facecolor(PLOT_BG)
    ax3.set_aspect('equal')
    ax3.axis('off')

    # Draw arc background zones
    # Red zone: 0–70
    red_arc = Wedge((0.5, 0.18), 0.38, 180, 180+70*1.8, width=0.12,
                    facecolor="#fef2f2", edgecolor="#fecaca", linewidth=1)
    ax3.add_patch(red_arc)
    # Yellow zone: 70–80
    yel_arc = Wedge((0.5, 0.18), 0.38, 180+70*1.8, 180+80*1.8, width=0.12,
                    facecolor="#fffbeb", edgecolor="#fde68a", linewidth=1)
    ax3.add_patch(yel_arc)
    # Green zone: 80–100
    grn_arc = Wedge((0.5, 0.18), 0.38, 180+80*1.8, 360, width=0.12,
                    facecolor="#f0fdf4", edgecolor="#bbf7d0", linewidth=1)
    ax3.add_patch(grn_arc)

    # Filled arc up to SOH
    fill_color = soh_color
    fill_end   = 180 + pred_soh * 1.8
    fill_arc   = Wedge((0.5, 0.18), 0.38, 180, fill_end, width=0.12,
                       facecolor=fill_color, edgecolor="none", alpha=0.85)
    ax3.add_patch(fill_arc)

    # Needle
    import math
    needle_angle_deg = 180 + pred_soh * 1.8
    needle_angle_rad = math.radians(needle_angle_deg)
    nx = 0.5 + 0.30 * math.cos(needle_angle_rad)
    ny = 0.18 + 0.30 * math.sin(needle_angle_rad)
    ax3.annotate("", xy=(nx, ny), xytext=(0.5, 0.18),
                 arrowprops=dict(arrowstyle="-|>", color=fill_color, lw=2.5,
                                 mutation_scale=14))

    # Center hub
    hub = plt.Circle((0.5, 0.18), 0.025, color="#1e293b", zorder=10)
    ax3.add_patch(hub)

    # Labels on arc
    for pct, label in [(0,"0%"), (25,"25%"), (50,"50%"), (70,"70% ⚠"), (80,"80%"), (100,"100%")]:
        a_rad = math.radians(180 + pct*1.8)
        lx = 0.5 + 0.42 * math.cos(a_rad)
        ly = 0.18 + 0.42 * math.sin(a_rad)
        ax3.text(lx, ly, label, ha='center', va='center', fontsize=6.5, color=TEXT_COL, fontweight='600')

    # Central SOH text
    ax3.text(0.5, -0.08, f"{pred_soh:.1f}%", ha='center', va='center',
             fontsize=24, fontweight='900', color=fill_color,
             transform=ax3.transAxes)
    ax3.text(0.5, -0.18, soh_label, ha='center', va='center',
             fontsize=9, color=TEXT_COL, transform=ax3.transAxes)

    ax3.set_xlim(0.04, 0.96)
    ax3.set_ylim(-0.25, 0.65)
    plt.tight_layout(pad=0.5)
    st.pyplot(fig3, use_container_width=True)
    plt.close()

# ── Graph 4: EV vs Petrol Annual Cost breakdown ──
with g4:
    st.markdown('<div class="sec-head">💰 EV vs Petrol — Annual Cost</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#64748b;font-size:0.8rem;margin-bottom:12px;">Annual fuel spend comparison between your EV (at your current charger rate) and a comparable petrol vehicle for the same distance. The difference is your annual saving by going electric.</div>', unsafe_allow_html=True)

    fig4, ax4 = plt.subplots(figsize=(6.5, 4.0))
    fig4.patch.set_facecolor(PLOT_BG)
    ax4.set_facecolor(PLOT_FG)

    cats   = ['EV\n(Your Charger)', 'Petrol\nEquivalent']
    vals   = [annual_cost, petrol_monthly * 12]
    bclrs  = ["#0891b2", "#f97316"]

    brs = ax4.bar(cats, vals, color=bclrs, width=0.4, zorder=3,
                  edgecolor=PLOT_BG, linewidth=2, capsize=4)

    for bar, val, col in zip(brs, vals, bclrs):
        ax4.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + max(vals)*0.02,
                 f"₹{val:,.0f}", ha='center', va='bottom',
                 fontsize=11, fontweight='900', color=col)

    # Savings arrow
    mid_x   = 0.5
    arrow_y = vals[0] + (vals[1] - vals[0])/2
    ax4.annotate("", xy=(1, vals[1]*0.98), xytext=(0, vals[0]*1.02),
                 arrowprops=dict(arrowstyle="<->", color="#16a34a", lw=2.5, connectionstyle="arc3,rad=0"))
    ax4.text(mid_x, arrow_y*1.05,
             f"Save ₹{annual_savings:,.0f}/yr ✅",
             ha='center', va='center', fontsize=9.5, fontweight='800',
             color="#16a34a", bbox=dict(boxstyle='round,pad=0.4', facecolor='#f0fdf4',
                                        edgecolor='#bbf7d0', linewidth=1.5))

    ax4.set_ylim(0, max(vals) * 1.35)
    ax4.yaxis.set_visible(False)
    ax4.tick_params(colors=TEXT_COL, labelsize=10)
    for spine in ax4.spines.values():
        spine.set_edgecolor(GRID_COL)
    ax4.grid(True, alpha=0.5, color=GRID_COL, axis='y', linewidth=0.8)
    ax4.set_title(f"Petrol assumed: ₹{petrol_cost_l}/L · {petrol_km_per_l} km/L", color=AXIS_COL, fontsize=8, pad=6)
    plt.tight_layout(pad=1.4)
    st.pyplot(fig4, use_container_width=True)
    plt.close()

st.markdown("<hr>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CHARGING COST TABLE
# ─────────────────────────────────────────────
st.markdown('<div class="sec-head">⚡ Full Charging Cost Breakdown — India 2026</div>', unsafe_allow_html=True)
rows = []
for name, c in CHARGING_MODELS.items():
    fc   = battery_kwh * c["rate"]
    cph  = battery_kwh / c["power_kw"]
    cpkm = fc / max(pred_range, 1)
    mc   = (monthly_km / max(pred_range, 1)) * fc
    rows.append({
        "Charging Type": name,
        "₹/kWh": f"₹{c['rate']:.0f}",
        "Full Charge Cost": f"₹{fc:.0f}",
        "Charge Time": f"{cph:.1f} hrs" if cph > 1 else f"{cph*60:.0f} min",
        "₹/km": f"₹{cpkm:.2f}",
        "Monthly Cost": f"₹{mc:,.0f}",
        "Selected": name == charging_model,
    })
df_cost = pd.DataFrame(rows)
table_html2 = '<table class="expiry-table"><thead><tr>'
for col in ["Charging Type","₹/kWh","Full Charge Cost","Charge Time","₹/km","Monthly Cost"]:
    table_html2 += f"<th>{col}</th>"
table_html2 += "</tr></thead><tbody>"
for _, row in df_cost.iterrows():
    row_cl = ' class="current-row"' if row["Selected"] else ""
    table_html2 += f"<tr{row_cl}>"
    for col in ["Charging Type","₹/kWh","Full Charge Cost","Charge Time","₹/km","Monthly Cost"]:
        bold = ' style="font-weight:800;color:#1d4ed8;"' if row["Selected"] and col != "Charging Type" else ""
        table_html2 += f"<td{bold}>{row[col]}</td>"
    table_html2 += "</tr>"
table_html2 += "</tbody></table>"
st.markdown(table_html2, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# BATTERY PROTECTION GUIDE — New Section
# ─────────────────────────────────────────────
st.markdown('<div class="sec-head">🛡️ How to Protect Your Battery from Degradation</div>', unsafe_allow_html=True)
st.markdown('<div style="color:#64748b;font-size:0.84rem;margin-bottom:18px;">Follow these proven practices to maximise battery lifespan and slow down degradation. Each tip below is AI-selected and ranked by impact for your vehicle and usage pattern.</div>', unsafe_allow_html=True)

protect_tips = [
    ("🔋", "Keep Charge Between 20% and 80%",
     "This is the single biggest thing you can do. Lithium-ion batteries age fastest at the extremes — charging to 100% daily creates cell stress, and letting it drain to 0% causes deep discharge damage. Set your app's charge limit to 80% for everyday use and only go to 100% before a long trip.",
     "#0369a1", "#eff6ff", "#bfdbfe"),
    ("❄️", "Never Charge in Extreme Heat",
     "Charging your battery when it's hot (>38°C ambient or after a long drive) multiplies degradation. The battery generates its own heat during charging — adding ambient heat on top shortens cell life. Wait 15–20 minutes after parking before plugging in, and park in shade or a covered garage.",
     "#0f766e", "#f0fdfa", "#99f6e4"),
    ("⚡", "Use DC Fast Charging Sparingly",
     "DC fast chargers push current at 3–5× the normal rate, generating significant internal heat. Cells that regularly experience fast-charge heat cycles degrade 2–3× faster. Reserve DC fast charging for highway emergencies. 70%+ of your charges should be slow home AC charging overnight.",
     "#7c3aed", "#faf5ff", "#ddd6fe"),
    ("🌙", "Charge Overnight, Not All Day",
     "Lithium cells sitting at 100% SOC for hours generates a constant low-level stress called 'calendar aging'. Overnight charging finishes just before you wake up. Use your EV app's scheduled departure feature to time this automatically — your phone can set it once and it runs itself.",
     "#0891b2", "#f0f9ff", "#bae6fd"),
    ("🐢", "Drive Smoothly — Avoid Harsh Acceleration",
     "Hard acceleration spikes current draw from the battery, creating heat spikes inside the cells. Indian highway driving with smooth acceleration and regenerative braking coasting can increase real-world range by 15–25% and significantly reduce thermal stress on the battery.",
     "#16a34a", "#f0fdf4", "#bbf7d0"),
    ("🌡️", "Pre-condition in Extreme Heat or Cold",
     "In Indian summers (>40°C), run your EV's pre-cooling function while still plugged into a charger — this cools the cabin AND the battery pack using grid power, not battery power. You'll get better range and less heat stress on cells the moment you unplug.",
     "#d97706", "#fffbeb", "#fde68a"),
    ("🔌", "Store at 50% if Not Using for Weeks",
     "Going on a long trip and leaving your EV parked? Don't leave it at 100% or 0%. Charge or discharge to ~50% SOC before parking. A battery sitting at full charge self-discharges while oxidising the anode — at 50% this process is slowest. Some EVs have a 'transport mode' for exactly this.",
     "#dc2626", "#fef2f2", "#fecaca"),
    ("📱", "Use Manufacturer App for Battery Health Reports",
     "Tata Motors, Ather, Ola Electric, and Mahindra all offer apps that show real-time SOH, charging history, and anomaly alerts. Review your battery health monthly and contact your service centre if SOH drops more than 3–4% per year — early intervention prevents irreversible damage.",
     "#475569", "#f8fafc", "#e2e8f0"),
]

cols_protect = st.columns(2, gap="large")
for i, (icon, title, body, tc, bgc, bc) in enumerate(protect_tips):
    with cols_protect[i % 2]:
        st.markdown(f"""
        <div style="background:{bgc};border:1.5px solid {bc};border-left:5px solid {tc};
        border-radius:16px;padding:18px 20px;margin-bottom:12px;">
            <div style="font-family:'Nunito',sans-serif;font-size:1rem;font-weight:900;color:{tc};margin-bottom:8px;">
                {icon} {title}
            </div>
            <div style="font-size:0.82rem;color:#475569;line-height:1.7;">{body}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# AI TIPS (personalised to your inputs)
# ─────────────────────────────────────────────
st.markdown('<div class="sec-head">💡 AI Recommendations — Personalised to Your Usage</div>', unsafe_allow_html=True)
tips = []
if fast_charge > 0.5:
    tips.append(("warn","⚡ <b>High DC fast-charge usage detected.</b> You're fast-charging {:.0f}% of the time. Limit fast charging to highway emergencies — frequent DC charging generates internal heat that degrades your battery 2–3× faster than AC charging.".format(fast_charge*100)))
if charge_level > 0.95:
    tips.append(("warn","🔋 <b>You're charging to 100% regularly.</b> Keep daily charge at 80–90% — this single habit can extend battery life by 20–30%. Use your EV app's charge-limit setting."))
if temperature > 38:
    tips.append(("warn",f"🌡️ <b>Ambient temp {temperature}°C is very high.</b> Park in shade or a covered garage. Heat above 35°C is the #1 enemy of lithium batteries in Indian summers — it accelerates both calendar and cycle aging."))
if avg_speed > 90 and is_scooter:
    tips.append(("warn","🚀 <b>Scooter speed above 90 km/h drains range fast.</b> Cruise at 55–70 km/h for best efficiency and lower thermal load on the battery."))
if avg_speed > 110 and not is_scooter:
    tips.append(("warn","🚀 <b>Highway speeds above 110 km/h cut range by up to 35%.</b> Eco mode at 90–100 km/h saves significantly and reduces cell stress."))
if hvac > 0.7:
    tips.append(("warn","❄️ <b>Heavy AC/heater usage detected ({:.0f}%).</b> This reduces effective range 25–30%. Pre-cool the cabin while plugged in before you start driving — grid power, not battery power.".format(hvac*100)))
if cm["speed"] in ["Fast","Ultra"] and fast_charge > 0.5:
    tips.append(("bad","🔴 <b>Using DC fast chargers as your primary charger</b> is the fastest way to age your battery. Switch to home AC charging for 70%+ of your charges to protect cell integrity."))
if charge_level < 0.7:
    tips.append(("warn","🔋 <b>Keeping charge below 70% habitually is too low.</b> 20–80% is the optimal daily range. Deep discharges below 20% cause lithium plating at the anode over time."))
if pred_soh < 75:
    tips.append(("bad",f"🔴 <b>Battery at {pred_soh:.1f}% SOH.</b> If within warranty period, visit your authorised service centre — most manufacturers cover battery replacement below 70–75% SOH within 5–8 years."))
if cm["speed"] == "Slow" and fast_charge < 0.2:
    tips.append(("good","✅ <b>Excellent charging habits!</b> AC-only / home charging is the gentlest on your battery. You're on the right path to maximising battery lifespan. Keep it up!"))
if not tips:
    tips.append(("good","✅ <b>Great habits!</b> All conditions look optimal. Keep charging at home, avoid extreme heat, maintain 20–80% charge range daily, and your battery will last well beyond 10 years."))

tip_cols = st.columns(2, gap="large")
for i, (kind, text) in enumerate(tips):
    css = "tip-good" if kind=="good" else "tip-bad" if kind=="bad" else "tip-warn"
    tip_cols[i % 2].markdown(f'<div class="{css}">{text}</div>', unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
b1, b2, b3 = st.columns(3, gap="large")
with b1:
    st.markdown(f"""<div class="footer-card">
    <div class="fc-title">🤖 Model Info</div>
    <div class="fc-row">
    Algorithm: Random Forest (2 models)<br>
    Training samples: 3,000 battery records<br>
    SOH accuracy: R² = {soh_r2:.3f}<br>
    Range accuracy: R² = {range_r2:.3f}
    </div></div>""", unsafe_allow_html=True)
with b2:
    st.markdown(f"""<div class="footer-card">
    <div class="fc-title">📊 Your Session Summary</div>
    <div class="fc-row">
    Vehicle: {selected_vehicle.split("(")[0].strip()}<br>
    Battery: {v['battery']} kWh · SOH: {pred_soh:.1f}%<br>
    Real range: {pred_range:.0f} km · Daily: {daily_km} km<br>
    Annual EV cost: ₹{annual_cost:,.0f}<br>
    Annual saving vs petrol: ₹{annual_savings:,.0f}
    </div></div>""", unsafe_allow_html=True)
with b3:
    st.markdown(f"""<div class="footer-card">
    <div class="fc-title">📡 Data Sources</div>
    <div class="fc-row">
    EV specs: CarWale & eScooterWale India 2026<br>
    Charging rates: Pulse Energy India 2026<br>
    Home AC: ₹6–10/kWh avg<br>
    Public DC Fast: ₹18–25/kWh avg<br>
    Petrol: ₹105/litre avg India 2026
    </div></div>""", unsafe_allow_html=True)

st.markdown('<div style="text-align:center;color:#94a3b8;font-size:0.72rem;margin-top:24px;padding-bottom:12px;">⚡ EV Battery AI India 2026 · Predictions are AI estimates based on typical usage patterns · Not a substitute for professional battery diagnosis</div>', unsafe_allow_html=True)
