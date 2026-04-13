import io

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sympy import symbols, integrate, lambdify, Piecewise, And
from sympy import solve as sym_solve, symbols as sym_symbols
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fpdf import FPDF

_VERSION = "1.0.0"
_N_POINTS = 1000  # number of evaluation points along the beam
_PDF_DPI = 150            # raster resolution for charts embedded in the PDF report
_COUNTER_DURATION_MS = 1200  # duration of the count-up animation in milliseconds
_FONT_FAMILY = '-apple-system, BlinkMacSystemFont, Inter, sans-serif'

# ── Shared color palette ──
_CLR_BLUE   = '#0071e3'
_CLR_RED    = '#ff3b30'
_CLR_GREEN  = '#34c759'
_CLR_ORANGE = '#ff9500'
_CLR_PURPLE = '#af52de'
_CLR_TEAL   = '#5ac8fa'
_CLR_GRAY   = '#6e6e73'
_CLR_LGRAY  = '#d1d1d6'
_CLR_TEXT   = '#1d1d1f'
_CLR_GRID   = '#f0f0f0'
_CLR_BEAM_FILL = '#e8f0fc'
_CLR_WALL_FILL = '#e8e8ed'

# ─────────────────────────────────────────────
#  APPLE CSS
# ─────────────────────────────────────────────

APPLE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #f5f5f7 !important;
    font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Inter', 'Segoe UI', sans-serif !important;
    color: #1d1d1f !important;
}
[data-testid="stAppViewContainer"] > .main { background: #f5f5f7 !important; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
.stDeployButton { display: none; }

[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.85) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border-right: 1px solid rgba(0,0,0,0.08) !important;
    box-shadow: 2px 0 40px rgba(0,0,0,0.06) !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 2rem; }

[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stNumberInput label {
    font-size: 0.78rem !important; font-weight: 600 !important;
    letter-spacing: 0.04em !important; text-transform: uppercase !important;
    color: #6e6e73 !important; margin-bottom: 4px !important;
}

[data-testid="stSidebar"] input,
[data-testid="stSidebar"] select,
[data-testid="stSidebar"] .stSelectbox > div > div {
    border-radius: 10px !important; border: 1.5px solid rgba(0,0,0,0.1) !important;
    background: #ffffff !important; font-size: 0.95rem !important;
    color: #1d1d1f !important; transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
[data-testid="stSidebar"] input:focus {
    border-color: #0071e3 !important;
    box-shadow: 0 0 0 3px rgba(0,113,227,0.15) !important; outline: none !important;
}

[data-testid="stSidebar"] [data-testid="stNumberInput"] *,
[data-testid="stSidebar"] .stNumberInput * {
    border-color: transparent !important; box-shadow: none !important; outline: none !important;
}
[data-testid="stSidebar"] [data-testid="stNumberInput"] input,
[data-testid="stSidebar"] .stNumberInput input {
    border: 1.5px solid rgba(0,0,0,0.1) !important;
    border-radius: 10px !important; background: #ffffff !important; color: #1d1d1f !important;
}
[data-testid="stSidebar"] [data-testid="stNumberInput"] input:focus,
[data-testid="stSidebar"] .stNumberInput input:focus {
    border-color: #0071e3 !important; box-shadow: 0 0 0 3px rgba(0,113,227,0.15) !important;
}
[data-testid="stSidebar"] [data-testid="stNumberInput"] button,
[data-testid="stSidebar"] .stNumberInput button {
    background: #f5f5f7 !important; border: 1px solid rgba(0,0,0,0.08) !important;
    color: #1d1d1f !important; border-radius: 8px !important;
    box-shadow: none !important; transition: background 0.18s ease !important;
}
[data-testid="stSidebar"] [data-testid="stNumberInput"] button:hover,
[data-testid="stSidebar"] .stNumberInput button:hover { background: #e8e8ed !important; }
[data-testid="stSidebar"] [data-testid="stNumberInput"],
[data-testid="stSidebar"] [data-testid="stNumberInput"] > div,
[data-testid="stSidebar"] [data-testid="stNumberInput"] > div > div,
[data-testid="stSidebar"] [data-testid="stNumberInput"] > div > div > div,
[data-testid="stSidebar"] .stNumberInput,
[data-testid="stSidebar"] .stNumberInput > div,
[data-testid="stSidebar"] .stNumberInput > div > div,
[data-testid="stSidebar"] .stNumberInput > div > div > div {
    background: transparent !important; border: none !important;
    box-shadow: none !important; outline: none !important;
}

.stButton > button {
    border-radius: 980px !important; font-weight: 500 !important; font-size: 0.9rem !important;
    padding: 0.55rem 1.4rem !important;
    transition: all 0.25s cubic-bezier(0.25, 0.46, 0.45, 0.94) !important;
    border: none !important; cursor: pointer !important; letter-spacing: -0.01em !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"],
.stButton > button[kind="primary"] {
    background: linear-gradient(180deg, #147ce5 0%, #0071e3 100%) !important;
    color: #ffffff !important; box-shadow: 0 2px 8px rgba(0,113,227,0.35) !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover,
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(180deg, #1a88f5 0%, #0077ed 100%) !important;
    box-shadow: 0 4px 16px rgba(0,113,227,0.45) !important; transform: translateY(-1px) !important;
}
[data-testid="stSidebar"] .stButton > button:not([kind="primary"]) {
    background: rgba(0,0,0,0.05) !important; color: #1d1d1f !important;
    border: 1.5px solid rgba(0,0,0,0.1) !important;
}
[data-testid="stSidebar"] .stButton > button:not([kind="primary"]):hover {
    background: rgba(0,113,227,0.08) !important; border-color: #0071e3 !important;
    color: #0071e3 !important; transform: translateY(-1px) !important;
}

hr { border: none !important; border-top: 1px solid rgba(0,0,0,0.08) !important; margin: 1.2rem 0 !important; }

[data-testid="stExpander"] {
    background: #ffffff !important; border-radius: 14px !important;
    border: 1px solid rgba(0,0,0,0.06) !important; box-shadow: 0 2px 16px rgba(0,0,0,0.05) !important;
    overflow: hidden !important; margin-bottom: 0.8rem !important; transition: box-shadow 0.2s ease !important;
}
[data-testid="stExpander"]:hover { box-shadow: 0 4px 24px rgba(0,0,0,0.08) !important; }
[data-testid="stExpander"] summary {
    font-weight: 600 !important; font-size: 0.95rem !important;
    color: #1d1d1f !important; padding: 1rem 1.2rem !important;
}
/* ── Nuclear expander fix — force white on every child ── */
:root [data-testid="stExpander"],
:root [data-testid="stExpander"] *:not(svg):not(path):not(circle):not(th):not(thead) {
    background-color: #ffffff !important;
    color: #1d1d1f !important;
}
:root [data-testid="stExpander"] th,
:root [data-testid="stExpander"] thead,
:root [data-testid="stExpander"] thead th {
    background-color: #f5f5f7 !important;
    color: #6e6e73 !important;
}
:root [data-testid="stExpander"] summary {
    background-color: #ffffff !important;
    color: #1d1d1f !important;
    font-weight: 600 !important;
}

[data-testid="stAlert"] { border-radius: 12px !important; border: none !important; animation: fadeIn 0.4s ease both !important; }

[data-testid="stTable"] table { border-radius: 12px !important; overflow: hidden !important; border: 1px solid rgba(0,0,0,0.06) !important; font-size: 0.9rem !important; }
[data-testid="stTable"] th,
[data-testid="stTable"] thead th,
[data-testid="stTable"] thead tr th {
    background: #f5f5f7 !important;
    color: #6e6e73 !important;
    font-weight: 600 !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
    padding: 0.8rem 1rem !important;
}
[data-testid="stTable"] td,
[data-testid="stTable"] tbody td,
[data-testid="stTable"] tbody tr td,
[data-testid="stTable"] tbody th {
    color: #1d1d1f !important;
    background: #ffffff !important;
    padding: 0.75rem 1rem !important;
    border-top: 1px solid rgba(0,0,0,0.04) !important;
}
[data-testid="stTable"] tbody tr:hover td {
    background: #f9f9fb !important;
}

[data-testid="stSidebar"] [data-testid="stTooltipIcon"] svg *,
[data-testid="stSidebar"] span[data-testid="tooltipHoverTarget"] svg * { fill: #ffffff !important; stroke: none !important; }
[data-testid="stSidebar"] [data-testid="stTooltipIcon"] svg circle,
[data-testid="stSidebar"] span[data-testid="tooltipHoverTarget"] svg circle { fill: #ffffff !important; stroke: #c7c7cc !important; stroke-width: 1 !important; }

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(0,0,0,0.15); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0,0,0,0.25); }

@keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
@keyframes slideUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
@keyframes heroShimmer { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
@keyframes countUp { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }

.hero-section { text-align: center !important; padding: 2.5rem 2rem 1.2rem; animation: fadeIn 0.8s ease both; width: 100%; display: flex; flex-direction: column; align-items: center; }
.hero-badge { display: inline-flex; align-items: center; gap: 6px; background: rgba(0,113,227,0.08); border: 1px solid rgba(0,113,227,0.2); color: #0071e3; font-size: 0.78rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; padding: 0.35rem 0.9rem; border-radius: 980px; margin-bottom: 1.2rem; animation: fadeIn 0.6s ease 0.1s both; }
.hero-title { font-size: clamp(2.4rem, 5vw, 3.6rem); font-weight: 800; letter-spacing: -0.04em; line-height: 1.05; color: #1d1d1f; margin: 0 auto 0.8rem auto; text-align: center !important; width: 100%; display: block; animation: fadeIn 0.7s ease 0.15s both; }
.hero-title span { background: linear-gradient(135deg, #0071e3 0%, #5ac8fa 50%, #34aadc 100%); background-size: 200% 200%; -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; animation: heroShimmer 4s ease infinite; }
.hero-subtitle { font-size: 1.15rem; font-weight: 400; color: #6e6e73; letter-spacing: -0.01em; margin-bottom: 2rem; animation: fadeIn 0.7s ease 0.2s both; }
.hero-pills { display: flex; justify-content: center; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 2rem; animation: fadeIn 0.7s ease 0.25s both; }
.hero-pill { background: #ffffff; border: 1px solid rgba(0,0,0,0.08); border-radius: 980px; font-size: 0.82rem; font-weight: 500; color: #1d1d1f; padding: 0.3rem 0.8rem; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }

.section-heading { font-size: 1.5rem; font-weight: 700; letter-spacing: -0.03em; color: #1d1d1f; margin-bottom: 1rem; margin-top: 1.5rem; animation: fadeIn 0.5s ease both; }
.section-subheading { font-size: 0.85rem; font-weight: 500; letter-spacing: 0.05em; text-transform: uppercase; color: #6e6e73; margin-bottom: 0.4rem; }

.glass-card { background: rgba(255,255,255,0.9); backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px); border-radius: 20px; border: 1px solid rgba(255,255,255,0.8); box-shadow: 0 4px 30px rgba(0,0,0,0.08); padding: 1.8rem 2rem; margin-bottom: 1.2rem; animation: slideUp 0.5s ease both; transition: transform 0.3s ease, box-shadow 0.3s ease; }
.glass-card:hover { transform: translateY(-2px); box-shadow: 0 8px 40px rgba(0,0,0,0.12); }

/* ── Animated Result Cards ── */
.results-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 1rem; margin: 1.2rem 0; }
.result-card { background: #ffffff; border-radius: 18px; padding: 1.4rem 1.2rem; border: 1px solid rgba(0,0,0,0.06); box-shadow: 0 2px 20px rgba(0,0,0,0.06); transition: transform 0.25s ease, box-shadow 0.25s ease; text-align: center; animation: slideUp 0.6s ease both; }
.result-card:nth-child(1) { animation-delay: 0.05s; }
.result-card:nth-child(2) { animation-delay: 0.1s; }
.result-card:nth-child(3) { animation-delay: 0.15s; }
.result-card:nth-child(4) { animation-delay: 0.2s; }
.result-card:nth-child(5) { animation-delay: 0.25s; }
.result-card:hover { transform: translateY(-4px); box-shadow: 0 12px 36px rgba(0,0,0,0.12); }
.result-card .rc-label { font-size: 0.7rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: #6e6e73; margin-bottom: 0.5rem; }
.result-card .rc-value { font-size: 1.55rem; font-weight: 800; letter-spacing: -0.03em; color: #1d1d1f; animation: countUp 0.5s ease both; }
.result-card .rc-unit { font-size: 0.78rem; font-weight: 500; color: #6e6e73; margin-top: 0.2rem; }
.result-card.blue  { border-top: 3px solid #0071e3; }
.result-card.green { border-top: 3px solid #34c759; }
.result-card.orange{ border-top: 3px solid #ff9500; }
.result-card.purple{ border-top: 3px solid #af52de; }
.result-card.teal  { border-top: 3px solid #5ac8fa; }

.load-tag { display: inline-flex; align-items: center; gap: 6px; background: rgba(0,113,227,0.07); border: 1px solid rgba(0,113,227,0.15); border-radius: 8px; padding: 0.4rem 0.7rem; font-size: 0.82rem; font-weight: 500; color: #0071e3; margin: 3px 0; width: 100%; justify-content: space-between; }

/* ── Download button ── */
[data-testid="stDownloadButton"] > button {
    background: #f5f5f7 !important;
    color: #1d1d1f !important;
    border: 1.5px solid rgba(0,0,0,0.1) !important;
    border-radius: 980px !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    padding: 0.55rem 1.4rem !important;
    box-shadow: none !important;
    transition: all 0.2s ease !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: #e8e8ed !important;
    border-color: rgba(0,0,0,0.18) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08) !important;
}

/* ── Remove buttons for loads ── */
[data-testid="stSidebar"] button[kind="secondary"] {
    padding: 0.25rem 0.6rem !important;
    font-size: 0.75rem !important;
    height: 28px !important;
    min-height: 28px !important;
    border-radius: 8px !important;
    background: rgba(255,59,48,0.07) !important;
    border: 1px solid rgba(255,59,48,0.18) !important;
    color: #ff3b30 !important;
    margin-top: 0 !important;
}
[data-testid="stSidebar"] button[kind="secondary"]:hover {
    background: rgba(255,59,48,0.14) !important;
    border-color: #ff3b30 !important;
    transform: none !important;
}
.sidebar-section-title { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: #6e6e73; margin: 0.8rem 0 0.5rem; padding-top: 0.4rem; }

.guide-step { display: flex; align-items: flex-start; gap: 1rem; padding: 1rem 0; border-bottom: 1px solid rgba(0,0,0,0.05); animation: fadeIn 0.5s ease both; }
.guide-step:last-child { border-bottom: none; }
.step-num { width: 28px; height: 28px; border-radius: 50%; background: linear-gradient(135deg, #0071e3, #5ac8fa); color: white; font-size: 0.78rem; font-weight: 700; display: flex; align-items: center; justify-content: center; flex-shrink: 0; margin-top: 1px; }
.step-body strong { font-size: 0.92rem; font-weight: 600; color: #1d1d1f; display: block; margin-bottom: 2px; }
.step-body span { font-size: 0.83rem; color: #6e6e73; }

.reaction-chip { display: inline-flex; align-items: center; gap: 8px; background: #f5f5f7; border: 1px solid rgba(0,0,0,0.06); border-radius: 12px; padding: 0.8rem 1.2rem; font-size: 0.9rem; font-weight: 500; color: #1d1d1f; width: 100%; animation: fadeIn 0.4s ease both; }
.reaction-chip .r-label { font-weight: 700; color: #0071e3; }

.plot-container { background: #ffffff; border-radius: 20px; border: 1px solid rgba(0,0,0,0.06); box-shadow: 0 4px 30px rgba(0,0,0,0.08); padding: 1rem; animation: slideUp 0.6s ease both; overflow: hidden; }

/* Style the Plotly chart block directly */
[data-testid="stPlotlyChart"] {
    background: #ffffff !important;
    border-radius: 20px !important;
    border: 1px solid rgba(0,0,0,0.06) !important;
    box-shadow: 0 4px 30px rgba(0,0,0,0.08) !important;
    padding: 0.5rem !important;
    animation: slideUp 0.6s ease both !important;
    overflow: hidden !important;
}

.beam-viz-title { font-size: 0.75rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; color: #6e6e73; margin-bottom: 0.4rem; margin-top: 1rem; }

[data-testid="stPyplotRootElement"] {
    background: #ffffff !important;
    border-radius: 20px !important;
    border: 1px solid rgba(0,0,0,0.06) !important;
    box-shadow: 0 4px 30px rgba(0,0,0,0.08) !important;
    padding: 0.5rem !important;
    animation: slideUp 0.4s ease both !important;
    overflow: hidden !important;
}

.download-btn-wrap { display: flex; justify-content: flex-end; margin-bottom: 0.8rem; }

.main .block-container { padding: 0 2.5rem 3rem !important; max-width: 1100px !important; }
</style>
"""

# ── Counter animation JS ──
def counter_js(values_dict):
    """Inject JS to animate .rc-value counters from 0 to target."""
    scripts = ""
    for el_id, (target, decimals) in values_dict.items():
        scripts += f"""
        (function() {{
            var el = document.getElementById('{el_id}');
            if (!el) return;
            var target = {target};
            var decimals = {decimals};
            var duration = {_COUNTER_DURATION_MS};
            var start = null;
            function step(ts) {{
                if (!start) start = ts;
                var progress = Math.min((ts - start) / duration, 1);
                var ease = 1 - Math.pow(1 - progress, 3);
                el.textContent = (ease * target).toFixed(decimals);
                if (progress < 1) requestAnimationFrame(step);
            }}
            requestAnimationFrame(step);
        }})();
        """
    return f"<script>{scripts}</script>"


# ─────────────────────────────────────────────
#  VALIDATION
# ─────────────────────────────────────────────

def validate_inputs(beam_length: float, E_GPa: float, I_cm4: float,
                     point_loads: list, udl_loads: list) -> list[str]:
    """Return a list of error strings for any invalid beam configuration inputs."""
    errors = []
    if beam_length <= 0:   errors.append("Beam length must be greater than 0.")
    if E_GPa <= 0:         errors.append("Young's Modulus (E) must be greater than 0.")
    if I_cm4 <= 0:         errors.append("Moment of Inertia (I) must be greater than 0.")
    for i, (pos, mag) in enumerate(point_loads):
        if not (0 <= pos <= beam_length):
            errors.append(f"Point load {i+1}: position {pos} m is outside beam length.")
    for i, (intensity, start, end) in enumerate(udl_loads):
        if start >= end:
            errors.append(f"UDL {i+1}: start must be less than end.")
        if not (0 <= start <= beam_length) or not (0 <= end <= beam_length):
            errors.append(f"UDL {i+1}: range [{start}, {end}] m is outside beam length.")
    if not point_loads and not udl_loads:
        errors.append("Please add at least one load.")
    return errors


# ─────────────────────────────────────────────
#  REACTIONS
# ─────────────────────────────────────────────

def compute_reactions(beam_length: float, support_type: str,
                      point_loads: list, udl_loads: list) -> tuple[float, float]:
    """Compute vertical support reactions using static equilibrium."""
    total_force, total_moment = 0.0, 0.0
    for pos, mag in point_loads:
        total_force += mag;  total_moment += mag * pos
    for intensity, start, end in udl_loads:
        F = intensity * (end - start);  total_force += F
        total_moment += F * (start + end) / 2.0
    if support_type == "Simply Supported":
        R_B = total_moment / beam_length if beam_length else 0
        R_A = total_force - R_B
    else:
        R_A, R_B = total_force, 0.0
    return R_A, R_B


# ─────────────────────────────────────────────
#  SFD / BMD
# ─────────────────────────────────────────────

def compute_sfd(beam_length: float, support_type: str, point_loads: list,
                udl_loads: list, R_A: float, R_B: float,
                n: int = _N_POINTS) -> tuple[np.ndarray, np.ndarray]:
    """Return (x, shear) arrays representing the shear force diagram."""
    x = np.linspace(0, beam_length, n)
    shear = np.full(n, R_A)
    for pos, mag in point_loads:
        shear[x >= pos] -= mag
    for intensity, start, end in udl_loads:
        within = (x >= start) & (x <= end)
        past   = x > end
        shear[within] -= intensity * (x[within] - start)
        shear[past]   -= intensity * (end - start)
    if support_type == "Simply Supported":
        shear[x >= beam_length] += R_B
    return x, shear


def compute_bmd(x: np.ndarray, shear: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (x, moment) arrays by numerically integrating the shear force."""
    dx = x[1] - x[0]
    moment = np.cumsum(shear) * dx
    moment[0] = 0.0
    return x, moment


# ─────────────────────────────────────────────
#  DEFLECTION (SymPy)
# ─────────────────────────────────────────────

def compute_deflection(beam_length: float, support_type: str, point_loads: list,
                       udl_loads: list, R_A: float, E_Pa: float, I_m4: float,
                       n: int = _N_POINTS) -> tuple[np.ndarray, np.ndarray]:
    """Return (x, deflection) arrays using SymPy double-integration with boundary conditions."""
    xi = symbols('xi')
    EI = E_Pa * I_m4
    M_expr = R_A * xi
    for pos, mag in point_loads:
        M_expr -= mag * Piecewise((xi - pos, xi >= pos), (0, True))
    for intensity, start, end in udl_loads:
        M_expr -= Piecewise(
            (intensity * (xi - start)**2 / 2, And(xi >= start, xi <= end)),
            (intensity * (end - start) * (xi - (start + end) / 2), xi > end),
            (0, True))
    slope_expr   = integrate(M_expr, xi) / EI
    deflect_expr = integrate(slope_expr, xi)
    C1, C2 = sym_symbols('C1 C2')
    slope_c   = slope_expr + C1
    deflect_c = deflect_expr + C1 * xi + C2
    if support_type == "Simply Supported":
        C2_val = 0
        C1_val = sym_solve(deflect_c.subs(C2, 0).subs(xi, beam_length), C1)[0]
    else:
        C2_val = 0
        C1_val = sym_solve(slope_c.subs(xi, 0), C1)[0]
    deflect_final = deflect_c.subs(C1, C1_val).subs(C2, C2_val)
    f = lambdify(xi, deflect_final, modules='numpy')
    x_vals = np.linspace(0, beam_length, n)
    try:
        y_vals = np.array(f(x_vals), dtype=float)
    except Exception:
        y_vals = np.array([float(deflect_final.subs(xi, xv)) for xv in x_vals])
    return x_vals, y_vals


# ─────────────────────────────────────────────
#  BEAM VISUALIZER  (Matplotlib schematic)
# ─────────────────────────────────────────────

def draw_beam_visualizer(beam_length, support_type, point_loads, udl_loads):
    """Render a Matplotlib schematic of the beam with supports and applied loads."""
    fig, ax = plt.subplots(figsize=(10, 3.2))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    ax.set_xlim(-0.5, beam_length + 0.5)
    ax.set_ylim(-1.8, 2.2)
    ax.axis('off')

    BEAM_Y    = 0.0
    BEAM_H    = 0.22
    BLUE      = _CLR_BLUE
    RED       = _CLR_RED
    ORANGE    = _CLR_ORANGE
    GREEN     = _CLR_GREEN
    GRAY      = _CLR_GRAY
    LGRAY     = _CLR_LGRAY

    # ── Beam body ──
    beam_rect = mpatches.FancyBboxPatch(
        (0, BEAM_Y - BEAM_H / 2), beam_length, BEAM_H,
        boxstyle="round,pad=0.02", linewidth=1.5,
        edgecolor=BLUE, facecolor=_CLR_BEAM_FILL)
    ax.add_patch(beam_rect)

    # ── Supports ──
    if support_type == "Simply Supported":
        for xp, col in [(0, BLUE), (beam_length, BLUE)]:
            tri = plt.Polygon(
                [[xp - 0.25, -BEAM_H/2 - 0.55],
                 [xp + 0.25, -BEAM_H/2 - 0.55],
                 [xp,         -BEAM_H/2]],
                closed=True, facecolor=col, edgecolor=col, alpha=0.85, zorder=4)
            ax.add_patch(tri)
            ax.plot([xp - 0.35, xp + 0.35],
                    [-BEAM_H/2 - 0.57, -BEAM_H/2 - 0.57],
                    color=col, linewidth=2.5, zorder=5)
        ax.text(0,           -1.05, 'A', ha='center', va='top', fontsize=8, fontweight='700', color=BLUE)
        ax.text(beam_length, -1.05, 'B', ha='center', va='top', fontsize=8, fontweight='700', color=BLUE)
    else:
        wall = mpatches.FancyBboxPatch(
            (-0.45, -0.9), 0.42, 1.8,
            boxstyle="round,pad=0.02",
            facecolor=_CLR_WALL_FILL, edgecolor=GRAY, linewidth=1.2, zorder=3)
        ax.add_patch(wall)
        for yy in np.arange(-0.75, 0.9, 0.22):
            ax.plot([-0.45, -0.03], [yy, yy + 0.18], color=LGRAY, linewidth=0.9, zorder=2)
        ax.text(-0.22, -1.05, 'Fixed', ha='center', va='top', fontsize=7.5,
                fontweight='600', color=GRAY)
        ax.text(beam_length + 0.1, 0, 'Free end', ha='left', va='center',
                fontsize=7.5, fontweight='500', color=GRAY, alpha=0.7)

    # ── Point loads ──
    arrow_h = 0.9
    for pos, mag in point_loads:
        ax.annotate('', xy=(pos, BEAM_H / 2),
                    xytext=(pos, BEAM_H / 2 + arrow_h),
                    arrowprops=dict(arrowstyle='->', color=RED,
                                   lw=2.2, mutation_scale=14))
        ax.text(pos, BEAM_H / 2 + arrow_h + 0.12,
                f'{mag} kN', ha='center', va='bottom',
                fontsize=8, fontweight='700', color=RED)

    # ── UDL ──
    udl_arrow_h = 0.55
    for intensity, start, end in udl_loads:
        xs = np.linspace(start, end, max(int((end - start) / 0.3) + 2, 4))
        top_y = BEAM_H / 2 + udl_arrow_h
        ax.plot([start, end], [top_y + 0.06, top_y + 0.06],
                color=ORANGE, linewidth=2, zorder=5)
        for xp in xs:
            ax.annotate('', xy=(xp, BEAM_H / 2),
                        xytext=(xp, top_y),
                        arrowprops=dict(arrowstyle='->', color=ORANGE,
                                        lw=1.5, mutation_scale=10))
        cx = (start + end) / 2
        ax.text(cx, top_y + 0.24,
                f'{intensity} kN/m', ha='center', va='bottom',
                fontsize=8, fontweight='700', color=ORANGE)

    # ── Dimension line ──
    dim_y = -1.3
    ax.annotate('', xy=(beam_length, dim_y), xytext=(0, dim_y),
                arrowprops=dict(arrowstyle='<->', color=GRAY, lw=1.2))
    ax.text(beam_length / 2, dim_y - 0.18, f'L = {beam_length} m',
            ha='center', va='top', fontsize=8, fontweight='600', color=GRAY)

    plt.tight_layout(pad=0.3)
    return fig


# ─────────────────────────────────────────────
#  PLOTLY INTERACTIVE CHARTS
# ─────────────────────────────────────────────

def plot_results_plotly(x, shear, moment, x_def, deflection):
    """Build and return an interactive Plotly figure with SFD, BMD, and deflection subplots."""
    BLUE   = _CLR_BLUE
    RED    = _CLR_RED
    GREEN  = _CLR_GREEN
    PURPLE = _CLR_PURPLE
    ORANGE = _CLR_ORANGE
    GRID   = _CLR_GRID
    TEXT   = _CLR_TEXT
    SUB    = _CLR_GRAY

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=False,
        subplot_titles=('Shear Force Diagram', 'Bending Moment Diagram', 'Deflection Curve'),
        vertical_spacing=0.10)

    hover_sfd  = 'x: %{x:.3f} m<br>Shear: %{y:.3f} kN<extra></extra>'
    hover_bmd  = 'x: %{x:.3f} m<br>Moment: %{y:.3f} kN·m<extra></extra>'
    hover_def  = 'x: %{x:.3f} m<br>Deflection: %{y:.5f} mm<extra></extra>'
    defl_mm    = deflection * 1000

    # ── SFD pos fill ──
    pos_mask = shear >= 0
    fig.add_trace(go.Scatter(
        x=np.concatenate([x[pos_mask], x[pos_mask][::-1]]),
        y=np.concatenate([shear[pos_mask], np.zeros(pos_mask.sum())]),
        fill='toself', fillcolor=f'rgba(0,113,227,0.12)',
        line=dict(width=0), showlegend=False, hoverinfo='skip'), row=1, col=1)
    # ── SFD neg fill ──
    neg_mask = shear < 0
    fig.add_trace(go.Scatter(
        x=np.concatenate([x[neg_mask], x[neg_mask][::-1]]),
        y=np.concatenate([shear[neg_mask], np.zeros(neg_mask.sum())]),
        fill='toself', fillcolor=f'rgba(255,59,48,0.12)',
        line=dict(width=0), showlegend=False, hoverinfo='skip'), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=x, y=shear, mode='lines', line=dict(color=BLUE, width=2.5),
        name='Shear Force', hovertemplate=hover_sfd), row=1, col=1)

    # ── BMD pos fill ──
    pos_m = moment >= 0
    fig.add_trace(go.Scatter(
        x=np.concatenate([x[pos_m], x[pos_m][::-1]]),
        y=np.concatenate([moment[pos_m], np.zeros(pos_m.sum())]),
        fill='toself', fillcolor='rgba(52,199,89,0.12)',
        line=dict(width=0), showlegend=False, hoverinfo='skip'), row=2, col=1)
    neg_m = moment < 0
    fig.add_trace(go.Scatter(
        x=np.concatenate([x[neg_m], x[neg_m][::-1]]),
        y=np.concatenate([moment[neg_m], np.zeros(neg_m.sum())]),
        fill='toself', fillcolor='rgba(175,82,222,0.12)',
        line=dict(width=0), showlegend=False, hoverinfo='skip'), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=x, y=moment, mode='lines', line=dict(color=GREEN, width=2.5),
        name='Bending Moment', hovertemplate=hover_bmd), row=2, col=1)

    # ── Deflection ──
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_def, x_def[::-1]]),
        y=np.concatenate([defl_mm, np.zeros(len(x_def))]),
        fill='toself', fillcolor='rgba(255,149,0,0.10)',
        line=dict(width=0), showlegend=False, hoverinfo='skip'), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=x_def, y=defl_mm, mode='lines', line=dict(color=ORANGE, width=2.5),
        name='Deflection', hovertemplate=hover_def), row=3, col=1)

    # ── Axis labels ──
    fig.update_yaxes(title_text='Shear (kN)',     title_font=dict(size=11, color=SUB), row=1, col=1)
    fig.update_yaxes(title_text='Moment (kN·m)',  title_font=dict(size=11, color=SUB), row=2, col=1)
    fig.update_yaxes(title_text='Deflection (mm)',title_font=dict(size=11, color=SUB), row=3, col=1)
    for r in [1, 2, 3]:
        fig.update_xaxes(title_text='Position (m)',
                         title_font=dict(size=11, color=SUB),
                         gridcolor=GRID, zeroline=True,
                         zerolinecolor='#d1d1d6', zerolinewidth=1.5,
                         row=r, col=1)
        fig.update_yaxes(gridcolor=GRID, row=r, col=1)

    # ── Title styling ──
    for ann in fig.layout.annotations:
        ann.update(font=dict(size=13, color=TEXT, family=_FONT_FAMILY),
                   x=0, xanchor='left', xref='paper')

    fig.update_layout(
        height=760,
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff',
        font=dict(family=_FONT_FAMILY, color=TEXT, size=11),
        margin=dict(l=60, r=30, t=60, b=40),
        hovermode='x unified',
        hoverlabel=dict(bgcolor='rgba(255,255,255,0.95)',
                        bordercolor='rgba(0,0,0,0.1)',
                        font_size=12,
                        font_family=_FONT_FAMILY),
        legend=dict(
            orientation='v',
            x=1.01, y=1,
            xanchor='left', yanchor='top',
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='rgba(0,0,0,0.06)',
            borderwidth=1,
            font=dict(color=SUB, size=11),
            tracegroupgap=8),
        showlegend=True)

    return fig


# ─────────────────────────────────────────────
#  PDF EXPORT
# ─────────────────────────────────────────────

def generate_pdf(beam_length, support_type, E_GPa, I_cm4,
                 point_loads, udl_loads, R_A, R_B,
                 max_sf, max_bm, max_def, def_pos,
                 x, shear, moment, x_def, deflection):
    """Generate and return the bytes of a PDF report containing results and diagrams."""
    # Build matplotlib figure for PDF (white, clean)
    BLUE, GREEN, ORANGE = _CLR_BLUE, _CLR_GREEN, _CLR_ORANGE
    fig_pdf, axes = plt.subplots(3, 1, figsize=(8, 10))
    fig_pdf.patch.set_facecolor('white')
    fig_pdf.subplots_adjust(hspace=0.45)
    defl_mm = deflection * 1000
    datasets = [
        (axes[0], x, shear, BLUE, 'Shear Force Diagram', 'Shear (kN)'),
        (axes[1], x, moment, GREEN, 'Bending Moment Diagram', 'Moment (kN·m)'),
        (axes[2], x_def, defl_mm, ORANGE, 'Deflection Curve', 'Deflection (mm)'),
    ]
    for ax, xd, yd, col, title, ylabel in datasets:
        ax.set_facecolor('white')
        ax.fill_between(xd, yd, 0, color=col, alpha=0.1)
        ax.plot(xd, yd, color=col, linewidth=2)
        ax.axhline(0, color='#d1d1d6', linewidth=1)
        ax.set_title(title, fontsize=11, fontweight='bold', loc='left', pad=8)
        ax.set_xlabel('Position (m)', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, color='#f0f0f0', linewidth=0.8)
        for sp in ax.spines.values():
            sp.set_edgecolor('#e8e8ed')
    plt.tight_layout()

    buf = io.BytesIO()
    fig_pdf.savefig(buf, format='png', dpi=_PDF_DPI, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig_pdf)

    # Build PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Title
    pdf.set_font('Helvetica', 'B', 20)
    pdf.set_text_color(0, 113, 227)
    pdf.cell(0, 12, 'PyBeam - Beam Analysis Report', ln=True)
    pdf.set_draw_color(0, 113, 227)
    pdf.set_line_width(0.6)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    # Beam config
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(29, 29, 31)
    pdf.cell(0, 8, 'Beam Configuration', ln=True)
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(110, 110, 115)
    rows = [
        ('Beam Length',    f'{beam_length} m'),
        ('Support Type',   support_type),
        ("Young's Modulus", f'{E_GPa} GPa'),
        ('Moment of Inertia', f'{I_cm4} cm4'),
    ]
    for label, val in rows:
        pdf.cell(70, 7, label + ':', border=0)
        pdf.set_text_color(29, 29, 31)
        pdf.cell(0, 7, val, ln=True)
        pdf.set_text_color(110, 110, 115)
    pdf.ln(3)

    # Reactions
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(29, 29, 31)
    pdf.cell(0, 8, 'Support Reactions', ln=True)
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(110, 110, 115)
    pdf.cell(70, 7, 'R_A (x = 0):');  pdf.set_text_color(29, 29, 31); pdf.cell(0, 7, f'{R_A:.3f} kN (up)', ln=True); pdf.set_text_color(110, 110, 115)
    if support_type == "Simply Supported":
        pdf.cell(70, 7, f'R_B (x = {beam_length} m):'); pdf.set_text_color(29, 29, 31); pdf.cell(0, 7, f'{R_B:.3f} kN (up)', ln=True)
    pdf.ln(3)

    # Results
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(29, 29, 31)
    pdf.cell(0, 8, 'Results Summary', ln=True)
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(110, 110, 115)
    results = [
        ('Max Shear Force',    f'{max_sf:.2f} kN'),
        ('Max Bending Moment', f'{max_bm:.2f} kN.m'),
        ('Max Deflection',     f'{max_def:.4f} mm'),
        ('Deflection Position',f'{def_pos:.2f} m'),
    ]
    for label, val in results:
        pdf.cell(70, 7, label + ':'); pdf.set_text_color(29, 29, 31); pdf.cell(0, 7, val, ln=True); pdf.set_text_color(110, 110, 115)
    pdf.ln(4)

    # Applied Loads table
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(29, 29, 31)
    pdf.cell(0, 8, 'Applied Loads', ln=True)

    if point_loads:
        pdf.set_font('Helvetica', 'B', 9)
        pdf.set_text_color(110, 110, 115)
        pdf.cell(0, 6, 'Point Loads', ln=True)
        pdf.set_font('Helvetica', 'B', 9)
        pdf.set_fill_color(245, 245, 247)
        pdf.set_text_color(110, 110, 115)
        pdf.cell(90, 7, 'Position (m)', border=1, fill=True)
        pdf.cell(90, 7, 'Magnitude (kN)', border=1, fill=True, ln=True)
        pdf.set_font('Helvetica', '', 9)
        pdf.set_text_color(29, 29, 31)
        for pos, mag in point_loads:
            pdf.cell(90, 7, f'  {pos}', border=1)
            pdf.cell(90, 7, f'  {mag}', border=1, ln=True)
        pdf.ln(3)

    if udl_loads:
        pdf.set_font('Helvetica', 'B', 9)
        pdf.set_text_color(110, 110, 115)
        pdf.cell(0, 6, 'UDLs', ln=True)
        pdf.set_fill_color(245, 245, 247)
        pdf.cell(63, 7, 'Intensity (kN/m)', border=1, fill=True)
        pdf.cell(63, 7, 'Start (m)', border=1, fill=True)
        pdf.cell(64, 7, 'End (m)', border=1, fill=True, ln=True)
        pdf.set_font('Helvetica', '', 9)
        pdf.set_text_color(29, 29, 31)
        for w, s, e in udl_loads:
            pdf.cell(63, 7, f'  {w}', border=1)
            pdf.cell(63, 7, f'  {s}', border=1)
            pdf.cell(64, 7, f'  {e}', border=1, ln=True)
        pdf.ln(4)

    # Beam Schematic
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(29, 29, 31)
    pdf.cell(0, 8, 'Beam Schematic', ln=True)
    viz_fig = draw_beam_visualizer(beam_length, support_type, point_loads, udl_loads)
    viz_buf = io.BytesIO()
    viz_fig.savefig(viz_buf, format='png', dpi=_PDF_DPI, bbox_inches='tight')
    viz_buf.seek(0)
    plt.close(viz_fig)
    pdf.image(viz_buf, x=10, w=190)
    pdf.ln(4)

    # Charts image
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(29, 29, 31)
    pdf.cell(0, 8, 'Diagrams', ln=True)
    pdf.image(buf, x=10, w=190)

    return pdf.output()


# ─────────────────────────────────────────────
#  STREAMLIT APP
# ─────────────────────────────────────────────

def main():
    """Entry point for the Streamlit app — renders sidebar, hero, and analysis output."""
    st.set_page_config(
        page_title="PyBeam — Beam Analysis",
        page_icon="⚗️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.markdown(APPLE_CSS, unsafe_allow_html=True)

    if "point_loads" not in st.session_state: st.session_state.point_loads = []
    if "udl_loads"   not in st.session_state: st.session_state.udl_loads   = []

    # ══════════════════════════════════════════
    #  SIDEBAR
    # ══════════════════════════════════════════
    with st.sidebar:
        st.markdown('<div class="sidebar-section-title">Beam Setup</div>', unsafe_allow_html=True)
        beam_length  = st.number_input("Beam Length (m)", min_value=0.5, max_value=100.0, value=5.0, step=0.5)
        support_type = st.selectbox("Support Type", ["Simply Supported", "Cantilever"])

        st.markdown('<div class="sidebar-section-title">Material & Section</div>', unsafe_allow_html=True)
        E_GPa = st.number_input("Young's Modulus E (GPa)", min_value=0.1, max_value=500.0, value=200.0, step=1.0,
                                 help="Steel ≈ 200 GPa · Concrete ≈ 30 GPa · Wood ≈ 12 GPa")
        I_cm4 = st.number_input("Moment of Inertia I (cm⁴)", min_value=0.1, max_value=1e7, value=8000.0, step=100.0,
                                  help="Typical I-beam: 5,000 – 50,000 cm⁴")

        st.markdown('<div class="sidebar-section-title">Point Loads</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            pl_pos = st.number_input("Position (m)", min_value=0.0, max_value=float(beam_length),
                                      value=min(2.5, beam_length), step=0.1, key="pl_pos")
        with c2:
            pl_mag = st.number_input("Magnitude (kN)", value=10.0, step=1.0, key="pl_mag", help="Positive = downward")
        if st.button("Add Point Load", use_container_width=True):
            st.session_state.point_loads.append((pl_pos, pl_mag))
        for i, (p, m) in enumerate(st.session_state.point_loads):
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:6px;margin:3px 0;">'
                f'<div class="load-tag" style="flex:1;">📍 {m} kN @ {p} m</div>'
                f'</div>', unsafe_allow_html=True)
            if st.button("✕ Remove", key=f"rpl_{i}", use_container_width=True):
                st.session_state.point_loads.pop(i); st.rerun()

        st.markdown('<div class="sidebar-section-title">Distributed Loads (UDL)</div>', unsafe_allow_html=True)
        udl_int = st.number_input("Intensity (kN/m)", value=5.0, step=0.5, key="udl_int")
        c3, c4 = st.columns(2)
        with c3:
            udl_start = st.number_input("Start (m)", min_value=0.0, max_value=float(beam_length),
                                         value=0.0, step=0.1, key="udl_start")
        with c4:
            udl_end = st.number_input("End (m)", min_value=0.0, max_value=float(beam_length),
                                       value=float(beam_length), step=0.1, key="udl_end")
        if st.button("Add UDL", use_container_width=True):
            if udl_start >= udl_end: st.error("Start must be less than end.")
            else: st.session_state.udl_loads.append((udl_int, udl_start, udl_end))
        for i, (w, s, e) in enumerate(st.session_state.udl_loads):
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:6px;margin:3px 0;">'
                f'<div class="load-tag" style="flex:1;">&#8596; {w} kN/m [{s}–{e} m]</div>'
                f'</div>', unsafe_allow_html=True)
            if st.button("✕ Remove", key=f"rudl_{i}", use_container_width=True):
                st.session_state.udl_loads.pop(i); st.rerun()

        st.markdown("")
        if st.button("Clear All Loads", use_container_width=True):
            st.session_state.point_loads = []; st.session_state.udl_loads = []; st.rerun()
        st.markdown("")
        analyze = st.button("Analyze Beam", type="primary", use_container_width=True)

    # ══════════════════════════════════════════
    #  HERO
    # ══════════════════════════════════════════
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title">Beam Analysis</h1>
        <p class="hero-subtitle">
            Compute shear forces, bending moments, and deflections<br>
            with real-time interactive visualization.
        </p>
        <div class="hero-pills">
            <span class="hero-pill">NumPy</span>
            <span class="hero-pill">SymPy</span>
            <span class="hero-pill">Plotly</span>
            <span class="hero-pill">Simply Supported</span>
            <span class="hero-pill">Cantilever</span>
            <span class="hero-pill">PDF Export</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ══════════════════════════════════════════
    #  IDLE STATE
    # ══════════════════════════════════════════
    if not analyze:
        st.markdown("""
        <div class="glass-card">
            <div class="section-heading" style="margin-top:0;">How to get started</div>
            <div class="guide-step"><div class="step-num">1</div><div class="step-body"><strong>Configure the beam</strong><span>Set beam length and support type in the sidebar.</span></div></div>
            <div class="guide-step"><div class="step-num">2</div><div class="step-body"><strong>Enter material properties</strong><span>Provide E and I for deflection calculation.</span></div></div>
            <div class="guide-step"><div class="step-num">3</div><div class="step-body"><strong>Add loads</strong><span>Add Point Loads and/or UDLs using the sidebar panels.</span></div></div>
            <div class="guide-step"><div class="step-num">4</div><div class="step-body"><strong>Analyze &amp; export</strong><span>Click Analyze Beam, hover over charts, and download a PDF report.</span></div></div>
        </div>
        <div class="glass-card">
            <div class="section-heading" style="margin-top:0;">Sign Convention</div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-top:0.5rem;">
                <div><div class="section-subheading">Loads</div><p style="font-size:0.88rem;color:#1d1d1f;margin:0;">Downward forces are <strong>positive</strong>.</p></div>
                <div><div class="section-subheading">Bending Moment</div><p style="font-size:0.88rem;color:#1d1d1f;margin:0;">Sagging (concave up) is <strong>positive</strong>.</p></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # ══════════════════════════════════════════
    #  VALIDATE
    # ══════════════════════════════════════════
    errors = validate_inputs(beam_length, E_GPa, I_cm4,
                              st.session_state.point_loads, st.session_state.udl_loads)
    if errors:
        for err in errors: st.error(err)
        return

    E_Pa = E_GPa * 1e9
    I_m4 = I_cm4 * 1e-8

    # ══════════════════════════════════════════
    #  COMPUTE
    # ══════════════════════════════════════════
    with st.spinner("Running analysis..."):
        R_A, R_B    = compute_reactions(beam_length, support_type,
                                         st.session_state.point_loads, st.session_state.udl_loads)
        x, shear    = compute_sfd(beam_length, support_type,
                                   st.session_state.point_loads, st.session_state.udl_loads, R_A, R_B)
        x, moment   = compute_bmd(x, shear)
        x_def, defl = compute_deflection(beam_length, support_type,
                                          st.session_state.point_loads, st.session_state.udl_loads,
                                          R_A, E_Pa, I_m4)

    max_sf  = float(np.max(np.abs(shear)))
    max_bm  = float(np.max(np.abs(moment)))
    max_def = float(np.max(np.abs(defl)) * 1000)
    def_pos = float(x_def[np.argmax(np.abs(defl))])

    # ══════════════════════════════════════════
    #  BEAM VISUALIZER
    # ══════════════════════════════════════════
    st.markdown('<p class="beam-viz-title">Beam Schematic</p>', unsafe_allow_html=True)
    viz_fig = draw_beam_visualizer(beam_length, support_type,
                                    st.session_state.point_loads, st.session_state.udl_loads)
    st.pyplot(viz_fig, use_container_width=True)
    plt.close(viz_fig)

    # ══════════════════════════════════════════
    #  ANIMATED RESULTS
    # ══════════════════════════════════════════
    st.markdown('<div class="section-heading">Results</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="results-grid">
        <div class="result-card blue">
            <div class="rc-label">Max Shear Force</div>
            <div class="rc-value" id="cnt_sf">{max_sf:.2f}</div>
            <div class="rc-unit">kN</div>
        </div>
        <div class="result-card green">
            <div class="rc-label">Max Bending Moment</div>
            <div class="rc-value" id="cnt_bm">{max_bm:.2f}</div>
            <div class="rc-unit">kN·m</div>
        </div>
        <div class="result-card orange">
            <div class="rc-label">Max Deflection</div>
            <div class="rc-value" id="cnt_def">{max_def:.4f}</div>
            <div class="rc-unit">mm</div>
        </div>
        <div class="result-card purple">
            <div class="rc-label">Deflection At</div>
            <div class="rc-value" id="cnt_pos">{def_pos:.2f}</div>
            <div class="rc-unit">m</div>
        </div>
        <div class="result-card teal">
            <div class="rc-label">Reaction R_A</div>
            <div class="rc-value" id="cnt_ra">{R_A:.2f}</div>
            <div class="rc-unit">kN ↑</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Inject counter animations
    st.markdown(counter_js({
        'cnt_sf':  (max_sf,  2),
        'cnt_bm':  (max_bm,  2),
        'cnt_def': (max_def, 4),
        'cnt_pos': (def_pos, 2),
        'cnt_ra':  (R_A,     2),
    }), unsafe_allow_html=True)

    # ══════════════════════════════════════════
    #  REACTIONS
    # ══════════════════════════════════════════
    with st.expander("Support Reactions"):
        rc1, rc2 = st.columns(2)
        rc1.markdown(f'<div class="reaction-chip"><span class="r-label">R_A</span>x = 0 m &nbsp;·&nbsp; {R_A:.3f} kN ↑</div>', unsafe_allow_html=True)
        if support_type == "Simply Supported":
            rc2.markdown(f'<div class="reaction-chip"><span class="r-label">R_B</span>x = {beam_length} m &nbsp;·&nbsp; {R_B:.3f} kN ↑</div>', unsafe_allow_html=True)
        else:
            rc2.markdown('<div class="reaction-chip"><span class="r-label">Fixed</span>Moment reaction at x = 0</div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════
    #  PLOTLY INTERACTIVE CHARTS
    # ══════════════════════════════════════════
    st.markdown('<div class="section-heading">Diagrams</div>', unsafe_allow_html=True)

    # PDF download button (top-right)
    pdf_bytes = generate_pdf(
        beam_length, support_type, E_GPa, I_cm4,
        st.session_state.point_loads, st.session_state.udl_loads,
        R_A, R_B, max_sf, max_bm, max_def, def_pos,
        x, shear, moment, x_def, defl)

    st.markdown('<div class="download-btn-wrap">', unsafe_allow_html=True)
    st.download_button(
        label="⬇ Download PDF Report",
        data=bytes(pdf_bytes),
        file_name="PyBeam_Report.pdf",
        mime="application/pdf")
    st.markdown('</div>', unsafe_allow_html=True)

    plotly_fig = plot_results_plotly(x, shear, moment, x_def, defl)
    st.plotly_chart(plotly_fig, use_container_width=True)

    # ══════════════════════════════════════════
    #  LOADS TABLE
    # ══════════════════════════════════════════
    with st.expander("Applied Loads"):
        if st.session_state.point_loads:
            st.markdown('<div class="section-subheading">Point Loads</div>', unsafe_allow_html=True)
            st.table({"Position (m)":   [p for p, _ in st.session_state.point_loads],
                      "Magnitude (kN)": [m for _, m in st.session_state.point_loads]})
        if st.session_state.udl_loads:
            st.markdown('<div class="section-subheading">UDLs</div>', unsafe_allow_html=True)
            st.table({"Intensity (kN/m)": [w for w, _, _ in st.session_state.udl_loads],
                      "Start (m)":        [s for _, s, _ in st.session_state.udl_loads],
                      "End (m)":          [e for _, _, e in st.session_state.udl_loads]})


if __name__ == "__main__":
    main()
