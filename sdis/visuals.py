"""
SDIS Visuals Module
====================

Chart layouts, styling, and UI rendering utilities for Streamlit.
"""

from typing import Dict, Optional
import streamlit as st
import os

from .config import get_config


# =============================================================================
# CHART LAYOUT
# =============================================================================

def get_chart_layout(
    custom_settings: Optional[Dict] = None
) -> Dict:
    """
    Return consistent chart layout settings with dark theme.

    Args:
        custom_settings: Optional dict to override default settings

    Returns:
        Dict with Plotly layout settings
    """
    config = get_config()
    ui = config.ui

    layout = dict(
        template='plotly_dark',
        paper_bgcolor=ui.paper_bgcolor,
        plot_bgcolor=ui.plot_bgcolor,
        font=dict(color=ui.font_color),
        title_font=dict(color=ui.font_color),
        legend=dict(
            bgcolor='rgba(45, 55, 72, 0.8)',
            font=dict(color=ui.font_color)
        ),
        xaxis=dict(
            gridcolor=ui.grid_color,
            tickfont=dict(color=ui.font_color)
        ),
        yaxis=dict(
            gridcolor=ui.grid_color,
            tickfont=dict(color=ui.font_color)
        )
    )

    if custom_settings:
        layout.update(custom_settings)

    return layout


def get_color_scale(
    value: float,
    thresholds: Optional[Dict[str, float]] = None
) -> str:
    """
    Get color based on value thresholds.

    Args:
        value: The value to color
        thresholds: Dict with 'low', 'mid', 'high' thresholds

    Returns:
        CSS color string
    """
    config = get_config()
    ui = config.ui

    if thresholds is None:
        thresholds = {'low': -3.0, 'mid': 0.0, 'high': 3.0}

    if value >= thresholds['high']:
        return ui.secondary_color  # Green
    elif value >= thresholds['mid']:
        return ui.warning_color    # Yellow/Orange
    else:
        return ui.danger_color     # Red


# =============================================================================
# CUSTOM CSS
# =============================================================================

def apply_custom_css():
    """Apply custom CSS styling for SDIS dark theme."""
    css = """
    <style>
    /* Main container styling */
    .stApp {
        background: linear-gradient(135deg, #0a1628 0%, #162d50 50%, #0a1628 100%);
    }

    /* Card styling */
    .info-box {
        background: rgba(26, 45, 74, 0.6);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4A90D9;
        margin-bottom: 1rem;
    }

    .metric-card {
        background: rgba(26, 45, 74, 0.8);
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        border: 1px solid rgba(74, 144, 217, 0.3);
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #E5E7EB;
    }

    .metric-label {
        font-size: 0.875rem;
        color: rgba(229, 231, 235, 0.7);
        margin-top: 0.25rem;
    }

    /* Risk level colors */
    .risk-low { color: #10B981; }
    .risk-medium { color: #F59E0B; }
    .risk-high { color: #EF4444; }

    /* Status badges */
    .status-elite {
        background: linear-gradient(135deg, #10B981, #059669);
        color: white;
        padding: 4px 12px;
        border-radius: 16px;
        font-weight: 600;
    }

    .status-good {
        background: linear-gradient(135deg, #3B82F6, #2563EB);
        color: white;
        padding: 4px 12px;
        border-radius: 16px;
        font-weight: 600;
    }

    .status-fatigue {
        background: linear-gradient(135deg, #EF4444, #DC2626);
        color: white;
        padding: 4px 12px;
        border-radius: 16px;
        font-weight: 600;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(10, 22, 40, 0.95);
    }

    /* Table styling */
    .dataframe {
        background: rgba(26, 45, 74, 0.6) !important;
    }

    .dataframe th {
        background: rgba(74, 144, 217, 0.3) !important;
        color: #E5E7EB !important;
    }

    .dataframe td {
        color: #E5E7EB !important;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #4A90D9, #2563EB);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #3B82F6, #1D4ED8);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(74, 144, 217, 0.4);
    }

    /* Slider styling */
    .stSlider > div > div > div {
        background: #4A90D9;
    }

    /* Selectbox styling */
    .stSelectbox > div > div {
        background: rgba(26, 45, 74, 0.8);
        border: 1px solid rgba(74, 144, 217, 0.3);
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def load_external_css(css_filename: str = "sdis_styles.css"):
    """
    Load external CSS file for styling.

    Args:
        css_filename: Name of CSS file in the same directory
    """
    css_path = os.path.join(os.path.dirname(__file__), "..", css_filename)
    try:
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        # Fall back to inline CSS
        apply_custom_css()


# =============================================================================
# HEADER RENDERING
# =============================================================================

def render_header(module_name: str = ""):
    """
    Render the SDIS header with optional module name.

    Args:
        module_name: Optional subtitle for the current module
    """
    module_display = f'''
        <div style="font-size: 0.9rem; color: rgba(100,181,246,0.9);
                    margin-top: 8px; font-weight: 500;">{module_name}</div>
    ''' if module_name else ''

    st.markdown(f'''
    <div style="position: relative; padding: 2rem; margin-bottom: 1.5rem;
                background: linear-gradient(135deg, #0a1628 0%, #162d50 50%, #0a1628 100%);
                border-radius: 16px; overflow: hidden;
                box-shadow: 0 8px 32px rgba(0,0,0,0.3);">
        <!-- Background grid pattern -->
        <div style="position: absolute; top: 0; left: 0; right: 0; bottom: 0; opacity: 0.1;
                    background-image:
                        linear-gradient(rgba(100,181,246,0.3) 1px, transparent 1px),
                        linear-gradient(90deg, rgba(100,181,246,0.3) 1px, transparent 1px);
                    background-size: 40px 40px;">
        </div>
        <!-- DNA Strand Left -->
        <svg style="position: absolute; left: 30px; top: 50%;
                    transform: translateY(-50%); opacity: 0.15;"
             width="60" height="120" viewBox="0 0 60 120">
            <path d="M30 0 Q50 15 30 30 Q10 45 30 60 Q50 75 30 90 Q10 105 30 120"
                  stroke="#64B5F6" stroke-width="2" fill="none"/>
            <path d="M30 0 Q10 15 30 30 Q50 45 30 60 Q10 75 30 90 Q50 105 30 120"
                  stroke="#64B5F6" stroke-width="2" fill="none"/>
            <line x1="15" y1="15" x2="45" y2="15" stroke="#64B5F6" stroke-width="1.5" opacity="0.6"/>
            <line x1="45" y1="30" x2="15" y2="30" stroke="#64B5F6" stroke-width="1.5" opacity="0.6"/>
            <line x1="15" y1="45" x2="45" y2="45" stroke="#64B5F6" stroke-width="1.5" opacity="0.6"/>
            <line x1="45" y1="60" x2="15" y2="60" stroke="#64B5F6" stroke-width="1.5" opacity="0.6"/>
            <line x1="15" y1="75" x2="45" y2="75" stroke="#64B5F6" stroke-width="1.5" opacity="0.6"/>
            <line x1="45" y1="90" x2="15" y2="90" stroke="#64B5F6" stroke-width="1.5" opacity="0.6"/>
            <line x1="15" y1="105" x2="45" y2="105" stroke="#64B5F6" stroke-width="1.5" opacity="0.6"/>
        </svg>
        <!-- DNA Strand Right -->
        <svg style="position: absolute; right: 30px; top: 50%;
                    transform: translateY(-50%); opacity: 0.15;"
             width="60" height="120" viewBox="0 0 60 120">
            <path d="M30 0 Q50 15 30 30 Q10 45 30 60 Q50 75 30 90 Q10 105 30 120"
                  stroke="#64B5F6" stroke-width="2" fill="none"/>
            <path d="M30 0 Q10 15 30 30 Q50 45 30 60 Q10 75 30 90 Q50 105 30 120"
                  stroke="#64B5F6" stroke-width="2" fill="none"/>
            <line x1="15" y1="15" x2="45" y2="15" stroke="#64B5F6" stroke-width="1.5" opacity="0.6"/>
            <line x1="45" y1="30" x2="15" y2="30" stroke="#64B5F6" stroke-width="1.5" opacity="0.6"/>
            <line x1="15" y1="45" x2="45" y2="45" stroke="#64B5F6" stroke-width="1.5" opacity="0.6"/>
            <line x1="45" y1="60" x2="15" y2="60" stroke="#64B5F6" stroke-width="1.5" opacity="0.6"/>
            <line x1="15" y1="75" x2="45" y2="75" stroke="#64B5F6" stroke-width="1.5" opacity="0.6"/>
            <line x1="45" y1="90" x2="15" y2="90" stroke="#64B5F6" stroke-width="1.5" opacity="0.6"/>
            <line x1="15" y1="105" x2="45" y2="105" stroke="#64B5F6" stroke-width="1.5" opacity="0.6"/>
        </svg>
        <!-- Content -->
        <div style="position: relative; display: flex; align-items: center;
                    justify-content: center; gap: 24px;">
            <!-- Logo -->
            <svg width="80" height="80" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
                <line x1="50" y1="5" x2="50" y2="95" stroke="white" stroke-width="5" stroke-linecap="round"/>
                <circle cx="50" cy="50" r="30" fill="none" stroke="white" stroke-width="5"/>
                <line x1="5" y1="50" x2="20" y2="50" stroke="white" stroke-width="5" stroke-linecap="round"/>
                <line x1="80" y1="50" x2="95" y2="50" stroke="white" stroke-width="5" stroke-linecap="round"/>
            </svg>
            <!-- Text -->
            <div style="text-align: left;">
                <div style="font-size: 1.1rem; font-weight: 400;
                            color: rgba(255,255,255,0.6); letter-spacing: 2px;
                            text-transform: uppercase;">
                    SDIS
                </div>
                <div style="font-size: 1.8rem; font-weight: 600; color: white;
                            line-height: 1.2; letter-spacing: 0.5px;">
                    Szklanny Decision<br>Intelligence System
                </div>
                {module_display}
            </div>
        </div>
        <!-- Decorative data lines -->
        <svg style="position: absolute; bottom: 10px; left: 100px; opacity: 0.2;"
             width="150" height="40" viewBox="0 0 150 40">
            <path d="M0 35 L20 35 L30 15 L40 30 L50 10 L60 25 L70 20 L80 35 L150 35"
                  stroke="#64B5F6" stroke-width="2" fill="none"/>
        </svg>
        <svg style="position: absolute; bottom: 10px; right: 100px; opacity: 0.2;"
             width="150" height="40" viewBox="0 0 150 40">
            <path d="M0 35 L30 35 L40 20 L50 30 L60 5 L70 25 L90 15 L110 30 L150 35"
                  stroke="#64B5F6" stroke-width="2" fill="none"/>
        </svg>
    </div>
    ''', unsafe_allow_html=True)


# =============================================================================
# METRIC CARDS
# =============================================================================

def render_metric_card(
    label: str,
    value: str,
    delta: Optional[str] = None,
    color: str = "#4A90D9"
):
    """
    Render a styled metric card.

    Args:
        label: Metric label
        value: Metric value (formatted string)
        delta: Optional change indicator
        color: Accent color
    """
    delta_html = f'''
        <div style="font-size: 0.875rem; color: {color}; margin-top: 4px;">
            {delta}
        </div>
    ''' if delta else ''

    st.markdown(f'''
    <div class="metric-card" style="border-color: {color}40;">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
        {delta_html}
    </div>
    ''', unsafe_allow_html=True)


def render_status_badge(
    status: str,
    spm_value: Optional[float] = None
) -> str:
    """
    Get HTML for a status badge based on SPM value.

    Args:
        status: Status text to display
        spm_value: Optional SPM value to determine color

    Returns:
        HTML string for the badge
    """
    config = get_config()
    benchmarks = config.benchmarks

    if spm_value is not None:
        if spm_value >= benchmarks.get('elite_spm', 3.0):
            badge_class = 'status-elite'
        elif spm_value >= benchmarks.get('good_spm', 1.0):
            badge_class = 'status-good'
        else:
            badge_class = 'status-fatigue'
    else:
        badge_class = 'status-good'

    return f'<span class="{badge_class}">{status}</span>'


# =============================================================================
# PROGRESS BARS
# =============================================================================

def render_cap_bar(
    current: float,
    cap: float,
    tax: float,
    first_apron: float,
    second_apron: float
):
    """
    Render a salary cap progress bar with threshold markers.

    Args:
        current: Current salary
        cap: Salary cap
        tax: Luxury tax threshold
        first_apron: First apron threshold
        second_apron: Second apron threshold
    """
    cap_pct = min(100, (current / second_apron) * 100)
    tax_pct = (tax / second_apron) * 100
    apron1_pct = (first_apron / second_apron) * 100

    st.markdown(f'''
    <div style="position: relative; height: 40px; background: #1a202c;
                border-radius: 8px; overflow: hidden; margin: 1rem 0;">
        <div style="position: absolute; left: 0; top: 0; height: 100%;
                    width: {cap_pct}%;
                    background: linear-gradient(90deg,
                        #10B981 0%, #F59E0B {tax_pct}%,
                        #EF4444 {apron1_pct}%, #7f1d1d 100%);
                    transition: width 0.3s;">
        </div>
        <div style="position: absolute; left: {tax_pct}%; top: 0;
                    height: 100%; width: 2px; background: white; opacity: 0.5;">
        </div>
        <div style="position: absolute; left: {apron1_pct}%; top: 0;
                    height: 100%; width: 2px; background: white; opacity: 0.5;">
        </div>
        <div style="position: absolute; left: 50%; top: 50%;
                    transform: translate(-50%, -50%); color: white;
                    font-weight: 600; text-shadow: 0 1px 3px rgba(0,0,0,0.5);">
            ${current/1e6:.1f}M / ${cap/1e6:.0f}M Cap
        </div>
    </div>
    <div style="display: flex; justify-content: space-between;
                font-size: 0.8rem; opacity: 0.7;">
        <span>Cap: ${cap/1e6:.0f}M</span>
        <span>Tax: ${tax/1e6:.0f}M</span>
        <span>1st Apron: ${first_apron/1e6:.0f}M</span>
        <span>2nd Apron: ${second_apron/1e6:.0f}M</span>
    </div>
    ''', unsafe_allow_html=True)


# =============================================================================
# SPM GAUGE
# =============================================================================

def render_spm_gauge(spm_value: float, player_name: str = ""):
    """
    Render an SPM gauge visualization.

    Args:
        spm_value: SPM score (-10 to +10)
        player_name: Optional player name for title
    """
    import plotly.graph_objects as go

    config = get_config()

    # Determine color based on SPM
    if spm_value >= config.benchmarks.get('elite_spm', 3.0):
        color = config.ui.secondary_color
        status = "Elite Resilience"
    elif spm_value >= 0:
        color = config.ui.primary_color
        status = "Good Resilience"
    elif spm_value >= config.benchmarks.get('fatigue_spm', -3.0):
        color = config.ui.warning_color
        status = "Moderate Fatigue"
    else:
        color = config.ui.danger_color
        status = "High Fatigue Risk"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=spm_value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': player_name or "SPM Score", 'font': {'size': 16}},
        gauge={
            'axis': {'range': [-10, 10], 'tickwidth': 1},
            'bar': {'color': color},
            'steps': [
                {'range': [-10, -3], 'color': 'rgba(239, 68, 68, 0.3)'},
                {'range': [-3, 0], 'color': 'rgba(245, 158, 11, 0.3)'},
                {'range': [0, 3], 'color': 'rgba(59, 130, 246, 0.3)'},
                {'range': [3, 10], 'color': 'rgba(16, 185, 129, 0.3)'},
            ],
            'threshold': {
                'line': {'color': 'white', 'width': 2},
                'thickness': 0.75,
                'value': spm_value
            }
        }
    ))

    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=50, b=20),
        **get_chart_layout()
    )

    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Status: **{status}**")
