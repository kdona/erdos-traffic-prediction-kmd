"""
Shared utilities for visualization scripts.

This module eliminates code duplication by providing common:
- Color schemes and markers
- Plot styling functions
- Directory setup
- Model family mappings
"""
from pathlib import Path
import matplotlib.pyplot as plt


# ============================================================================
# Directory Configuration
# ============================================================================

OUTPUT_DIR = Path("images")


def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    return OUTPUT_DIR


# ============================================================================
# Day-of-Week Visualization Styling
# ============================================================================

# Markers for each day of the week
DOW_MARKERS = ['o', 's', '^', 'D', 'v', 'P', '*']

# Color gradients: blue shades for weekdays, orange shades for weekends
DOW_COLORS = [
    '#08519c',  # Monday - darkest blue
    '#3182bd',  # Tuesday - dark blue
    '#6baed6',  # Wednesday - medium blue
    '#9ecae1',  # Thursday - light blue
    '#c6dbef',  # Friday - lightest blue
    '#d94801',  # Saturday - darker orange
    '#fd8d3c',  # Sunday - lighter orange
]

# Day name mappings
DOW_SHORT_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
DOW_FULL_NAMES = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']


def get_dow_style(day_index):
    """
    Get marker, color, and styling for a specific day of week.

    Args:
        day_index: 0-6 (Monday-Sunday)

    Returns:
        dict with keys: marker, color, linewidth, alpha, markersize
    """
    is_weekend = day_index >= 5

    return {
        'marker': DOW_MARKERS[day_index],
        'color': DOW_COLORS[day_index],
        'linewidth': 2.5 if is_weekend else 1.5,
        'alpha': 1.0 if is_weekend else 0.8,
        'markersize': 6 if is_weekend else 4,
    }


def add_rush_hour_shading(ax):
    """Add subtle background shading for rush hours to a plot."""
    ax.axvspan(7, 9, alpha=0.05, color='red', label='_nolegend_')   # Morning rush
    ax.axvspan(16, 18, alpha=0.05, color='red', label='_nolegend_')  # Evening rush


def style_time_plot(ax, xlabel='Hour of Day', ylabel='Value', title=''):
    """Apply consistent styling to time-based plots."""
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='best', ncol=7, framealpha=0.9, edgecolor='gray')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, 23)
    ax.set_xticks(range(0, 24, 2))


# ============================================================================
# Model Comparison Styling
# ============================================================================

# Model family colors for comparison charts
MODEL_FAMILY_COLORS = {
    'Linear Models': '#3182bd',
    'Tree Models': '#31a354',
    'Sequence Models': '#8856a7'
}

# Specific model colors for ablation study
MODEL_COLORS = {
    'Linear Regression': '#3182bd',
    'Random Forest': '#31a354',
    'Gradient Boosting': '#fd8d3c',
    'XGBoost': '#8856a7',
}

# Model name mappings
MODEL_TYPE_MAP = {
    'lr': 'Linear Regression',
    'rf': 'Random Forest',
    'xgb': 'XGBoost',
    'gbrt': 'Gradient Boosting',
    'ridge': 'Ridge',
    'lasso': 'Lasso'
}

# Feature set display names
FEATURE_DISPLAY_MAP = {
    'base': 'road',
    'base_lags': 'road + lag',
    'evt': 'road + evt',
    'evt_lags': 'road + evt + lag',
    'cyc': 'road + cyc',
    'cyc_lags': 'road + cyc + lag',
    'full': 'full'
}


def classify_model_family(model_display_name):
    """Classify a model into its family for coloring."""
    if model_display_name in ['Linear Regression', 'Ridge', 'Lasso']:
        return 'Linear Models'
    elif model_display_name in ['Random Forest', 'Gradient Boosting', 'XGBoost']:
        return 'Tree Models'
    else:
        return 'Sequence Models'


# ============================================================================
# Travel Time Visualization Settings
# ============================================================================

# TMC order for I-10 westbound
TMC_ORDER_WB = [
    "115P04188", "115+04188", "115P04187", "115+04187", "115P04186", "115+04186",
    "115P04185", "115+04185", "115P04184", "115+04184", "115P04183", "115+04183",
    "115P04182", "115+04182", "115P04181", "115+04181", "115P04180", "115+04180",
    "115P04179", "115+04179", "115P04178", "115+04178", "115P04177", "115+04177",
    "115P05165"
]

# Travel time heatmap color scale
TT_VMIN, TT_VMAX = 40, 110


# ============================================================================
# Figure Saving
# ============================================================================

def save_figure(fig, filename, dpi=300):
    """
    Save a figure to the output directory with consistent settings.

    Args:
        fig: matplotlib Figure object
        filename: Name of the output file (with extension)
        dpi: Resolution (default 300 for publication quality)

    Returns:
        Path to the saved file
    """
    output_path = ensure_output_dir() / filename
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return output_path


# ============================================================================
# Printing Utilities
# ============================================================================

def print_section_header(title):
    """Print a formatted section header."""
    print("\n" + "="*60)
    print(title)
    print("="*60)


def print_completion_message(script_name):
    """Print completion message for a visualization script."""
    print("\n" + "="*60)
    print(f"{script_name} complete!")
    print("="*60)
