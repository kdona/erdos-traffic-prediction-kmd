# Visualization Scripts

This document describes the consolidated visualization scripts used to generate plots for the project README.

## Quick Start

**Generate ALL plots with one command:**
```bash
python generate_all_plots.py
```

**Generate only fast plots (skip day-of-week and event impact):**
```bash
python generate_all_plots.py --skip-slow
```

**Generate specific plot type:**
```bash
python generate_all_plots.py --only MODEL_COMP
```

---

## Recent Consolidation (2025-10-31)

The visualization scripts have been consolidated to eliminate code duplication and provide a single entry point for all plot generation.

### Major Changes

1. **Created `generate_all_plots.py`** - Master script that generates all visualizations in one run
2. **Created `visualization_utils.py`** - Shared utilities module
3. **Consolidated day-of-week scripts** - Combined speed + accident into one script
4. **Refactored all scripts** - All use shared utilities for consistency

### Changes Made

1. **Created `visualization_utils.py`** - Shared utilities module containing:
   - Day-of-week color schemes and markers
   - Model family color mappings
   - Travel time visualization constants
   - Common plotting helper functions
   - Figure saving utilities
   - Print formatting functions

2. **Combined day-of-week scripts** into `create_day_of_week_viz.py`:
   - Merged `create_speed_visualization.py` + `create_accident_visualization.py`
   - Both scripts shared identical marker/color definitions and plotting patterns
   - New combined script handles both speed and accident visualizations

3. **Refactored existing scripts** to use shared utilities:
   - `create_model_comparison_viz.py` - Now uses shared model family colors and utility functions
   - `create_travel_time_viz.py` - Now uses shared TMC order and color scale constants

### Benefits

- **Eliminated code duplication**: Marker definitions, color schemes, and plotting utilities are defined once
- **Consistent styling**: All visualizations use the same color schemes and styling
- **Easier maintenance**: Changes to colors or styling only need to be made in one place
- **Reduced file count**: 4 scripts → 3 scripts (+ 1 utilities module)

## Current Visualization Scripts

### 1. `create_day_of_week_viz.py`
**Purpose**: Generate traffic speed and accident/incident reporting patterns by day of week

**Output files**:
- `images/speed_by_dow_lines.png` - Traffic speed patterns by hour and day of week
- `images/accidents_by_dow_lines.png` - Accident reporting patterns by hour and day of week

**Usage**:
```bash
python create_day_of_week_viz.py
```

**Features**:
- Unique markers for each day of the week
- Blue color gradients for weekdays, orange for weekends
- Rush hour shading (7-9 AM, 4-6 PM)

---

### 2. `create_model_comparison_viz.py`
**Purpose**: Generate model performance comparison charts

**Output files**:
- `images/tabular_models_cv_rmse.png` - Feature ablation study (grouped bars)
- `images/full_feature_models_cv_rmse.png` - Full-feature model comparison (sorted)
- `images/model_rmse_heatmap.png` - Complete performance overview heatmap

**Usage**:
```bash
python create_model_comparison_viz.py
```

**Features**:
- Includes tabular models (Linear, Ridge, Lasso, RF, XGBoost, GBRT)
- Includes sequence models (LSTM, ST-GNN) if metrics available
- Color-coded by model family (Linear, Tree, Sequence)
- Gold highlighting for best model

---

### 3. `create_travel_time_viz.py`
**Purpose**: Generate travel time prediction visualizations for ST-GNN model

**Output files**:
- `images/travel_time_comparison_st-gnn.png` - Side-by-side truth vs prediction
- `images/prediction_errors_st-gnn.png` - Error heatmap showing where model struggles
- `images/error_analysis_st-gnn.png` - 4-panel error analysis breakdown
- `images/detailed_comparison_st-gnn.png` - Detailed one-week view

**Usage**:
```bash
python create_travel_time_viz.py
```

**Features**:
- Diverging colormaps for errors (red = overprediction, blue = underprediction)
- Error analysis by hour and road segment
- Focused detail views

---

### 4. `evt_impact_analysis.py`
**Purpose**: Event impact counterfactual analysis with boxplots, correlations, and delay heatmaps

**Output files**:
- `images/diff_by_event_type_boxplot.png` - Distribution of event-induced delay
- `images/corr_planned_event_rate_vs_delay_lag.png` - Lagged correlation analysis
- `images/extra_delay_by_hour.png` - Average delay by hour of day
- `images/extra_delay_heatmap_dow_hour.png` - Delay heatmap by day and hour

**Usage**:
```bash
python evt_impact_analysis.py --dir WB
python evt_impact_analysis.py --dir EB --buffer 3
```

**Features**:
- Trains separate models for no-event vs planned-event scenarios
- Estimates event impact through counterfactual analysis
- Analyzes optimal timing for planned work

---

### 5. `model_comparison.py` (via `manage.py`)
**Purpose**: Comprehensive model comparison with prediction heatmaps

**Output files**:
- Bar charts comparing all models (tabular + sequence)
- Heatmaps for ground truth and each model's predictions

**Usage**:
```bash
python manage.py compare --direction WB --save-figs
python manage.py compare --direction EB --save-figs --skip-heatmaps
```

**Features**:
- Loads all trained models (tabular, LSTM, ST-GNN)
- Computes fair RMSE comparisons on log(tt_per_mile)
- Generates time × TMC heatmaps for visual comparison

---

### 6. `create_feature_importance_viz.py`
**Purpose**: Generate XGBoost feature importance visualizations

**Output files**:
- `images/xgb_feature_importance_bars.png` - Top 20 features (horizontal bars)
- `images/xgb_feature_importance_grouped.png` - Grouped by category
- `images/xgb_feature_importance_summary.png` - Compact summary (used in README)

**Usage**:
```bash
# Via master script (recommended)
python generate_all_plots.py --only FEATURE_IMPORTANCE

# Or directly
python create_feature_importance_viz.py
```

**Features**:
- Shows relative importance of lag, time, event, and road features
- Multiple visualization styles (detailed, grouped, compact)
- Requires trained XGBoost model with metrics.json

---

### 7. `create_severity_visualization.py`
**Purpose**: Generate event severity distribution visualizations

**Output files**:
- `images/severity_stacked_bars.png` - Stacked horizontal bars (top 15 event types)
- `images/severity_grouped_bars.png` - Grouped bars for comparison
- `images/severity_summary.png` - Overall distribution + missing data analysis (used in README)

**Usage**:
```bash
# Via master script (recommended)
python generate_all_plots.py --only SEVERITY

# Or directly
python create_severity_visualization.py
```

**Features**:
- Shows severity distribution (None, Minor, Major)
- Highlights data quality issues (62% missing severity)
- Identifies event types with missing data

---

## Shared Utilities (`visualization_utils.py`)

All visualization scripts import from this module to ensure consistency.

### Key Constants

```python
# Day-of-week styling
DOW_MARKERS = ['o', 's', '^', 'D', 'v', 'P', '*']
DOW_COLORS = ['#08519c', '#3182bd', '#6baed6', '#9ecae1', '#c6dbef', '#d94801', '#fd8d3c']

# Model family colors
MODEL_FAMILY_COLORS = {
    'Linear Models': '#3182bd',
    'Tree Models': '#31a354',
    'Sequence Models': '#8856a7'
}

# Travel time visualization
TMC_ORDER_WB = [...]  # I-10 westbound segment order
TT_VMIN, TT_VMAX = 40, 110  # Color scale limits
```

### Key Functions

- `ensure_output_dir()` - Create images/ directory if needed
- `save_figure(fig, filename, dpi=300)` - Consistent figure saving
- `get_dow_style(day_index)` - Get marker/color for day of week
- `add_rush_hour_shading(ax)` - Add shaded rush hour regions
- `style_time_plot(ax, ...)` - Apply consistent plot styling
- `classify_model_family(model_name)` - Classify models into families
- `print_section_header(title)` - Formatted section headers
- `print_completion_message(script_name)` - Completion messages

## Master Script: `generate_all_plots.py`

**The recommended way to generate all plots is using the master script.**

### Basic Usage

```bash
# Generate ALL plots (takes ~5-10 minutes)
python generate_all_plots.py

# Generate all plots except slow ones (day-of-week, event impact)
python generate_all_plots.py --skip-slow

# Generate only specific plot type
python generate_all_plots.py --only MODEL_COMP
python generate_all_plots.py --only TRAVEL_TIME

# Skip specific plot types
python generate_all_plots.py --skip EVENT_IMPACT
python generate_all_plots.py --skip DOW --skip EVENT_IMPACT

# Use eastbound TMC order for comparison heatmaps
python generate_all_plots.py --direction EB

# Skip heatmaps in comparison plots (faster)
python generate_all_plots.py --skip-heatmaps
```

### Plot Types

The master script generates 7 categories of plots:

1. **DOW** - Day-of-week patterns (speed + accidents) [SLOW ~2-5 min]
2. **MODEL_COMP** - Model comparison charts (feature ablation, full comparison, heatmap)
3. **TRAVEL_TIME** - Travel time prediction visualizations (ST-GNN comparisons, error analysis)
4. **EVENT_IMPACT** - Event impact analysis (boxplots, correlations, delay heatmaps) [SLOW ~2-3 min]
5. **COMPARISON_HEATMAPS** - Model comparison with prediction heatmaps
6. **FEATURE_IMPORTANCE** - XGBoost feature importance visualizations (requires trained model)
7. **SEVERITY** - Event severity distribution

### Typical Workflows

**After training new models:**
```bash
# Quick check of model performance (< 1 minute)
python generate_all_plots.py --only MODEL_COMP

# Full regeneration including analysis plots (~ 5-10 minutes)
python generate_all_plots.py
```

**For README/presentation:**
```bash
# Generate all plots with best quality
python generate_all_plots.py
```

**Quick iteration during development:**
```bash
# Skip slow data processing, just regenerate ML plots
python generate_all_plots.py --skip-slow
```

## Individual Scripts (Advanced)

If you need fine-grained control, you can run individual scripts directly:

## Design Principles

1. **Single Source of Truth**: Color schemes, markers, and styling defined once in `visualization_utils.py`
2. **Consistency**: All plots use the same styling conventions
3. **Modularity**: Each script focuses on one type of visualization
4. **Reusability**: Utility functions can be used across scripts
5. **Clear Output**: Scripts print progress and save locations
