"""
Create improved model comparison visualizations for README.

Generates clearer, more informative charts showing:
1. Feature ablation study (grouped horizontal bars)
2. Full-feature model comparison (sorted horizontal bars with model family colors)
3. Heatmap overview of all models and features
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Import shared utilities
from visualization_utils import (
    ensure_output_dir, save_figure, print_section_header, print_completion_message,
    MODEL_FAMILY_COLORS, MODEL_COLORS, MODEL_TYPE_MAP, FEATURE_DISPLAY_MAP,
    classify_model_family
)

# Configuration
METRICS_PATH = Path("models/tabular_run/metrics.csv")


def load_metrics():
    """Load model metrics from CSV file."""
    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"Metrics file not found: {METRICS_PATH}")

    df = pd.read_csv(METRICS_PATH)
    print(f"Loaded {len(df)} model results from {METRICS_PATH}")
    return df


def parse_model_info(df):
    """Extract model family and feature set from model names using shared mappings."""
    # Store original combined name
    df['model_feat'] = df['model']

    # Extract model type and features
    df['model_type'] = df['model'].str.extract(r'^(lr|rf|xgb|gbrt|ridge|lasso)_?')[0]
    df['features'] = df['model'].str.replace(r'^(lr|rf|xgb|gbrt|ridge|lasso)_?', '', regex=True)

    # Map to display names using shared utilities
    df['model_display'] = df['model_type'].map(MODEL_TYPE_MAP)
    df['features_display'] = df['features'].map(FEATURE_DISPLAY_MAP).fillna(df['features'])

    # Use cv_rmse_mean as the metric
    df['cv_rmse'] = df['cv_rmse_mean']

    return df


def create_feature_ablation_chart(df):
    """
    Create a grouped horizontal bar chart showing feature ablation study.
    Groups models by feature set, colored by model family.
    """
    # Filter out ridge/lasso for cleaner comparison
    df_plot = df[~df['model_type'].isin(['ridge', 'lasso'])].copy()

    # Define feature order (from simple to complex)
    feature_order = ['road', 'road + evt', 'road + cyc', 'road + lag',
                     'road + evt + lag', 'road + cyc + lag', 'full']

    fig, ax = plt.subplots(figsize=(12, 7))

    # Pivot data for grouped plotting
    pivot_data = df_plot.pivot(index='features_display', columns='model_display', values='cv_rmse')
    pivot_data = pivot_data.reindex(feature_order)

    # Create horizontal bar chart using shared color scheme
    pivot_data.plot(kind='barh', ax=ax, width=0.7,
                   color=[MODEL_COLORS.get(col, '#888888') for col in pivot_data.columns])

    ax.set_xlabel('CV RMSE (lower is better)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature Set', fontsize=12, fontweight='bold')
    ax.set_title('Feature Ablation Study: Adding Features Improves Performance',
                fontsize=14, fontweight='bold', pad=15)

    # Move legend outside plot area to avoid covering values
    ax.legend(title='Model', bbox_to_anchor=(1.02, 1), loc='upper left',
             framealpha=1, edgecolor='gray', fontsize=10)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.invert_yaxis()  # Simple features at top

    # Set x-axis limit to give space for values
    ax.set_xlim(0, pivot_data.max().max() * 1.05)

    plt.tight_layout()
    return fig


def create_full_feature_comparison(df):
    """
    Create a sorted horizontal bar chart comparing all models with full features.
    Includes LSTM and ST-GNN if metrics available.
    """
    # Get full-feature models
    df_full = df[df['features'] == 'full'].copy()

    # Try to add sequence models from their metrics files
    lstm_metrics_path = Path("models/lstm_run/metrics.json")
    gcn_metrics_path = Path("models/gcn/gcn_lstm_i10_wb/metrics.json")

    extra_models = []

    # Load LSTM metrics
    if lstm_metrics_path.exists():
        with open(lstm_metrics_path, 'r') as f:
            data = json.load(f)
            lstm_val = data.get("best_val", None)
            if lstm_val is not None:
                extra_models.append({
                    'model_feat': 'lstm_full',
                    'cv_rmse': float(np.sqrt(lstm_val)),
                    'model_display': 'LSTM',
                    'family': 'Sequence Models'
                })

    # Load ST-GNN (GCN-LSTM) metrics
    if gcn_metrics_path.exists():
        with open(gcn_metrics_path, 'r') as f:
            data = json.load(f)
            gcn_val = data.get("best_val", None)
            if gcn_val is not None:
                extra_models.append({
                    'model_feat': 'gnn_full',
                    'cv_rmse': float(np.sqrt(gcn_val)),
                    'model_display': 'ST-GNN',
                    'family': 'Sequence Models'
                })

    # Combine with tabular models
    if extra_models:
        df_extra = pd.DataFrame(extra_models)
        df_full = pd.concat([df_full, df_extra], ignore_index=True)

    # Add model family for coloring using shared function
    df_full['family'] = df_full['model_display'].apply(classify_model_family)

    # Sort by RMSE (ASCENDING - best at top)
    df_full = df_full.sort_values('cv_rmse', ascending=True)

    # Use shared color scheme for model families
    colors = [MODEL_FAMILY_COLORS.get(family, '#888888') for family in df_full['family']]

    # Highlight best model with gold color
    colors[0] = '#ffd700'  # Gold for best model

    fig, ax = plt.subplots(figsize=(10, 7))

    bars = ax.barh(df_full['model_display'], df_full['cv_rmse'],
                   color=colors, alpha=0.85, edgecolor='black', linewidth=1.2)

    # Add value labels INSIDE bars (right-aligned)
    for i, (bar, val) in enumerate(zip(bars, df_full['cv_rmse'])):
        # Place text inside bar, right-aligned
        ax.text(val - 0.05, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', ha='right', fontsize=11,
                fontweight='bold', color='white')

    # Add "BEST" annotation to top model
    best_idx = 0
    best_bar = bars[best_idx]
    ax.text(best_bar.get_width() + 0.15, best_bar.get_y() + best_bar.get_height()/2,
            '★ BEST', va='center', fontsize=12, fontweight='bold', color='#ffd700')

    ax.set_xlabel('CV RMSE (lower is better)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')
    ax.set_title('Full-Feature Model Comparison: Best to Worst',
                fontsize=14, fontweight='bold', pad=15)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.invert_yaxis()  # Best model at top

    # Set x-axis limit to give space for labels
    ax.set_xlim(0, df_full['cv_rmse'].max() * 1.15)

    # Create custom legend for model families using shared colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#ffd700', label='Best Model', edgecolor='black'),
        Patch(facecolor=MODEL_FAMILY_COLORS['Linear Models'], label='Linear Models', edgecolor='black'),
        Patch(facecolor=MODEL_FAMILY_COLORS['Tree Models'], label='Tree Models', edgecolor='black'),
        Patch(facecolor=MODEL_FAMILY_COLORS['Sequence Models'], label='Sequence Models', edgecolor='black')
    ]
    ax.legend(handles=legend_elements, title='Model Family', loc='lower right',
             framealpha=1, edgecolor='gray', fontsize=10)

    plt.tight_layout()
    return fig


def create_heatmap_comparison(df):
    """
    Create a heatmap showing RMSE across all feature sets and models.
    Provides a complete overview at a glance.
    """
    # Filter out ridge/lasso
    df_plot = df[~df['model_type'].isin(['ridge', 'lasso'])].copy()

    # Pivot to create heatmap
    heatmap_data = df_plot.pivot(index='model_display', columns='features_display', values='cv_rmse')

    # Reorder columns by feature complexity
    feature_order = ['road', 'road + evt', 'road + cyc', 'road + lag',
                     'road + evt + lag', 'road + cyc + lag', 'full']
    heatmap_data = heatmap_data[feature_order]

    # Sort rows by performance on 'full' features (best at top)
    row_order = heatmap_data['full'].sort_values().index
    heatmap_data = heatmap_data.loc[row_order]

    fig, ax = plt.subplots(figsize=(13, 6))

    # Use a better color scheme - lower values (better) are greener
    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn_r',
                cbar_kws={'label': 'CV RMSE (lower is better)'}, linewidths=1.5,
                linecolor='white', ax=ax,
                vmin=heatmap_data.min().min(), vmax=heatmap_data.max().max(),
                annot_kws={'fontsize': 10, 'fontweight': 'bold'})

    ax.set_xlabel('Feature Set (simple → complex)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model (best → worst)', fontsize=12, fontweight='bold')
    ax.set_title('Complete Model Performance Overview: Lower is Better',
                fontsize=14, fontweight='bold', pad=15)

    # Rotate x-axis labels for better readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    return fig


def main():
    """Main workflow."""
    print_section_header("Generating Model Comparison Visualizations")

    # Load and parse metrics
    df = load_metrics()
    df = parse_model_info(df)

    # 1. Feature ablation study
    print("\n1. Creating feature ablation chart...")
    fig1 = create_feature_ablation_chart(df)
    output_path1 = save_figure(fig1, "tabular_models_cv_rmse.png")
    print(f"✓ Saved: {output_path1}")

    # 2. Full-feature model comparison
    print("\n2. Creating full-feature comparison chart...")
    fig2 = create_full_feature_comparison(df)
    output_path2 = save_figure(fig2, "full_feature_models_cv_rmse.png")
    print(f"✓ Saved: {output_path2}")

    # 3. Heatmap overview
    print("\n3. Creating heatmap overview...")
    fig3 = create_heatmap_comparison(df)
    output_path3 = save_figure(fig3, "model_rmse_heatmap.png")
    print(f"✓ Saved: {output_path3}")

    print_completion_message("Model Comparison Visualization")


if __name__ == "__main__":
    main()
