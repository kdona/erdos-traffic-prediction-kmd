"""
Generate clear, professional feature importance visualizations for XGBoost model.

Creates multiple views:
1. Horizontal bar chart (top 20 features) - clearest view
2. Grouped bar chart by feature category - shows category importance
3. Compact summary chart - for presentations
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Configuration
MODELS_DIR = Path("models/tabular_run")
OUTPUT_DIR = Path("images")
OUTPUT_DIR.mkdir(exist_ok=True)

# Define feature categories with nice names
FEATURE_GROUPS = {
    'lag': {
        'features': ['log_lag1_tt_per_mile', 'log_lag2_tt_per_mile', 'log_lag3_tt_per_mile',
                     'lag1_tt_per_mile', 'lag2_tt_per_mile', 'lag3_tt_per_mile'],
        'display_name': 'Lag Features',
        'color': '#2ecc71',
        'description': 'Previous hours travel time'
    },
    'cyc': {
        'features': ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
                     'hour_of_week_sin', 'hour_of_week_cos', 'is_weekend'],
        'display_name': 'Time Features',
        'color': '#3498db',
        'description': 'Hour/day patterns'
    },
    'evt': {
        'features': ['evt_cat_planned', 'evt_cat_unplanned', 'evt_total'],
        'display_name': 'Event Features',
        'color': '#e74c3c',
        'description': 'Roadwork & accidents'
    },
    'road': {
        'features': ['miles', 'reference_speed', 'curve', 'onramp', 'offramp'],
        'display_name': 'Road Features',
        'color': '#9b59b6',
        'description': 'Segment geometry'
    }
}

# Nice display names for features
FEATURE_DISPLAY_NAMES = {
    'log_lag1_tt_per_mile': 'Lag 1hr (log)',
    'log_lag2_tt_per_mile': 'Lag 2hr (log)',
    'log_lag3_tt_per_mile': 'Lag 3hr (log)',
    'lag1_tt_per_mile': 'Lag 1hr',
    'lag2_tt_per_mile': 'Lag 2hr',
    'lag3_tt_per_mile': 'Lag 3hr',
    'hour_sin': 'Hour (sin)',
    'hour_cos': 'Hour (cos)',
    'dow_sin': 'Day of Week (sin)',
    'dow_cos': 'Day of Week (cos)',
    'hour_of_week_sin': 'Hour of Week (sin)',
    'hour_of_week_cos': 'Hour of Week (cos)',
    'is_weekend': 'Is Weekend',
    'evt_cat_planned': 'Planned Events',
    'evt_cat_unplanned': 'Unplanned Events',
    'evt_total': 'Total Events',
    'miles': 'Segment Length',
    'reference_speed': 'Free-flow Speed',
    'curve': 'Has Curve',
    'onramp': 'Has On-ramp',
    'offramp': 'Has Off-ramp'
}


def load_feature_importance():
    """Load feature importance from saved XGBoost model metrics."""
    metrics_file = MODELS_DIR / "metrics.json"

    if not metrics_file.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_file}")

    with open(metrics_file, 'r') as f:
        metrics = json.load(f)

    # metrics is a list of model results
    # Find XGBoost full model
    xgb_models = [m for m in metrics if m.get('model', '').startswith('xgb_full')]

    if not xgb_models:
        raise ValueError("No XGBoost full model found in metrics")

    # Use first XGBoost full model
    model_metrics = xgb_models[0]
    model_name = model_metrics['model']

    if 'top_features' not in model_metrics:
        raise ValueError(f"No feature importance found for {model_name}")

    # Convert to DataFrame
    # top_features is a list of {feature, score} dicts
    features_list = model_metrics['top_features']
    df = pd.DataFrame(features_list)
    df.rename(columns={'score': 'importance'}, inplace=True)

    # Normalize to percentages
    df['importance_pct'] = (df['importance'] / df['importance'].sum()) * 100

    print(f"Loaded feature importance from: {model_name}")
    print(f"Total features: {len(df)}")

    return df


def assign_feature_groups(df):
    """Assign each feature to its group."""
    def get_group(feature):
        for group_name, group_info in FEATURE_GROUPS.items():
            if feature in group_info['features']:
                return group_name
        return 'other'

    df['group'] = df['feature'].apply(get_group)
    df['group_name'] = df['group'].map(lambda g: FEATURE_GROUPS.get(g, {}).get('display_name', 'Other'))
    df['color'] = df['group'].map(lambda g: FEATURE_GROUPS.get(g, {}).get('color', '#95a5a6'))
    df['display_name'] = df['feature'].map(lambda f: FEATURE_DISPLAY_NAMES.get(f, f))

    return df


def create_horizontal_bar_chart(df, top_n=20):
    """
    Create a clean horizontal bar chart showing top N most important features.
    This is the clearest way to show feature importance.
    """
    # Sort by importance and take top N
    df_top = df.nlargest(top_n, 'importance_pct').sort_values('importance_pct')

    fig, ax = plt.subplots(figsize=(12, 10))

    # Create bars with colors by group
    bars = ax.barh(range(len(df_top)), df_top['importance_pct'],
                    color=df_top['color'], alpha=0.8, edgecolor='black', linewidth=0.5)

    # Set labels
    ax.set_yticks(range(len(df_top)))
    ax.set_yticklabels(df_top['display_name'], fontsize=11)
    ax.set_xlabel('Importance (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'XGBoost Feature Importance (Top {top_n})',
                 fontsize=14, fontweight='bold', pad=20)

    # Add percentage labels on bars
    for i, (idx, row) in enumerate(df_top.iterrows()):
        ax.text(row['importance_pct'] + 0.3, i, f"{row['importance_pct']:.1f}%",
                va='center', fontsize=9, fontweight='bold')

    # Add grid
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Create legend for feature groups
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=info['color'], label=f"{info['display_name']}")
        for group, info in FEATURE_GROUPS.items()
    ]
    ax.legend(handles=legend_elements, loc='lower right',
              framealpha=0.9, edgecolor='black')

    plt.tight_layout()
    return fig


def create_grouped_summary(df):
    """
    Create a summary showing total importance by feature group.
    Shows the big picture - which categories matter most.
    """
    # Calculate group totals
    group_totals = df.groupby(['group', 'group_name', 'color'])['importance_pct'].sum().reset_index()
    group_totals = group_totals.sort_values('importance_pct', ascending=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Grouped bar chart with individual features
    for group_name in group_totals['group_name'].unique():
        group_data = df[df['group_name'] == group_name].nlargest(5, 'importance_pct')
        if len(group_data) == 0:
            continue

        group_color = group_data['color'].iloc[0]

        # Show top 5 features per group
        y_pos = np.arange(len(group_data))
        ax1.barh(y_pos, group_data['importance_pct'],
                label=group_name, color=group_color, alpha=0.7)

    ax1.set_xlabel('Importance (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Feature Importance by Category (Top 5 per Category)',
                  fontsize=13, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3, linestyle='--')

    # Right: Category-level summary
    bars = ax2.barh(range(len(group_totals)), group_totals['importance_pct'],
                    color=group_totals['color'], alpha=0.8,
                    edgecolor='black', linewidth=1.5)

    ax2.set_yticks(range(len(group_totals)))
    ax2.set_yticklabels(group_totals['group_name'], fontsize=12, fontweight='bold')
    ax2.set_xlabel('Total Importance (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Feature Category Importance', fontsize=13, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3, linestyle='--')

    # Add percentage labels
    for i, (idx, row) in enumerate(group_totals.iterrows()):
        ax2.text(row['importance_pct'] + 1, i,
                f"{row['importance_pct']:.1f}%",
                va='center', fontsize=11, fontweight='bold')

    # Add description annotations
    for i, (idx, row) in enumerate(group_totals.iterrows()):
        group = row['group']
        desc = FEATURE_GROUPS.get(group, {}).get('description', '')
        ax2.text(0.5, i, desc, va='center', fontsize=9,
                style='italic', color='gray')

    plt.tight_layout()
    return fig


def create_compact_summary(df):
    """
    Create a compact, publication-ready chart for papers/presentations.
    Shows category breakdown and top 10 individual features.
    """
    fig = plt.figure(figsize=(16, 8))

    # Create grid for subplots with more space
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.5], wspace=0.3)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # Left: Category totals (cleaner than pie chart)
    group_totals = df.groupby(['group_name', 'color'])['importance_pct'].sum().reset_index()
    group_totals = group_totals.sort_values('importance_pct')

    bars1 = ax1.barh(range(len(group_totals)), group_totals['importance_pct'],
                     color=group_totals['color'], alpha=0.85,
                     edgecolor='black', linewidth=1.2)

    ax1.set_yticks(range(len(group_totals)))
    ax1.set_yticklabels(group_totals['group_name'], fontsize=11, fontweight='bold')
    ax1.set_xlabel('Category Importance (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Feature Categories', fontsize=12, fontweight='bold', pad=15)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')

    # Add percentage labels
    for i, (idx, row) in enumerate(group_totals.iterrows()):
        ax1.text(row['importance_pct'] + 1.5, i,
                f"{row['importance_pct']:.1f}%",
                va='center', fontsize=10, fontweight='bold')

    # Set x-axis limit to ensure labels fit
    ax1.set_xlim(0, group_totals['importance_pct'].max() * 1.15)

    # Right: Top 10 individual features
    df_top10 = df.nlargest(10, 'importance_pct').sort_values('importance_pct')

    bars2 = ax2.barh(range(len(df_top10)), df_top10['importance_pct'],
                     color=df_top10['color'], alpha=0.85,
                     edgecolor='black', linewidth=0.8)

    ax2.set_yticks(range(len(df_top10)))
    ax2.set_yticklabels(df_top10['display_name'], fontsize=10)
    ax2.set_xlabel('Feature Importance (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Top 10 Most Important Features', fontsize=12, fontweight='bold', pad=15)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')

    # Add percentage labels
    for i, (idx, row) in enumerate(df_top10.iterrows()):
        ax2.text(row['importance_pct'] + 0.5, i,
                f"{row['importance_pct']:.1f}%",
                va='center', fontsize=9, fontweight='bold')

    # Set x-axis limit to ensure labels fit
    ax2.set_xlim(0, df_top10['importance_pct'].max() * 1.12)

    # Add title to figure with more space
    fig.suptitle('XGBoost Feature Importance Summary',
                 fontsize=14, fontweight='bold', y=0.96)

    plt.tight_layout(rect=[0, 0, 1, 0.94])  # Leave room for suptitle
    return fig


def print_summary_stats(df):
    """Print summary statistics about feature importance."""
    print("\n" + "="*60)
    print("Feature Importance Summary")
    print("="*60)

    # Group totals
    group_totals = df.groupby('group_name')['importance_pct'].sum().sort_values(ascending=False)
    print("\nImportance by Category:")
    for group, pct in group_totals.items():
        print(f"  {group}: {pct:.1f}%")

    # Top features
    print(f"\nTop 10 Most Important Features:")
    for i, (idx, row) in enumerate(df.nlargest(10, 'importance_pct').iterrows(), 1):
        print(f"  {i:2d}. {row['display_name']:30s} {row['importance_pct']:5.1f}% ({row['group_name']})")


def main():
    """Main workflow."""
    print("="*60)
    print("Generating Feature Importance Visualizations")
    print("="*60)

    # Load and process data
    df = load_feature_importance()
    df = assign_feature_groups(df)

    # Print summary
    print_summary_stats(df)

    # Create visualizations
    print("\n\nCreating visualizations...")

    # 1. Horizontal bar chart (clearest)
    print("  1. Creating horizontal bar chart (top 20)...")
    fig1 = create_horizontal_bar_chart(df, top_n=20)
    output1 = OUTPUT_DIR / "xgb_feature_importance_bars.png"
    fig1.savefig(output1, dpi=300, bbox_inches='tight')
    print(f"     ✓ Saved: {output1}")
    plt.close(fig1)

    # 2. Grouped summary
    print("  2. Creating grouped category summary...")
    fig2 = create_grouped_summary(df)
    output2 = OUTPUT_DIR / "xgb_feature_importance_grouped.png"
    fig2.savefig(output2, dpi=300, bbox_inches='tight')
    print(f"     ✓ Saved: {output2}")
    plt.close(fig2)

    # 3. Compact summary (best for README)
    print("  3. Creating compact summary...")
    fig3 = create_compact_summary(df)
    output3 = OUTPUT_DIR / "xgb_feature_importance_summary.png"
    fig3.savefig(output3, dpi=300, bbox_inches='tight')
    print(f"     ✓ Saved: {output3}")
    plt.close(fig3)

    print("\n" + "="*60)
    print("Visualization complete!")
    print("="*60)
    print("\nRecommendation for README:")
    print("  Use: xgb_feature_importance_summary.png (compact, publication-ready)")
    print("  Or:  xgb_feature_importance_bars.png (detailed, easier to read)")


if __name__ == "__main__":
    main()
