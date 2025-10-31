"""
Create improved travel time prediction visualizations for README.

Instead of separate heatmaps that are hard to compare, this creates:
1. Side-by-side comparison (Truth vs Best Model)
2. Error/residual heatmap showing where predictions fail
3. Comprehensive error analysis by time and segment
4. Focused one-week detail view
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
    TMC_ORDER_WB, TT_VMIN, TT_VMAX
)

# Configuration
DATA_DIR = Path("database/i10-broadway")
MODELS_DIR = Path("models")


def load_ground_truth():
    """Load ground truth travel times from the dataset."""
    X_full = pd.read_parquet(DATA_DIR / "X_full_1h.parquet")

    # Extract travel time and reshape to time × TMC
    time_index = X_full.index.get_level_values('time_bin').unique().tolist()
    Y_tt = (
        X_full.reset_index()[['time_bin', 'tmc_code', 'tt_per_mile']]
        .pivot(index='time_bin', columns='tmc_code', values='tt_per_mile')
        .reindex(index=time_index, columns=TMC_ORDER_WB)
    )

    return Y_tt, time_index


def load_best_model_predictions():
    """Load predictions from the best performing model (ST-GNN/GCN-LSTM)."""
    # Try GCN-LSTM (ST-GNN) first - has proper format
    gcn_path = MODELS_DIR / "gcn" / "gcn_lstm_i10_wb" / "predictions.npz"

    if gcn_path.exists():
        d = np.load(gcn_path, allow_pickle=True)
        preds = d["preds"]  # [T_eval, N]
        Y_all = d["Y"]  # [T, N]
        seq_len = int(d["seq_len"])

        # ST-GNN predictions don't cover first seq_len timesteps
        # Pad with NaNs to align with full timeline
        T_eval = preds.shape[0]

        Y_pred_full = np.full_like(Y_all, fill_value=np.nan)
        Y_pred_full[seq_len:seq_len + T_eval, :] = preds

        return Y_pred_full, "ST-GNN", seq_len

    # Fallback - no predictions available
    return None, None, None


def create_side_by_side_comparison(Y_true, Y_pred, model_name, time_index, seq_len=0):
    """
    Create a side-by-side comparison of ground truth vs prediction.
    Much clearer than separate plots.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    # Use shared color scale
    vmin, vmax = TT_VMIN, TT_VMAX

    # Left: Ground Truth
    df_true = pd.DataFrame(Y_true, index=time_index, columns=TMC_ORDER_WB)
    sns.heatmap(df_true.T, cmap='plasma', vmin=vmin, vmax=vmax,
                cbar_kws={'label': 'Travel Time (sec/mile)'}, ax=axes[0],
                xticklabels=False, yticklabels=True)
    axes[0].set_title('Ground Truth', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Time', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Road Segment (TMC)', fontsize=11, fontweight='bold')

    # Right: Prediction
    df_pred = pd.DataFrame(Y_pred, index=time_index, columns=TMC_ORDER_WB)
    sns.heatmap(df_pred.T, cmap='plasma', vmin=vmin, vmax=vmax,
                cbar_kws={'label': 'Travel Time (sec/mile)'}, ax=axes[1],
                xticklabels=False, yticklabels=False)
    axes[1].set_title(f'{model_name} Prediction', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Time', fontsize=11, fontweight='bold')

    if seq_len > 0:
        # Add vertical line showing where predictions start
        axes[1].axvline(x=seq_len, color='red', linestyle='--', linewidth=2, alpha=0.7)
        axes[1].text(seq_len + 5, 2, 'Predictions start', color='red',
                    fontsize=10, fontweight='bold', rotation=90, va='bottom')

    plt.suptitle(f'Travel Time Comparison: Ground Truth vs {model_name}',
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()

    return fig


def create_error_heatmap(Y_true, Y_pred, model_name, time_index, seq_len=0):
    """
    Create a heatmap showing prediction errors (residuals).
    Clearly shows where and when the model fails.
    """
    # Calculate errors
    errors = Y_pred - Y_true

    fig, ax = plt.subplots(figsize=(14, 6))

    # Use diverging colormap centered at 0
    # Positive = overprediction (red), Negative = underprediction (blue)
    df_error = pd.DataFrame(errors, index=time_index, columns=TMC_ORDER_WB)

    # Set symmetric scale around 0
    max_error = np.nanmax(np.abs(errors))

    sns.heatmap(df_error.T, cmap='RdBu_r', center=0,
                vmin=-max_error, vmax=max_error,
                cbar_kws={'label': 'Prediction Error (sec/mile)\nRed=Overprediction, Blue=Underprediction'},
                ax=ax, xticklabels=False, yticklabels=True)

    if seq_len > 0:
        # Mark where predictions start
        ax.axvline(x=seq_len, color='black', linestyle='--', linewidth=2, alpha=0.7)

    ax.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax.set_ylabel('Road Segment (TMC)', fontsize=12, fontweight='bold')
    ax.set_title(f'{model_name} Prediction Errors: Where Does The Model Struggle?',
                fontsize=14, fontweight='bold', pad=15)

    # Add summary statistics
    mae = np.nanmean(np.abs(errors))
    rmse = np.sqrt(np.nanmean(errors**2))
    textstr = f'MAE: {mae:.2f} sec/mile\nRMSE: {rmse:.2f} sec/mile'
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    return fig


def create_error_analysis(Y_true, Y_pred, model_name, time_index):
    """
    Create detailed error analysis showing:
    - Error distribution by time of day
    - Error distribution by road segment
    """
    # Convert to numpy arrays if they're DataFrames
    if isinstance(Y_true, pd.DataFrame):
        Y_true = Y_true.values
    if isinstance(Y_pred, pd.DataFrame):
        Y_pred = Y_pred.values

    errors = Y_pred - Y_true

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Error by hour of day
    times = pd.DatetimeIndex(time_index)
    hours = times.hour
    error_by_hour = pd.DataFrame({
        'hour': np.repeat(hours, errors.shape[1]),
        'error': errors.flatten()
    }).dropna()

    error_by_hour_stats = error_by_hour.groupby('hour')['error'].agg(['mean', 'std', 'count'])

    axes[0, 0].plot(error_by_hour_stats.index, error_by_hour_stats['mean'],
                    marker='o', linewidth=2, markersize=8, color='#e74c3c')
    axes[0, 0].fill_between(error_by_hour_stats.index,
                            error_by_hour_stats['mean'] - error_by_hour_stats['std'],
                            error_by_hour_stats['mean'] + error_by_hour_stats['std'],
                            alpha=0.3, color='#e74c3c')
    axes[0, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[0, 0].set_xlabel('Hour of Day', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('Mean Prediction Error (sec/mile)', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Error by Hour: When Does Model Struggle?', fontsize=12, fontweight='bold')
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].set_xticks(range(0, 24, 3))

    # Add rush hour shading
    axes[0, 0].axvspan(7, 9, alpha=0.1, color='red', label='Morning rush')
    axes[0, 0].axvspan(16, 18, alpha=0.1, color='red', label='Evening rush')
    axes[0, 0].legend(loc='best')

    # 2. Absolute error by hour of day
    error_by_hour['abs_error'] = error_by_hour['error'].abs()
    abs_error_by_hour = error_by_hour.groupby('hour')['abs_error'].mean()

    axes[0, 1].bar(abs_error_by_hour.index, abs_error_by_hour.values,
                   color='#3498db', alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Hour of Day', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Mean Absolute Error (sec/mile)', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Absolute Error by Hour', fontsize=12, fontweight='bold')
    axes[0, 1].grid(axis='y', alpha=0.3)
    axes[0, 1].set_xticks(range(0, 24, 3))

    # 3. Error by road segment
    error_by_segment = pd.DataFrame(errors, columns=TMC_ORDER_WB).mean().sort_values()

    axes[1, 0].barh(range(len(error_by_segment)), error_by_segment.values,
                    color=['#e74c3c' if x > 0 else '#3498db' for x in error_by_segment.values],
                    alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(x=0, color='black', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Mean Prediction Error (sec/mile)', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Road Segment', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('Error by Segment: Which Roads Are Hard to Predict?',
                        fontsize=12, fontweight='bold')
    axes[1, 0].set_yticks([])  # Too many segments to label
    axes[1, 0].grid(axis='x', alpha=0.3)

    # 4. Error distribution histogram
    axes[1, 1].hist(errors.flatten()[~np.isnan(errors.flatten())], bins=50,
                    color='#9b59b6', alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Perfect prediction')
    axes[1, 1].set_xlabel('Prediction Error (sec/mile)', fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('Error Distribution', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(axis='y', alpha=0.3)

    plt.suptitle(f'{model_name} Error Analysis: Comprehensive Breakdown',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    return fig


def create_focused_comparison(Y_true, Y_pred, model_name, time_index, seq_len=0):
    """
    Create a focused view on a specific time period (e.g., one week) for detail.
    """
    # Convert to numpy arrays if they're DataFrames
    if isinstance(Y_true, pd.DataFrame):
        Y_true = Y_true.values
    if isinstance(Y_pred, pd.DataFrame):
        Y_pred = Y_pred.values

    # Focus on middle portion of data (skip warmup period)
    start_idx = max(seq_len + 24*7, 0)  # Skip first week after warmup
    end_idx = start_idx + 24*7  # Show one week

    if end_idx > len(time_index):
        end_idx = len(time_index)
        start_idx = max(0, end_idx - 24*7)

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

    # Use shared color scale
    vmin, vmax = TT_VMIN, TT_VMAX

    # Slice data
    Y_true_slice = Y_true[start_idx:end_idx, :]
    Y_pred_slice = Y_pred[start_idx:end_idx, :]
    time_slice = time_index[start_idx:end_idx]

    # 1. Ground truth
    df_true = pd.DataFrame(Y_true_slice, index=time_slice, columns=TMC_ORDER_WB)
    sns.heatmap(df_true.T, cmap='plasma', vmin=vmin, vmax=vmax,
                cbar_kws={'label': 'Travel Time (sec/mile)'}, ax=axes[0],
                xticklabels=24, yticklabels=True)  # Show every 24th label (daily)
    axes[0].set_title('Ground Truth (One Week Detail)', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Road Segment', fontsize=11, fontweight='bold')

    # 2. Prediction
    df_pred = pd.DataFrame(Y_pred_slice, index=time_slice, columns=TMC_ORDER_WB)
    sns.heatmap(df_pred.T, cmap='plasma', vmin=vmin, vmax=vmax,
                cbar_kws={'label': 'Travel Time (sec/mile)'}, ax=axes[1],
                xticklabels=24, yticklabels=True)
    axes[1].set_title(f'{model_name} Prediction (One Week Detail)', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Road Segment', fontsize=11, fontweight='bold')

    # 3. Error
    errors_slice = Y_pred_slice - Y_true_slice
    df_error = pd.DataFrame(errors_slice, index=time_slice, columns=TMC_ORDER_WB)
    max_error = np.nanmax(np.abs(errors_slice))
    sns.heatmap(df_error.T, cmap='RdBu_r', center=0, vmin=-max_error, vmax=max_error,
                cbar_kws={'label': 'Error (sec/mile)'}, ax=axes[2],
                xticklabels=24, yticklabels=True)
    axes[2].set_title('Prediction Error (One Week Detail)', fontsize=13, fontweight='bold')
    axes[2].set_xlabel('Time', fontsize=11, fontweight='bold')
    axes[2].set_ylabel('Road Segment', fontsize=11, fontweight='bold')

    # Format x-axis labels to show dates
    for ax in axes:
        labels = ax.get_xticklabels()
        ax.set_xticklabels([label.get_text()[:10] for label in labels], rotation=45, ha='right')

    plt.suptitle(f'Detailed View: {model_name} Performance Over One Week',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    return fig


def main():
    """Main workflow."""
    print_section_header("Generating Travel Time Prediction Visualizations")

    # Load data
    print("\nLoading ground truth data...")
    Y_true, time_index = load_ground_truth()
    print(f"Loaded {Y_true.shape[0]} timesteps × {Y_true.shape[1]} segments")

    print("\nLoading best model predictions...")
    Y_pred, model_name, seq_len = load_best_model_predictions()

    if Y_pred is None:
        print("❌ Could not load model predictions. Exiting.")
        return

    print(f"Loaded {model_name} predictions")

    # 1. Side-by-side comparison
    print("\n1. Creating side-by-side comparison...")
    fig1 = create_side_by_side_comparison(Y_true, Y_pred, model_name, time_index, seq_len)
    output_path1 = save_figure(fig1, f"travel_time_comparison_{model_name.lower()}.png")
    print(f"✓ Saved: {output_path1}")

    # 2. Error heatmap
    print("\n2. Creating error heatmap...")
    fig2 = create_error_heatmap(Y_true, Y_pred, model_name, time_index, seq_len)
    output_path2 = save_figure(fig2, f"prediction_errors_{model_name.lower()}.png")
    print(f"✓ Saved: {output_path2}")

    # 3. Error analysis
    print("\n3. Creating error analysis...")
    fig3 = create_error_analysis(Y_true, Y_pred, model_name, time_index)
    output_path3 = save_figure(fig3, f"error_analysis_{model_name.lower()}.png")
    print(f"✓ Saved: {output_path3}")

    # 4. Focused one-week view
    print("\n4. Creating focused one-week comparison...")
    fig4 = create_focused_comparison(Y_true, Y_pred, model_name, time_index, seq_len)
    output_path4 = save_figure(fig4, f"detailed_comparison_{model_name.lower()}.png")
    print(f"✓ Saved: {output_path4}")

    print_completion_message("Travel Time Visualization")
    print("\nGenerated files:")
    print(f"  • {output_path1.name} - Side-by-side truth vs prediction")
    print(f"  • {output_path2.name} - Error heatmap (where model fails)")
    print(f"  • {output_path3.name} - Comprehensive error analysis")
    print(f"  • {output_path4.name} - Detailed one-week view")


if __name__ == "__main__":
    main()
