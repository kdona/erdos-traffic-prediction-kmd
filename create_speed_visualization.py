"""
Generate improved traffic speed visualizations for README.

This script creates an overlaid line plot showing speed patterns by day of week.
Much clearer than a heatmap for showing temporal patterns.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from pathlib import Path

# Configuration
INRIX_FILE_NAME = "I10-and-I17-1year"  # Change to your data file
DATA_DIR = Path(f'database/inrix-traffic-speed/{INRIX_FILE_NAME}')
CHUNKS_DIR = DATA_DIR / "chunks"
OUTPUT_DIR = Path("images")
OUTPUT_DIR.mkdir(exist_ok=True)


def process_chunk_file(file_path):
    """Read and aggregate a chunk file to hourly intervals."""
    chunk_df = pd.read_csv(file_path)
    chunk_df['measurement_tstamp'] = pd.to_datetime(chunk_df['measurement_tstamp'])
    chunk_df = chunk_df[['tmc_code', 'measurement_tstamp', 'speed']]

    # Round to hour
    chunk_df['hour'] = chunk_df['measurement_tstamp'].dt.floor('h')

    # Aggregate by tmc_code and hour
    agg_df = chunk_df.groupby(['tmc_code', 'hour']).agg(
        speed_mean=('speed', 'mean'),
        count=('speed', 'count')
    ).reset_index()

    return agg_df


def load_all_data():
    """Load and aggregate all INRIX data chunks."""
    all_chunk_files = sorted(glob.glob(str(CHUNKS_DIR / "*.csv")))

    if not all_chunk_files:
        # If no chunks, try loading the main file directly (slower)
        main_file = DATA_DIR / f"{INRIX_FILE_NAME}.csv"
        if main_file.exists():
            print(f"Loading data from {main_file}...")
            df = pd.read_csv(main_file, parse_dates=['measurement_tstamp'])
            df = df[['tmc_code', 'measurement_tstamp', 'speed']]
            df['hour'] = df['measurement_tstamp'].dt.floor('h')
            df_1y = df.groupby(['tmc_code', 'hour'])['speed'].mean().reset_index()
            df_1y.columns = ['tmc_code', 'hour', 'speed_mean']
        else:
            raise FileNotFoundError(f"No data found in {CHUNKS_DIR} or {main_file}")
    else:
        print(f"Found {len(all_chunk_files)} chunk files. Processing...")
        df_chunks = []

        for i, file_path in enumerate(all_chunk_files):
            if (i + 1) % 5 == 0 or i == 0:
                print(f"  Processing {i+1}/{len(all_chunk_files)}")
            df_chunks.append(process_chunk_file(file_path))

        # Combine all chunks
        df_1y = pd.concat(df_chunks, ignore_index=True)

        # Re-aggregate in case of duplicates
        df_1y = df_1y.groupby(['tmc_code', 'hour']).agg(
            speed_mean=('speed_mean', 'mean'),
            count=('count', 'sum')
        ).reset_index()

    # Add time features
    df_1y['hour_of_day'] = df_1y['hour'].dt.hour
    df_1y['day_of_week'] = df_1y['hour'].dt.dayofweek
    df_1y['day_name'] = df_1y['hour'].dt.day_name()

    print(f"Loaded {len(df_1y)} aggregated hourly records")
    return df_1y


def create_overlaid_speed_plot(df):
    """
    Creates an overlaid line plot showing speed by hour for each day of week.
    Weekdays in blue, weekends in orange - makes patterns jump out visually.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Define colors: blue for weekdays, orange for weekends
    colors = ['#1f77b4'] * 5 + ['#ff7f0e'] * 2
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    day_full_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    for idx, (day_name, day_short, color) in enumerate(zip(day_full_names, days, colors)):
        # Filter data for this day
        day_data = df[df['day_name'] == day_name]

        if len(day_data) == 0:
            continue

        # Calculate hourly average
        hourly_avg = day_data.groupby('hour_of_day')['speed_mean'].mean().sort_index()

        # Thicker lines for weekends, thinner for weekdays
        linewidth = 3 if idx >= 5 else 1.5
        alpha = 1.0 if idx >= 5 else 0.7

        ax.plot(hourly_avg.index, hourly_avg.values,
                label=day_short, linewidth=linewidth, alpha=alpha, color=color,
                marker='o' if idx >= 5 else None, markersize=4)

    ax.set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Speed (mph)', fontsize=12, fontweight='bold')
    ax.set_title('Traffic Speed Patterns by Day of Week', fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='best', ncol=7, framealpha=0.9, edgecolor='gray')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, 23)
    ax.set_xticks(range(0, 24, 2))

    # Add subtle background shading for rush hours
    ax.axvspan(7, 9, alpha=0.05, color='red', label='_nolegend_')  # Morning rush
    ax.axvspan(16, 18, alpha=0.05, color='red', label='_nolegend_')  # Evening rush

    plt.tight_layout()
    return fig


def main():
    """Main workflow."""
    print("="*60)
    print("Generating Traffic Speed Visualizations")
    print("="*60)

    # Load data
    df = load_all_data()

    # Create overlaid line plot (better than heatmap for time series)
    print("\nCreating overlaid speed plot by day of week...")
    fig = create_overlaid_speed_plot(df)

    # Save the figure
    output_path = OUTPUT_DIR / "speed_by_dow_lines.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")

    plt.close(fig)

    print("\n" + "="*60)
    print("Visualization complete!")
    print("="*60)


if __name__ == "__main__":
    main()
