"""
Generate day-of-week pattern visualizations for README.

This script combines traffic speed and accident/incident reporting patterns,
both showing temporal patterns by day of week with unique markers and colors.

Replaces: create_speed_visualization.py + create_accident_visualization.py
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import glob
from pathlib import Path
from zoneinfo import ZoneInfo

# Import shared utilities
from visualization_utils import (
    ensure_output_dir, save_figure, print_section_header, print_completion_message,
    DOW_FULL_NAMES, DOW_SHORT_NAMES, get_dow_style, add_rush_hour_shading, style_time_plot
)


# Configuration
INRIX_FILE_NAME = "I10-and-I17-1year"  # Change to your data file
DATA_DIR = Path(f'database/inrix-traffic-speed/{INRIX_FILE_NAME}')
CHUNKS_DIR = DATA_DIR / "chunks"
DB_PATH = Path("database/az511.db")
AZ_TZ = ZoneInfo("America/Phoenix")  # Arizona timezone (no DST)


# ============================================================================
# Traffic Speed Data Loading
# ============================================================================

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


def load_speed_data():
    """Load and aggregate all INRIX traffic speed data."""
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

    print(f"Loaded {len(df_1y)} aggregated hourly speed records")
    return df_1y


# ============================================================================
# Accident Data Loading
# ============================================================================

def load_accident_data():
    """Load accident and incident events from AZ511 database."""
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    print(f"Loading accident/incident data from {DB_PATH}...")

    conn = sqlite3.connect(DB_PATH)

    # Query for accidents and incidents
    query = """
    SELECT
        ID,
        EventType,
        EventSubType,
        Reported,
        StartDate,
        LastUpdated,
        Severity,
        Description
    FROM events
    WHERE EventType = 'accidentsAndIncidents'
    """

    df = pd.read_sql(query, conn)
    conn.close()

    print(f"Loaded {len(df)} accident/incident records")

    # Convert Unix epoch timestamps to datetime
    for col in ['Reported', 'StartDate', 'LastUpdated']:
        if col in df.columns:
            # Auto-detect if seconds or milliseconds based on magnitude
            if df[col].notna().any():
                median_val = df[col].dropna().median()
                unit = 'ms' if median_val > 1e12 else 's'
                df[col] = pd.to_datetime(df[col], unit=unit, errors='coerce', utc=True)
                # Convert to Arizona time
                df[col] = df[col].dt.tz_convert(AZ_TZ)

    # Use Reported as the primary timestamp (when the event was first reported)
    df = df.dropna(subset=['Reported'])

    # Extract time components
    df['hour'] = df['Reported'].dt.hour
    df['day_of_week'] = df['Reported'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['day_name'] = df['Reported'].dt.day_name()
    df['date'] = df['Reported'].dt.date

    print(f"Date range: {df['Reported'].min()} to {df['Reported'].max()}")
    print(f"Total unique dates: {df['date'].nunique()}")

    return df


# ============================================================================
# Visualization Functions
# ============================================================================

def create_speed_by_dow_plot(df):
    """
    Create overlaid line plot showing speed patterns by day of week.
    Each day gets a unique marker and color. Shows rush hour patterns clearly.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    for idx, (day_name, day_short) in enumerate(zip(DOW_FULL_NAMES, DOW_SHORT_NAMES)):
        # Filter data for this day
        day_data = df[df['day_name'] == day_name]
        if len(day_data) == 0:
            continue

        # Calculate hourly average
        hourly_avg = day_data.groupby('hour_of_day')['speed_mean'].mean().sort_index()

        # Get styling for this day
        style = get_dow_style(idx)

        ax.plot(hourly_avg.index, hourly_avg.values,
                label=day_short,
                marker=style['marker'],
                color=style['color'],
                linewidth=style['linewidth'],
                alpha=style['alpha'],
                markersize=style['markersize'],
                markevery=2)  # Show markers every 2 hours to avoid clutter

    # Apply consistent styling
    add_rush_hour_shading(ax)
    style_time_plot(ax,
                   xlabel='Hour of Day',
                   ylabel='Average Speed (mph)',
                   title='Traffic Speed Patterns by Day of Week and Hour')

    plt.tight_layout()
    return fig


def create_accidents_by_dow_plot(df):
    """
    Create overlaid line plot showing accident reports by day of week.
    Each day gets a unique marker and color. Shows reporting patterns throughout the day.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Calculate total days for each day of week to normalize counts
    day_counts = df.groupby('day_name')['date'].nunique()

    for idx, (day_name, day_short) in enumerate(zip(DOW_FULL_NAMES, DOW_SHORT_NAMES)):
        # Filter data for this day
        day_data = df[df['day_name'] == day_name]
        if len(day_data) == 0:
            continue

        # Count accidents per hour
        hourly_counts = day_data.groupby('hour').size()

        # Normalize by number of occurrences of this day of week
        num_days = day_counts.get(day_name, 1)
        hourly_avg = hourly_counts / num_days

        # Fill in missing hours with 0
        hourly_avg = hourly_avg.reindex(range(24), fill_value=0)

        # Get styling for this day
        style = get_dow_style(idx)

        ax.plot(hourly_avg.index, hourly_avg.values,
                label=day_short,
                marker=style['marker'],
                color=style['color'],
                linewidth=style['linewidth'],
                alpha=style['alpha'],
                markersize=style['markersize'],
                markevery=2)  # Show markers every 2 hours to avoid clutter

    # Apply consistent styling
    add_rush_hour_shading(ax)
    style_time_plot(ax,
                   xlabel='Hour of Day',
                   ylabel='Average Accident Reports per Hour',
                   title='Accident/Incident Reporting Patterns by Day of Week and Hour')

    plt.tight_layout()
    return fig


def print_accident_summary_stats(df):
    """Print summary statistics about accident reporting."""
    print_section_header("Accident/Incident Reporting Summary")

    # Total counts
    print(f"Total accident/incident reports: {len(df)}")
    print(f"Date range: {df['Reported'].min().strftime('%Y-%m-%d')} to {df['Reported'].max().strftime('%Y-%m-%d')}")
    print(f"Number of days covered: {df['date'].nunique()}")

    # By day of week
    print("\nReports by Day of Week:")
    day_counts = df['day_name'].value_counts().reindex(DOW_FULL_NAMES)
    for day, count in day_counts.items():
        pct = (count / len(df)) * 100
        print(f"  {day}: {count} ({pct:.1f}%)")

    # Peak hours
    print("\nTop 5 Peak Reporting Hours:")
    hourly = df.groupby('hour').size().sort_values(ascending=False).head(5)
    for hour, count in hourly.items():
        pct = (count / len(df)) * 100
        print(f"  {hour:02d}:00 - {count} reports ({pct:.1f}%)")

    # Weekend vs weekday
    df['is_weekend'] = df['day_of_week'].isin([5, 6])
    weekend_count = df['is_weekend'].sum()
    weekday_count = len(df) - weekend_count
    print(f"\nWeekday reports: {weekday_count} ({weekday_count/len(df)*100:.1f}%)")
    print(f"Weekend reports: {weekend_count} ({weekend_count/len(df)*100:.1f}%)")


# ============================================================================
# Main Workflow
# ============================================================================

def main():
    """Main workflow."""
    print_section_header("Generating Day-of-Week Pattern Visualizations")

    # ========== Traffic Speed Visualization ==========
    print("\n[1/2] Processing traffic speed data...")
    try:
        df_speed = load_speed_data()
        print("\nCreating speed by day-of-week plot...")
        fig_speed = create_speed_by_dow_plot(df_speed)
        output_speed = save_figure(fig_speed, "speed_by_dow_lines.png")
        print(f"✓ Saved: {output_speed}")
    except Exception as e:
        print(f"⚠ Warning: Could not create speed visualization: {e}")

    # ========== Accident Reporting Visualization ==========
    print("\n[2/2] Processing accident data...")
    try:
        df_accidents = load_accident_data()
        print_accident_summary_stats(df_accidents)
        print("\nCreating accidents by day-of-week plot...")
        fig_accidents = create_accidents_by_dow_plot(df_accidents)
        output_accidents = save_figure(fig_accidents, "accidents_by_dow_lines.png")
        print(f"✓ Saved: {output_accidents}")
    except Exception as e:
        print(f"⚠ Warning: Could not create accident visualization: {e}")

    print_completion_message("Day-of-Week Visualization")


if __name__ == "__main__":
    main()
