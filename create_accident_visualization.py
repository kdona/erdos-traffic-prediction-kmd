"""
Generate accident/incident reporting patterns visualization for README.

This script creates an overlaid line plot showing when accidents are reported by day of week and hour.
Helps identify reporting patterns and potential gaps in real-time incident capture.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
from pathlib import Path
from zoneinfo import ZoneInfo

# Configuration
DB_PATH = Path("database/az511.db")
OUTPUT_DIR = Path("images")
OUTPUT_DIR.mkdir(exist_ok=True)

# Arizona timezone (no DST)
AZ_TZ = ZoneInfo("America/Phoenix")


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


def create_overlaid_accident_plot(df):
    """
    Creates an overlaid line plot showing accident reports by hour for each day of week.
    Similar to the speed visualization - weekdays in blue, weekends in orange.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Define colors: blue for weekdays, orange for weekends
    colors = ['#1f77b4'] * 5 + ['#ff7f0e'] * 2
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    day_full_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # Calculate total days for each day of week to normalize counts
    day_counts = df.groupby('day_name')['date'].nunique()

    for idx, (day_name, day_short, color) in enumerate(zip(day_full_names, days, colors)):
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

        # Thicker lines for weekends, thinner for weekdays
        linewidth = 3 if idx >= 5 else 1.5
        alpha = 1.0 if idx >= 5 else 0.7

        ax.plot(hourly_avg.index, hourly_avg.values,
                label=day_short, linewidth=linewidth, alpha=alpha, color=color,
                marker='o' if idx >= 5 else None, markersize=4)

    ax.set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Accident Reports per Hour', fontsize=12, fontweight='bold')
    ax.set_title('Accident/Incident Reporting Patterns by Day of Week', fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='best', ncol=7, framealpha=0.9, edgecolor='gray')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, 23)
    ax.set_xticks(range(0, 24, 2))

    # Add subtle background shading for typical rush hours
    ax.axvspan(7, 9, alpha=0.05, color='red', label='_nolegend_')  # Morning rush
    ax.axvspan(16, 18, alpha=0.05, color='red', label='_nolegend_')  # Evening rush

    plt.tight_layout()
    return fig


def create_summary_stats(df):
    """Print summary statistics about accident reporting."""
    print("\n" + "="*60)
    print("Accident/Incident Reporting Summary")
    print("="*60)

    # Total counts
    print(f"Total accident/incident reports: {len(df)}")
    print(f"Date range: {df['Reported'].min().strftime('%Y-%m-%d')} to {df['Reported'].max().strftime('%Y-%m-%d')}")
    print(f"Number of days covered: {df['date'].nunique()}")

    # By day of week
    print("\nReports by Day of Week:")
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_counts = df['day_name'].value_counts().reindex(day_order)
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

    # Reporting delay (if StartDate is available)
    if 'StartDate' in df.columns:
        df['report_delay'] = (df['Reported'] - df['StartDate']).dt.total_seconds() / 60  # minutes
        valid_delays = df[df['report_delay'] >= 0]['report_delay']
        if len(valid_delays) > 0:
            print(f"\nReporting Delay Statistics:")
            print(f"  Median delay: {valid_delays.median():.1f} minutes")
            print(f"  Mean delay: {valid_delays.mean():.1f} minutes")
            print(f"  95th percentile: {valid_delays.quantile(0.95):.1f} minutes")


def main():
    """Main workflow."""
    print("="*60)
    print("Generating Accident/Incident Reporting Visualizations")
    print("="*60)

    # Load data
    df = load_accident_data()

    # Print summary statistics
    create_summary_stats(df)

    # Create overlaid line plot
    print("\n\nCreating overlaid accident reporting plot by day of week...")
    fig = create_overlaid_accident_plot(df)

    # Save the figure
    output_path = OUTPUT_DIR / "accidents_by_dow_lines.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")

    plt.close(fig)

    print("\n" + "="*60)
    print("Visualization complete!")
    print("="*60)


if __name__ == "__main__":
    main()
