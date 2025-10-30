"""
Generate improved severity visualization showing relationship between event types and severity.

Creates a clearer alternative to the sparse heatmap.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from pathlib import Path

# Configuration
DB_PATH = Path("database/az511.db")
OUTPUT_DIR = Path("images")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_event_data():
    """Load accident/incident events from AZ511 database."""
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    print(f"Loading event data from {DB_PATH}...")

    conn = sqlite3.connect(DB_PATH)

    # Query for accidents and incidents
    query = """
    SELECT
        EventType,
        EventSubType,
        Severity,
        Description
    FROM events
    WHERE EventType = 'accidentsAndIncidents'
    """

    df = pd.read_sql(query, conn)
    conn.close()

    print(f"Loaded {len(df)} accident/incident records")

    # Clean up severity values
    df['Severity'] = df['Severity'].fillna('None')

    # Clean up EventSubType
    df['EventSubType'] = df['EventSubType'].fillna('Unknown')

    return df


def create_severity_stacked_bars(df, top_n=15):
    """
    Creates a stacked horizontal bar chart showing severity distribution
    for the most common event types.
    """
    # Count events by subtype and severity
    severity_counts = df.groupby(['EventSubType', 'Severity']).size().reset_index(name='count')

    # Get top N event subtypes by total count
    top_subtypes = df['EventSubType'].value_counts().head(top_n).index.tolist()
    severity_counts = severity_counts[severity_counts['EventSubType'].isin(top_subtypes)]

    # Pivot for stacked bar chart
    pivot = severity_counts.pivot(index='EventSubType', columns='Severity', values='count')
    pivot = pivot.fillna(0)

    # Sort by total count
    pivot['total'] = pivot.sum(axis=1)
    pivot = pivot.sort_values('total', ascending=True)
    pivot = pivot.drop('total', axis=1)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Color scheme: None (gray), Minor (yellow), Major (red)
    colors = {
        'None': '#CCCCCC',
        'Minor': '#FDB462',
        'Major': '#E41A1C'
    }

    # Order severity levels
    severity_order = ['None', 'Minor', 'Major']
    columns_to_plot = [col for col in severity_order if col in pivot.columns]

    # Create stacked horizontal bars
    left = np.zeros(len(pivot))
    for severity in columns_to_plot:
        if severity in pivot.columns:
            ax.barh(range(len(pivot)), pivot[severity], left=left,
                   label=severity, color=colors.get(severity, '#999999'),
                   edgecolor='black', linewidth=0.5)
            left += pivot[severity]

    # Formatting
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index, fontsize=10)
    ax.set_xlabel('Number of Events', fontsize=12, fontweight='bold')
    ax.set_title(f'Event Severity Distribution (Top {top_n} Event Types)',
                fontsize=14, fontweight='bold', pad=15)
    ax.legend(title='Severity', loc='lower right', framealpha=0.9, edgecolor='black')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    plt.tight_layout()
    return fig


def create_severity_grouped_bars(df, top_n=10):
    """
    Creates a grouped bar chart showing each severity level separately.
    Easier to compare severity levels across event types.
    """
    # Count events by subtype and severity
    severity_counts = df.groupby(['EventSubType', 'Severity']).size().reset_index(name='count')

    # Get top N event subtypes
    top_subtypes = df['EventSubType'].value_counts().head(top_n).index.tolist()
    plot_data = severity_counts[severity_counts['EventSubType'].isin(top_subtypes)]

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Colors
    colors = {
        'None': '#CCCCCC',
        'Minor': '#FDB462',
        'Major': '#E41A1C'
    }

    # Get unique severities and subtypes
    severities = ['None', 'Minor', 'Major']
    severities = [s for s in severities if s in plot_data['Severity'].unique()]

    # Calculate bar positions
    x = np.arange(len(top_subtypes))
    width = 0.25

    # Plot bars for each severity
    for i, severity in enumerate(severities):
        severity_data = plot_data[plot_data['Severity'] == severity]
        counts = [severity_data[severity_data['EventSubType'] == st]['count'].sum()
                 for st in top_subtypes]

        offset = width * (i - len(severities)/2 + 0.5)
        ax.bar(x + offset, counts, width, label=severity,
              color=colors.get(severity, '#999999'),
              edgecolor='black', linewidth=0.5)

    # Formatting
    ax.set_xlabel('Event Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Events', fontsize=12, fontweight='bold')
    ax.set_title(f'Event Counts by Severity (Top {top_n} Event Types)',
                fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(top_subtypes, rotation=45, ha='right', fontsize=10)
    ax.legend(title='Severity', loc='upper right', framealpha=0.9, edgecolor='black')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    plt.tight_layout()
    return fig


def create_severity_summary(df):
    """
    Creates a simple summary chart showing overall severity distribution
    and the top issue with missing severity data.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Overall severity distribution
    severity_counts = df['Severity'].value_counts()
    colors = {'None': '#CCCCCC', 'Minor': '#FDB462', 'Major': '#E41A1C'}
    bar_colors = [colors.get(sev, '#999999') for sev in severity_counts.index]

    ax1.bar(range(len(severity_counts)), severity_counts.values,
           color=bar_colors, edgecolor='black', linewidth=1.5, alpha=0.85)
    ax1.set_xticks(range(len(severity_counts)))
    ax1.set_xticklabels(severity_counts.index, fontsize=11, fontweight='bold')
    ax1.set_ylabel('Number of Events', fontsize=12, fontweight='bold')
    ax1.set_title('Overall Severity Distribution', fontsize=13, fontweight='bold', pad=15)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Add percentage labels
    total = severity_counts.sum()
    for i, (count, pct) in enumerate(zip(severity_counts.values, severity_counts.values/total*100)):
        ax1.text(i, count + max(severity_counts.values)*0.02,
                f'{count}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Right: Missing severity by event type (top 10)
    missing_severity = df[df['Severity'] == 'None']
    top_missing = missing_severity['EventSubType'].value_counts().head(10)

    y_pos = range(len(top_missing))
    ax2.barh(y_pos, top_missing.values, color='#CCCCCC',
            edgecolor='black', linewidth=0.8, alpha=0.85)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(top_missing.index, fontsize=10)
    ax2.set_xlabel('Events Missing Severity', fontsize=12, fontweight='bold')
    ax2.set_title('Event Types with Missing Severity Data', fontsize=13, fontweight='bold', pad=15)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)

    # Add count labels
    for i, count in enumerate(top_missing.values):
        ax2.text(count + max(top_missing.values)*0.02, i,
                f'{count}', va='center', fontsize=9, fontweight='bold')

    fig.suptitle('Severity Data Quality Issues', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    return fig


def print_summary_stats(df):
    """Print summary statistics about severity distribution."""
    print("\n" + "="*60)
    print("Severity Distribution Summary")
    print("="*60)

    total = len(df)
    severity_counts = df['Severity'].value_counts()

    print(f"\nTotal accident/incident events: {total}")
    print("\nSeverity breakdown:")
    for severity, count in severity_counts.items():
        pct = (count / total) * 100
        print(f"  {severity}: {count} ({pct:.1f}%)")

    # Most common event types
    print("\nTop 10 most common event types:")
    for i, (event_type, count) in enumerate(df['EventSubType'].value_counts().head(10).items(), 1):
        pct = (count / total) * 100
        print(f"  {i:2d}. {event_type:30s} {count:5d} ({pct:.1f}%)")

    # Severity by top event types
    print("\nSeverity distribution for top 5 event types:")
    top_5 = df['EventSubType'].value_counts().head(5).index
    for event_type in top_5:
        subset = df[df['EventSubType'] == event_type]
        severity_dist = subset['Severity'].value_counts()
        print(f"\n  {event_type}:")
        for sev, count in severity_dist.items():
            pct = (count / len(subset)) * 100
            print(f"    {sev}: {count} ({pct:.1f}%)")


def main():
    """Main workflow."""
    print("="*60)
    print("Generating Severity Visualizations")
    print("="*60)

    # Load data
    df = load_event_data()

    # Print summary
    print_summary_stats(df)

    # Create visualizations
    print("\n\nCreating visualizations...")

    # 1. Stacked horizontal bars (best overview)
    print("  1. Creating stacked bar chart...")
    fig1 = create_severity_stacked_bars(df, top_n=15)
    output1 = OUTPUT_DIR / "severity_stacked_bars.png"
    fig1.savefig(output1, dpi=300, bbox_inches='tight')
    print(f"     ✓ Saved: {output1}")
    plt.close(fig1)

    # 2. Grouped bars (easier comparison)
    print("  2. Creating grouped bar chart...")
    fig2 = create_severity_grouped_bars(df, top_n=10)
    output2 = OUTPUT_DIR / "severity_grouped_bars.png"
    fig2.savefig(output2, dpi=300, bbox_inches='tight')
    print(f"     ✓ Saved: {output2}")
    plt.close(fig2)

    # 3. Summary with data quality issues
    print("  3. Creating summary with data quality analysis...")
    fig3 = create_severity_summary(df)
    output3 = OUTPUT_DIR / "severity_summary.png"
    fig3.savefig(output3, dpi=300, bbox_inches='tight')
    print(f"     ✓ Saved: {output3}")
    plt.close(fig3)

    print("\n" + "="*60)
    print("Visualization complete!")
    print("="*60)
    print("\nRecommendation for README:")
    print("  Use: severity_stacked_bars.png (shows distribution clearly)")
    print("  Or:  severity_summary.png (highlights missing data issue)")


if __name__ == "__main__":
    main()
