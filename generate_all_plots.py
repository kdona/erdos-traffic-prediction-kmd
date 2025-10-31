"""
Master script to generate ALL visualization plots for the project.

This consolidates plot generation from:
1. Day-of-week patterns (speed + accidents)
2. Model comparison charts (feature ablation, full comparison, heatmaps)
3. Travel time predictions (ST-GNN comparisons, error analysis)
4. Event impact analysis (boxplots, correlation, delay heatmaps)
5. Model comparison with prediction heatmaps

Usage:
    python generate_all_plots.py                    # Generate all plots
    python generate_all_plots.py --skip-slow        # Skip slow plots (day-of-week, event impact)
    python generate_all_plots.py --only MODEL_COMP  # Only generate specific plot types

Plot types: DOW, MODEL_COMP, TRAVEL_TIME, EVENT_IMPACT, COMPARISON_HEATMAPS
"""
import sys
import argparse
from pathlib import Path
import importlib

# Import visualization utilities
from visualization_utils import print_section_header, print_completion_message


# =============================================================================
# Plot Generation Functions
# =============================================================================

def generate_dow_patterns():
    """Generate day-of-week pattern visualizations (speed + accidents)."""
    print_section_header("Generating Day-of-Week Patterns")
    try:
        # Import and run the combined day-of-week script
        dow_viz = importlib.import_module("create_day_of_week_viz")
        dow_viz.main()
        print("✓ Day-of-week patterns generated successfully")
    except Exception as e:
        print(f"✗ Error generating day-of-week patterns: {e}")
        return False
    return True


def generate_model_comparison():
    """Generate model comparison charts (feature ablation, full comparison, heatmap)."""
    print_section_header("Generating Model Comparison Charts")
    try:
        # Import and run model comparison visualization
        mc_viz = importlib.import_module("create_model_comparison_viz")
        mc_viz.main()
        print("✓ Model comparison charts generated successfully")
    except Exception as e:
        print(f"✗ Error generating model comparison charts: {e}")
        return False
    return True


def generate_travel_time_viz():
    """Generate travel time prediction visualizations (ST-GNN)."""
    print_section_header("Generating Travel Time Visualizations")
    try:
        # Import and run travel time visualization
        tt_viz = importlib.import_module("create_travel_time_viz")
        tt_viz.main()
        print("✓ Travel time visualizations generated successfully")
    except Exception as e:
        print(f"✗ Error generating travel time visualizations: {e}")
        return False
    return True


def generate_event_impact():
    """Generate event impact analysis plots (boxplots, correlations, delay heatmaps)."""
    print_section_header("Generating Event Impact Analysis")
    try:
        # Import and run event impact analysis
        evt_analysis = importlib.import_module("evt_impact_analysis")
        result = evt_analysis.main()
        if result == 0:
            print("✓ Event impact analysis generated successfully")
            return True
        else:
            print(f"✗ Event impact analysis returned non-zero exit code: {result}")
            return False
    except Exception as e:
        print(f"✗ Error generating event impact analysis: {e}")
        return False


def generate_comparison_heatmaps(direction="WB", skip_heatmaps=False):
    """Generate model comparison with prediction heatmaps."""
    print_section_header(f"Generating Model Comparison Heatmaps ({direction})")
    try:
        # Import model_comparison module
        model_comp = importlib.import_module("model_comparison")

        # Build argv for model_comparison
        old_argv = sys.argv
        try:
            mc_argv = ["model_comparison.py", "--direction", direction, "--save-figs"]
            if skip_heatmaps:
                mc_argv.append("--skip-heatmaps")

            sys.argv = mc_argv
            model_comp.main()
            print(f"✓ Model comparison heatmaps ({direction}) generated successfully")
            return True
        finally:
            sys.argv = old_argv

    except Exception as e:
        print(f"✗ Error generating comparison heatmaps: {e}")
        return False


def generate_feature_importance():
    """Generate XGBoost feature importance visualizations."""
    print_section_header("Generating Feature Importance Visualizations")
    try:
        # Import and run feature importance script
        fi_viz = importlib.import_module("create_feature_importance_viz")
        fi_viz.main()
        print("✓ Feature importance visualizations generated successfully")
        return True
    except Exception as e:
        print(f"✗ Error generating feature importance visualizations: {e}")
        return False


def generate_severity_viz():
    """Generate event severity distribution visualizations."""
    print_section_header("Generating Severity Visualizations")
    try:
        # Import and run severity visualization script
        sev_viz = importlib.import_module("create_severity_visualization")
        sev_viz.main()
        print("✓ Severity visualizations generated successfully")
        return True
    except Exception as e:
        print(f"✗ Error generating severity visualizations: {e}")
        return False


# =============================================================================
# Main Workflow
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate all visualization plots for the traffic prediction project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Plot Types:
  DOW                 - Day-of-week patterns (speed + accidents) [SLOW]
  MODEL_COMP          - Model comparison charts (feature ablation, full comparison, heatmap)
  TRAVEL_TIME         - Travel time prediction visualizations (ST-GNN)
  EVENT_IMPACT        - Event impact analysis (boxplots, correlations, delay heatmaps) [SLOW]
  COMPARISON_HEATMAPS - Model comparison with prediction heatmaps
  FEATURE_IMPORTANCE  - XGBoost feature importance (requires trained model)
  SEVERITY            - Event severity distribution

Examples:
  python generate_all_plots.py                           # Generate all plots
  python generate_all_plots.py --skip-slow               # Skip slow plots (DOW, EVENT_IMPACT)
  python generate_all_plots.py --only MODEL_COMP         # Only generate model comparison
  python generate_all_plots.py --skip EVENT_IMPACT       # Skip event impact analysis
  python generate_all_plots.py --direction EB            # Use eastbound TMC order
        """
    )

    p.add_argument(
        "--only",
        choices=["DOW", "MODEL_COMP", "TRAVEL_TIME", "EVENT_IMPACT", "COMPARISON_HEATMAPS", "FEATURE_IMPORTANCE", "SEVERITY"],
        help="Only generate a specific plot type"
    )

    p.add_argument(
        "--skip",
        action="append",
        choices=["DOW", "MODEL_COMP", "TRAVEL_TIME", "EVENT_IMPACT", "COMPARISON_HEATMAPS", "FEATURE_IMPORTANCE", "SEVERITY"],
        help="Skip specific plot types (can be used multiple times)"
    )

    p.add_argument(
        "--skip-slow",
        action="store_true",
        help="Skip slow plot generation (DOW, EVENT_IMPACT)"
    )

    p.add_argument(
        "--skip-heatmaps",
        action="store_true",
        help="Skip heatmaps in comparison plots (faster)"
    )

    p.add_argument(
        "--direction",
        choices=["WB", "EB"],
        default="WB",
        help="TMC direction for comparison heatmaps (default: WB)"
    )

    return p.parse_args()


def main():
    args = parse_args()

    # Determine which plots to generate
    skip_set = set(args.skip or [])
    if args.skip_slow:
        skip_set.update(["DOW", "EVENT_IMPACT"])

    # Track results
    results = {}

    print("="*70)
    print(" "*15 + "TRAFFIC PREDICTION VISUALIZATION GENERATOR")
    print("="*70)
    print()

    # If --only is specified, generate only that plot type
    if args.only:
        plot_type = args.only
        print(f"Generating only: {plot_type}\n")

        if plot_type == "DOW":
            results["DOW"] = generate_dow_patterns()
        elif plot_type == "MODEL_COMP":
            results["MODEL_COMP"] = generate_model_comparison()
        elif plot_type == "TRAVEL_TIME":
            results["TRAVEL_TIME"] = generate_travel_time_viz()
        elif plot_type == "EVENT_IMPACT":
            results["EVENT_IMPACT"] = generate_event_impact()
        elif plot_type == "COMPARISON_HEATMAPS":
            results["COMPARISON_HEATMAPS"] = generate_comparison_heatmaps(
                direction=args.direction,
                skip_heatmaps=args.skip_heatmaps
            )
        elif plot_type == "FEATURE_IMPORTANCE":
            results["FEATURE_IMPORTANCE"] = generate_feature_importance()
        elif plot_type == "SEVERITY":
            results["SEVERITY"] = generate_severity_viz()
    else:
        # Generate all plots (except skipped ones)

        if "DOW" not in skip_set:
            print("\n[1/5] Day-of-Week Patterns")
            results["DOW"] = generate_dow_patterns()
        else:
            print("\n[1/5] Day-of-Week Patterns - SKIPPED")

        if "MODEL_COMP" not in skip_set:
            print("\n[2/5] Model Comparison Charts")
            results["MODEL_COMP"] = generate_model_comparison()
        else:
            print("\n[2/5] Model Comparison Charts - SKIPPED")

        if "TRAVEL_TIME" not in skip_set:
            print("\n[3/5] Travel Time Visualizations")
            results["TRAVEL_TIME"] = generate_travel_time_viz()
        else:
            print("\n[3/5] Travel Time Visualizations - SKIPPED")

        if "EVENT_IMPACT" not in skip_set:
            print("\n[4/5] Event Impact Analysis")
            results["EVENT_IMPACT"] = generate_event_impact()
        else:
            print("\n[4/5] Event Impact Analysis - SKIPPED")

        if "COMPARISON_HEATMAPS" not in skip_set:
            print("\n[5/7] Model Comparison Heatmaps")
            results["COMPARISON_HEATMAPS"] = generate_comparison_heatmaps(
                direction=args.direction,
                skip_heatmaps=args.skip_heatmaps
            )
        else:
            print("\n[5/7] Model Comparison Heatmaps - SKIPPED")

        if "FEATURE_IMPORTANCE" not in skip_set:
            print("\n[6/7] Feature Importance")
            results["FEATURE_IMPORTANCE"] = generate_feature_importance()
        else:
            print("\n[6/7] Feature Importance - SKIPPED")

        if "SEVERITY" not in skip_set:
            print("\n[7/7] Severity Distributions")
            results["SEVERITY"] = generate_severity_viz()
        else:
            print("\n[7/7] Severity Distributions - SKIPPED")

    # Print summary
    print("\n")
    print_section_header("SUMMARY")

    successful = [k for k, v in results.items() if v]
    failed = [k for k, v in results.items() if not v]
    skipped = [k for k in ["DOW", "MODEL_COMP", "TRAVEL_TIME", "EVENT_IMPACT", "COMPARISON_HEATMAPS", "FEATURE_IMPORTANCE", "SEVERITY"]
               if k not in results]

    if successful:
        print(f"\n✓ Successful ({len(successful)}):")
        for name in successful:
            print(f"  • {name}")

    if failed:
        print(f"\n✗ Failed ({len(failed)}):")
        for name in failed:
            print(f"  • {name}")

    if skipped:
        print(f"\n⊘ Skipped ({len(skipped)}):")
        for name in skipped:
            print(f"  • {name}")

    print("\nAll output files saved to: images/")

    print_completion_message("Visualization Generation")

    # Return exit code based on results
    if failed:
        return 1  # Some plots failed
    return 0  # All attempted plots succeeded


if __name__ == "__main__":
    sys.exit(main())
