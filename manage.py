"""Tiny CLI wrapper to consolidate common script entrypoints.

This lightweight wrapper provides a `compare` subcommand that calls
the existing `model_comparison.py` script while keeping its CLI intact.

Usage examples:
  python manage.py compare --direction WB --save-figs
  python manage.py compare --direction EB
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser(prog="manage.py")
    sub = p.add_subparsers(dest="cmd")

    c = sub.add_parser("compare", help="Run the model_comparison.py script (plots/eval)")
    c.add_argument("--direction", choices=["WB", "EB"], default="WB")
    c.add_argument("--save-figs", action="store_true")
    c.add_argument("--skip-heatmaps", action="store_true")

    args = p.parse_args(argv)

    if args.cmd == "compare":
        # Build argv as model_comparison.py expects, then import and run its main()
        mc_argv = ["model_comparison.py", "--direction", args.direction]
        if args.save_figs:
            mc_argv.append("--save-figs")
        if args.skip_heatmaps:
            mc_argv.append("--skip-heatmaps")

        # Importing and calling the script's main() allows reuse without spawning a subprocess.
        # We temporarily replace sys.argv so its internal argparse parses correctly.
        import importlib
        try:
            mc = importlib.import_module("model_comparison")
        except Exception as exc:
            print(f"Could not import model_comparison.py: {exc}")
            raise

        old_argv = sys.argv
        try:
            sys.argv = mc_argv
            mc.main()
        finally:
            sys.argv = old_argv
    else:
        p.print_help()


if __name__ == "__main__":
    main()
