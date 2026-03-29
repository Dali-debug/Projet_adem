"""
run_nilm.py
-----------
Main entry point for the REFIT NILM pipeline.

This script orchestrates the full pipeline:
  1. Load and preprocess a REFIT house CSV.
  2. Train per-appliance Gaussian HMMs (saved to ``models/``).
  3. Decode the state of each appliance at every time step.
  4. Generate and save plots to ``plots/``.
  5. Print a state-summary table.

Usage
-----
    # Full pipeline (train + disaggregate):
    python run_nilm.py --house ../Processed_Data_CSV/House_3.csv

    # Training only:
    python run_nilm.py --house ../Processed_Data_CSV/House_3.csv --mode train

    # Disaggregation only (models must exist):
    python run_nilm.py --house ../Processed_Data_CSV/House_3.csv --mode disaggregate

    # True NILM mode (use aggregate signal only, no sub-metering):
    python run_nilm.py --house ../Processed_Data_CSV/House_3.csv --nilm

    # Limit data for a quick test:
    python run_nilm.py --house ../Processed_Data_CSV/House_3.csv --limit 5000

    # Custom appliance selection:
    python run_nilm.py --house ../Processed_Data_CSV/House_3.csv \\
                       --appliances kettle microwave fridge tv
"""

import argparse
import sys
import os
import pandas as pd

from refit_metadata import parse_house_number, get_appliance_column, HOUSE_APPLIANCES
from train_hmm import run_training
from disaggregate import run_disaggregation

DEFAULT_APPLIANCES = ["kettle", "microwave", "fridge", "tv"]


# ---------------------------------------------------------------------------
# State-summary printer
# ---------------------------------------------------------------------------

def print_state_summary(results: pd.DataFrame, target_appliances: list,
                        house_number: int):
    """Print a concise state-occupancy table for every appliance."""
    print("\n" + "=" * 60)
    print(f"  STATE SUMMARY — House {house_number}")
    print("=" * 60)

    rows = []
    for appliance in target_appliances:
        label_col = f"{appliance}_state_label"
        if label_col not in results.columns:
            continue
        counts = results[label_col].value_counts()
        total = len(results)
        row = {"Appliance": appliance.capitalize()}
        for state, cnt in counts.items():
            row[state] = f"{cnt:>8,}  ({100 * cnt / total:5.1f}%)"
        rows.append(row)

    if not rows:
        print("  No results to display.")
        return

    summary_df = pd.DataFrame(rows).set_index("Appliance")
    summary_df = summary_df.fillna("—")
    print(summary_df.to_string())
    print("=" * 60)


def print_recent_states(results: pd.DataFrame, target_appliances: list,
                        n_rows: int = 10):
    """Print the last *n_rows* state labels for each appliance."""
    label_cols = [f"{a}_state_label" for a in target_appliances
                  if f"{a}_state_label" in results.columns]
    if not label_cols:
        return

    print(f"\n--- Last {n_rows} samples ---")
    display = results[label_cols].tail(n_rows)
    display.columns = [c.replace("_state_label", "").capitalize()
                       for c in display.columns]
    print(display.to_string())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "REFIT NILM Pipeline — Disaggregate household appliances "
            "using Hidden Markov Models."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--house", required=True,
        help="Path to a REFIT house CSV file, e.g. "
             "'../Processed_Data_CSV/House_3.csv'."
    )
    parser.add_argument(
        "--appliances", nargs="+", default=DEFAULT_APPLIANCES,
        metavar="APPLIANCE",
        help=(
            "Canonical appliance names to process. "
            f"Default: {DEFAULT_APPLIANCES}. "
            "Supported: kettle, microwave, fridge, tv, washing_machine, "
            "dishwasher, tumble_dryer, toaster, freezer, computer."
        ),
    )
    parser.add_argument(
        "--mode", choices=["all", "train", "disaggregate"], default="all",
        help=(
            "'all' = train then disaggregate (default); "
            "'train' = training only; "
            "'disaggregate' = disaggregation only (requires saved models)."
        ),
    )
    parser.add_argument(
        "--nilm", action="store_true",
        help=(
            "Run in true NILM mode: estimate appliance states from the "
            "aggregate signal only (no sub-metering columns used)."
        ),
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N samples (useful for quick tests)."
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Disable plot generation."
    )
    parser.add_argument(
        "--n-states", type=int, default=None,
        help="Override number of HMM hidden states for all appliances."
    )
    parser.add_argument(
        "--sample-limit", type=int, default=50_000,
        help="Maximum training samples per appliance (default: 50000)."
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    house_csv = args.house
    if not os.path.isfile(house_csv):
        print(f"\n[ERROR] File not found: {house_csv!r}")
        print(
            "Please download the REFIT dataset and place the CSV files in "
            "Processed_Data_CSV/ — see Processed_Data_CSV/README.md for "
            "instructions."
        )
        sys.exit(1)

    house_number = parse_house_number(house_csv)
    appliances = args.appliances

    # Inform which appliances are available in this house
    if house_number in HOUSE_APPLIANCES:
        house_map = {
            f"Appliance{i+1}": name
            for i, name in enumerate(HOUSE_APPLIANCES[house_number])
        }
        print(f"\nHouse {house_number} appliance map:")
        for col, name in house_map.items():
            print(f"  {col}: {name}")
    print(f"\nTarget appliances: {appliances}")

    # ---- Training ----
    if args.mode in ("all", "train"):
        run_training(
            house_csv=house_csv,
            target_appliances=appliances,
            n_states_override=args.n_states,
            sample_limit=args.sample_limit,
        )

    # ---- Disaggregation ----
    if args.mode in ("all", "disaggregate"):
        results = run_disaggregation(
            house_csv=house_csv,
            target_appliances=appliances,
            nilm_mode=args.nilm,
            limit=args.limit,
            plot=not args.no_plot,
        )

        print_state_summary(results, appliances, house_number)
        print_recent_states(results, appliances)

        print("\n[DONE] Results available in the returned DataFrame.")
        print("       Plots saved to: plots/")
        print("       Models saved to: models/")
        return results


if __name__ == "__main__":
    main()
