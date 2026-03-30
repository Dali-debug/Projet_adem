"""
disaggregate.py
---------------
Estimate the on/off state and power level of individual appliances from
REFIT sub-metering data (or from the aggregate signal in NILM mode).

Two modes are supported:

1. **Sub-metering mode** (default, recommended for evaluation):
   Each appliance's own power column is decoded through its trained HMM.
   This gives ground-truth-quality state sequences and is suitable for
   evaluating clustering / HMM quality.

2. **NILM mode** (``--nilm`` flag):
   Only the aggregate power column is used.  A simple combinatorial
   search finds the combination of per-appliance states whose sum best
   matches the aggregate at each time step.  This is the true NILM
   scenario.

Usage
-----
    python disaggregate.py --house ../Processed_Data_CSV/House_3.csv
    python disaggregate.py --house ../Processed_Data_CSV/House_3.csv --nilm
    python disaggregate.py --house ../Processed_Data_CSV/House_3.csv \
                           --appliances kettle microwave fridge tv \
                           --limit 5000 --plot
"""

import os
import argparse
import itertools

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from preprocessing import preprocess_house
from refit_metadata import get_appliance_column, parse_house_number
from train_hmm import (
    load_models, reconstruct_hmm, DEFAULT_N_STATES, _state_labels
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode_states(model, series: pd.Series) -> np.ndarray:
    """Run Viterbi on *series* using *model* and return the state sequence."""
    X = series.values.reshape(-1, 1)
    return model.predict(X)


def _state_to_label(state_idx: int, labels: list) -> str:
    if 0 <= state_idx < len(labels):
        return labels[state_idx]
    return str(state_idx)


def _mean_power_per_state(model, state_seq: np.ndarray) -> dict:
    """Map state index → mean emission power (W) from the fitted means."""
    return {i: float(model.means_[i, 0]) for i in range(model.n_components)}


# ---------------------------------------------------------------------------
# Sub-metering disaggregation
# ---------------------------------------------------------------------------

def disaggregate_submetering(df: pd.DataFrame, models: dict,
                              house_number: int,
                              target_appliances: list) -> pd.DataFrame:
    """Decode hidden states using each appliance's own power column.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed REFIT DataFrame (all columns available).
    models : dict
        Mapping ``appliance → dict`` (from ``load_models``).
    house_number : int
    target_appliances : list of str

    Returns
    -------
    results : pd.DataFrame
        Columns: ``<appliance>_power``, ``<appliance>_state``,
        ``<appliance>_state_label`` for each appliance.
    """
    results = pd.DataFrame(index=df.index)

    for appliance in target_appliances:
        if appliance not in models:
            print(f"  [skip disagg] No model for '{appliance}'")
            continue

        col = get_appliance_column(house_number, appliance)
        if col is None or col not in df.columns:
            print(f"  [skip disagg] Column for '{appliance}' not in DataFrame")
            continue

        model_data = models[appliance]
        hmm = reconstruct_hmm(model_data)
        labels = model_data.get("state_labels", _state_labels(hmm.n_components))

        print(f"  Decoding states for '{appliance}' (col={col}) …", end=" ")
        series = df[col]
        state_seq = _decode_states(hmm, series)
        state_labels_seq = [_state_to_label(s, labels) for s in state_seq]

        results[f"{appliance}_power"] = series.values
        results[f"{appliance}_state"] = state_seq
        results[f"{appliance}_state_label"] = state_labels_seq
        print(f"OK  | unique states: {sorted(set(state_labels_seq))}")

    return results


# ---------------------------------------------------------------------------
# NILM disaggregation (aggregate-only)
# ---------------------------------------------------------------------------

def disaggregate_nilm(df: pd.DataFrame, models: dict,
                      house_number: int,
                      target_appliances: list) -> pd.DataFrame:
    """Estimate appliance states from the aggregate power signal only (NILM).

    For each time step the combination of per-appliance states whose
    predicted total power is closest to the observed aggregate is chosen.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed REFIT DataFrame.
    models : dict
    house_number : int
    target_appliances : list of str

    Returns
    -------
    results : pd.DataFrame
    """
    # Build list of (appliance, hmm, state_means, state_labels) tuples
    app_info = []
    for appliance in target_appliances:
        if appliance not in models:
            continue
        model_data = models[appliance]
        hmm = reconstruct_hmm(model_data)
        labels = model_data.get("state_labels", _state_labels(hmm.n_components))
        means = hmm.means_.flatten().tolist()
        app_info.append((appliance, hmm, means, labels))

    if not app_info:
        print("  [nilm] No models available — skipping.")
        return pd.DataFrame(index=df.index)

    aggregate = df["Aggregate"].values
    n_samples = len(aggregate)

    # Pre-compute all combinations of states
    state_ranges = [range(len(info[2])) for info in app_info]
    combinations = list(itertools.product(*state_ranges))

    # Vectorised computation of predicted aggregate power for every combination.
    # means_matrix: shape (n_appliances, n_combos) — power of each appliance in each combo
    means_matrix = np.array([
        [app_info[j][2][combo[j]] for combo in combinations]
        for j in range(len(app_info))
    ])
    combo_powers = means_matrix.sum(axis=0)  # (n_combos,)

    # Vectorised nearest-combo selection
    # aggregate: (n_samples,)  combo_powers: (n_combos,)
    diffs = np.abs(aggregate[:, None] - combo_powers[None, :])  # (n, n_combos)
    best_combo_idx = np.argmin(diffs, axis=1)                    # (n,)

    results = pd.DataFrame(index=df.index)
    results["aggregate_power"] = aggregate

    for j, (appliance, hmm, means, labels) in enumerate(app_info):
        state_seq = np.array([combinations[ci][j] for ci in best_combo_idx])
        state_labels_seq = [_state_to_label(s, labels) for s in state_seq]
        estimated_power = np.array([means[s] for s in state_seq])

        results[f"{appliance}_power_est"] = estimated_power
        results[f"{appliance}_state"] = state_seq
        results[f"{appliance}_state_label"] = state_labels_seq
        print(f"  NILM '{appliance}': unique states: {sorted(set(state_labels_seq))}")

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(results: pd.DataFrame, target_appliances: list,
                 house_number: int, plots_dir: str = None,
                 limit: int = 2000):
    """Save time-series plots for each appliance's power and decoded states.

    Parameters
    ----------
    results : pd.DataFrame
        Output from ``disaggregate_submetering`` or ``disaggregate_nilm``.
    target_appliances : list of str
    house_number : int
    plots_dir : str, optional
        Directory to write PNG files.
    limit : int
        Maximum number of samples to plot (for readability).
    """
    if plots_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        plots_dir = os.path.join(script_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    plot_df = results.iloc[:limit]
    time_axis = range(len(plot_df))

    for appliance in target_appliances:
        power_col = (f"{appliance}_power"
                     if f"{appliance}_power" in results.columns
                     else f"{appliance}_power_est")
        label_col = f"{appliance}_state_label"

        if power_col not in results.columns:
            continue

        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

        # Power plot
        axes[0].plot(time_axis, plot_df[power_col].values,
                     color="steelblue", linewidth=0.8, label="Power (W)")
        axes[0].set_ylabel("Power (W)")
        axes[0].set_title(
            f"House {house_number} — {appliance.capitalize()}: Power consumption"
        )
        axes[0].legend(loc="upper right")
        axes[0].grid(True, alpha=0.4)

        # State plot
        if label_col in plot_df.columns:
            unique_labels = sorted(set(plot_df[label_col]))
            label_to_int = {lbl: i for i, lbl in enumerate(unique_labels)}
            state_int = plot_df[label_col].map(label_to_int).values

            axes[1].step(time_axis, state_int, where="post",
                         color="darkorange", linewidth=1.2)
            axes[1].set_yticks(range(len(unique_labels)))
            axes[1].set_yticklabels(unique_labels)
            axes[1].set_ylabel("State")
            axes[1].set_xlabel("Time (samples)")
            axes[1].set_title(
                f"House {house_number} — {appliance.capitalize()}: Decoded state"
            )
            axes[1].grid(True, alpha=0.4)

        plt.tight_layout()
        out_path = os.path.join(plots_dir,
                                f"house{house_number}_{appliance}_states.png")
        plt.savefig(out_path, dpi=100)
        plt.close(fig)
        print(f"  Plot saved → {out_path}")


# ---------------------------------------------------------------------------
# Evaluation (when ground truth is available)
# ---------------------------------------------------------------------------

def evaluate_results(results: pd.DataFrame, df: pd.DataFrame,
                     models: dict, house_number: int,
                     target_appliances: list):
    """Print a simple accuracy summary comparing decoded states with a
    ground-truth state sequence derived from the same data.

    Ground-truth states are obtained by thresholding the sub-meter power:
    a sample is ON if its power exceeds 10 W.

    Parameters
    ----------
    results : pd.DataFrame
        Disaggregation output.
    df : pd.DataFrame
        Preprocessed REFIT DataFrame (contains sub-meter columns).
    models : dict
    house_number : int
    target_appliances : list of str
    """
    from refit_metadata import get_appliance_column

    print("\n--- Evaluation Summary ---")
    for appliance in target_appliances:
        state_col = f"{appliance}_state_label"
        if state_col not in results.columns:
            continue

        col = get_appliance_column(house_number, appliance)
        if col is None or col not in df.columns:
            continue

        model_data = models.get(appliance, {})
        hmm_labels = model_data.get("state_labels", ["OFF", "ON"])

        # Simple threshold ground truth: ON if power > 10 W
        gt_on = (df[col] > 10).astype(int)
        pred_on = (results[state_col] != "OFF").astype(int)

        # Align indices
        common_idx = results.index.intersection(df.index)
        gt_on = gt_on.loc[common_idx]
        pred_on = pred_on.loc[common_idx]

        accuracy = (gt_on.values == pred_on.values).mean()
        print(f"  {appliance:15s}: ON/OFF accuracy = {accuracy:.3f} "
              f"({len(common_idx):,} samples)")


# ---------------------------------------------------------------------------
# Main disaggregation pipeline
# ---------------------------------------------------------------------------

def run_disaggregation(house_csv: str,
                       target_appliances: list | None = None,
                       nilm_mode: bool = False,
                       limit: int | None = None,
                       plot: bool = True,
                       models_dir: str | None = None,
                       plots_dir: str | None = None,
                       train_house_number: int | None = None) -> pd.DataFrame:
    """Full disaggregation pipeline.

    Parameters
    ----------
    house_csv : str
        Path to REFIT house CSV (test/evaluation house).
    target_appliances : list of str, optional
        Defaults to ``["kettle", "microwave", "fridge", "tv"]``.
    nilm_mode : bool
        If True use only the aggregate column (true NILM).
    limit : int, optional
        Truncate data to this many samples (speed-up for testing).
    plot : bool
        Generate and save plots.
    models_dir : str, optional
        Directory containing the saved model JSON files.  When *None* the
        directory is derived from *train_house_number* (or *house_csv*).
    plots_dir : str, optional
        Directory to write plot PNG files.
    train_house_number : int, optional
        House number from which the models were trained.  Used to locate
        the correct ``models/<train_house_number>/`` directory for cross-house
        evaluation (train on House 9, test on House 3).  When *None* the
        house number is derived from *house_csv*.

    Returns
    -------
    pd.DataFrame
        State and power estimates for each appliance.
    """
    if target_appliances is None:
        target_appliances = ["kettle", "microwave", "fridge", "tv"]

    test_house_number = parse_house_number(house_csv)
    # If no explicit train house given, assume same-house evaluation.
    if train_house_number is None:
        train_house_number = test_house_number

    cross_house = train_house_number != test_house_number
    if cross_house:
        print(f"\n=== Disaggregating House {test_house_number} "
              f"(models from House {train_house_number}) ===")
    else:
        print(f"\n=== Disaggregating House {test_house_number} ===")

    # 1. Load and preprocess
    df = preprocess_house(house_csv)
    if limit:
        df = df.iloc[:limit]
        print(f"  Data limited to first {limit} samples.")

    # 2. Load models (from train house directory)
    print("\nLoading models …")
    models = load_models(train_house_number, target_appliances,
                         models_dir=models_dir)
    if not models:
        raise RuntimeError(
            f"No trained models found for House {train_house_number}. "
            "Run train_hmm.py first."
        )

    # 3. Disaggregate (column mapping uses test house)
    print("\nDecoding states …")
    if nilm_mode:
        results = disaggregate_nilm(df, models, test_house_number,
                                    target_appliances)
    else:
        results = disaggregate_submetering(df, models, test_house_number,
                                           target_appliances)

    # 4. Evaluate (only meaningful in sub-metering mode)
    if not nilm_mode:
        evaluate_results(results, df, models, test_house_number,
                         target_appliances)

    # 5. Plot
    if plot:
        print("\nGenerating plots …")
        plot_results(results, target_appliances, test_house_number,
                     plots_dir=plots_dir)

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Disaggregate REFIT appliances using trained HMMs."
    )
    parser.add_argument("--house", required=True,
                        help="Path to the REFIT house CSV file.")
    parser.add_argument("--appliances", nargs="+",
                        default=["kettle", "microwave", "fridge", "tv"])
    parser.add_argument("--nilm", action="store_true",
                        help="Use aggregate-only NILM mode.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of samples processed.")
    parser.add_argument("--no-plot", action="store_true",
                        help="Disable plot generation.")
    args = parser.parse_args()

    results = run_disaggregation(
        house_csv=args.house,
        target_appliances=args.appliances,
        nilm_mode=args.nilm,
        limit=args.limit,
        plot=not args.no_plot,
    )

    # Print first few rows
    print("\n--- Sample Output (first 5 rows) ---")
    print(results.head().to_string())
