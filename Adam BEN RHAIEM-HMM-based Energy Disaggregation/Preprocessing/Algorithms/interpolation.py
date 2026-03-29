"""
interpolation.py
----------------
Comparison of interpolation methods for filling missing values in power
consumption data.

Run standalone:
    python interpolation.py --csv ../../../Processed_Data_CSV/House_3.csv

Requires the REFIT dataset to be available (see Processed_Data_CSV/README.md).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error


def compare_interpolations(df_smd: pd.DataFrame,
                           out_prefix: str = "interpolation"):
    """Compare interpolation methods on a power-consumption DataFrame.

    Parameters
    ----------
    df_smd : pd.DataFrame
        DataFrame with at least an 'Aggregate' column and a DatetimeIndex.
    out_prefix : str
        Prefix for output PNG filenames.
    """
    # take a segment of data points
    start = 0
    stop = 200
    start_missing = 18
    stop_missing = 22
    excerpt = df_smd.iloc[start:stop].copy()

    # remove data for 4 consecutive data points
    missing = excerpt.copy()
    missing = missing.copy(); missing.loc[missing.index[start_missing:stop_missing], 'Aggregate'] = np.nan

    # create figure
    fig, ax = plt.subplots(figsize=(14, 5))

    # plot different interpolations
    ax.plot(excerpt['Aggregate'], label='original data')
    ax.plot(missing['Aggregate'].interpolate().iloc[start_missing - 1:stop_missing + 1], label='linear')
    ax.plot(missing['Aggregate'].interpolate(method='polynomial', order=3).iloc[start_missing - 1:stop_missing + 1], label='polynomial')
    ax.plot(missing['Aggregate'].interpolate(method='spline', order=3).iloc[start_missing - 1:stop_missing + 1], label='spline')
    ax.axvspan(excerpt.index[start_missing - 1], excerpt.index[stop_missing], color='red', alpha=0.1)

    # additional formatting
    ax.xaxis.set_major_locator(mdates.HourLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    ax.xaxis.set_minor_locator(mdates.MinuteLocator(range(15, 60, 15)))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter(":%M"))
    ax.tick_params(axis='x', which='minor', labelsize=8, colors='grey')
    ax.grid()
    ax.set_xlabel('Time')
    ax.set_ylabel('Power (Watts)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_visual.png", dpi=100)
    plt.close(fig)
    print(f"Saved {out_prefix}_visual.png")

    # Randomly remove 2% of the data
    indices = np.random.choice(len(df_smd.index), size=int(0.02 * len(df_smd.index)), replace=False)
    masked_df = df_smd['Aggregate'].copy()
    masked_df.iloc[indices] = np.nan

    # Define the parameters for the different interpolations
    parameters = [
        {'method': 'linear', 'order': None},
        {'method': 'polynomial', 'order': 3},
        {'method': 'polynomial', 'order': 5},
        {'method': 'spline', 'order': 3},
    ]

    # Calculate the mean absolute error of each interpolation
    print('Mean absolute error:\n')
    for d in parameters:
        kwargs = {k: v for k, v in d.items() if v is not None}
        interpolated = masked_df.interpolate(**kwargs)
        mae = mean_absolute_error(interpolated.iloc[indices], df_smd['Aggregate'].iloc[indices])
        print(f"{d['method']} (order {d['order']}): {np.round(mae, 4)}")

    # For comparison, simply fill with zero
    mae_zeros = mean_absolute_error(
        masked_df.fillna(0).iloc[indices], df_smd['Aggregate'].iloc[indices]
    )
    print('fill zeros:', np.round(mae_zeros, 4))


if __name__ == "__main__":
    import argparse
    import sys
    import os

    parser = argparse.ArgumentParser(
        description="Compare interpolation methods on a REFIT house CSV."
    )
    parser.add_argument(
        "--csv",
        default=os.path.join(
            os.path.dirname(__file__),
            "../../../Processed_Data_CSV/House_3.csv",
        ),
        help="Path to a REFIT house CSV file.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        print(f"[ERROR] File not found: {args.csv!r}")
        print("Please place a REFIT CSV in Processed_Data_CSV/ — see that folder's README.")
        sys.exit(1)

    df = pd.read_csv(args.csv)
    time_col = "Time" if "Time" in df.columns else df.columns[0]
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).set_index(time_col)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    compare_interpolations(df)
