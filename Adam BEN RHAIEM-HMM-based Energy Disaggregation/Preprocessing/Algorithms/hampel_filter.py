"""
hampel_filter.py
----------------
Hampel filter for outlier detection and correction in power-consumption
time series.

Run standalone:
    python hampel_filter.py --csv ../../../Processed_Data_CSV/House_3.csv

Requires the REFIT dataset to be available (see Processed_Data_CSV/README.md).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ------------------------ #
# DEFINE HELPER FUNCTION  #
# ------------------------ #

def hampel_filter(series, window_size, n_sigmas=3):
    """Apply Hampel filter to detect and correct outliers in a time series.

    Parameters
    ----------
    series : pd.Series
        1-D power/energy time series.
    window_size : int
        Half-width of the sliding window (total window = 2*window_size).
    n_sigmas : float
        Detection threshold in number of MAD-based sigmas.

    Returns
    -------
    new_series : pd.Series
        Series with outliers replaced by the local median.
    mask : pd.Series of bool
        Boolean mask; True where an outlier was detected.
    """
    k = 1.4826  # Scale factor for Gaussian distribution

    new_series = series.copy()

    rolling_median = series.rolling(window=2 * window_size, center=True).median()

    rolling_mad = k * series.rolling(window=2 * window_size, center=True).apply(
        lambda x: np.median(np.abs(x - np.median(x))), raw=True
    )

    diff = np.abs(series - rolling_median)
    mask = diff > (n_sigmas * rolling_mad)
    new_series[mask] = rolling_median[mask]

    print('Share of outliers in the series:', mask.mean())
    return new_series, mask


def plot_hampel(series: pd.Series, filtered: pd.Series, mask: pd.Series,
                out_path: str = "hampel_filter.png"):
    """Plot original series with outliers highlighted, plus a zoomed inset."""
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(filtered, label='Filtered data')
    ax.scatter(series.index[mask], series[mask], c='r', label='Original outliers')

    ax.set_xlabel('Time')
    ax.set_ylabel('Power (W)')
    ax.legend()

    # Zoomed inset
    start = int(1.5 * 96)
    stop = start + 50

    zoomed_index = series.index[start:stop]
    zoomed_excerpt = series.iloc[start:stop]
    zoomed_filtered = filtered.iloc[start:stop]
    zoomed_mask = mask.iloc[start:stop]

    axins = fig.add_axes([0.12, 0.55, 0.25, 0.35])
    axins.plot(range(len(zoomed_filtered)), zoomed_filtered.values,
               label='Filtered', color='blue')
    axins.plot(range(len(zoomed_excerpt)), zoomed_excerpt.values,
               linestyle='--', color='gray', label='Original')
    axins.scatter(
        [i for i, m in enumerate(zoomed_mask) if m],
        zoomed_excerpt.values[[i for i, m in enumerate(zoomed_mask) if m]],
        color='red', label='Outliers'
    )
    axins.set_xticks([])
    axins.set_title("Zoomed View", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close(fig)
    print(f"Plot saved to {out_path}")


# ------------------------ #
# STANDALONE ENTRY POINT  #
# ------------------------ #

if __name__ == "__main__":
    import argparse
    import sys
    import os

    parser = argparse.ArgumentParser(
        description="Apply Hampel filter to a REFIT house CSV."
    )
    parser.add_argument(
        "--csv",
        default=os.path.join(
            os.path.dirname(__file__),
            "../../../Processed_Data_CSV/House_3.csv",
        ),
        help="Path to a REFIT house CSV file.",
    )
    parser.add_argument(
        "--window", type=int, default=15,
        help="Half-window size for the Hampel filter (default: 15).",
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
    df = df.sort_index()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Apply to first month of Aggregate data
    excerpt = df['Aggregate'].iloc[:96 * 30]
    filtered, mask = hampel_filter(excerpt, window_size=args.window)
    plot_hampel(excerpt, filtered, mask)
