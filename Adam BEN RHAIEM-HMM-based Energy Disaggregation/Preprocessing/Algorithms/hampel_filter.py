import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------ #
# DEFINE HELPER FUNCTION  #
# ------------------------ #
def hampel_filter(series, window_size, n_sigmas=3):
    """
    Apply Hampel filter to detect and correct outliers in a time series.
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

# ------------------------ #
# APPLY FILTER TO YOUR DF #
# ------------------------ #

df = df.sort_index()

# Example: 1 month = 96 samples/day * 30 days
excerpt = df['Aggregate'].iloc[:96 * 30]

filtered, mask = hampel_filter(excerpt, window_size=15)

# ------------------------ #
# PLOT MAIN RESULTS        #
# ------------------------ #
fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(filtered, label='Filtered data')
ax.scatter(excerpt.index[mask], excerpt[mask], c='r', label='Original outliers')

ax.set_xlabel('Time')
ax.set_ylabel('Energy (kWh)')
ax.legend()

# ------------------------ #
# INSET OUTSIDE LEFT VIEW  #
# ------------------------ #
start = int(1.5 * 96)
stop = start + 50

zoomed_index = excerpt.index[start:stop]
zoomed_excerpt = excerpt.iloc[start:stop]
zoomed_filtered = filtered.iloc[start:stop]
zoomed_mask = mask.iloc[start:stop]

# Create inset OUTSIDE to the left of main plot
axins = fig.add_axes([-0.3, 0.4, 0.3, 0.5])  # (left, bottom, width, height)

# Plot data in zoomed view
axins.plot(zoomed_index, zoomed_filtered, label='Filtered', color='blue')
axins.plot(zoomed_index, zoomed_excerpt, linestyle='--', color='gray', label='Original')

# Highlight outliers in zoomed view
axins.scatter(zoomed_index[zoomed_mask], zoomed_excerpt[zoomed_mask], color='red', label='Outliers')

axins.set_xlim(zoomed_index[0], zoomed_index[-1])
axins.set_ylim(
    min(zoomed_excerpt.min(), zoomed_filtered.min()) - 0.2,
    max(zoomed_excerpt.max(), zoomed_filtered.max()) + 0.2
)

axins.set_xticks([])
axins.set_yticks([])
axins.set_title("Zoomed View")

# Draw line connection from zoom to main
ax.indicate_inset_zoom(axins, edgecolor="black")

plt.tight_layout()
plt.show()
