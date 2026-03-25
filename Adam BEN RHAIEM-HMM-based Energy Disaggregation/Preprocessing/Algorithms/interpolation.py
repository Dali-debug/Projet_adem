import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

# Assuming df_smd is already defined and has a column named 'Aggregate'


# take a segment of data points
start = 0
stop = 200
start_missing = 18
stop_missing = 22
excerpt = df_smd.iloc[start:stop].copy()

# remove data for 4 consecutive data points
missing = excerpt.copy()
missing['Aggregate'].iloc[start_missing:stop_missing] = np.nan

# create figure
fig, ax = plt.subplots(figsize=(14, 5))

# plot different interpolations
ax.plot(excerpt['Aggregate'], label='original data')
ax.plot(missing['Aggregate'].interpolate().iloc[start_missing - 1:stop_missing + 1], label='linear')
ax.plot(missing['Aggregate'].interpolate(method='polynomial', order=3).iloc[start_missing - 1:stop_missing + 1], label='polynomial')
ax.plot(missing['Aggregate'].interpolate(method='spline', order=3).iloc[start_missing - 1:stop_missing + 1], label='spline')
ax.axvspan(excerpt.index[start_missing - 1], excerpt.index[stop_missing], color='red', alpha=0.1)

# additional formatting and show plot
# additional formatting and show plot
ax.xaxis.set_major_locator(mdates.HourLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
ax.xaxis.set_minor_locator(mdates.MinuteLocator(range(15, 60, 15)))
ax.xaxis.set_minor_formatter(mdates.DateFormatter(":%M"))
ax.tick_params(axis='x', which='minor', labelsize=8, colors='grey')
ax.grid()
ax.set_xlabel('Time')
ax.set_ylabel('Power (Watts)')
ax.legend()
fig.show()








# Randomly remove 2% of the data
indices = np.random.choice(len(df_smd.index), size=int(0.02 * len(df_smd.index)), replace=False)
masked_df = df_smd['Aggregate'].copy()
masked_df[indices] = np.nan

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
    interpolated = masked_df.interpolate(**d)
    mae = mean_absolute_error(interpolated[indices], df_smd['Aggregate'][indices])
    print(f"{d['method']} (order {d['order']}): {np.round(mae, 4)}")

# For comparison, simply fill with zero
mae_zeros = mean_absolute_error(masked_df.fillna(0)[indices], df_smd['Aggregate'][indices])
print('fill zeros:', np.round(mae_zeros, 4))




