import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def detect_and_plot_dishwasher_events(power_series, window_size=1000, a=1, b=1, plot=True):
    """
    Detect dishwasher events and optionally plot the results.
    
    Parameters:
    - power_series: pd.Series with power values
    - window_size: size of the sliding window (n=60 in the original)
    - a, b: coefficients for the threshold calculation
    - plot: whether to display the plot
    
    Returns:
    - List of event indices (timestamps or indices where events start)
    - If plot=True, also displays a visualization
    """
    # Step 1: Apply median filter to aggregated power
    p_med = power_series.rolling(window=60, center=True).median()  # Using window=3 for median filter
    
    # Step 2: Calculate absolute power differences
    delta_p = p_med.diff().abs().dropna()
    
    # Initialize lists to store window statistics
    mu_w = []
    sigma_w = []
    delta_power_windows = []
    
    # Step 3: Slide window through delta_p and calculate statistics
    for i in range(len(delta_p) - window_size + 1):
        window = delta_p.iloc[i:i+window_size]
        delta_power_windows.append(window)
        mu_w.append(window.mean())
        sigma_w.append(window.std())
    
    # Convert to numpy arrays for easier manipulation
    mu_w = np.array(mu_w)
    sigma_w = np.array(sigma_w)
    
    # Step 4: Calculate thresholds
    s_p = np.mean(sigma_w) + np.mean(mu_w)
    s_a = s_p / 2
    
    # Step 5: Find windows where sigma > s_a
    w_p = np.where(sigma_w > s_a)[0]
    
    # Step 6: Perform peak detection in selected windows
    event_indices = []
    
    for i in w_p:
        window_data = delta_power_windows[i]
        
        # Calculate threshold for this window
        threshold = a * mu_w[i] + b * sigma_w[i]
        
        # Find peaks in this window that exceed the threshold
        peaks, _ = find_peaks(window_data, height=threshold)
        
        # Convert window-relative peaks to absolute indices
        for peak_pos in peaks:
            absolute_index = (i * 1) + peak_pos  # *1 since we slide 1 point at a time
            # Get the actual index from the original series
            event_index = delta_p.index[absolute_index]
            event_indices.append(event_index)
    
    # Remove duplicates (peaks might be detected in overlapping windows)
    event_indices = sorted(list(set(event_indices)))
    
    if plot:
        plt.figure(figsize=(15, 6))
        
        # Plot the original power series
        plt.plot(power_series.index, power_series, label='Dishwasher Power', alpha=0.7)
        
        
        # Mark the detected events 
        if len(event_indices) > 0:
     # Get the y-values (power) at event times
          event_values = power_series.loc[event_indices]
    
        # Plot scatter points for events
          plt.scatter(event_indices, event_values, color='red', s=100, marker='o', 
               edgecolor='black', linewidth=1,
               label='Detected Events', zorder=5)
 
    
        plt.title('Dishwasher Power Consumption with Detected Events by Local threshold based event detection ')
        plt.xlabel('Time')
        plt.ylabel('Power (W)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    return event_indices

# Usage example:
# Assuming df is your DataFrame with a 'Dishwasher' column containing power data
event_times = detect_and_plot_dishwasher_events(df['Dishwasher'])

print(f"Detected events at indices/timestamps: {event_times}")