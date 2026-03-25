import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ca_cfar(signal, num_guard_cells, num_ref_cells, alpha):
    """
    """
    num_cells = len(signal)
    detection_mask = np.zeros(num_cells, dtype=int)

    for i in range(num_guard_cells + num_ref_cells, num_cells - num_guard_cells - num_ref_cells):
        # Reference cells: leading and lagging
        leading_ref = signal[i - num_guard_cells - num_ref_cells : i - num_guard_cells]
        lagging_ref = signal[i + num_guard_cells : i + num_guard_cells + num_ref_cells]
        noise_level = np.mean(np.concatenate((leading_ref, lagging_ref)))

        # Set threshold
        threshold = alpha * noise_level

        # Detect target
        if signal[i] > threshold:
            detection_mask[i] = 1

    return detection_mask

# Load your signal from the DataFrame
signal = df['Dishwasher'].values  # Convert the column to a NumPy array

# CFAR parameters
num_guard_cells =10  # Adjust based on your signal characteristics
num_ref_cells = 20 # Adjust based on your signal characteristics
alpha =2  # Adjust to control the false alarm rate

# Perform CFAR detection
detection_mask = ca_cfar(signal, num_guard_cells, num_ref_cells, alpha)

# Add the detection results to the DataFrame
df['Detection'] = detection_mask

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(signal, label="Dishwasher Signal")
plt.plot(np.where(detection_mask == 1, signal, np.nan), 'ro', label="Detected Targets")
plt.legend()
plt.title("CA-CFAR Detection on Dishwasher Signal")
plt.xlabel("Time/Sample Index")
plt.ylabel("Signal Amplitude")
plt.show()