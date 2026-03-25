import numpy as np
import pandas as pd
import ruptures as rpt
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from itertools import combinations

# Assuming df['Dishwasher'] exists
dishwasher_series = df['Dishwasher'].values
time_index = np.arange(len(dishwasher_series))
n = len(dishwasher_series)
dim = 1  # Since we are working with a single time series
sigma = np.std(dishwasher_series)

### Step 1: Event Detection (Unknown number of change points) ###
model = "l2"  # Cost function for change detection
algo = rpt.Binseg(model=model).fit(dishwasher_series)

# Choose either penalty or epsilon-based change point detection
my_bkps = algo.predict(pen=np.log(n) * dim * sigma**2)
# Alternatively:
# my_bkps = algo.predict(epsilon=3 * n * sigma**2)

# Extract event signatures
event_signatures = []
for i in range(len(my_bkps) - 1):
    start, end = my_bkps[i], my_bkps[i+1]
    mean_power = np.mean(dishwasher_series[start:end])
    event_signatures.append([start, mean_power])

event_signatures = np.array(event_signatures)

### Step 2: Clustering of Event Signatures ###
num_clusters = 2  # Assuming two states (ON and OFF)
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
labels = kmeans.fit_predict(event_signatures[:, 1].reshape(-1, 1))

# Determine which cluster corresponds to ON and OFF
cluster_means = {i: np.mean(event_signatures[labels == i, 1]) for i in range(num_clusters)}
on_cluster = max(cluster_means, key=cluster_means.get)  # Higher power -> ON
off_cluster = min(cluster_means, key=cluster_means.get)  # Lower power -> OFF

# Assign clusters
event_clusters = {"ON": [], "OFF": []}
for i, label in enumerate(labels):
    state = "ON" if label == on_cluster else "OFF"
    event_clusters[state].append(event_signatures[i])

### Step 3: Event Pairing for Cycle Identification ###
paired_cycles = []
sorted_on_events = sorted(event_clusters["ON"], key=lambda x: x[0])
sorted_off_events = sorted(event_clusters["OFF"], key=lambda x: x[0])

i, j = 0, 0
while i < len(sorted_on_events) and j < len(sorted_off_events):
    on_event, off_event = sorted_on_events[i], sorted_off_events[j]
    if off_event[0] > on_event[0]:  # Ensure OFF follows ON
        paired_cycles.append({
            "Appliance": "Dishwasher",
            "ON Time": on_event[0],
            "OFF Time": off_event[0],
            "Duration": abs(off_event[0] - on_event[0]),
            "Power Level": on_event[1]
        })
        i += 1  # Move to the next ON event
    j += 1  # Always check the next OFF event

# Convert to DataFrame
cycles_df = pd.DataFrame(paired_cycles)

### Visualization ###

# 1. Plot dishwasher power consumption with detected events
plt.figure(figsize=(12, 5))
plt.plot(time_index, dishwasher_series, label="Dishwasher Power Consumption", color="blue", alpha=0.3)
for cp in my_bkps:
    plt.axvline(x=cp, color='red', linestyle='--', label="Detected Event" if cp == my_bkps[0] else "")
plt.xlabel("Time")
plt.ylabel("Power (W)")
plt.title("Dishwasher Power Consumption with Event Detection")
plt.legend()
plt.show()

# 2. Scatter plot of event clustering
plt.figure(figsize=(10, 5))
for state in ["ON", "OFF"]:
    cluster_events = np.array(event_clusters[state])
    plt.scatter(cluster_events[:, 0], cluster_events[:, 1], label=f"{state} State")
plt.xlabel("Time")
plt.ylabel("Power Level")
plt.title("Clustering of Dishwasher Events")
plt.legend()
plt.show()

# 3. Mark ON/OFF events and dishwasher cycles
plt.figure(figsize=(12, 5))
plt.plot(t