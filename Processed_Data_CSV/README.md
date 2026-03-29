# REFIT Dataset — Placement Instructions

## About REFIT

The **REFIT** (*Real world Energy Flexible Intelligent metering dataset*) is a publicly
available dataset of electrical energy measurements collected from 20 UK homes at an
8-second sampling rate.

**Source:** https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements-cleaned

---

## How to Download

1. Visit the link above and download the **Cleaned REFIT dataset**.
2. Extract the ZIP archive — you should get CSV files named:
   - `CLEAN_House1.csv`
   - `CLEAN_House2.csv`
   - …up to `CLEAN_House21.csv` (note: `House_14` does not exist in the dataset)

---

## How to Place the Files Here

Rename (or copy) each file as follows and place it in **this directory**:

| Downloaded file      | Rename to       |
|----------------------|-----------------|
| `CLEAN_House1.csv`   | `House_1.csv`   |
| `CLEAN_House2.csv`   | `House_2.csv`   |
| `CLEAN_House3.csv`   | `House_3.csv`   |
| …                    | …               |
| `CLEAN_House21.csv`  | `House_21.csv`  |

**Final layout of this directory:**

```
Processed_Data_CSV/
├── README.md        ← this file
├── House_1.csv
├── House_2.csv
├── House_3.csv
├── ...
└── House_21.csv
```

---

## CSV Format

Each CSV file has the following columns:

| Column      | Description                                    |
|-------------|------------------------------------------------|
| `Time`      | ISO-8601 timestamp (e.g. `2013-09-18 01:12:42`) |
| `Unix`      | Unix epoch timestamp (seconds)                 |
| `Aggregate` | Total household power consumption (Watts)      |
| `Appliance1`| Power of appliance 1 (Watts)                   |
| `Appliance2`| Power of appliance 2 (Watts)                   |
| …           | …                                              |
| `Appliance9`| Power of appliance 9 (Watts)                   |

The mapping of `Appliance1`–`Appliance9` to real device names varies per house and is
defined in `../Projet_NILM/refit_metadata.py`.

---

## Quick Start

Once the CSV files are in this directory, run the disaggregation pipeline:

```bash
# From the repository root:
cd Projet_NILM

# Run the full pipeline on House 3 (trains + disaggregates + plots):
python run_nilm.py --house ../Processed_Data_CSV/House_3.csv

# Run only training (saves models to models/):
python run_nilm.py --house ../Processed_Data_CSV/House_3.csv --mode train

# Run only disaggregation (requires saved models):
python run_nilm.py --house ../Processed_Data_CSV/House_3.csv --mode disaggregate

# Specify which appliances to focus on (default: kettle microwave fridge tv):
python run_nilm.py --house ../Processed_Data_CSV/House_3.csv --appliances kettle microwave fridge tv
```

Plots are saved to `Projet_NILM/plots/`.
