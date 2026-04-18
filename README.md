# Synthetic HAR Digital Twin Generator (IMU + Room Context + Vitals)

A lightweight **digital-twin style simulator** for **Human Activity Recognition (HAR)** that generates synthetic time-series data from:
- **IMU** (accelerometer + gyroscope)
- **Room / context** (bedroom, kitchen, toilet, etc.)
- **Vitals** (heart rate, optional SpO₂ and HRV)

This is designed for:
- prototyping HAR pipelines quickly,
- bootstrapping datasets when real data is limited,
- testing visualization / streaming dashboards,
- experimenting with context-aware activity + health monitoring.

> Note: This is **not** a biomechanical physics engine. It is a controllable, realistic-ish generator for ML experiments and demos.

---

## Features

### HAR / IMU
- Accelerometer: `ax, ay, az` in **m/s²**
- Gyroscope: `gx, gy, gz` in **rad/s**
- Activities (default):
  - `stand`, `sit`, `walk`, `run`, `stairs_up`, `stairs_down`, `fall`
- Sensor placements (default):
  - `waist`, `wrist`, `pocket`
- Includes:
  - sensor noise, bias
  - placement gains (e.g., wrist has more motion)
  - random sensor orientation misalignment (rotation matrix)

### Room / Context (indoor location)
Adds a `room` label per segment (simple schedule-based model), e.g.:
- bedroom, kitchen, toilet, living_room, hallway, outside

> This is a simple context model meant to mimic “room detection” systems such as BLE-beacon or Wi‑Fi fingerprinting solutions.

### Vitals (health monitoring)
Adds:
- `heart_rate_bpm` (activity-dependent, smooth dynamics + noise)
Optional:
- `spo2_pct` (mostly stable, small variation)
- `hrv_rmssd_ms` (lower during intense activity)

---

## Output files

Running the script (demo mode) writes:

- `synthetic_har_context_vitals_long.csv`  
  Long-form sample-by-sample dataset:
  - `time_s, ax, ay, az, gx, gy, gz, activity, room, heart_rate_bpm, spo2_pct, hrv_rmssd_ms, subject_id, placement, fs_hz`

- `synthetic_har_context_vitals_windows_meta.csv`  
  Per-window metadata (start/end times, label, subject, placement)

- `synthetic_har_context_vitals_windows_X.npy`  
  Windowed sensor data shaped:
  - `(num_windows, T, 6)` where channels are `[ax, ay, az, gx, gy, gz]`

- `synthetic_har_context_vitals_windows_y.npy`  
  Window labels (strings)

---

## Requirements

- Python 3.9+ (recommended)
- Packages:
  - `numpy`
  - `pandas`

Optional (for visualization scripts):
- `matplotlib`

---

## Install

### Recommended (Windows / macOS / Linux): virtual environment
```bash
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install numpy pandas
```

If you also want plots:
```bash
python -m pip install matplotlib
```

---

## Run

```bash
python synthetic_har_digital_twin_generator.py
```

---

## Use as a module

```python
from synthetic_har_digital_twin_generator import generate_dataset, make_windows

df = generate_dataset(
    num_subjects=5,
    minutes_per_activity=1.0,
    fs=50,
    seed=7,
    include_spo2=True,
    include_hrv=True,
)

X, y, meta = make_windows(df, window_s=2.0, stride_s=1.0)
```

---

## Main functions

### `generate_dataset(...) -> pandas.DataFrame`
Generates labeled synthetic data across subjects, activities, and sensor placements.

Key parameters:
- `num_subjects`: number of synthetic users
- `minutes_per_activity`: duration per activity **per subject and placement**
- `fs`: sampling rate (Hz)
- `activities`: list of activities (optional)
- `placements`: list of placements (optional)
- `seed`: reproducibility
- `include_spo2`, `include_hrv`: add/remove vitals channels

### `make_windows(df, window_s=2.0, stride_s=1.0)`
Creates fixed-length windows for ML training.

Returns:
- `X`: `(num_windows, T, 6)` float32
- `y`: `(num_windows,)` labels
- `meta`: window metadata DataFrame

### `stream_twin_state(df)`
Yields one row at a time (dict), like a real-time “digital twin state” stream.

---

## Data columns (long-form)

| Column | Meaning | Unit |
|---|---|---|
| `time_s` | time from segment start | seconds |
| `ax, ay, az` | accelerometer | m/s² |
| `gx, gy, gz` | gyroscope | rad/s |
| `activity` | activity label | string |
| `room` | inferred room/context | string |
| `heart_rate_bpm` | heart rate | bpm |
| `spo2_pct` | oxygen saturation (optional) | % |
| `hrv_rmssd_ms` | HRV RMSSD (optional) | ms |
| `subject_id` | synthetic subject id | int |
| `placement` | sensor placement | string |
| `fs_hz` | sampling rate | Hz |

---

## Common issues

### `ModuleNotFoundError: No module named 'numpy'`
Install dependencies in the same Python you run:
```bash
python -m pip install numpy pandas
```

### `SyntaxError` near `if __name__`
Make sure the file ends with:
```python
if __name__ == "__main__":
    demo()
```

---

## License
Add your preferred license (MIT is common) in a `LICENSE` file.

---

## Next improvements (ideas)
- Add activity transitions (sit→stand, stand→walk)
- Simulate BLE beacon RSSI and infer `room` from RSSI (more realistic)
- Add orientation drift over time
- Add additional health events / risk flags (tachycardia at rest, fall confirmation, etc.)
- Add training pipeline (PyTorch/TensorFlow) and evaluation metrics

---

## GitHub upload checklist
- [ ] `synthetic_har_digital_twin_generator.py`
- [ ] `README.md`
- [ ] (optional) `requirements.txt`
- [ ] (optional) `LICENSE`
- [ ] (optional) `visualize_*.py` scripts