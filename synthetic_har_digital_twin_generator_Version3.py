"""
Synthetic HAR + Context (Room) + Vitals (Heart Rate, etc.) generator using a simple "digital twin" concept.

Adds to previous version:
- room (bedroom/kitchen/toilet/living_room/hallway/outside)
- heart_rate_bpm (activity-dependent with noise)
- optional spo2_pct and hrv_rmssd_ms
- helper to stream "digital twin state" events

Dependencies:
- numpy, pandas

Outputs:
- DataFrame columns:
  time_s, ax, ay, az, gx, gy, gz,
  activity, room,
  heart_rate_bpm, spo2_pct, hrv_rmssd_ms,
  subject_id, placement, fs_hz
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Iterator
import numpy as np
import pandas as pd


# -----------------------------
# Digital-twin parameterization
# -----------------------------

@dataclass
class SubjectTwin:
    subject_id: int
    height_m: float
    mass_kg: float
    gait_cadence_hz: float
    run_cadence_hz: float
    tremor_hz: float
    device_bias_acc: np.ndarray
    device_bias_gyr: np.ndarray

    # Vitals baseline (personalization)
    resting_hr_bpm: float         # e.g. 55-85
    max_hr_bpm: float             # e.g. 160-200
    baseline_spo2_pct: float      # e.g. 96-99
    baseline_hrv_rmssd_ms: float  # e.g. 20-80


@dataclass
class SensorTwin:
    placement: str
    R_sb: np.ndarray
    noise_std_acc: float
    noise_std_gyr: float
    env_vibration_std: float


# -----------------------------
# Helpers
# -----------------------------

def _rng(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(seed)


def random_rotation_matrix(r: np.random.Generator, max_deg: float = 25.0) -> np.ndarray:
    max_rad = np.deg2rad(max_deg)
    angles = r.uniform(-max_rad, max_rad, size=3)  # roll, pitch, yaw
    cr, sr = np.cos(angles[0]), np.sin(angles[0])
    cp, sp = np.cos(angles[1]), np.sin(angles[1])
    cy, sy = np.cos(angles[2]), np.sin(angles[2])

    Rz = np.array([[cy, -sy, 0],
                   [sy,  cy, 0],
                   [ 0,   0, 1]])
    Ry = np.array([[ cp, 0, sp],
                   [  0, 1,  0],
                   [-sp, 0, cp]])
    Rx = np.array([[1,  0,   0],
                   [0, cr, -sr],
                   [0, sr,  cr]])
    return Rz @ Ry @ Rx


def rotate(R: np.ndarray, x: np.ndarray) -> np.ndarray:
    return x @ R.T


def bandlimited_noise(
    r: np.random.Generator,
    n: int,
    fs: float,
    low_hz: float,
    high_hz: float,
    scale: float
) -> np.ndarray:
    x = r.normal(0.0, 1.0, size=n)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    X[~mask] = 0
    y = np.fft.irfft(X, n=n)
    y = y / (np.std(y) + 1e-9) * scale
    return y


# -----------------------------
# Context model (room)
# -----------------------------

DEFAULT_ROOMS = ["bedroom", "kitchen", "toilet", "living_room", "hallway", "outside"]

# Simple mapping: activity -> likely rooms (weighted)
ACTIVITY_ROOMS: Dict[str, List[str]] = {
    "stand": ["living_room", "hallway", "kitchen"],
    "sit": ["living_room", "bedroom"],
    "walk": ["hallway", "outside", "kitchen"],
    "run": ["outside", "hallway"],
    "stairs_up": ["hallway"],
    "stairs_down": ["hallway"],
    "fall": ["bathroom", "bedroom", "kitchen", "hallway"],  # "bathroom" alias
}

ROOM_ALIASES = {"bathroom": "toilet"}


def pick_room_for_activity(activity: str, r: np.random.Generator) -> str:
    candidates = ACTIVITY_ROOMS.get(activity, ["living_room"])
    choice = candidates[int(r.integers(0, len(candidates)))]
    return ROOM_ALIASES.get(choice, choice)


# -----------------------------
# Vitals model (HR, SpO2, HRV)
# -----------------------------

def simulate_vitals(
    activity: str,
    t: np.ndarray,
    subject: SubjectTwin,
    r: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      hr_bpm: (n,)
      spo2_pct: (n,)
      hrv_rmssd_ms: (n,)
    Simple but realistic-ish relationships:
      - HR increases with activity intensity
      - HRV decreases with intensity
      - SpO2 mostly stable, slight dips during run (optional)
    """
    n = len(t)

    rest = subject.resting_hr_bpm
    hr_max = subject.max_hr_bpm

    # Target HR by activity (rough)
    if activity in ("sit", "stand"):
        target = rest + 5
        noise = 2.0
    elif activity == "walk":
        target = rest + 25
        noise = 3.0
    elif activity in ("stairs_up", "stairs_down"):
        target = rest + 35
        noise = 4.0
    elif activity == "run":
        target = min(hr_max - 5, rest + 70)
        noise = 5.0
    elif activity == "fall":
        # spike then settle
        target = rest + 20
        noise = 4.0
    else:
        target = rest + 10
        noise = 3.0

    # Smooth HR approach (1st order)
    tau = 6.0  # seconds time constant
    hr = np.empty(n, dtype=float)
    hr[0] = rest + r.normal(0, 1.0)
    for i in range(1, n):
        dt = t[i] - t[i - 1] if i > 0 else 0.02
        alpha = 1.0 - np.exp(-dt / tau)
        hr[i] = hr[i - 1] + alpha * (target - hr[i - 1])

    # Fall: sudden spike around mid segment
    if activity == "fall":
        mid = int(0.55 * n)
        width = max(2, int(0.04 * n))
        k = np.arange(-width, width + 1)
        spike = np.exp(-(k / (0.25 * width + 1e-9)) ** 2) * (20 + 20 * r.random())
        j0 = max(0, mid - width)
        j1 = min(n, mid + width + 1)
        hr[j0:j1] += spike[(width - (mid - j0)):(width - (mid - j0)) + (j1 - j0)]

    # Add noise + clamp
    hr += r.normal(0.0, noise, size=n)
    hr = np.clip(hr, 35, hr_max)

    # SpO2 (mostly stable)
    spo2 = np.full(n, subject.baseline_spo2_pct, dtype=float)
    spo2 += r.normal(0.0, 0.15, size=n)
    if activity == "run":
        spo2 -= 0.3 + 0.3 * np.sin(2 * np.pi * 0.05 * t)  # tiny dips
    spo2 = np.clip(spo2, 90.0, 100.0)

    # HRV RMSSD (drops with intensity)
    hrv_base = subject.baseline_hrv_rmssd_ms
    if activity in ("sit", "stand"):
        hrv_target = hrv_base
    elif activity == "walk":
        hrv_target = 0.75 * hrv_base
    elif activity in ("stairs_up", "stairs_down"):
        hrv_target = 0.65 * hrv_base
    elif activity == "run":
        hrv_target = 0.45 * hrv_base
    elif activity == "fall":
        hrv_target = 0.60 * hrv_base
    else:
        hrv_target = 0.8 * hrv_base

    hrv = hrv_target + bandlimited_noise(r, n, fs=1.0 / np.median(np.diff(t)), low_hz=0.02, high_hz=0.2, scale=2.0)
    hrv += r.normal(0.0, 1.0, size=n)
    hrv = np.clip(hrv, 5.0, 200.0)

    return hr, spo2, hrv


# -----------------------------
# Activity signal models (body frame)
# -----------------------------

def simulate_activity_body(
    activity: str,
    duration_s: float,
    fs: float,
    subject: SubjectTwin,
    r: np.random.Generator,
    g: float = 9.81
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(duration_s * fs)
    t = np.arange(n) / fs

    acc = np.zeros((n, 3), dtype=float)
    acc[:, 2] = g
    gyr = np.zeros((n, 3), dtype=float)

    trem = subject.tremor_hz
    trem_phase = r.uniform(0, 2 * np.pi)
    trem_signal = 0.05 * np.sin(2 * np.pi * trem * t + trem_phase)
    acc += trem_signal[:, None] * np.array([0.5, 0.3, 0.2])[None, :]

    if activity == "stand":
        sway = bandlimited_noise(r, n, fs, 0.1, 0.6, scale=0.08)
        acc[:, 0] += sway
        acc[:, 1] += 0.6 * sway
        gyr[:, 2] += bandlimited_noise(r, n, fs, 0.1, 0.6, scale=0.02)

    elif activity == "sit":
        sway = bandlimited_noise(r, n, fs, 0.1, 0.5, scale=0.06)
        acc[:, 0] += 0.7 * sway
        acc[:, 1] += 0.4 * sway
        gyr[:, 2] += bandlimited_noise(r, n, fs, 0.1, 0.5, scale=0.015)

    elif activity == "walk":
        f = np.clip(subject.gait_cadence_hz + r.normal(0, 0.05), 1.2, 2.4)
        intensity = (0.9 + 0.2 * r.random()) * (subject.mass_kg / 75.0)
        acc[:, 2] += intensity * 1.2 * np.sin(2 * np.pi * f * t)
        acc[:, 0] += intensity * 0.6 * np.sin(2 * np.pi * f * t + np.pi / 3)
        acc[:, 1] += intensity * 0.3 * np.sin(2 * np.pi * 0.5 * f * t + np.pi / 8)
        gyr[:, 0] += 0.25 * intensity * np.sin(2 * np.pi * f * t + np.pi / 5)
        gyr[:, 2] += 0.18 * intensity * np.sin(2 * np.pi * f * t + np.pi / 2)

    elif activity == "run":
        f = np.clip(subject.run_cadence_hz + r.normal(0, 0.07), 2.2, 3.6)
        intensity = (1.3 + 0.35 * r.random()) * (subject.mass_kg / 75.0)
        acc[:, 2] += intensity * 2.6 * np.sin(2 * np.pi * f * t)
        acc[:, 0] += intensity * 1.1 * np.sin(2 * np.pi * f * t + np.pi / 4)
        acc[:, 1] += intensity * 0.5 * np.sin(2 * np.pi * 0.5 * f * t + np.pi / 7)
        gyr[:, 0] += 0.45 * intensity * np.sin(2 * np.pi * f * t + np.pi / 6)
        gyr[:, 2] += 0.35 * intensity * np.sin(2 * np.pi * f * t + np.pi / 2)

    elif activity == "stairs_up":
        f = np.clip(subject.gait_cadence_hz - 0.15 + r.normal(0, 0.05), 1.0, 2.2)
        intensity = (1.0 + 0.25 * r.random()) * (subject.mass_kg / 75.0)
        acc[:, 2] += intensity * 1.7 * np.sin(2 * np.pi * f * t)
        acc[:, 0] += intensity * 0.55 * np.sin(2 * np.pi * f * t + np.pi / 3)
        gyr[:, 0] += 0.30 * intensity * np.sin(2 * np.pi * f * t + np.pi / 9)
        gyr[:, 2] += 0.20 * intensity * np.sin(2 * np.pi * f * t + np.pi / 2)

    elif activity == "stairs_down":
        f = np.clip(subject.gait_cadence_hz - 0.05 + r.normal(0, 0.05), 1.0, 2.4)
        intensity = (1.05 + 0.3 * r.random()) * (subject.mass_kg / 75.0)
        acc[:, 2] += intensity * 1.5 * np.sin(2 * np.pi * f * t)
        acc[:, 0] += intensity * 0.5 * np.sin(2 * np.pi * f * t + np.pi / 4)

        step_times = np.arange(0, duration_s, 1.0 / f)
        for ts in step_times:
            idx = int(ts * fs)
            if 0 <= idx < n:
                width = int(0.06 * fs)
                k = np.arange(-width, width + 1)
                pulse = np.exp(-(k / (0.015 * fs + 1e-9)) ** 2) * (0.9 * intensity)
                j0 = max(0, idx - width)
                j1 = min(n, idx + width + 1)
                kk0 = width - (idx - j0)
                kk1 = kk0 + (j1 - j0)
                acc[j0:j1, 2] += pulse[kk0:kk1]

        gyr[:, 0] += 0.28 * intensity * np.sin(2 * np.pi * f * t + np.pi / 7)
        gyr[:, 2] += 0.22 * intensity * np.sin(2 * np.pi * f * t + np.pi / 2)

    elif activity == "fall":
        acc[:, 0] += bandlimited_noise(r, n, fs, 0.1, 0.6, scale=0.05)
        acc[:, 1] += bandlimited_noise(r, n, fs, 0.1, 0.6, scale=0.05)

        t0 = duration_s * (0.25 + 0.1 * r.random())
        t1 = t0 + duration_s * (0.15 + 0.05 * r.random())
        t2 = t1 + duration_s * (0.08 + 0.03 * r.random())
        i0, i1, i2 = int(t0 * fs), int(t1 * fs), int(t2 * fs)

        burst = bandlimited_noise(r, n, fs, 1.0, 6.0, scale=1.8)
        gyr[:, 0] += 0.7 * burst
        gyr[:, 1] += 0.5 * burst
        gyr[:, 2] += 0.6 * burst

        if i0 < i1:
            drop = np.linspace(0, 1, i1 - i0)
            acc[i0:i1, 2] = g * (1.0 - 0.75 * drop)

        if 0 <= i2 < n:
            width = int(0.08 * fs)
            k = np.arange(-width, width + 1)
            pulse = np.exp(-(k / (0.01 * fs + 1e-9)) ** 2) * (22.0 + 8.0 * r.random())
            j0 = max(0, i2 - width)
            j1 = min(n, i2 + width + 1)
            kk0 = width - (i2 - j0)
            kk1 = kk0 + (j1 - j0)
            acc[j0:j1, 2] += pulse[kk0:kk1]

        if i2 < n:
            post = slice(i2, n)
            acc[post, 0] *= 0.2
            acc[post, 1] *= 0.2
            gyr[post, :] *= 0.25

    else:
        raise ValueError(f"Unknown activity: {activity}")

    return t, acc, gyr


# -----------------------------
# Placement model (body -> sensor)
# -----------------------------

def placement_gain(placement: str) -> Tuple[float, float]:
    if placement == "waist":
        return 1.0, 1.0
    if placement == "wrist":
        return 1.15, 1.45
    if placement == "pocket":
        return 1.05, 1.10
    return 1.0, 1.0


def make_sensor_twin(r: np.random.Generator, placement: str, fs: float) -> SensorTwin:
    R_sb = random_rotation_matrix(r, max_deg=30.0 if placement == "pocket" else 20.0)

    noise_acc = 0.12 if placement == "waist" else (0.18 if placement == "wrist" else 0.25)
    noise_gyr = 0.015 if placement == "waist" else (0.025 if placement == "wrist" else 0.03)
    env_vib = 0.03 if placement == "waist" else (0.05 if placement == "wrist" else 0.07)

    if fs < 30:
        noise_acc *= 1.2
        noise_gyr *= 1.2

    return SensorTwin(
        placement=placement,
        R_sb=R_sb,
        noise_std_acc=noise_acc,
        noise_std_gyr=noise_gyr,
        env_vibration_std=env_vib
    )


def make_subject_twin(r: np.random.Generator, subject_id: int) -> SubjectTwin:
    height = float(np.clip(r.normal(1.72, 0.09), 1.45, 2.05))
    mass = float(np.clip(r.normal(78.0, 14.0), 45.0, 125.0))

    base_walk = 1.85 - 0.35 * (height - 1.72) + float(r.normal(0, 0.08))
    base_walk = float(np.clip(base_walk, 1.2, 2.4))

    base_run = 2.9 - 0.45 * (height - 1.72) + float(r.normal(0, 0.10))
    base_run = float(np.clip(base_run, 2.2, 3.6))

    trem = float(np.clip(r.normal(8.0, 1.3), 5.0, 11.0))

    bias_acc = r.normal(0.0, 0.08, size=3)
    bias_gyr = r.normal(0.0, 0.004, size=3)

    # Vitals personalization
    resting_hr = float(np.clip(r.normal(70.0, 8.0), 50.0, 90.0))
    max_hr = float(np.clip(r.normal(185.0, 12.0), 160.0, 210.0))
    spo2 = float(np.clip(r.normal(97.5, 0.8), 95.0, 99.5))
    hrv = float(np.clip(r.normal(45.0, 15.0), 15.0, 110.0))

    return SubjectTwin(
        subject_id=subject_id,
        height_m=height,
        mass_kg=mass,
        gait_cadence_hz=base_walk,
        run_cadence_hz=base_run,
        tremor_hz=trem,
        device_bias_acc=bias_acc,
        device_bias_gyr=bias_gyr,
        resting_hr_bpm=resting_hr,
        max_hr_bpm=max_hr,
        baseline_spo2_pct=spo2,
        baseline_hrv_rmssd_ms=hrv
    )


def apply_sensor_model(
    acc_b: np.ndarray,
    gyr_b: np.ndarray,
    subject: SubjectTwin,
    sensor: SensorTwin,
    r: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    acc_gain, gyr_gain = placement_gain(sensor.placement)

    acc_s = rotate(sensor.R_sb, acc_b) * acc_gain
    gyr_s = rotate(sensor.R_sb, gyr_b) * gyr_gain

    acc_s += r.normal(0.0, sensor.env_vibration_std, size=acc_s.shape)
    acc_s += subject.device_bias_acc[None, :]
    gyr_s += subject.device_bias_gyr[None, :]

    acc_s += r.normal(0.0, sensor.noise_std_acc, size=acc_s.shape)
    gyr_s += r.normal(0.0, sensor.noise_std_gyr, size=gyr_s.shape)

    return acc_s, gyr_s


# -----------------------------
# Dataset generation
# -----------------------------

DEFAULT_ACTIVITIES = ["stand", "sit", "walk", "run", "stairs_up", "stairs_down", "fall"]
DEFAULT_PLACEMENTS = ["waist", "wrist", "pocket"]


def generate_dataset(
    num_subjects: int = 5,
    minutes_per_activity: float = 1.0,
    fs: float = 50.0,
    activities: Optional[List[str]] = None,
    placements: Optional[List[str]] = None,
    seed: Optional[int] = 7,
    include_spo2: bool = True,
    include_hrv: bool = True,
) -> pd.DataFrame:
    activities = activities or DEFAULT_ACTIVITIES
    placements = placements or DEFAULT_PLACEMENTS

    r = _rng(seed)
    all_rows: List[pd.DataFrame] = []
    duration_s = float(minutes_per_activity * 60.0)

    for sid in range(1, num_subjects + 1):
        subj = make_subject_twin(r, sid)

        for placement in placements:
            sensor = make_sensor_twin(r, placement, fs)

            for act in activities:
                seg_seed = int(r.integers(0, 2**32 - 1, dtype=np.uint32).item())
                rs = _rng(seg_seed)

                room = pick_room_for_activity(act, rs)

                t, acc_b, gyr_b = simulate_activity_body(
                    activity=act,
                    duration_s=duration_s,
                    fs=fs,
                    subject=subj,
                    r=rs,
                    g=9.81
                )
                acc_s, gyr_s = apply_sensor_model(acc_b, gyr_b, subj, sensor, rs)

                hr_bpm, spo2_pct, hrv_rmssd = simulate_vitals(act, t, subj, rs)

                seg = pd.DataFrame({
                    "time_s": t,
                    "ax": acc_s[:, 0],
                    "ay": acc_s[:, 1],
                    "az": acc_s[:, 2],
                    "gx": gyr_s[:, 0],
                    "gy": gyr_s[:, 1],
                    "gz": gyr_s[:, 2],
                    "activity": act,
                    "room": room,
                    "heart_rate_bpm": hr_bpm,
                    "subject_id": sid,
                    "placement": placement,
                    "fs_hz": fs
                })

                if include_spo2:
                    seg["spo2_pct"] = spo2_pct
                if include_hrv:
                    seg["hrv_rmssd_ms"] = hrv_rmssd

                all_rows.append(seg)

    return pd.concat(all_rows, ignore_index=True)


def make_windows(
    df: pd.DataFrame,
    window_s: float = 2.0,
    stride_s: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    if df.empty:
        return np.zeros((0, 0, 6), dtype=np.float32), np.array([], dtype=object), pd.DataFrame()

    fs = float(df["fs_hz"].iloc[0])
    w = int(window_s * fs)
    s = int(stride_s * fs)
    cols = ["ax", "ay", "az", "gx", "gy", "gz"]

    X_list: List[np.ndarray] = []
    y_list: List[str] = []
    meta_rows: List[dict] = []

    for (sid, placement, act), g in df.groupby(["subject_id", "placement", "activity"], sort=False):
        g = g.reset_index(drop=True)
        n = len(g)
        for start in range(0, n - w + 1, s):
            end = start + w
            X_list.append(g.loc[start:end - 1, cols].to_numpy(dtype=np.float32))
            y_list.append(act)
            meta_rows.append({
                "subject_id": int(sid),
                "placement": str(placement),
                "activity": str(act),
                "start_time_s": float(g.loc[start, "time_s"]),
                "end_time_s": float(g.loc[end - 1, "time_s"]),
            })

    X = np.stack(X_list, axis=0) if X_list else np.zeros((0, w, 6), dtype=np.float32)
    y = np.array(y_list, dtype=object)
    meta = pd.DataFrame(meta_rows)
    return X, y, meta


def stream_twin_state(df: pd.DataFrame) -> Iterator[dict]:
    """
    Yields one sample at a time (like real-time) with a "digital twin state" dict.
    """
    for row in df.itertuples(index=False):
        d = {
            "time_s": float(row.time_s),
            "activity": str(row.activity),
            "room": str(row.room),
            "heart_rate_bpm": float(row.heart_rate_bpm),
            "subject_id": int(row.subject_id),
            "placement": str(row.placement),
            "ax": float(row.ax), "ay": float(row.ay), "az": float(row.az),
            "gx": float(row.gx), "gy": float(row.gy), "gz": float(row.gz),
        }
        if hasattr(row, "spo2_pct"):
            d["spo2_pct"] = float(row.spo2_pct)
        if hasattr(row, "hrv_rmssd_ms"):
            d["hrv_rmssd_ms"] = float(row.hrv_rmssd_ms)
        yield d


def demo() -> None:
    df = generate_dataset(
        num_subjects=2,
        minutes_per_activity=0.15,  # ~9 seconds each
        fs=50,
        seed=7,
        include_spo2=True,
        include_hrv=True,
    )

    print("Generated rows:", len(df))
    print(df.head(3))

    X, y, meta = make_windows(df, window_s=2.0, stride_s=1.0)
    print("Windows X shape:", X.shape)
    print("Labels distribution:\n", pd.Series(y).value_counts())

    df.to_csv("synthetic_har_context_vitals_long.csv", index=False)
    meta.to_csv("synthetic_har_context_vitals_windows_meta.csv", index=False)
    np.save("synthetic_har_context_vitals_windows_X.npy", X)
    np.save("synthetic_har_context_vitals_windows_y.npy", y)

    print("Saved:")
    print(" - synthetic_har_context_vitals_long.csv")
    print(" - synthetic_har_context_vitals_windows_meta.csv")
    print(" - synthetic_har_context_vitals_windows_X.npy")
    print(" - synthetic_har_context_vitals_windows_y.npy")


if __name__ == "__main__":
    demo()