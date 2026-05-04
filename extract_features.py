import numpy as np
if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz

import pickle
import pandas as pd
import neurokit2 as nk
from pathlib import Path
from scipy.signal import butter, filtfilt
import warnings
warnings.filterwarnings('ignore')

SUBJECTS = [f"S{i}" for i in range(2, 18) if i != 12]
DATA_DIR = Path("data/WESAD")
LABEL_MAP = {1: "baseline", 2: "stress", 3: "amusement"}

def bandpass(sig, fs=64, low=0.5, high=8.0, order=4):
    """Filter PPG to heart-rate frequency band."""
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, sig)

records = []
for subj in SUBJECTS:
    path = DATA_DIR / subj / f"{subj}.pkl"
    if not path.exists():
        continue
    print(f"Processing {subj}...", end=" ", flush=True)
    
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    
    bvp = d["signal"]["wrist"]["BVP"].flatten()
    bvp = bandpass(bvp, fs=64)  # clean the signal first
    
    labels = d["label"].flatten()
    label_ratio = len(labels) // len(bvp)
    labels_ds = labels[::label_ratio][:len(bvp)]
    
    # LONGER windows: 60 sec instead of 30 (better for frequency-domain HRV)
    WIN = 64 * 60
    STEP = 64 * 10   # 10-sec stride
    
    n_success = 0
    for start in range(0, len(bvp) - WIN, STEP):
        seg = bvp[start:start + WIN]
        lbl_seg = labels_ds[start:start + WIN]
        valid = lbl_seg[np.isin(lbl_seg, list(LABEL_MAP.keys()))]
        if len(valid) < WIN * 0.7:  # stricter: 70% of window must have valid label
            continue
        lbl = int(np.bincount(valid).argmax())
        
        try:
            signals, _ = nk.ppg_process(seg, sampling_rate=64)
            hrv_t = nk.hrv_time(signals, sampling_rate=64, show=False)
            
            row = {
                "subject": subj,
                "label":   LABEL_MAP[lbl],
                "MeanNN":  hrv_t["HRV_MeanNN"].values[0],
                "SDNN":    hrv_t["HRV_SDNN"].values[0],
                "RMSSD":   hrv_t["HRV_RMSSD"].values[0],
                "pNN50":   hrv_t["HRV_pNN50"].values[0],
                "pNN20":   hrv_t["HRV_pNN20"].values[0],
                "MedianNN": hrv_t["HRV_MedianNN"].values[0],
                "IQRNN":   hrv_t["HRV_IQRNN"].values[0],
                "CVNN":    hrv_t["HRV_CVNN"].values[0],
                "MinNN":   hrv_t["HRV_MinNN"].values[0],
                "MaxNN":   hrv_t["HRV_MaxNN"].values[0],
            }
            
            # Derived: heart rate from MeanNN (ms → bpm)
            if row["MeanNN"] > 0:
                row["HR_mean"] = 60000.0 / row["MeanNN"]
            else:
                row["HR_mean"] = np.nan
            
            # Try frequency-domain on 60-sec window (should now work)
            try:
                hrv_f = nk.hrv_frequency(signals, sampling_rate=64, show=False)
                row["LF"] = hrv_f["HRV_LF"].values[0]
                row["HF"] = hrv_f["HRV_HF"].values[0]
                row["LF_HF"] = hrv_f["HRV_LFHF"].values[0]
            except Exception:
                row["LF"] = np.nan
                row["HF"] = np.nan
                row["LF_HF"] = np.nan
            
            # Only keep if time-domain features are all present
            time_feats = ["MeanNN", "SDNN", "RMSSD", "pNN50", "HR_mean"]
            if not any(pd.isna(row[k]) for k in time_feats):
                records.append(row)
                n_success += 1
        except Exception:
            continue
    
    print(f"{n_success} windows")

df = pd.DataFrame(records)

# Report NaN counts per feature
print("\nNaN counts per feature:")
for col in df.columns:
    if col not in ["subject", "label"]:
        n_nan = df[col].isna().sum()
        if n_nan > 0:
            print(f"  {col}: {n_nan} ({100*n_nan/len(df):.1f}%)")

# Impute remaining NaNs with per-column median
for col in df.columns:
    if col not in ["subject", "label"] and df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())

df.to_csv("hrv_features.csv", index=False)
print(f"\n{'='*50}")
print(f"Saved {len(df)} windows across {df['subject'].nunique()} subjects")
print(f"\nLabel distribution:")
print(df["label"].value_counts())
print(f"\nFeature columns: {[c for c in df.columns if c not in ['subject', 'label']]}")