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
FS_ECG = 700  # chest ECG sampling rate in WESAD

records = []
for subj in SUBJECTS:
    path = DATA_DIR / subj / f"{subj}.pkl"
    if not path.exists():
        continue
    print(f"Processing {subj}...", end=" ", flush=True)
    
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    
    ecg = d["signal"]["chest"]["ECG"].flatten()
    labels = d["label"].flatten()  # also 700Hz, matches ECG
    
    WIN = FS_ECG * 60    # 60-sec windows
    STEP = FS_ECG * 10   # 10-sec stride
    
    n_success = 0
    for start in range(0, len(ecg) - WIN, STEP):
        seg = ecg[start:start + WIN]
        lbl_seg = labels[start:start + WIN]
        valid = lbl_seg[np.isin(lbl_seg, list(LABEL_MAP.keys()))]
        if len(valid) < WIN * 0.7:
            continue
        lbl = int(np.bincount(valid).argmax())
        
        try:
            signals, _ = nk.ecg_process(seg, sampling_rate=FS_ECG)
            hrv_t = nk.hrv_time(signals, sampling_rate=FS_ECG, show=False)
            
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
            row["HR_mean"] = 60000.0 / row["MeanNN"] if row["MeanNN"] > 0 else np.nan
            
            try:
                hrv_f = nk.hrv_frequency(signals, sampling_rate=FS_ECG, show=False)
                row["LF"] = hrv_f["HRV_LF"].values[0]
                row["HF"] = hrv_f["HRV_HF"].values[0]
                row["LF_HF"] = hrv_f["HRV_LFHF"].values[0]
            except Exception:
                row["LF"] = np.nan
                row["HF"] = np.nan
                row["LF_HF"] = np.nan
            
            time_feats = ["MeanNN", "SDNN", "RMSSD", "pNN50", "HR_mean"]
            if not any(pd.isna(row[k]) for k in time_feats):
                records.append(row)
                n_success += 1
        except Exception:
            continue
    
    print(f"{n_success} windows")

df = pd.DataFrame(records)
for col in df.columns:
    if col not in ["subject", "label"] and df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())

df.to_csv("hrv_features_ecg.csv", index=False)
print(f"\nSaved {len(df)} windows across {df['subject'].nunique()} subjects")
print(df["label"].value_counts())