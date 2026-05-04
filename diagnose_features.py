import numpy as np
if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz

import pickle
import pandas as pd
import neurokit2 as nk
import warnings
warnings.filterwarnings('ignore')

with open("data/WESAD/S2/S2.pkl", "rb") as f:
    d = pickle.load(f, encoding="latin1")

bvp = d["signal"]["wrist"]["BVP"].flatten()
labels = d["label"].flatten()
label_ratio = len(labels) // len(bvp)
labels_ds = labels[::label_ratio][:len(bvp)]

# Grab one clearly-labeled window
WIN = 64 * 30
for start in range(0, len(bvp) - WIN, 64 * 5):
    lbl_seg = labels_ds[start:start + WIN]
    if np.all(lbl_seg == 2):  # pure stress window
        print(f"Found pure stress window at {start/64:.0f}s")
        seg = bvp[start:start + WIN]
        break

signals, _ = nk.ppg_process(seg, sampling_rate=64)
hrv_t = nk.hrv_time(signals, sampling_rate=64, show=False)
hrv_f = nk.hrv_frequency(signals, sampling_rate=64, show=False)

print("\nTime-domain HRV:")
for col in hrv_t.columns:
    val = hrv_t[col].values[0]
    print(f"  {col}: {val}")

print("\nFrequency-domain HRV:")
for col in hrv_f.columns:
    val = hrv_f[col].values[0]
    print(f"  {col}: {val}")