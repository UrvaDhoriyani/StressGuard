import pickle
import numpy as np

with open("data/WESAD/S2/S2.pkl", "rb") as f:
    d = pickle.load(f, encoding="latin1")

print("=" * 50)
print("TOP-LEVEL STRUCTURE")
print("=" * 50)
print("Keys:", list(d.keys()))

print("\n" + "=" * 50)
print("SIGNAL STRUCTURE")
print("=" * 50)
print("Signal sources:", list(d["signal"].keys()))
print("Wrist signals:", list(d["signal"]["wrist"].keys()))

bvp = d["signal"]["wrist"]["BVP"]
print(f"\nBVP shape: {bvp.shape}")
print(f"BVP dtype: {bvp.dtype}")
flat_bvp = bvp.flatten()
print(f"BVP flat length: {len(flat_bvp)}")
print(f"BVP duration at 64Hz: {len(flat_bvp)/64:.1f} seconds")
print(f"BVP min: {flat_bvp.min():.3f}, max: {flat_bvp.max():.3f}")
print(f"BVP first 5 values: {flat_bvp[:5]}")

print("\n" + "=" * 50)
print("LABEL STRUCTURE")
print("=" * 50)
labels = d["label"]
print(f"Labels shape: {labels.shape}")
flat_labels = labels.flatten()
print(f"Labels dtype: {flat_labels.dtype}")
print(f"Labels unique values: {np.unique(flat_labels)}")
print(f"Label counts:")
unique, counts = np.unique(flat_labels, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  Label {u}: {c} samples")
print(f"Labels duration at 700Hz: {len(flat_labels)/700:.1f} seconds")

print("\n" + "=" * 50)
print("RATIO CHECK")
print("=" * 50)
ratio = len(flat_labels) / len(flat_bvp)
print(f"Labels/BVP ratio: {ratio:.3f} (expect ~10.9 = 700/64)")

# Check if downsampled labels at BVP sample rate contain class labels
label_ratio = len(flat_labels) // len(flat_bvp)
labels_ds = flat_labels[::label_ratio][:len(flat_bvp)]
print(f"\nDownsampled labels unique values: {np.unique(labels_ds)}")

# Quick window-level check
WIN = 64 * 30
STEP = 64 * 5
print("\n" + "=" * 50)
print("WINDOW-LEVEL ANALYSIS (first 10 windows)")
print("=" * 50)
count_valid = 0
count_total = 0
for start in range(0, len(flat_bvp) - WIN, STEP):
    count_total += 1
    lbl_seg = labels_ds[start:start + WIN]
    valid = lbl_seg[np.isin(lbl_seg, [1, 2, 3])]
    if count_total <= 10:
        print(f"Window {start}: unique labels = {np.unique(lbl_seg)}, "
              f"valid samples = {len(valid)}/{WIN}")
    if len(valid) > 0:
        count_valid += 1

print(f"\nTotal windows: {count_total}")
print(f"Windows with valid label (1/2/3): {count_valid}")