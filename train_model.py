import numpy as np
if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv("hrv_features.csv")
FEATS = [c for c in df.columns if c not in ["subject", "label"]]
print(f"Using features: {FEATS}\n")

def per_subject_zscore(df, features):
    """Normalize each subject's features by their own mean/std."""
    df_norm = df.copy()
    for subj in df["subject"].unique():
        mask = df["subject"] == subj
        for feat in features:
            vals = df.loc[mask, feat]
            mean, std = vals.mean(), vals.std()
            if std > 0:
                df_norm.loc[mask, feat] = (vals - mean) / std
            else:
                df_norm.loc[mask, feat] = 0.0
    return df_norm

df_norm = per_subject_zscore(df, FEATS)

X = df_norm[FEATS].values
groups = df_norm["subject"].values

# ===================================================================
# 3-CLASS EVALUATION with per-subject normalization
# ===================================================================
print("#" * 60)
print("# 3-CLASS (baseline / amusement / stress) — LOSO-CV with per-subject normalization")
print("#" * 60)

y_3c = df_norm["label"].values
le_3c = LabelEncoder()
y_3c_enc = le_3c.fit_transform(y_3c)

logo = LeaveOneGroupOut()
rf_scores, lr_scores, svm_scores = [], [], []

for train_idx, test_idx in logo.split(X, y_3c_enc, groups):
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y_3c_enc[train_idx], y_3c_enc[test_idx]

    rf = RandomForestClassifier(n_estimators=300, max_depth=None,
                                class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    rf_scores.append(accuracy_score(y_te, rf.predict(X_te)))

    lr = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    lr.fit(X_tr, y_tr)
    lr_scores.append(accuracy_score(y_te, lr.predict(X_te)))

    svm = SVC(kernel="rbf", class_weight="balanced", random_state=42)
    svm.fit(X_tr, y_tr)
    svm_scores.append(accuracy_score(y_te, svm.predict(X_te)))

print(f"\nRandom Forest LOSO-CV:       {np.mean(rf_scores):.4f} ± {np.std(rf_scores):.4f}")
print(f"Logistic Regression LOSO-CV: {np.mean(lr_scores):.4f} ± {np.std(lr_scores):.4f}")
print(f"SVM LOSO-CV:                 {np.mean(svm_scores):.4f} ± {np.std(svm_scores):.4f}")

# ===================================================================
# BINARY EVALUATION with per-subject normalization
# ===================================================================
print("\n" + "#" * 60)
print("# BINARY (stress vs non-stress) — LOSO-CV with per-subject normalization")
print("#" * 60)

y_bin = np.where(df_norm["label"].values == "stress", 1, 0)

rf_scores_b, lr_scores_b, svm_scores_b = [], [], []
rf_f1_b, lr_f1_b, svm_f1_b = [], [], []

for train_idx, test_idx in logo.split(X, y_bin, groups):
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y_bin[train_idx], y_bin[test_idx]

    rf = RandomForestClassifier(n_estimators=300, max_depth=None,
                                class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    pred = rf.predict(X_te)
    rf_scores_b.append(accuracy_score(y_te, pred))
    rf_f1_b.append(f1_score(y_te, pred))

    lr = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    lr.fit(X_tr, y_tr)
    pred = lr.predict(X_te)
    lr_scores_b.append(accuracy_score(y_te, pred))
    lr_f1_b.append(f1_score(y_te, pred))

    svm = SVC(kernel="rbf", class_weight="balanced", random_state=42)
    svm.fit(X_tr, y_tr)
    pred = svm.predict(X_te)
    svm_scores_b.append(accuracy_score(y_te, pred))
    svm_f1_b.append(f1_score(y_te, pred))

print(f"\nRandom Forest:       acc={np.mean(rf_scores_b):.4f} ± {np.std(rf_scores_b):.4f}, F1={np.mean(rf_f1_b):.4f}")
print(f"Logistic Regression: acc={np.mean(lr_scores_b):.4f} ± {np.std(lr_scores_b):.4f}, F1={np.mean(lr_f1_b):.4f}")
print(f"SVM:                 acc={np.mean(svm_scores_b):.4f} ± {np.std(svm_scores_b):.4f}, F1={np.mean(svm_f1_b):.4f}")

# ===================================================================
# FINAL MODEL for Pi deployment
# ===================================================================
print("\n" + "#" * 60)
print("# FINAL MODEL (trained on all data, for Pi deployment)")
print("#" * 60)

# Train on everything with per-subject normalization applied
rf_final = RandomForestClassifier(n_estimators=300, max_depth=None,
                                  class_weight="balanced", random_state=42, n_jobs=-1)
rf_final.fit(X, y_3c_enc)

# Held-out single-split for confusion matrix visualization
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y_3c_enc, groups))
rf_demo = RandomForestClassifier(n_estimators=300, max_depth=None,
                                 class_weight="balanced", random_state=42, n_jobs=-1)
rf_demo.fit(X[train_idx], y_3c_enc[train_idx])
pred = rf_demo.predict(X[test_idx])
demo_acc = accuracy_score(y_3c_enc[test_idx], pred)
print(f"\nHeld-out test (single split): acc={demo_acc:.4f}")
print(classification_report(y_3c_enc[test_idx], pred, target_names=le_3c.classes_))

print("\nFEATURE IMPORTANCE:")
for f, imp in sorted(zip(FEATS, rf_final.feature_importances_), key=lambda t: -t[1]):
    print(f"  {f:<10} {imp:.4f}")

# Save plots
cm = confusion_matrix(y_3c_enc[test_idx], pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le_3c.classes_, yticklabels=le_3c.classes_)
plt.title(f"Random Forest Confusion Matrix (acc={demo_acc:.2%})")
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("\nSaved confusion_matrix.png")

plt.figure(figsize=(7, 5))
imp_df = pd.DataFrame({
    "feature": FEATS,
    "importance": rf_final.feature_importances_
}).sort_values("importance", ascending=True)
plt.barh(imp_df["feature"], imp_df["importance"])
plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
print("Saved feature_importance.png")

# Save feature stats per-subject for Pi normalization at inference time
# For deployment, use population mean/std as approximation
pop_mean = df[FEATS].mean()
pop_std = df[FEATS].std()

joblib.dump({
    "model": rf_final,
    "encoder": le_3c,
    "features": FEATS,
    "normalize_mean": pop_mean.to_dict(),
    "normalize_std": pop_std.to_dict(),
}, "stress_model.joblib")
print("Saved stress_model.joblib")

# Also save a binary-classification version
rf_bin_final = RandomForestClassifier(n_estimators=300, max_depth=None,
                                       class_weight="balanced", random_state=42, n_jobs=-1)
rf_bin_final.fit(X, y_bin)
joblib.dump({
    "model": rf_bin_final,
    "features": FEATS,
    "normalize_mean": pop_mean.to_dict(),
    "normalize_std": pop_std.to_dict(),
    "labels": ["non-stress", "stress"],
}, "stress_model_binary.joblib")
print("Saved stress_model_binary.joblib")