import numpy as np
if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, f1_score)
from sklearn.preprocessing import StandardScaler

# === Load data ===
df = pd.read_csv("hrv_features.csv")
FEATS = ['MeanNN', 'SDNN', 'RMSSD', 'pNN50', 'pNN20',
         'MedianNN', 'IQRNN', 'CVNN', 'MinNN', 'MaxNN',
         'HR_mean', 'LF', 'HF', 'LF_HF']
X = df[FEATS].values
groups = df["subject"].values
y_bin = np.where(df["label"].values == "stress", 1, 0)

print(f"Dataset: {len(df)} windows from {df['subject'].nunique()} subjects")
print(f"Class balance: stress={y_bin.sum()} ({100*y_bin.mean():.1f}%), "
      f"non-stress={(1-y_bin).sum()}")
print()

logo = LeaveOneGroupOut()

# ===================================================================
# XGBoost + SMOTE (your main classifier)
# ===================================================================
print("#" * 60)
print("# XGBoost + SMOTE — BINARY LOSO-CV")
print("#" * 60)

xgb_acc, xgb_f1 = [], []
all_true, all_pred = [], []

for fold, (tr, te) in enumerate(logo.split(X, y_bin, groups)):
    subj = np.unique(groups[te])[0]
    X_tr, X_te = X[tr], X[te]
    y_tr, y_te = y_bin[tr], y_bin[te]

    smote = SMOTE(random_state=42)
    X_tr_sm, y_tr_sm = smote.fit_resample(X_tr, y_tr)

    xgb = XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        eval_metric='logloss', random_state=42)
    xgb.fit(X_tr_sm, y_tr_sm)
    pred = xgb.predict(X_te)
    a = accuracy_score(y_te, pred)
    f = f1_score(y_te, pred, zero_division=0)
    xgb_acc.append(a)
    xgb_f1.append(f)
    all_true.extend(y_te)
    all_pred.extend(pred)
    print(f"  Fold {fold+1:2d} ({subj}): acc={a:.4f}, F1={f:.4f}")

print(f"\nXGBoost + SMOTE LOSO:  acc={np.mean(xgb_acc):.4f} ± {np.std(xgb_acc):.4f}, "
      f"F1={np.mean(xgb_f1):.4f}")

print("\nAggregated classification report (XGBoost):")
print(classification_report(all_true, all_pred,
                            target_names=["non-stress", "stress"], digits=4))

print("Confusion matrix (XGBoost, aggregated across folds):")
cm = confusion_matrix(all_true, all_pred)
print(pd.DataFrame(cm,
                   index=["true non-stress", "true stress"],
                   columns=["pred non-stress", "pred stress"]))

# ===================================================================
# Baselines: Random Forest, Logistic Regression, SVM
# ===================================================================
print("\n" + "#" * 60)
print("# BASELINE MODELS — BINARY LOSO-CV")
print("#" * 60)

rf_acc, lr_acc, svm_acc = [], [], []
rf_f1, lr_f1, svm_f1 = [], [], []

for tr, te in logo.split(X, y_bin, groups):
    X_tr, X_te = X[tr], X[te]
    y_tr, y_te = y_bin[tr], y_bin[te]

    # All baselines get SMOTE too for fair comparison
    smote = SMOTE(random_state=42)
    X_tr_sm, y_tr_sm = smote.fit_resample(X_tr, y_tr)

    scaler = StandardScaler().fit(X_tr_sm)
    X_tr_sc = scaler.transform(X_tr_sm)
    X_te_sc = scaler.transform(X_te)

    rf = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_tr_sm, y_tr_sm)
    p = rf.predict(X_te)
    rf_acc.append(accuracy_score(y_te, p))
    rf_f1.append(f1_score(y_te, p, zero_division=0))

    lr = LogisticRegression(max_iter=2000, random_state=42)
    lr.fit(X_tr_sc, y_tr_sm)
    p = lr.predict(X_te_sc)
    lr_acc.append(accuracy_score(y_te, p))
    lr_f1.append(f1_score(y_te, p, zero_division=0))

    svm = SVC(kernel="rbf", random_state=42)
    svm.fit(X_tr_sc, y_tr_sm)
    p = svm.predict(X_te_sc)
    svm_acc.append(accuracy_score(y_te, p))
    svm_f1.append(f1_score(y_te, p, zero_division=0))

print(f"Random Forest:       acc={np.mean(rf_acc):.4f} ± {np.std(rf_acc):.4f}, F1={np.mean(rf_f1):.4f}")
print(f"Logistic Regression: acc={np.mean(lr_acc):.4f} ± {np.std(lr_acc):.4f}, F1={np.mean(lr_f1):.4f}")
print(f"SVM (RBF):           acc={np.mean(svm_acc):.4f} ± {np.std(svm_acc):.4f}, F1={np.mean(svm_f1):.4f}")

# ===================================================================
# Confusion matrix plot (XGBoost)
# ===================================================================
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["non-stress", "stress"],
            yticklabels=["non-stress", "stress"])
plt.title(f"XGBoost+SMOTE Confusion Matrix (LOSO-CV, acc={np.mean(xgb_acc):.2%})")
plt.ylabel("True")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("xgb_confusion_matrix.png", dpi=150)
print("\nSaved xgb_confusion_matrix.png")

# Normalized version (better for class-imbalanced viewing)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
plt.figure(figsize=(5, 4))
sns.heatmap(cm_norm, annot=True, fmt=".1%", cmap="Blues",
            xticklabels=["non-stress", "stress"],
            yticklabels=["non-stress", "stress"])
plt.title("XGBoost+SMOTE Confusion Matrix (normalized by true class)")
plt.ylabel("True")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("xgb_confusion_matrix_normalized.png", dpi=150)
print("Saved xgb_confusion_matrix_normalized.png")

# ===================================================================
# Feature importance (from saved XGBoost model if exists, else train fresh)
# ===================================================================
print("\n" + "#" * 60)
print("# FEATURE IMPORTANCE (XGBoost)")
print("#" * 60)

# Train final model on all data (matches what you'd deploy on Pi)
smote_final = SMOTE(random_state=42)
X_final, y_final = smote_final.fit_resample(X, y_bin)
xgb_final = XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8,
                           eval_metric='logloss', random_state=42)
xgb_final.fit(X_final, y_final)

imp = xgb_final.feature_importances_
imp_df = pd.DataFrame({"feature": FEATS, "importance": imp}).sort_values("importance", ascending=False)
for _, r in imp_df.iterrows():
    print(f"  {r['feature']:<10} {r['importance']:.4f}")

plt.figure(figsize=(7, 5))
imp_asc = imp_df.sort_values("importance", ascending=True)
plt.barh(imp_asc["feature"], imp_asc["importance"])
plt.title("XGBoost Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("xgb_feature_importance.png", dpi=150)
print("\nSaved xgb_feature_importance.png")

# ===================================================================
# Save the deployable model
# ===================================================================
joblib.dump({"model": xgb_final, "features": FEATS,
             "labels": ["non-stress", "stress"]},
            "stress_model_xgb.joblib")
print("Saved stress_model_xgb.joblib")

# ===================================================================
# Summary for report
# ===================================================================
print("\n" + "#" * 60)
print("# SUMMARY FOR FINAL REPORT")
print("#" * 60)
print(f"Dataset:            WESAD, {len(df)} windows across {df['subject'].nunique()} subjects")
print(f"Features:           {len(FEATS)} HRV features (time + frequency domain)")
print(f"Task:               Binary classification (stress vs non-stress)")
print(f"Validation:         Leave-One-Subject-Out cross-validation (15 folds)")
print(f"Class balancing:    SMOTE (training folds only)")
print()
print(f"XGBoost + SMOTE:    acc={np.mean(xgb_acc):.4f} ± {np.std(xgb_acc):.4f}, F1={np.mean(xgb_f1):.4f}")
print(f"Random Forest:      acc={np.mean(rf_acc):.4f} ± {np.std(rf_acc):.4f}, F1={np.mean(rf_f1):.4f}")
print(f"SVM (RBF):          acc={np.mean(svm_acc):.4f} ± {np.std(svm_acc):.4f}, F1={np.mean(svm_f1):.4f}")
print(f"Logistic Reg:       acc={np.mean(lr_acc):.4f} ± {np.std(lr_acc):.4f}, F1={np.mean(lr_f1):.4f}")
print()
print(f"Top 3 features: {list(imp_df.head(3)['feature'])}")