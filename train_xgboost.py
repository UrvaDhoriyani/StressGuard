import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv("hrv_features.csv")
FEATS = ['MeanNN', 'SDNN', 'RMSSD', 'pNN50', 'pNN20', 'MedianNN', 'IQRNN', 'CVNN', 'MinNN', 'MaxNN', 'HR_mean', 'LF', 'HF', 'LF_HF']
X = df[FEATS].values
groups = df["subject"].values

# Create Binary Labels: 1 for stress, 0 for non-stress (baseline/amusement)
y_str = df["label"].values
y_bin = np.where(y_str == "stress", 1, 0)

print("############################################################")
print("# BINARY (stress vs non-stress) — LOSO-CV (XGBoost + SMOTE)")
print("############################################################")

logo = LeaveOneGroupOut()
accs, f1s = [], []

for train_idx, test_idx in logo.split(X, y_bin, groups):
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y_bin[train_idx], y_bin[test_idx]
    
    # 1. Apply SMOTE to the training fold ONLY
    smote = SMOTE(random_state=42)
    X_tr_sm, y_tr_sm = smote.fit_resample(X_tr, y_tr)
    
    # 2. Train XGBoost
    xgb = XGBClassifier(
        n_estimators=150, 
        max_depth=5, 
        learning_rate=0.05,
        subsample=0.8, 
        colsample_bytree=0.8, 
        eval_metric='logloss',
        random_state=42
    )
    xgb.fit(X_tr_sm, y_tr_sm)
    
    # 3. Predict on the held-out test subject
    preds = xgb.predict(X_te)
    accs.append(accuracy_score(y_te, preds))
    f1s.append(f1_score(y_te, preds, zero_division=0))

print(f"XGBoost + SMOTE LOSO-CV Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
print(f"XGBoost + SMOTE LOSO-CV F1-Score: {np.mean(f1s):.4f}\n")

print("############################################################")
print("# FINAL MODEL (trained on all data, for Pi deployment)")
print("############################################################")

# Train final deployment model using all data balanced by SMOTE
smote_final = SMOTE(random_state=42)
X_sm_final, y_bin_sm_final = smote_final.fit_resample(X, y_bin)

xgb_final = XGBClassifier(
    n_estimators=150, 
    max_depth=5, 
    learning_rate=0.05,
    subsample=0.8, 
    colsample_bytree=0.8, 
    random_state=42
)
xgb_final.fit(X_sm_final, y_bin_sm_final)

# Export the trained model
joblib.dump({"model": xgb_final, "features": FEATS}, "stress_model_xgb.joblib")
print("Saved stress_model_xgb.joblib")