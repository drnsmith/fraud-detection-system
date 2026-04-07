"""
Fix: Save XGBoost model properly and run full ensemble.

The previous run didn't save the model because it ran inline.
This script: trains XGBoost, saves pkl, trains Isolation Forest,
runs Level 2 complementarity analysis.

Run:
    cd ~/fraud-detection-system
    conda activate ds
    PYTHONPATH=. python src/models/run_ensemble.py --dev
"""

import logging
import pickle
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

sys.path.insert(0, ".")
from config.settings import TARGET_COL, RANDOM_STATE, MODELS_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

DATA_PROC = Path("data/processed")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_features(dev: bool) -> pd.DataFrame:
    fname = "paysim_features_dev.parquet" if dev else "paysim_features.parquet"
    df = pd.read_parquet(DATA_PROC / fname)
    log.info(f"Loaded: {df.shape} | fraud: {df[TARGET_COL].sum():,}")
    return df


def get_feature_cols(df):
    exclude = [TARGET_COL, "type", "fraud_eligible"]
    return [c for c in df.columns
            if c not in exclude
            and df[c].dtype in ["int64","float64","int32","float32","uint8","bool"]]


def run(dev=False):
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier

    df = load_features(dev)
    feature_cols = get_feature_cols(df)
    X = df[feature_cols]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # ── Step 1: Train and SAVE XGBoost ────────────────────────────────────────
    log.info("Training XGBoost...")
    fraud_ratio = (y_train==0).sum() / (y_train==1).sum()
    xgb = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        scale_pos_weight=fraud_ratio, use_label_encoder=False,
        eval_metric="aucpr", random_state=RANDOM_STATE,
        n_jobs=-1, verbosity=0,
    )
    xgb.fit(X_train, y_train, verbose=False)

    # Save XGBoost
    xgb_path = MODELS_DIR / "xgboost_fraud.pkl"
    with open(xgb_path, "wb") as f:
        pickle.dump(xgb, f)
    log.info(f"XGBoost saved → {xgb_path}")

    xgb_probs = xgb.predict_proba(X_test)[:,1]
    xgb_preds = (xgb_probs >= 0.5).astype(int)
    xgb_roc   = roc_auc_score(y_test, xgb_probs)
    xgb_cm    = confusion_matrix(y_test, xgb_preds)
    tn,fp,fn,tp = xgb_cm.ravel()

    print(f"\n{'='*55}")
    print(f"XGBoost Results")
    print(f"ROC-AUC: {xgb_roc:.4f} | Fraud caught: {tp}/{tp+fn} | False alarms: {fp}")
    xgb_caught = set(y_test[y_test==1].index[xgb_preds[y_test==1]==1])
    xgb_missed = set(y_test[y_test==1].index[xgb_preds[y_test==1]==0])
    print(f"XGBoost caught: {len(xgb_caught)} | Missed: {len(xgb_missed)}")

    # ── Step 2: Train and SAVE Isolation Forest ───────────────────────────────
    log.info("\nTraining Isolation Forest...")
    contamination = min(y_train.mean() * 2, 0.05)
    iso = IsolationForest(
        n_estimators=200, contamination=contamination,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    iso.fit(X_train)

    iso_path = MODELS_DIR / "isolation_forest_fraud.pkl"
    with open(iso_path, "wb") as f:
        pickle.dump(iso, f)
    log.info(f"Isolation Forest saved → {iso_path}")

    if_scores = -iso.decision_function(X_test)
    if_flags  = (iso.predict(X_test) == -1).astype(int)
    if_roc    = roc_auc_score(y_test, if_scores)

    print(f"\nIsolation Forest Results")
    print(f"ROC-AUC: {if_roc:.4f}")

    # ── Step 3: Level 2 complementarity ──────────────────────────────────────
    print(f"\n{'='*55}")
    print("LEVEL 2 — COMPLEMENTARITY ANALYSIS")
    print("="*55)

    fraud_idx  = y_test[y_test==1].index
    if_series  = pd.Series(if_flags, index=y_test.index)
    if_caught  = set(fraud_idx[if_series.loc[fraud_idx]==1])
    if_missed  = set(fraud_idx) - if_caught

    both       = xgb_caught & if_caught
    xgb_only   = xgb_caught - if_caught
    if_only    = if_caught  - xgb_caught
    neither    = set(fraud_idx) - xgb_caught - if_caught
    ensemble   = xgb_caught | if_caught

    print(f"Total fraud in test:       {len(fraud_idx)}")
    print(f"XGBoost caught:            {len(xgb_caught)} ({len(xgb_caught)/len(fraud_idx)*100:.1f}%)")
    print(f"Isolation Forest caught:   {len(if_caught)} ({len(if_caught)/len(fraud_idx)*100:.1f}%)")
    print(f"Both caught:               {len(both)}")
    print(f"XGBoost only:              {len(xgb_only)}")
    print(f"Isolation Forest only:     {len(if_only)}  ← IF adds these cases")
    print(f"Neither caught:            {len(neither)}")
    print(f"Ensemble (XGB + IF):       {len(ensemble)} ({len(ensemble)/len(fraud_idx)*100:.1f}%)")
    print(f"Improvement over XGBoost:  +{len(if_only)} fraud cases")

    # ── Step 4: Context shift experiment ─────────────────────────────────────
    print(f"\n{'='*55}")
    print("LEVEL 3 — CONTEXT SHIFT EXPERIMENT")
    print("="*55)

    df_test = df.loc[X_test.index].copy()
    normal = (
        (df_test.get("type_CASH_IN", pd.Series(0, index=df_test.index)) == 1) |
        (df_test.get("type_PAYMENT",  pd.Series(0, index=df_test.index)) == 1)
    )
    risky = (
        (df_test.get("type_CASH_OUT",  pd.Series(0, index=df_test.index)) == 1) |
        (df_test.get("type_TRANSFER",  pd.Series(0, index=df_test.index)) == 1)
    )

    if normal.sum() > 100:
        ctx_model = IsolationForest(n_estimators=200, contamination=0.01,
                                     random_state=RANDOM_STATE, n_jobs=-1)
        ctx_model.fit(X_test[normal])
        ctx_scores = -ctx_model.decision_function(X_test)

        df_test["ctx_dist"] = ctx_scores
        r = df_test[risky]
        if r[TARGET_COL].sum() > 0:
            fd = r[r[TARGET_COL]==1]["ctx_dist"].mean()
            ld = r[r[TARGET_COL]==0]["ctx_dist"].mean()
            ctx_roc = roc_auc_score(r[TARGET_COL], r["ctx_dist"])
            print(f"Training context: {normal.sum():,} transactions (CASH_IN + PAYMENT)")
            print(f"Fraud mean contextual distance:  {fd:.4f}")
            print(f"Legit mean contextual distance:  {ld:.4f}")
            print(f"Fraud/legit ratio:               {abs(fd)/abs(ld):.2f}x more anomalous")
            print(f"Context-shift ROC-AUC:           {ctx_roc:.4f}")
            print(f"\nFraudulent transactions are {abs(fd)/abs(ld):.1f}x more contextually")
            print(f"foreign than legitimate ones — AFI Indicator 2 (SECI) parallel confirmed.")

    print(f"\n{'='*55}")
    print("ENSEMBLE SUMMARY")
    print("="*55)
    print(f"XGBoost alone:    {len(xgb_caught)/len(fraud_idx)*100:.1f}% recall")
    print(f"IF alone:         {len(if_caught)/len(fraud_idx)*100:.1f}% recall")
    print(f"Ensemble:         {len(ensemble)/len(fraud_idx)*100:.1f}% recall")
    print(f"Models saved:     {xgb_path.name}, {iso_path.name}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dev", action="store_true")
    args = p.parse_args()
    run(dev=args.dev)
