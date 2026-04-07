"""
Supervised Fraud Detection — XGBoost + Random Forest.

Layer 1 of the three-layer detection system.
Trained on labelled PaySim mobile money transactions.

Key design decisions:
  - Optimised for RECALL (missing fraud >> false alarm)
  - scale_pos_weight handles severe class imbalance
  - SHAP explanations for every prediction
  - Threshold tunable for different risk appetites
"""

import logging
import pickle
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score,
    average_precision_score, confusion_matrix,
    precision_recall_curve
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import (
    XGBOOST_PARAMS, RANDOM_FOREST_PARAMS,
    TARGET_COL, RANDOM_STATE, TEST_SIZE, MODELS_DIR
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

DATA_PROC  = Path(__file__).resolve().parents[2] / "data" / "processed"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_features(dev_mode: bool = False) -> pd.DataFrame:
    fname = "paysim_features_dev.parquet" if dev_mode else "paysim_features.parquet"
    path  = DATA_PROC / fname
    df    = pd.read_parquet(path)
    log.info(f"Loaded features: {df.shape} | fraud: {df[TARGET_COL].sum():,} ({df[TARGET_COL].mean()*100:.3f}%)")
    return df


def get_feature_cols(df: pd.DataFrame) -> list:
    exclude = [TARGET_COL, "type", "fraud_eligible",
               "mobile_subscriptions_norm"]
    return [c for c in df.columns
            if c not in exclude
            and df[c].dtype in ["int64","float64","int32","float32","uint8","bool","int8"]]


def split_data(df: pd.DataFrame, feature_cols: list):
    X = df[feature_cols]
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=y
    )
    log.info(f"Train: {X_train.shape} | Test: {X_test.shape}")
    log.info(f"Train fraud: {y_train.sum()} | Test fraud: {y_test.sum()}")
    return X_train, X_test, y_train, y_test


def train_xgboost(X_train, y_train):
    try:
        from xgboost import XGBClassifier
    except ImportError:
        log.error("Install xgboost: pip install xgboost")
        return None

    fraud_ratio = (y_train == 0).sum() / (y_train == 1).sum()
    params = {**XGBOOST_PARAMS, "scale_pos_weight": fraud_ratio}

    log.info(f"Training XGBoost (scale_pos_weight={fraud_ratio:.0f})...")
    model = XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train)],
        verbose=False,
    )
    log.info("XGBoost training complete ✓")
    return model


def train_random_forest(X_train, y_train):
    log.info("Training Random Forest...")
    model = RandomForestClassifier(**RANDOM_FOREST_PARAMS)
    model.fit(X_train, y_train)
    log.info("Random Forest training complete ✓")
    return model


def evaluate(model, X_test, y_test, model_name: str) -> dict:
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    roc_auc  = roc_auc_score(y_test, y_prob)
    avg_prec = average_precision_score(y_test, y_prob)
    cm       = confusion_matrix(y_test, y_pred)

    # Find optimal threshold for F1
    prec, rec, thresholds = precision_recall_curve(y_test, y_prob)
    f1_scores = 2 * prec * rec / (prec + rec + 1e-8)
    best_thresh = thresholds[f1_scores[:-1].argmax()]
    y_pred_opt  = (y_prob >= best_thresh).astype(int)

    print(f"\n{'='*55}")
    print(f"{model_name} — Evaluation")
    print(f"{'='*55}")
    print(f"ROC-AUC:              {roc_auc:.4f}")
    print(f"Average Precision:    {avg_prec:.4f}")
    print(f"Optimal threshold:    {best_thresh:.3f}")
    print(f"\nAt threshold 0.5:")
    print(classification_report(y_test, y_pred,
                                target_names=["Legit","Fraud"]))
    print(f"Confusion matrix:\n{cm}")
    tn, fp, fn, tp = cm.ravel()
    print(f"\nFraud caught:  {tp}/{tp+fn} ({tp/(tp+fn)*100:.1f}%)")
    print(f"False alarms:  {fp}/{tn+fp} ({fp/(tn+fp)*100:.2f}% of legit)")
    print(f"{'='*55}")

    return {
        "model_name":      model_name,
        "roc_auc":         roc_auc,
        "avg_precision":   avg_prec,
        "optimal_threshold": best_thresh,
        "fraud_recall":    tp / (tp + fn),
        "false_alarm_rate": fp / (tn + fp),
    }


def compute_shap(model, X_test, feature_cols: list, model_name: str):
    try:
        import shap
    except ImportError:
        log.warning("Install shap for explanations: pip install shap")
        return

    log.info("Computing SHAP values...")
    try:
        explainer  = shap.TreeExplainer(model)
        shap_vals  = explainer.shap_values(X_test)

        # For binary classification shap_values may be list
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]

        mean_abs   = np.abs(shap_vals).mean(axis=0)
        importance = pd.DataFrame({
            "feature":    feature_cols,
            "shap_importance": mean_abs
        }).sort_values("shap_importance", ascending=False)

        print(f"\nTop 15 features by SHAP importance ({model_name}):")
        print(importance.head(15).to_string(index=False))

        # Save SHAP importance
        out = MODELS_DIR / f"shap_importance_{model_name.lower().replace(' ','_')}.csv"
        importance.to_csv(out, index=False)
        log.info(f"SHAP importance saved → {out}")

    except Exception as e:
        log.warning(f"SHAP computation failed: {e}")


def save_model(model, name: str) -> Path:
    path = MODELS_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    log.info(f"Model saved → {path}")
    return path


def run_supervised(dev_mode: bool = False, skip_rf: bool = False) -> dict:
    df           = load_features(dev_mode)
    feature_cols = get_feature_cols(df)
    log.info(f"Features: {len(feature_cols)} → {feature_cols[:8]}...")

    X_train, X_test, y_train, y_test = split_data(df, feature_cols)

    results = {}

    # ── XGBoost ───────────────────────────────────────────────────────────────
    xgb = train_xgboost(X_train, y_train)
    if xgb:
        results["xgboost"] = evaluate(xgb, X_test, y_test, "XGBoost")
        compute_shap(xgb, X_test.sample(min(1000, len(X_test)),
                    random_state=RANDOM_STATE), feature_cols, "XGBoost")
        save_model(xgb, "xgboost_fraud")

    # ── Random Forest ─────────────────────────────────────────────────────────
    if not skip_rf:
        rf = train_random_forest(X_train, y_train)
        results["random_forest"] = evaluate(rf, X_test, y_test, "Random Forest")
        compute_shap(rf, X_test.sample(min(500, len(X_test)),
                    random_state=RANDOM_STATE), feature_cols, "Random Forest")
        save_model(rf, "random_forest_fraud")

    # ── Model comparison ──────────────────────────────────────────────────────
    if len(results) > 1:
        print("\n" + "="*55)
        print("MODEL COMPARISON")
        print("="*55)
        for name, r in results.items():
            print(f"{name:20s}  ROC-AUC: {r['roc_auc']:.4f}  "
                  f"AvgPrec: {r['avg_precision']:.4f}  "
                  f"Recall: {r['fraud_recall']:.3f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev",     action="store_true",
                        help="Use dev sample (fast)")
    parser.add_argument("--skip-rf", action="store_true",
                        help="Skip Random Forest (faster)")
    args = parser.parse_args()

    # Install dependencies if needed
    try:
        import xgboost
    except ImportError:
        import subprocess, sys
        subprocess.run([sys.executable, "-m", "pip", "install",
                       "xgboost", "shap", "--break-system-packages", "-q"])

    run_supervised(dev_mode=args.dev, skip_rf=args.skip_rf)
