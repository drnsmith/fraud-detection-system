"""
Isolation Forest Anomaly Detection — Layer 2 of fraud detection system.

Validation framework:
  Level 1 — Extrinsic: ROC-AUC and Precision@K against known labels
  Level 2 — Complementarity: catches fraud that XGBoost missed?
  Level 3 — Context shift experiment (AFI connection):
             Train on normal-context transactions (CASH_IN, PAYMENT),
             score all transactions. Anomaly score = contextual distance.
             Structurally identical to AFI Indicator 2 (SECI):
             systematic error concentration in contextually distant cases.

Connection to Algorithmic Foreignness research:
  The context shift experiment demonstrates that an AI system trained
  in one transactional context (normal payments) generates systematically
  higher anomaly scores in a different context (CASH_OUT/TRANSFER).
  This is the fraud detection analogue of cross-border construct shift:
  the model's learned representation of 'normal' does not transfer.
"""

import logging
import pickle
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import (
    ISOLATION_FOREST_PARAMS, TARGET_COL,
    RANDOM_STATE, MODELS_DIR
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

DATA_PROC = Path(__file__).resolve().parents[2] / "data" / "processed"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_features(dev_mode: bool = False) -> pd.DataFrame:
    fname = "paysim_features_dev.parquet" if dev_mode else "paysim_features.parquet"
    df = pd.read_parquet(DATA_PROC / fname)
    log.info(f"Loaded: {df.shape} | fraud: {df[TARGET_COL].sum():,}")
    return df


def get_feature_cols(df: pd.DataFrame) -> list:
    exclude = [TARGET_COL, "type", "fraud_eligible"]
    return [c for c in df.columns
            if c not in exclude
            and df[c].dtype in ["int64","float64","int32","float32","uint8","bool"]]


def train_isolation_forest(
    X_train: pd.DataFrame,
    contamination: float = 0.002,
    label: str = "standard",
) -> IsolationForest:
    """
    Train Isolation Forest on unlabelled data.
    contamination: expected fraction of anomalies (slightly above true fraud rate).
    """
    params = {**ISOLATION_FOREST_PARAMS, "contamination": contamination}
    log.info(f"Training Isolation Forest ({label}, contamination={contamination})...")
    model = IsolationForest(**params)
    model.fit(X_train)
    log.info(f"Training complete ✓  ({len(X_train):,} observations)")
    return model


def evaluate_anomaly_detection(
    model: IsolationForest,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    top_k: int = 500,
) -> dict:
    """
    Evaluate anomaly detection against known fraud labels.

    Anomaly score convention:
      decision_function() returns higher = more normal (more negative = anomaly)
      We negate so that higher score = more anomalous = more fraud-like
    """
    # Raw anomaly scores (negated: high = anomalous)
    anomaly_scores = -model.decision_function(X)
    predictions    = model.predict(X)   # -1 = anomaly, 1 = normal
    fraud_flags    = (predictions == -1).astype(int)

    n_flagged = fraud_flags.sum()
    n_fraud   = y.sum()

    # ROC-AUC using continuous anomaly score
    roc_auc  = roc_auc_score(y, anomaly_scores)
    avg_prec = average_precision_score(y, anomaly_scores)

    # Confusion matrix
    cm = confusion_matrix(y, fraud_flags)
    tn, fp, fn, tp = cm.ravel()

    # Precision@K — of top K most anomalous, how many are fraud?
    top_k_actual = min(top_k, len(y))
    top_k_idx    = np.argsort(anomaly_scores)[-top_k_actual:]
    precision_at_k = y.iloc[top_k_idx].mean()

    # Recall@K — what fraction of all fraud is in top K?
    recall_at_k = y.iloc[top_k_idx].sum() / n_fraud if n_fraud > 0 else 0

    print(f"\n{'='*58}")
    print(f"{model_name} — Anomaly Detection Evaluation")
    print(f"{'='*58}")
    print(f"ROC-AUC:                {roc_auc:.4f}")
    print(f"Average Precision:      {avg_prec:.4f}")
    print(f"Transactions flagged:   {n_flagged:,} ({n_flagged/len(y)*100:.2f}%)")
    print(f"\nAt IF threshold:")
    print(f"  Fraud caught:         {tp}/{n_fraud} ({tp/n_fraud*100:.1f}%)")
    print(f"  False alarms:         {fp:,} ({fp/(tn+fp)*100:.2f}% of legit)")
    print(f"\nPrecision@{top_k_actual}:          {precision_at_k:.3f}")
    print(f"Recall@{top_k_actual}:             {recall_at_k:.3f}")
    print(f"  (of top {top_k_actual} most anomalous: {int(precision_at_k*top_k_actual)} are fraud)")
    print(f"{'='*58}")

    return {
        "model_name":      model_name,
        "roc_auc":         roc_auc,
        "avg_precision":   avg_prec,
        "fraud_recall":    tp / n_fraud if n_fraud > 0 else 0,
        "false_alarm_rate": fp / (tn + fp) if (tn + fp) > 0 else 0,
        "precision_at_k":  precision_at_k,
        "recall_at_k":     recall_at_k,
        "n_flagged":       n_flagged,
        "anomaly_scores":  anomaly_scores,
        "fraud_flags":     fraud_flags,
    }


def complementarity_analysis(
    xgb_predictions: pd.Series,
    if_results: dict,
    y: pd.Series,
) -> None:
    """
    Level 2 validation: does Isolation Forest catch fraud that XGBoost missed?

    This is the key ensemble justification:
    - XGBoost missed 7 frauds in our test set
    - Does IF catch any of those?
    - Venn diagram of what each model catches
    """
    print(f"\n{'='*58}")
    print("COMPLEMENTARITY ANALYSIS")
    print("XGBoost vs Isolation Forest — what each catches")
    print(f"{'='*58}")

    if_flags = if_results["fraud_flags"]
    fraud_idx = y[y == 1].index

    xgb_caught    = set(fraud_idx[xgb_predictions.loc[fraud_idx] == 1])
    if_caught     = set(fraud_idx[pd.Series(if_flags, index=y.index).loc[fraud_idx] == 1])

    both_caught   = xgb_caught & if_caught
    xgb_only      = xgb_caught - if_caught
    if_only       = if_caught - xgb_caught
    neither       = set(fraud_idx) - xgb_caught - if_caught

    print(f"Total fraud in test:    {len(fraud_idx)}")
    print(f"XGBoost caught:         {len(xgb_caught)} ({len(xgb_caught)/len(fraud_idx)*100:.1f}%)")
    print(f"Isolation Forest caught:{len(if_caught)} ({len(if_caught)/len(fraud_idx)*100:.1f}%)")
    print(f"Both caught:            {len(both_caught)}")
    print(f"XGBoost only:           {len(xgb_only)}")
    print(f"Isolation Forest only:  {len(if_only)}  ← IF adds these")
    print(f"Neither caught:         {len(neither)}")

    ensemble_caught = xgb_caught | if_caught
    print(f"\nEnsemble (XGB + IF):    {len(ensemble_caught)} ({len(ensemble_caught)/len(fraud_idx)*100:.1f}%)")
    print(f"Improvement over XGB:   +{len(if_only)} fraud cases")
    print(f"{'='*58}")

    return ensemble_caught


def context_shift_experiment(df: pd.DataFrame, feature_cols: list) -> dict:
    """
    Level 3 validation: the AFI connection.

    Train Isolation Forest ONLY on normal-context transactions:
    CASH_IN and PAYMENT (zero fraud in these types).

    Then score ALL transactions. The anomaly score measures
    how unlike the 'normal context' each transaction is.

    Hypothesis:
    - Fraudulent CASH_OUT/TRANSFER should have highest anomaly scores
    - The anomaly score IS the contextual distance measure
    - This is structurally identical to SECI in the AFI paper:
      systematic error concentration in contextually distant observations

    Research implication:
    An AI system trained on normal payment behaviour (the majority context)
    will generate systematically higher anomaly scores for CASH_OUT and
    TRANSFER transactions — a form of contextual foreignness. Within those
    types, fraudulent transactions are the most contextually foreign of all.
    """
    print(f"\n{'='*58}")
    print("CONTEXT SHIFT EXPERIMENT")
    print("AFI connection: training context = CASH_IN/PAYMENT")
    print("Scoring context = all transaction types")
    print(f"{'='*58}")

    # Training context: only normal payment types (no fraud)
    normal_mask  = df["type"].isin(["CASH_IN", "PAYMENT"]) if "type" in df.columns \
                   else (df.get("type_CASH_IN",0) | df.get("type_PAYMENT",0)).astype(bool)

    # Use type columns if original type is not available
    if "type" not in df.columns:
        normal_mask = (df.get("type_CASH_IN", pd.Series(0, index=df.index)) == 1) | \
                      (df.get("type_PAYMENT", pd.Series(0, index=df.index)) == 1)

    X_train_context = df[normal_mask][feature_cols]
    log.info(f"Training context: {len(X_train_context):,} transactions (CASH_IN + PAYMENT)")
    log.info(f"Training context fraud rate: {df[normal_mask][TARGET_COL].mean():.4f} (should be 0)")

    # Train on normal context only
    context_model = IsolationForest(
        n_estimators=200,
        contamination=0.01,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    context_model.fit(X_train_context)

    # Score all transactions
    X_all = df[feature_cols]
    contextual_distance = -context_model.decision_function(X_all)  # higher = more anomalous

    df = df.copy()
    df["contextual_distance"] = contextual_distance

    # Analyse by transaction type
    print("\nMean contextual distance by transaction type:")
    if "type" in df.columns:
        type_col = "type"
    else:
        # Reconstruct type from dummies
        type_cols = [c for c in df.columns if c.startswith("type_")]
        if type_cols:
            df["_type"] = "OTHER"
            for tc in type_cols:
                tname = tc.replace("type_","")
                df.loc[df[tc]==1, "_type"] = tname
            type_col = "_type"
        else:
            type_col = None

    if type_col:
        type_dist = df.groupby(type_col).agg(
            mean_distance=("contextual_distance","mean"),
            fraud_rate=(TARGET_COL,"mean"),
            n=("contextual_distance","count"),
        ).sort_values("mean_distance", ascending=False)
        print(type_dist.to_string())

    # Key finding: within CASH_OUT and TRANSFER, do fraud cases have higher contextual distance?
    cashout_transfer = df[df.get("type_CASH_OUT", pd.Series(0,index=df.index)) +
                          df.get("type_TRANSFER", pd.Series(0,index=df.index)) > 0] \
                       if "type_CASH_OUT" in df.columns \
                       else df[df.get(type_col,"") .isin(["CASH_OUT","TRANSFER"])] \
                       if type_col and type_col != "_type" else df

    if len(cashout_transfer) > 0 and cashout_transfer[TARGET_COL].sum() > 0:
        fraud_dist  = cashout_transfer[cashout_transfer[TARGET_COL]==1]["contextual_distance"].mean()
        legit_dist  = cashout_transfer[cashout_transfer[TARGET_COL]==0]["contextual_distance"].mean()
        print(f"\nWithin CASH_OUT + TRANSFER:")
        print(f"  Fraud mean contextual distance:  {fraud_dist:.4f}")
        print(f"  Legit mean contextual distance:  {legit_dist:.4f}")
        print(f"  Ratio (fraud/legit):             {fraud_dist/legit_dist:.2f}x")
        print(f"\n  → Fraudulent transactions are {fraud_dist/legit_dist:.1f}x more contextually")
        print(f"    foreign than legitimate ones in the same type.")
        print(f"    This parallels AFI Indicator 2 (SECI): AI systems trained")
        print(f"    in one institutional context generate systematically larger")
        print(f"    errors in contextually distant deployment contexts.")

        # ROC-AUC of contextual distance as fraud predictor within risky types
        roc_context = roc_auc_score(
            cashout_transfer[TARGET_COL],
            cashout_transfer["contextual_distance"]
        )
        print(f"\n  Context-shift ROC-AUC (within risky types): {roc_context:.4f}")

    print(f"{'='*58}")

    # Save context model
    path = MODELS_DIR / "isolation_forest_context_shift.pkl"
    with open(path, "wb") as f:
        pickle.dump(context_model, f)

    return {"context_model": context_model, "contextual_distance": contextual_distance}


def run_isolation_forest(
    dev_mode: bool = False,
    load_xgb: bool = True,
) -> dict:
    """Full Isolation Forest pipeline with three-level validation."""
    df = load_features(dev_mode)
    feature_cols = get_feature_cols(df)

    y = df[TARGET_COL]
    X = df[feature_cols]

    # ── Standard Isolation Forest ──────────────────────────────────────────────
    true_fraud_rate = y.mean()
    contamination   = min(true_fraud_rate * 2, 0.05)

    model = train_isolation_forest(X, contamination=contamination)

    # Level 1: extrinsic validation
    results = evaluate_anomaly_detection(model, X, y, "Isolation Forest (standard)")

    # Level 2: complementarity with XGBoost
    xgb_path = MODELS_DIR / "xgboost_fraud.pkl"
    if load_xgb and xgb_path.exists():
        with open(xgb_path, "rb") as f:
            xgb = pickle.load(f)
        xgb_probs = xgb.predict_proba(X)[:, 1]
        xgb_preds = pd.Series((xgb_probs >= 0.5).astype(int), index=y.index)
        complementarity_analysis(xgb_preds, results, y)
    else:
        log.info("XGBoost model not found — skipping complementarity analysis.")
        log.info("Run xgboost_model.py first to enable complementarity analysis.")

    # Level 3: context shift experiment
    context_results = context_shift_experiment(df, feature_cols)

    # Save standard model
    model_path = MODELS_DIR / "isolation_forest_fraud.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    log.info(f"Model saved → {model_path}")

    # Save anomaly scores
    scores_df = pd.DataFrame({
        "anomaly_score": results["anomaly_scores"],
        "is_fraud":      y.values,
        "if_flag":       results["fraud_flags"],
        "contextual_distance": context_results["contextual_distance"],
    })
    scores_path = DATA_PROC / "anomaly_scores.parquet"
    scores_df.to_parquet(scores_path, index=False)
    log.info(f"Anomaly scores saved → {scores_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev",       action="store_true", help="Use dev sample")
    parser.add_argument("--no-xgb",   action="store_true", help="Skip XGBoost comparison")
    args = parser.parse_args()
    run_isolation_forest(dev_mode=args.dev, load_xgb=not args.no_xgb)
