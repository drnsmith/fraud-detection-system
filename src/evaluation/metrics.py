"""
Evaluation metrics for fraud detection models.
Handles class imbalance — fraud is 0.13% of transactions.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix,
    precision_recall_curve, roc_curve
)
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[2]))


def evaluate_model(y_true, y_pred_proba, y_pred_binary=None,
                   threshold=0.5, model_name='Model'):
    """Full evaluation suite for a fraud detection model."""
    if y_pred_binary is None:
        y_pred_binary = (y_pred_proba >= threshold).astype(int)

    auc_roc = roc_auc_score(y_true, y_pred_proba)
    auc_pr  = average_precision_score(y_true, y_pred_proba)

    cm = confusion_matrix(y_true, y_pred_binary)
    tn, fp, fn, tp = cm.ravel()

    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    results = {
        'model':      model_name,
        'roc_auc':    round(auc_roc, 4),
        'pr_auc':     round(auc_pr, 4),
        'recall':     round(recall, 4),
        'precision':  round(precision, 4),
        'f1':         round(f1, 4),
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'threshold':  threshold,
    }

    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"{'='*50}")
    print(f"  ROC-AUC:   {auc_roc:.4f}")
    print(f"  PR-AUC:    {auc_pr:.4f}")
    print(f"  Recall:    {recall:.4f}  ({tp} fraud caught, {fn} missed)")
    print(f"  Precision: {precision:.4f}  ({fp} false alarms)")
    print(f"  F1:        {f1:.4f}")
    print(f"  Threshold: {threshold}")

    return results


def context_shift_evaluation(model, X_test, y_test, levels=3):
    """
    Evaluate model under distributional shift.
    Mimics real-world deployment drift.

    Level 1: mild amount scaling (±10%)
    Level 2: moderate - merchant category permutation
    Level 3: severe - full temporal + amount redistribution
    """
    results = []
    rng = np.random.RandomState(42)

    for level in range(1, levels + 1):
        X_shifted = X_test.copy()

        if level >= 1:
            # Mild: scale amounts
            if 'amount' in X_shifted.columns:
                X_shifted['amount'] *= rng.uniform(0.9, 1.1, len(X_shifted))

        if level >= 2:
            # Moderate: permute category features
            cat_cols = [c for c in X_shifted.columns if 'type' in c.lower()]
            for col in cat_cols:
                X_shifted[col] = rng.permutation(X_shifted[col].values)

        if level >= 3:
            # Severe: full feature redistribution
            X_shifted = X_shifted.apply(
                lambda col: col.sample(frac=1, random_state=42).values
                if col.dtype in ['float64', 'int64'] else col
            )

        y_proba = model.predict_proba(X_shifted)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        results.append({'level': level, 'roc_auc': round(auc, 4)})
        print(f"  Level {level} context shift ROC-AUC: {auc:.4f}")

    return pd.DataFrame(results)


if __name__ == '__main__':
    # Load saved models and evaluate
    BASE = Path(__file__).parents[2]
    models_dir = BASE / 'models_saved'

    print("Loading models...")
    xgb_model = joblib.load(models_dir / 'xgboost_fraud.pkl')
    if_model  = joblib.load(models_dir / 'isolation_forest_fraud.pkl')

    print("Models loaded. Run with test data to evaluate.")
    print("Usage: PYTHONPATH=. python src/evaluation/metrics.py")
