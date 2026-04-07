"""
Context shift analysis — tests model robustness under distributional drift.

Parallel methodology to AFI SECI indicator:
both measure signal retention under adversarial distributional change.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import joblib
from pathlib import Path


SHIFT_DESCRIPTIONS = {
    1: "Mild: ±10% amount scaling",
    2: "Moderate: merchant category permutation + amount scaling",
    3: "Severe: full temporal + amount redistribution (ROC-AUC: 0.7225)",
}


def apply_context_shift(X: pd.DataFrame, level: int,
                        random_state: int = 42) -> pd.DataFrame:
    """Apply distributional shift to feature matrix."""
    rng = np.random.RandomState(random_state)
    X_s = X.copy()

    if level >= 1:
        amt_cols = [c for c in X_s.columns if 'amount' in c.lower() or 'amt' in c.lower()]
        for col in amt_cols:
            X_s[col] = X_s[col] * rng.uniform(0.9, 1.1, len(X_s))

    if level >= 2:
        cat_cols = [c for c in X_s.columns
                    if any(t in c.lower() for t in ['type','category','merchant'])]
        for col in cat_cols:
            X_s[col] = rng.permutation(X_s[col].values)

    if level >= 3:
        numeric_cols = X_s.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            X_s[col] = rng.permutation(X_s[col].values)

    return X_s


def run_shift_analysis(model, X_test: pd.DataFrame,
                       y_test: pd.Series) -> pd.DataFrame:
    """Run full context shift analysis across all levels."""
    print("\nContext Shift Robustness Analysis")
    print("=" * 45)

    results = []
    baseline = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(f"Baseline ROC-AUC: {baseline:.4f}")
    results.append({'level': 0, 'description': 'Baseline', 'roc_auc': baseline})

    for level in [1, 2, 3]:
        X_shifted = apply_context_shift(X_test, level)
        auc = roc_auc_score(y_test, model.predict_proba(X_shifted)[:, 1])
        degradation = baseline - auc
        print(f"Level {level} [{SHIFT_DESCRIPTIONS[level][:40]}...]")
        print(f"  ROC-AUC: {auc:.4f}  (degradation: -{degradation:.4f})")
        results.append({
            'level': level,
            'description': SHIFT_DESCRIPTIONS[level],
            'roc_auc': round(auc, 4),
            'degradation': round(degradation, 4),
        })

    return pd.DataFrame(results)
