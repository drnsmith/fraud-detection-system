"""
Feature engineering for mobile money fraud detection.

Builds behavioural features that capture the patterns
characteristic of fraudulent transactions in mobile money systems.

Key insight: fraudsters typically empty the origin account completely
and move funds to a destination that either doesn't receive the full
amount (mule account layering) or shows unusual balance patterns.
"""

import logging
from pathlib import Path
import sys

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import DATA_PROC, TARGET_COL

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def build_balance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Balance movement features — most discriminative for fraud.

    Fraudulent CASH_OUT/TRANSFER transactions typically:
    - Empty the origin account completely
    - Show destination balance that doesn't increase as expected
    """
    df["balance_diff_orig"]      = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["balance_diff_dest"]      = df["newbalanceDest"] - df["oldbalanceDest"]
    df["orig_balance_zero_after"] = (df["newbalanceOrig"] == 0).astype(int)
    df["orig_fully_drained"]     = (
        (df["oldbalanceOrg"] > 0) & (df["newbalanceOrig"] == 0)
    ).astype(int)

    # Destination balance didn't increase — classic money mule sign
    df["dest_balance_unchanged"] = (
        (df["oldbalanceDest"] == df["newbalanceDest"]) & (df["amount"] > 0)
    ).astype(int)

    df["dest_was_empty"] = (df["oldbalanceDest"] == 0).astype(int)

    # Ratio: how much of the origin balance was moved
    df["amount_to_orig_balance"] = np.where(
        df["oldbalanceOrg"] > 0,
        df["amount"] / df["oldbalanceOrg"],
        0.0
    )

    # Accounting errors — should be zero for legitimate transactions
    df["orig_balance_error"] = (df["balance_diff_orig"] - df["amount"]).abs()
    df["dest_balance_error"] = (df["balance_diff_dest"] - df["amount"]).abs()

    return df


def build_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transaction amount features.
    Fraudsters often use round amounts (structuring) or very large transfers.
    """
    df["amount_log"]      = np.log1p(df["amount"])
    df["is_round_amount"] = (df["amount"] % 1000 == 0).astype(int)

    # Large transaction — above 99th percentile
    p99 = df["amount"].quantile(0.99)
    df["is_large_transaction"] = (df["amount"] > p99).astype(int)

    # Very small transaction — below 1st percentile (also suspicious)
    p01 = df["amount"].quantile(0.01)
    df["is_micro_transaction"] = (df["amount"] < p01).astype(int)

    return df


def build_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Time-based features from the simulation step (1 step = 1 hour).
    Fraud patterns often concentrate at night or specific times.
    """
    df["hour_of_day"] = df["step"] % 24
    df["day_of_sim"]  = df["step"] // 24
    df["is_night"]    = ((df["hour_of_day"] >= 0) & (df["hour_of_day"] <= 6)).astype(int)
    df["is_weekend"]  = (df["day_of_sim"] % 7 >= 5).astype(int)

    return df


def build_type_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode transaction type.
    Only CASH_OUT and TRANSFER carry fraud in PaySim.
    """
    type_dummies = pd.get_dummies(df["type"], prefix="type")
    # Ensure all expected columns exist even if type absent in subset
    for t in ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]:
        col = f"type_{t}"
        if col not in type_dummies.columns:
            type_dummies[col] = 0

    df = pd.concat([df, type_dummies.astype(int)], axis=1)
    return df


def build_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Interaction features combining type, balance, and amount signals.
    These are the strongest fraud indicators in mobile money.
    """
    # TRANSFER with full account drainage — highest risk combination
    df["transfer_and_drained"] = (
        df.get("type_TRANSFER", 0) & df["orig_fully_drained"]
    ).astype(int)

    # CASH_OUT to empty destination — common fraud exit pattern
    df["cashout_to_empty_dest"] = (
        df.get("type_CASH_OUT", 0) & df["dest_was_empty"]
    ).astype(int)

    # Large amount with unchanged destination balance
    df["large_with_no_dest_change"] = (
        df["is_large_transaction"] & df["dest_balance_unchanged"]
    ).astype(int)

    # Fully draining with accounting error (model inconsistency in data)
    df["drain_with_error"] = (
        df["orig_fully_drained"] & (df["orig_balance_error"] > 0)
    ).astype(int)

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return the list of feature columns to use for modelling."""
    exclude = [
        TARGET_COL, "type", "fraud_eligible",
        "step",   # raw step — replaced by temporal features
    ]
    return [c for c in df.columns if c not in exclude
            and df[c].dtype in ["int64", "float64", "int32", "float32",
                                 "uint8", "bool"]]


def run_feature_engineering(
    input_path: Path | None = None,
    dev_mode:   bool = False,
) -> pd.DataFrame:
    """
    Full feature engineering pipeline.

    Loads clean PaySim data, builds all feature groups,
    and saves the modelling-ready dataset.

    Parameters
    ----------
    input_path : Path or None
        Path to clean parquet. Defaults to data/processed/paysim_clean.parquet
    dev_mode : bool
        Use dev sample for fast iteration.
    """
    if input_path is None:
        fname = "paysim_dev_sample.parquet" if dev_mode else "paysim_clean.parquet"
        input_path = DATA_PROC / fname

    log.info(f"Loading clean data from {input_path} ...")
    df = pd.read_parquet(input_path)
    log.info(f"Loaded: {df.shape[0]:,} rows")

    log.info("Building features...")
    df = build_balance_features(df)
    df = build_amount_features(df)
    df = build_temporal_features(df)
    df = build_type_features(df)
    df = build_interaction_features(df)

    feature_cols = get_feature_columns(df)
    log.info(f"Total features built: {len(feature_cols)}")
    log.info(f"Features: {feature_cols}")

    # Save
    suffix = "_dev" if dev_mode else ""
    out_path = DATA_PROC / f"paysim_features{suffix}.parquet"
    df.to_parquet(out_path, index=False)
    log.info(f"Feature dataset saved → {out_path}  ({df.shape})")

    # Quick fraud signal check
    fraud = df[df[TARGET_COL] == 1]
    legit = df[df[TARGET_COL] == 0]
    log.info("\nKey feature means — fraud vs legitimate:")
    key_cols = ["orig_fully_drained", "dest_balance_unchanged",
                "amount_to_orig_balance", "is_large_transaction",
                "transfer_and_drained", "cashout_to_empty_dest"]
    for c in key_cols:
        if c in df.columns:
            log.info(
                f"  {c:35s}  fraud: {fraud[c].mean():.3f}  "
                f"legit: {legit[c].mean():.3f}"
            )

    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", action="store_true",
                        help="Use dev sample for fast iteration")
    args = parser.parse_args()
    run_feature_engineering(dev_mode=args.dev)
