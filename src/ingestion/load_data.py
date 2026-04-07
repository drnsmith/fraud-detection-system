"""
Data ingestion for PaySim mobile money transaction dataset.

Loads, validates, and performs initial profiling of the raw data.
Saves a clean processed version ready for feature engineering.
"""

import logging
from pathlib import Path
import sys

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import (
    PAYSIM_FILE, DATA_PROC, TARGET_COL,
    TRANSACTION_TYPES, FRAUD_ELIGIBLE_TYPES, RANDOM_STATE
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def load_raw(filepath: Path = PAYSIM_FILE, nrows: int | None = None) -> pd.DataFrame:
    """
    Load raw PaySim CSV.

    Parameters
    ----------
    nrows : int or None
        Load a subset for development. None = full 6.3M rows.
    """
    log.info(f"Loading PaySim from {filepath} ...")
    df = pd.read_csv(filepath, nrows=nrows)
    log.info(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate schema, types, and basic integrity.
    Raises ValueError if critical checks fail.
    """
    required = ["step", "type", "amount", "nameOrig", "oldbalanceOrg",
                "newbalanceOrig", "nameDest", "oldbalanceDest",
                "newbalanceDest", "isFraud", "isFlaggedFraud"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Types check
    assert df["amount"].min() >= 0, "Negative amounts found"
    assert df[TARGET_COL].isin([0, 1]).all(), "Target not binary"
    assert df["type"].isin(TRANSACTION_TYPES).all(), "Unknown transaction types"

    # Fraud only in eligible types
    fraud_types = df[df[TARGET_COL] == 1]["type"].unique()
    unexpected = set(fraud_types) - set(FRAUD_ELIGIBLE_TYPES)
    if unexpected:
        log.warning(f"Fraud found in unexpected types: {unexpected}")

    log.info("Validation passed ✓")
    return df


def profile(df: pd.DataFrame) -> None:
    """Print a concise data profile."""
    n = len(df)
    fraud = df[TARGET_COL].sum()

    print("\n" + "="*55)
    print("PAYSIM DATASET PROFILE")
    print("="*55)
    print(f"Total transactions:  {n:>12,}")
    print(f"Fraud transactions:  {fraud:>12,}  ({fraud/n*100:.3f}%)")
    print(f"Legitimate:          {n-fraud:>12,}  ({(n-fraud)/n*100:.3f}%)")
    print(f"Time steps (hours):  {df['step'].min()} – {df['step'].max()}")
    print(f"Amount range:        {df['amount'].min():,.0f} – {df['amount'].max():,.0f}")
    print(f"Missing values:      {df.isnull().sum().sum()}")

    print("\nTransactions by type:")
    type_counts = df.groupby("type").agg(
        count=("type", "count"),
        fraud_count=(TARGET_COL, "sum"),
    )
    type_counts["fraud_rate_%"] = (
        type_counts["fraud_count"] / type_counts["count"] * 100
    ).round(3)
    print(type_counts.to_string())

    print("\nFraud by type:")
    fraud_df = df[df[TARGET_COL] == 1]
    print(fraud_df["type"].value_counts().to_string())
    print("="*55 + "\n")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply cleaning steps:
    - Drop isFlaggedFraud (system flag, not target — would cause leakage)
    - Drop name columns (identifiers, not predictive)
    - Ensure correct dtypes
    - Add a fraud_eligible flag
    """
    # Drop identifier and leakage columns
    drop_cols = ["nameOrig", "nameDest", "isFlaggedFraud"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Correct dtypes
    df["step"]   = df["step"].astype(int)
    df["type"]   = df["type"].astype("category")
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    # Flag fraud-eligible transaction types
    df["fraud_eligible"] = df["type"].isin(FRAUD_ELIGIBLE_TYPES).astype(int)

    log.info(f"Cleaned dataset: {df.shape}")
    return df


def sample_for_dev(
    df: pd.DataFrame,
    n_fraud: int = 8_000,
    fraud_ratio: float = 0.01,
) -> pd.DataFrame:
    """
    Create a balanced development sample for fast iteration.
    Keeps all fraud cases + a random sample of legitimate transactions.

    Parameters
    ----------
    n_fraud : int
        Number of fraud cases to include (or all if fewer exist).
    fraud_ratio : float
        Desired fraud ratio in the sample (still imbalanced but workable).
    """
    fraud_df = df[df[TARGET_COL] == 1].sample(
        min(n_fraud, df[TARGET_COL].sum()), random_state=RANDOM_STATE
    )
    n_legit = int(len(fraud_df) / fraud_ratio)
    legit_df = df[df[TARGET_COL] == 0].sample(n_legit, random_state=RANDOM_STATE)
    sample = pd.concat([fraud_df, legit_df]).sample(frac=1, random_state=RANDOM_STATE)
    log.info(
        f"Dev sample: {len(sample):,} rows | "
        f"fraud: {sample[TARGET_COL].sum():,} ({sample[TARGET_COL].mean()*100:.2f}%)"
    )
    return sample.reset_index(drop=True)


def run_ingestion(dev_mode: bool = False) -> pd.DataFrame:
    """
    Full ingestion pipeline.

    Parameters
    ----------
    dev_mode : bool
        If True, loads a fast development sample (~800K rows).
        If False, loads all 6.3M transactions.

    Returns
    -------
    pd.DataFrame
        Clean, validated dataset saved to data/processed/.
    """
    # Load
    df = load_raw(nrows=800_000 if dev_mode else None)

    # Validate
    df = validate(df)

    # Profile
    profile(df)

    # Clean
    df = clean(df)

    # Save full clean version
    out_path = DATA_PROC / "paysim_clean.parquet"
    df.to_parquet(out_path, index=False)
    log.info(f"Clean data saved → {out_path}")

    # Save dev sample separately
    dev_sample = sample_for_dev(df)
    dev_path = DATA_PROC / "paysim_dev_sample.parquet"
    dev_sample.to_parquet(dev_path, index=False)
    log.info(f"Dev sample saved → {dev_path}")

    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", action="store_true",
                        help="Load 800K row subset for fast development")
    args = parser.parse_args()
    run_ingestion(dev_mode=args.dev)
