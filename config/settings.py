"""
Central configuration for the fraud detection system.

Dataset: PaySim — synthetic mobile money transactions
modelled on real M-Pesa data from Africa.

Dual purpose:
  ADB portfolio → fraud detection for mobile money platforms in DMCs
  IB research   → cross-border model validity (does a fraud model trained
                   on African M-Pesa generalise to other DMC mobile money
                   systems? — Indicator 2, Algorithmic Foreignness)
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parents[1]
DATA_RAW      = BASE_DIR / "data" / "raw"
DATA_PROC     = BASE_DIR / "data" / "processed"
DATA_HARM     = BASE_DIR / "data" / "harmonised"
MODELS_DIR    = BASE_DIR / "models_saved"
REPORTS_DIR   = BASE_DIR / "docs" / "reports"

for d in [DATA_PROC, DATA_HARM, MODELS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Raw data ──────────────────────────────────────────────────────────────────
PAYSIM_FILE   = DATA_RAW / "paysim_transactions.csv"

# ── Dataset characteristics ───────────────────────────────────────────────────
TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

# Only CASH_OUT and TRANSFER can be fraudulent in PaySim
FRAUD_ELIGIBLE_TYPES = ["CASH_OUT", "TRANSFER"]

TARGET_COL    = "isFraud"
FRAUD_RATE    = 0.0013   # ~0.13% — severe class imbalance

# ── Feature groups ────────────────────────────────────────────────────────────
# Raw features from PaySim
RAW_FEATURES = [
    "step",           # hour of simulation (1-744, ~30 days)
    "type",           # transaction type
    "amount",         # transaction amount
    "oldbalanceOrg",  # sender balance before
    "newbalanceOrig", # sender balance after
    "oldbalanceDest", # recipient balance before
    "newbalanceDest", # recipient balance after
]

# Engineered behavioural features (built in src/features/)
ENGINEERED_FEATURES = [
    "amount_log",
    "balance_diff_orig",       # oldbalanceOrg - newbalanceOrig
    "balance_diff_dest",       # newbalanceDest - oldbalanceDest
    "orig_balance_zero_after", # flag: sender emptied account
    "dest_balance_unchanged",  # flag: dest balance didn't change (suspicious)
    "amount_to_orig_balance",  # ratio: amount / oldbalanceOrg
    "hour_of_day",             # step % 24
    "is_night",                # hour 0-6
    "is_round_amount",         # amount divisible by 1000
    "type_CASH_OUT",           # one-hot
    "type_TRANSFER",           # one-hot
    "type_PAYMENT",
    "type_CASH_IN",
    "type_DEBIT",
]

# ── Model settings ────────────────────────────────────────────────────────────
RANDOM_STATE  = 42
TEST_SIZE     = 0.2
VAL_SIZE      = 0.1

# XGBoost — optimised for recall (missing fraud >> false alarm)
XGBOOST_PARAMS = {
    "n_estimators":      500,
    "max_depth":         6,
    "learning_rate":     0.05,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "scale_pos_weight":  100,  # ~1/fraud_rate — handles imbalance
    "eval_metric":       "aucpr",
    "random_state":      RANDOM_STATE,
    "n_jobs":            -1,
}

RANDOM_FOREST_PARAMS = {
    "n_estimators":  300,
    "max_depth":     10,
    "class_weight":  "balanced",
    "random_state":  RANDOM_STATE,
    "n_jobs":        -1,
}

# Isolation Forest — anomaly detection layer
ISOLATION_FOREST_PARAMS = {
    "n_estimators":   200,
    "contamination":  0.002,   # slightly above true fraud rate
    "random_state":   RANDOM_STATE,
    "n_jobs":         -1,
}

# Autoencoder — reconstruction error threshold
AUTOENCODER_PARAMS = {
    "encoding_dims":  [32, 16, 8],   # encoder layers
    "epochs":         50,
    "batch_size":     512,
    "learning_rate":  0.001,
}

# ── Ensemble decision thresholds ──────────────────────────────────────────────
# Configurable risk appetite:
#   conservative → catch more fraud, more false positives
#   balanced     → default operating point
#   aggressive   → fewer false positives, miss some fraud
ENSEMBLE_THRESHOLDS = {
    "conservative": 0.3,
    "balanced":     0.5,
    "aggressive":   0.7,
}
DEFAULT_THRESHOLD = "balanced"

# ── Monitoring ────────────────────────────────────────────────────────────────
DRIFT_WINDOW_SIZE    = 10_000   # transactions per monitoring window
PSI_THRESHOLD        = 0.2      # Population Stability Index — triggers retrain
PERFORMANCE_MIN_AUC  = 0.85     # below this → alert
