# Financial Fraud Detection System

**Live demo:** [https://drnsmith-fraud-detection-system.hf.space](https://huggingface.co/spaces/drnsmith/fraud-detection-system)

---

## Why I built this

Fraud detection is one of those problems that looks solved until you look closely. The standard approach — train a classifier, report AUC, ship it — misses the thing that actually matters in financial risk contexts: whether the model's measured performance is a valid representation of its real-world behaviour.

I built this project because I wanted to work through that question properly. The PaySim dataset (6.3 million synthetic financial transactions, 0.13% fraud rate) gave me a realistic class imbalance problem to solve, but my real interest was in what happens *after* you get a good AUC. Does the model actually generalise? Does it complement other detection methods, or just replicate them? What happens when the transaction environment shifts?

These questions sit at the heart of what I call evaluation validity — the degree to which the instruments we use to measure model performance are themselves valid. This is under-theorised in machine learning, inherited as it is from software engineering rather than psychometrics or econometrics. I've been developing a formal framework for it (Valimetrica) and this project is a working implementation of those ideas.

---

## What I built

I trained two models on the PaySim dataset and then subjected them to a three-level validation framework that goes beyond standard benchmark reporting.

The supervised model is XGBoost, which achieves ROC-AUC 0.9931 and 92.4% recall on the held-out test set. The unsupervised model is Isolation Forest, which achieves ROC-AUC 0.8794 without any labelled fraud examples. The ensemble (union of both) reaches 93.5% recall — catching one additional fraud per 100 that XGBoost alone misses.

But the numbers I'm most interested in are at Level 3: when I simulate a severe context shift (transaction volume spike, new merchant categories, altered fraud patterns), the ensemble ROC-AUC drops to 0.7225. The model degrades gracefully rather than catastrophically — which is itself an important property to measure and report honestly.

The dashboard has four tabs: Model Performance, SHAP Explainability, Context Shift Analysis, and Ensemble Complementarity. Each tab corresponds to a layer of the validation framework.

---

## The validation framework

I apply three levels of validation, each answering a different question:

**Level 1 — Extrinsic validity.** Does the model discriminate? Standard AUC, precision-recall, confusion matrix at operationally relevant thresholds. Necessary but not sufficient.

**Level 2 — Complementarity validity.** Does the ensemble add information beyond either component alone? I measure this by computing the overlap between XGBoost and Isolation Forest predictions — they catch different fraud patterns, which is why the ensemble outperforms either model alone.

**Level 3 — Context shift validity.** Does performance hold when the deployment environment changes? I simulate three severity levels of distributional shift and track degradation curves. A model that achieves 0.99 AUC in development and 0.72 under moderate drift is not the same model as one that holds at 0.91 across all levels.

This framework is the methodological core of the project. The XGBoost model is good. The validation is what makes it trustworthy.

---

## Model performance

| Metric | XGBoost | Isolation Forest | Ensemble |
|---|---|---|---|
| ROC-AUC | 0.9931 | 0.8794 | 0.9935 |
| Recall | 92.4% | 74.1% | 93.5% |
| Precision | 87.3% | 12.4% | 61.2% |
| Context shift AUC (Level 3) | 0.7225 | 0.6841 | 0.7380 |

---

## Tech stack

- Python 3.11, XGBoost 2.0.3, SHAP 0.44.1, scikit-learn 1.4.0
- Plotly Dash 4.1.0, dash-bootstrap-components 2.0.4
- Gunicorn, deployed on Render

---

## Data

PaySim synthetic financial transaction dataset — 6,362,620 transactions, 8,213 fraudulent (0.13%). Generated from real mobile money transaction logs. Available on Kaggle.

---

## Run locally

```bash
git clone https://github.com/drnsmith/fraud-detection-system.git
cd fraud-detection-system
pip install -r requirements.txt

PYTHONPATH=. python src/ingestion.py
PYTHONPATH=. python src/features.py
PYTHONPATH=. python src/models.py
PYTHONPATH=. python src/evaluation.py

PYTHONPATH=. python dashboard/app.py
# → http://127.0.0.1:8050
```

---

## Project structure

```
fraud-detection-system/
├── src/
│   ├── ingestion.py        # Data loading and preprocessing
│   ├── features.py         # Feature engineering
│   ├── models.py           # XGBoost + Isolation Forest training
│   ├── evaluation.py       # Three-level validation framework
│   └── monitoring.py       # Context shift simulation
├── dashboard/
│   └── app.py              # Plotly Dash, 4 tabs
├── models_saved/
│   ├── xgboost_fraud.pkl
│   └── isolation_forest_fraud.pkl
└── requirements.txt
```

---
## License

MIT — see [LICENSE](LICENSE).
