---
title: Fraud Detection System
emoji: 🔍
colorFrom: red
colorTo: red
sdk: docker
pinned: false
---

# Financial Fraud Detection System

Production-grade fraud detection pipeline on PaySim dataset (6.3M transactions, 0.13% fraud rate).

- XGBoost supervised model: ROC-AUC 0.9931, 92.4% recall
- Isolation Forest unsupervised anomaly detection
- Ensemble: 93.5% recall
- Three-level validation framework (Valimetrica)
- SHAP explainability

**GitHub:** https://github.com/drnsmith/fraud-detection-system
