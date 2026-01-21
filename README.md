# Business-Driven Demand Forecasting System

---

## Objective
To develop a business-driven demand forecasting system at the store–department–time level, where forecasts serve as the core signal and are translated into operational insights to support planning and decision-making.

---

## Problem Context
Retail demand is highly heterogeneous across stores and departments. A single global forecast is insufficient; decision-makers need:
- Reliable baseline forecasts
- Clear understanding of model risk
- Visibility into where human intervention is required

---

## Approach
This project focuses on:
- Robust feature engineering with strict feature contract locking
- A primary forecasting model benchmarked against an SMA-4 baseline
- Model diagnostics designed for operational risk awareness

---

## Key Highlights
- Primary model outperforms SMA-4 baseline on validation WAPE
- Stable performance across most stores
- Error concentration identified at specific department segments
- Diagnostics explicitly designed to support downstream decision rules

---

## Outputs
- Trained forecasting model and preprocessing pipeline
- Validation and test performance metrics
- Residual-based diagnostics at store and department level
- Feature importance analysis to explain model behavior

---

## Next Step (Planned)
A separate decision-support layer will translate forecasts and diagnostics into:
- Risk-based prioritization
- Store–department ranking
- Actionable operational rules

---

## Tech Stack
Python, Pandas, scikit-learn, LightGBM, joblib