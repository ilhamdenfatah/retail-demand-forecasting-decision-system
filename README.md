# Business-Driven Demand Forecasting & Decision System

## Executive Summary

This project demonstrates an **end-to-end, business-driven demand forecasting system** designed to support real operational planning decisions in a retail context. Rather than treating forecasts as purely numerical predictions, this system explicitly positions forecasts as **decision signals**—used to identify risk, prioritize attention, and guide human intervention.

The system operates at the **store–department–time** level, aligning closely with how planning and operational decisions are typically made in real retail organizations. The final output is a **decision-ready priority list** that planners can directly act upon.

---

## Business Problem

Retail demand behavior is inherently heterogeneous across stores and departments. In practice, planners face several recurring challenges:

* Forecast accuracy varies significantly across segments
* Percentage-based errors can be misleading when demand is low
* Limited operational bandwidth requires clear prioritization
* Forecast outputs often lack clear guidance on *where* to intervene

Traditional forecasting pipelines stop at model evaluation metrics, leaving a critical gap between model outputs and real business decisions.

This project addresses that gap.

---

## System Objective

The primary objective of this system is to:

> **Transform validated demand forecasts into actionable, decision-ready signals that support operational prioritization and planning.**

Key design goals:

* Focus on *decision usability*, not model complexity
* Ensure outputs are interpretable by non-technical stakeholders
* Preserve human judgment as part of the decision loop

---

## High-Level Architecture

The system is structured into four clear stages:

1. **Data Preparation & EDA**
   Cleaning, validation, and exploratory analysis of multi-source retail data.

2. **Feature Engineering**
   Business-informed feature construction with strict feature contract locking to ensure training–inference consistency.

3. **Forecast Modeling**
   A primary forecasting model benchmarked against a simple SMA baseline, with performance evaluated using WAPE and stability diagnostics.

4. **Decision Layer & Business Actions**
   A rule-based decision layer that converts forecast residuals into risk signals, priority rankings, and concrete action recommendations.

---

## Decision Layer: Core Design Philosophy

The decision layer is the defining component of this project.

Key principles:

* **Forecasts are directional signals, not absolute truths**
  The system prioritizes relative risk and ranking over point accuracy.

* **Actionability over precision**
  Slightly imperfect but interpretable signals are preferred over complex, opaque logic.

* **Explainability by design**
  All decision rules are transparent and traceable.

* **Human-in-the-loop**
  The system supports planners; it does not replace managerial judgment.

---

## Decision Unit & Granularity

All decisions are evaluated at the **store–department** level.

This granularity was intentionally chosen because it:

* Reflects how inventory and planning decisions are executed in practice
* Provides more stable signals than SKU-level analysis
* Avoids excessive noise while remaining operationally actionable

SKU-level execution is assumed to be handled downstream by operational teams.

---

## Risk Signals & Prioritization Logic

The system derives risk signals from forecast residuals on the validation and test splits.

Two primary risk dimensions are defined:

* **Under-forecast risk** (stockout proxy)
* **Over-forecast risk** (overstock proxy)

These are aggregated across time to produce store–department–level indicators, including:

* Frequency of under- and over-forecasting
* Sustained error magnitude (WAPE)
* Minimum historical support (period threshold)

Guardrails are applied to ensure decision safety (e.g., minimum periods, robust denominators).

The final output is a **ranked priority list** with tier classification (HIGH / MEDIUM / LOW) and explicit recommended actions.

---

## Final Outputs

The system produces:

* A trained forecasting model and preprocessing pipeline
* Residual-based diagnostics aligned with operational risk
* A prioritized store–department decision table
* Tiered risk classification
* Human-readable, actionable recommendations

These outputs are designed to be consumed directly by planning and operations teams.

---

## Project Structure

```
RETAIL-DEMAND-FORECASTING-DECISION-SYSTEM
├── data_raw/
├── data_clean/
├── data_modeling/
├── artifacts/
│   ├── preprocess.joblib
│   ├── primary_model.joblib
│   ├── forecast_residuals_val_test.parquet
│   └── ...
├── src/
│   ├── feature_engineering.py
│   └── time_split.py
├── 01_data_merge_eda.ipynb
├── 02_feature_engineering.ipynb
├── 03_modeling.ipynb
├── 04_decision_layer_and_business_actions.ipynb
└── README.md
```

---

## Tech Stack

* **Python**
* **Pandas / NumPy**
* **scikit-learn**
* **LightGBM**
* **joblib**

---

## Intended Audience

This project is designed for:

* **Business, Planning, and Operations stakeholders** seeking decision-ready insights
* **Data Scientists / Analysts** interested in bridging modeling and real-world decision-making
* **Recruiters** evaluating applied, business-aligned data science work

---

## Key Takeaway

This project intentionally goes beyond traditional forecasting pipelines.

It demonstrates how demand forecasts can be systematically transformed into **prioritized, explainable, and actionable business decisions**—closing the gap between modeling and real operational impact.
