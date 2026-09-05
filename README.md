# Retail Analytics & ML Forecasting Platform

End-to-end **Data Engineering, Analytics and Machine Learning** project built on Databricks for retail sales analysis and store-level sales forecasting.

**Landing → Bronze → Silver → Gold → Analytics → Feature Engineering → Training → Inference → ML Monitoring → Dashboards & Alerts**

## 1. Objective

The platform covers the complete lifecycle of a retail data and ML solution:

- historical and incremental ingestion;
- Medallion Architecture;
- dimensional modeling for analytics;
- daily store-level net-sales forecasting;
- MLflow model tracking and versioning;
- forecast-vs-actual evaluation;
- store-level model performance;
- Feature Drift and PSI monitoring;
- persistent monitoring history in Delta;
- dashboards and alerts;
- end-to-end orchestration with Lakeflow Jobs.

## 2. Technology Stack

- Databricks
- Apache Spark / PySpark / Spark SQL
- Delta Lake
- Unity Catalog and Volumes
- Auto Loader
- Lakeflow Jobs
- MLflow
- Spark ML
- Random Forest Regression
- Databricks SQL
- Databricks Dashboards

## 3. Architecture

```text
Source Files
    ↓
Landing / Unity Catalog Volumes
    ↓
Bronze
    ↓
Silver
    ↓
Gold
    ├──────────────→ SQL Analytics
    ↓
ML Feature Engineering
    ↓
Model Training / Tuning
    ↓
MLflow + Unity Catalog
    ↓
Sales Forecast Inference
    ↓
ML Monitoring
    ├──────────────→ Delta Monitoring History
    ├──────────────→ Dashboards
    └──────────────→ Alerts
```

## 4. Data Engineering

### Landing
Source files for customers, products, stores and sales are stored in Unity Catalog Volumes.

### Bronze
Raw ingestion layer. The design supports idempotent and incremental processing while preserving source-level information.

### Silver
Validated and standardized data layer. It handles schema normalization, data types, business rules, duplicate control, null analysis and consistency checks. Anonymous sales are intentionally preserved.

### Gold
Analytics-ready dimensional **Star Schema**.

Dimensions:
- `dim_customer`
- `dim_product`
- `dim_store`
- `dim_date`

Fact:
- `fact_sales`

The fact-table grain is a sales line identified by `sale_id + line_id`.

## 5. Analytics

Gold is consumed through Databricks SQL views and dashboards for sales trends, product/store performance, regional analysis, gross/net sales, tickets, units, average ticket, discounts and identified vs anonymous customers.

## 6. Machine Learning

The ML use case forecasts **daily net sales at store level**.

Temporal features include:

- `lag_1`
- `lag_7`
- `lag_14`
- `lag_28`
- `rolling_mean_7`
- `rolling_mean_28`
- `rolling_std_7`
- `lag1_minus_lag7`
- `lag1_vs_mean7`

Features are built using information available before the prediction date to avoid target leakage. Time-series datasets are split chronologically rather than randomly.

Random Forest regression configurations are trained and compared. Model selection considers predictive performance and unnecessary complexity.

MLflow and Unity Catalog provide experiment tracking, metrics, parameters, model artifacts, registration, versioning and inference traceability.

## 7. Forecast Inference

Forecasts are persisted by `forecast_date`, `store_id`, `model_name` and `model_version`.

When actual sales are unavailable, a forecast remains `PENDING`. Once actual sales arrive, the original prediction can be evaluated without regenerating it.

## 8. ML Monitoring

The monitoring layer evaluates:

- Forecast Quality
- Model Performance
- Store-Level Performance
- Feature Drift
- Population Stability Index (PSI)

Operational PSI thresholds:

```text
PSI < 0.10          → STABLE
0.10 <= PSI < 0.25  → WARNING
PSI >= 0.25         → DRIFT
```

A feature that cannot be evaluated reliably can be classified as `NOT_EVALUABLE`.

## 9. Pipeline Health vs Model Health

A core design decision is the separation between technical execution and functional ML health.

A functional `CRITICAL` monitoring state:

- is persisted;
- can be visualized;
- can trigger alerts;
- does not automatically fail a technically correct Lakeflow Job.

Technical failures are reserved for genuine execution problems.

## 10. Lakeflow Orchestration

Validated workflow:

```text
bronze_ingestion
        ↓
silver_transformation
        ↓
gold_modeling
   ├────────────→ sql_analytics
   ↓
ml_feature_refresh
        ↓
sales_forecast_inference
        ↓
ml_monitoring
```

The complete end-to-end Job has been executed successfully.

## 11. Monitoring Dashboard

### ML Monitoring Overview
Executive monitoring:
- Forecast Quality
- Feature Drift
- Global Status
- Model Performance
- Feature Drift distribution

### ML Monitoring Analysis
Operational analysis:
- PSI by Feature
- Feature Drift details
- Predicted vs Actual Sales by Store
- Absolute Error by Store
- Forecast Monitoring details

## 12. Project Status

| Component | Status |
|---|---|
| Data ingestion | Complete |
| Bronze / Silver / Gold | Complete |
| Incremental processing | Complete |
| SQL Analytics | Complete |
| ML Feature Engineering | Complete |
| Model Training / Tuning | Complete |
| MLflow / Model Registry | Complete |
| Forecast Inference | Complete |
| ML Monitoring | Complete |
| Monitoring persistence | Complete |
| Dashboards & Alerts | Complete |
| Lakeflow orchestration | Complete |
| End-to-end validation | Complete |

## 13. Next Evolution

The next step is to move from a workspace-based implementation toward reproducible deployment and version control with:

**Git + Databricks Asset Bundles + CI/CD**

## Documentation

Detailed documentation is available under `/docs`.
