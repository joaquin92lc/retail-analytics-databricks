# 05. Results and Engineering Decisions

## Purpose

Document the main outcomes and design decisions so the project can be understood without reading every notebook.

## 1. Medallion Architecture

Decision:
Separate ingestion, curation and consumption into Bronze, Silver and Gold.

Reason:
Each layer has a distinct responsibility and quality contract.

## 2. Preserve Anonymous Sales

Decision:
Do not drop anonymous transactions simply because `customer_id` is unavailable.

Reason:
They remain valid sales and are required for complete revenue forecasting and analytics.

## 3. Regularize Store-Day History

Decision:
Build a complete temporal grid for stores and days and fill true no-sales days with 0.

Reason:
Lag and rolling features require consistent temporal spacing.

## 4. Prevent Target Leakage

Decision:
Rolling windows exclude the current observation and lag features use previous days only.

Reason:
The model must not use information unavailable at prediction time.

## 5. Chronological ML Splits

Decision:
Use TRAIN, VALIDATION, TEST and OOT periods rather than random splitting.

Reason:
Forecasting performance must be evaluated on future unseen periods.

## 6. Keep TEST and OOT Isolated

Decision:
Hyperparameter tuning uses only TRAIN + VALIDATION.

Reason:
TEST must remain an unbiased final evaluation set and OOT is preserved for temporal robustness checks.

## 7. Materiality Rule for Tuning

Decision:
Do not select a more complex configuration when the improvement over the base model is immaterial.

Observed result:
The technical winner improved WAPE only marginally, below the selected materiality threshold, so the base configuration was retained.

## 8. Final Random Forest

Selected configuration:

- `RF_01_BASE`
- `numTrees = 60`
- `maxDepth = 8`
- `minInstancesPerNode = 3`
- `featureSubsetStrategy = auto`
- `seed = 42`

Final TEST metrics:

- MAE: 2,126.96
- RMSE: 2,710.35
- WAPE: 35.7727%

## 9. Use MLflow and Unity Catalog

Decision:
Track experiments and register the production model.

Reason:
Reproducibility, traceability and version-controlled inference.

Current validated model:

`retail_analytics.5_ml.sales_forecasting_random_forest` — Version 2

## 10. Forecast History Is Multi-Version

Decision:
Preserve forecasts from historical model versions.

Logical key:

`forecast_date + store_id + model_version`

Reason:
Allows model-version traceability and historical comparison without treating valid multi-version predictions as duplicates.

## 11. Current Forecast Can Be Pending

Decision:
A forecast remains `PENDING` until Gold contains actual data for its forecast date.

Reason:
Evaluation should not fabricate actuals for future dates.

## 12. Store With No Sales on an Available Date Means Actual = 0

Decision:
If the date exists in Gold but the store has no transaction rows, actual sales are treated as 0.

Reason:
This matches the regularized feature-engineering semantics.

## 13. PSI for Feature Drift

Decision:
Monitor feature distribution changes using PSI in addition to descriptive statistics.

Thresholds:

- < 0.10 → STABLE
- 0.10–0.25 → WARNING
- >= 0.25 → DRIFT

## 14. Functional CRITICAL Does Not Mean Technical Failure

Decision:
Do not fail Lakeflow solely because ML Monitoring reports `CRITICAL`.

Reason:
The monitoring system is working correctly when it detects and persists degradation.

Technical pipeline health and model health must remain separate.

## 15. Idempotent Persistence

Decision:
Use MERGE logic for forecasts and monitoring history.

Reason:
Safe re-execution without logical duplicates.

## 16. Dashboard Design

Decision:
Use two pages with a maximum of five visualizations each.

Reason:
Keep executive monitoring separate from technical analysis and avoid overcrowding.

## 17. End-to-End Result

The complete Lakeflow DAG has been executed successfully with all tasks green.

The project currently covers:

- Data Engineering
- Analytics
- ML Feature Engineering
- Training and Tuning
- MLflow / Registry
- Inference
- Monitoring
- Dashboards
- Alerts
- Lakeflow orchestration

## 18. Next Step

Move toward deployment-oriented engineering with:

**Git → Databricks Asset Bundles → CI/CD**

This is the next stage of the project rather than adding more isolated notebook functionality.
