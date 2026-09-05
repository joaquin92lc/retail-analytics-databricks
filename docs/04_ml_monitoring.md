# 04. ML Monitoring

## Objective

Provide production-oriented observability for the sales-forecasting workflow.

Monitoring covers:

1. Forecast Quality
2. Model Performance
3. Store-Level Performance
4. Feature Drift
5. PSI
6. Monitoring Quality Gates
7. Persistent monitoring history
8. Dashboard consumption
9. Alerts

## Forecast Quality

Controls include:

- no NULL predictions;
- no NaN predictions;
- no negative predictions;
- mandatory identifiers available;
- no logical duplicates;
- complete store coverage by `forecast_date + model_version`.

Logical key:

`forecast_date + store_id + model_version`

## Forecast vs Actual

A forecast is evaluable when:

`forecast_date <= MAX(Gold date)`

If a forecast date already exists in Gold but a store has no sales rows, actual sales are interpreted as 0, consistent with feature temporal regularization.

## Model Performance

When actuals exist, the monitoring layer calculates:

- MAE
- RMSE
- WAPE
- Forecast Bias

Metrics remain `PENDING` if actuals are unavailable.

## Store-Level Performance

Performance is also calculated per:

`store_id + model_name + model_version`

This helps identify local degradation hidden by aggregate metrics.

## Feature Drift Windows

### Reference Window

Historical feature distribution before the current monitoring window.

### Current Window

Latest 30 available days.

The validated current implementation checks:

- complete store coverage;
- exact 30-day window;
- expected row count;
- no duplicates.

## Drift Features

- `lag_1`
- `lag_7`
- `lag_14`
- `lag_28`
- `rolling_mean_7`
- `rolling_mean_28`
- `rolling_std_7`
- `lag1_minus_lag7`
- `lag1_vs_mean7`

## PSI

Operational thresholds:

- `PSI < 0.10` → `STABLE`
- `0.10 <= PSI < 0.25` → `WARNING`
- `PSI >= 0.25` → `DRIFT`
- insufficient variability → `NOT_EVALUABLE`

Bins are derived exclusively from the Reference Window.

## Functional Status

Monitoring combines:

- Forecast Quality;
- Feature Drift;
- Model Performance.

Global functional states:

- `HEALTHY`
- `WARNING`
- `CRITICAL`
- `PENDING_PERFORMANCE`

## Technical vs Functional State

A `CRITICAL` model-monitoring state is not automatically a Lakeflow failure.

Functional degradation is:

- persisted;
- visible in dashboards;
- eligible for alerts.

Technical failure is reserved for genuine execution/integrity problems.

## Persistence

Global runs:

`retail_analytics.5_ml.ml_monitoring_runs`

Feature drift:

`retail_analytics.5_ml.feature_drift_monitoring`

Persistence uses idempotent MERGE logic and supports schema evolution while preserving legacy historical rows.

## Consumption Views

- `vw_ml_monitoring_overview`
- `vw_feature_drift_history`
- `vw_forecast_monitoring`

## Dashboard

### Page 1 — ML Monitoring Overview

Maximum 5 visualizations:

- Forecast Quality
- Feature Drift
- Global Status
- Model Performance
- Feature Drift distribution

### Page 2 — ML Monitoring Analysis

Maximum 5 visualizations:

- PSI by Feature
- Feature Drift Details
- Predicted vs Actual Sales by Store
- Absolute Error by Store
- Forecast Monitoring Details

## Alerts

Alerts are already configured on monitoring states.

## End-to-End Validation

The complete Lakeflow Job executed successfully while the monitoring layer preserved a functional `CRITICAL` status.

This validates the intended separation between technical orchestration health and model/data health.
