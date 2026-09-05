# 03. ML Pipeline

## Objective

Forecast daily `net_sales` at store level using a time-aware Machine Learning workflow.

## Target

`net_sales`

## Granularity

`1 row = sale_date + store_id`

## Model Features

### Calendar Features

- `year`
- `month`
- `day`
- `day_of_week`
- `week_of_year`
- `is_weekend`

### Lag Features

- `lag_1`
- `lag_7`
- `lag_14`
- `lag_28`

### Rolling Features

- `rolling_mean_7`
- `rolling_mean_28`
- `rolling_std_7`

### Trend Features

- `lag1_minus_lag7`
- `lag1_vs_mean7`

All lag and rolling calculations use only information prior to the prediction date.

## Temporal Split

The dataset is split chronologically.

Validated periods:

- TRAIN: 2024-01-29 → 2025-12-31
- VALIDATION: 2026-01-01 → 2026-04-30
- TEST: 2026-05-01 → 2026-07-31
- OOT: 2026-08-01 → 2026-08-31

TEST and OOT remain isolated during tuning.

## Hyperparameter Tuning

Six Random Forest configurations were evaluated on TRAIN and compared on VALIDATION.

The selection process applied a materiality rule so that a marginal metric improvement would not automatically justify unnecessary model complexity.

Selected configuration:

- `config_name = RF_01_BASE`
- `numTrees = 60`
- `maxDepth = 8`
- `minInstancesPerNode = 3`
- `featureSubsetStrategy = auto`
- `seed = 42`

## Final Training

The final model is trained on:

`TRAIN + VALIDATION`

The TEST period is used once for final evaluation.

Validated TEST metrics:

- MAE: 2,126.96
- RMSE: 2,710.35
- WAPE: 35.7727%

Validation WAPE:

- 35.4789%

The Validation vs TEST comparison was classified as stable.

## MLflow

MLflow stores:

- parameters;
- validation metrics;
- TEST metrics;
- generalization metrics;
- tags;
- model artifact.

Registered model:

`retail_analytics.5_ml.sales_forecasting_random_forest`

Validated current model version:

`Version 2`

## Inference

Inference:

- loads the registered Spark ML model from Unity Catalog;
- calculates the next forecast date dynamically;
- prepares one row per store;
- generates predictions;
- performs sanity checks;
- persists results through an idempotent Delta MERGE.

Forecast grain:

`forecast_date + store_id + model_version`

MLflow Run ID is retained for traceability on the current model version.
