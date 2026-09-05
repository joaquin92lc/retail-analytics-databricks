# 01. Architecture

## Purpose

This document describes the end-to-end architecture of the Retail Analytics & ML Forecasting Platform.

## End-to-End Flow

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
ML Feature Refresh
    ↓
Feature Table
    ↓
Model Training / Tuning
    ↓
MLflow Tracking
    ↓
Unity Catalog Model Registry
    ↓
Sales Forecast Inference
    ↓
Forecast Delta History
    ↓
ML Monitoring
    ├──────────────→ Monitoring Delta Tables
    ├──────────────→ Monitoring Views
    ├──────────────→ Dashboards
    └──────────────→ Alerts
```

## Architectural Layers

### Landing

Unity Catalog Volumes act as the controlled file-entry layer for source data.

### Bronze

Purpose: preserve ingested data with minimal transformation and maintain a reliable technical ingestion layer.

### Silver

Purpose: apply quality, standardization and business-key controls.

### Gold

Purpose: expose a consumption-ready Star Schema and analytical views.

### ML Feature Layer

Purpose: reconstruct and persist the feature dataset consumed by forecasting.

Table:

`retail_analytics.5_ml.daily_store_features`

### Model Management

MLflow is used for experiment tracking and model lifecycle management. The selected Spark ML Random Forest model is registered in Unity Catalog.

### Inference

The inference notebook loads the registered model and generates daily store-level forecasts.

Forecast table:

`retail_analytics.5_ml.sales_forecast_predictions`

### Monitoring

Monitoring compares predictions against actuals, measures technical forecast quality, calculates performance and detects feature drift.

Main persistence tables:

- `retail_analytics.5_ml.ml_monitoring_runs`
- `retail_analytics.5_ml.feature_drift_monitoring`

## Lakeflow Job

Validated DAG:

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

The complete Job has been executed successfully end-to-end.

## Key Architectural Principle

Technical pipeline state and functional model state are separate concepts.

A monitoring status such as `CRITICAL` indicates model/data behavior requiring attention, but does not imply that the Lakeflow Job itself failed.
