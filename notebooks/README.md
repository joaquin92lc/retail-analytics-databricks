# Notebooks

This folder will contain the Databricks notebook source files that form the project pipeline.

Planned structure:

```text
02_bronze_ingestion
03_silver_transformation
04_gold_modeling
05_sql_analytics
06_incremental_validation
07_performance_optimization
08_ml_sales_forecasting
08_ml_sales_forecasting_tuning
09_ml_model_optimization
10_ml_sales_forecasting_inference
11_ml_feature_refresh
12_ml_monitoring
```

The notebooks will be exported from Databricks as source files before they are added to Git.

The Lakeflow Job orchestration is documented separately and will later be represented as code when Databricks Asset Bundles are introduced.
