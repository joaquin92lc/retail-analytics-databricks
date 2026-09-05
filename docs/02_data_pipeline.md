# 02. Data Pipeline

## Objective

Transform retail source data into trusted, analytics-ready datasets using a Medallion Architecture.

## Landing

Source domains:

- customers
- products
- stores
- sales

Files are stored in Unity Catalog Volumes.

## Bronze

Bronze is the raw ingestion layer.

Main design goals:

- preserve source information;
- provide idempotent ingestion behavior;
- support incremental file arrival;
- avoid applying heavy business logic too early.

## Silver

Silver is the curated transformation layer.

Typical controls include:

- schema normalization;
- data type conversion;
- mandatory-field checks;
- duplicate control;
- business-key validation;
- null analysis;
- referential checks;
- preservation of anonymous sales.

## Gold

Gold exposes a dimensional Star Schema.

### Dimensions

- `dim_customer`
- `dim_product`
- `dim_store`
- `dim_date`

### Fact

- `fact_sales`

Grain:

`1 row = sale_id + line_id`

## Daily Store Aggregation

For ML, Gold sales are aggregated to:

`1 row = sale_date + store_id`

Daily metrics include:

- `net_sales`
- `gross_sales`
- `total_discount`
- `total_units`
- `total_tickets`

## Temporal Regularization

The daily panel is regularized to guarantee a complete store/day grid.

Days without sales are represented with sales values equal to 0.

This is important because lag and rolling features require continuous temporal spacing.

## Feature Table

The final feature dataset is rebuilt from Gold and persisted as:

`retail_analytics.5_ml.daily_store_features`

Current validated coverage:

- 31 stores
- no `sale_date + store_id` duplicates
- no NULL/NaN in model features
- aligned maximum date between Gold and Feature Table

## Operational Quality Gates

Before inference, the feature-refresh process validates:

- source tables exist;
- store coverage is complete;
- Gold and Feature Table are temporally aligned;
- no duplicate store/day keys exist;
- all model features are available;
- no NULL/NaN values exist;
- enough history exists per store.
