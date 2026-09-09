# 06. CI/CD and Multi-Environment Deployment

## Purpose

This document describes the deployment and CI/CD architecture of the Retail Analytics & ML Forecasting Platform.

The objective is to move from a workspace-based implementation to a reproducible, version-controlled deployment model using:

- Databricks Asset Bundles
- Git
- GitHub
- GitHub Actions
- isolated DEV and PROD targets

## Environment Architecture

The project defines two Databricks Asset Bundle targets.

| Target | Catalog | Mode | Deployment |
|---|---|---|---|
| DEV | `retail_analytics_dev` | development | Automatic |
| PROD | `retail_analytics` | production | Manual |

Both environments use the same source code and Lakeflow Job definition.

Environment-specific catalog references are injected using:

```text
${var.catalog}
```

This avoids maintaining separate notebook implementations for DEV and PROD.

## Databricks Asset Bundle

The Bundle entry point is:

```text
databricks.yml
```

It defines:

- Bundle name;
- included resource files;
- environment variables;
- DEV target;
- PROD target;
- environment-specific workspace paths;
- DEV schema resources.

Resource definitions are maintained separately under:

```text
resources/
```

The Lakeflow Job is defined in:

```text
resources/jobs.yml
```

## DEV Environment

DEV uses the Unity Catalog catalog:

```text
retail_analytics_dev
```

The Bundle manages the following schemas:

```text
1_bronze
2_silver
3_gold
5_ml
```

Managing these schemas through the Bundle makes the development environment reproducible.

The deployed development Job uses the same pipeline structure as PROD while receiving `retail_analytics_dev` through the `catalog` notebook parameter.

## PROD Environment

PROD uses:

```text
retail_analytics
```

The production schemas existed before the Asset Bundle implementation.

Therefore, they are intentionally not declared as Bundle-managed schema resources for the PROD target.

This prevents the deployment process from attempting to recreate existing Unity Catalog schemas.

The Bundle deploys and manages the production Lakeflow Job while reusing the existing production data structures.

## Lakeflow Job Parameterization

Each environment-dependent task receives:

```yaml
base_parameters:
  catalog: ${var.catalog}
```

The notebooks retrieve the parameter and construct their catalog references dynamically.

This allows a single codebase to execute against:

```text
DEV  → retail_analytics_dev
PROD → retail_analytics
```

without maintaining duplicated environment-specific notebooks.

## Branching Strategy

Development work is performed outside `main`.

Typical lifecycle:

```text
main
  ↓
feature/* or fix/*
  ↓
changes
  ↓
commit
  ↓
push
  ↓
Pull Request
  ↓
CI validation
  ↓
merge to main
```

Examples used during implementation include:

```text
feature/multi-environment
feature/cicd-deployment
feature/prod-deployment
fix/prod-existing-schemas
```

The `main` branch represents the stable deployable repository state.

## Continuous Integration

Workflow:

```text
.github/workflows/ci.yml
```

Trigger:

```text
Pull Request → main
```

The workflow:

1. checks out the repository;
2. installs the Databricks CLI;
3. authenticates against Databricks using GitHub Secrets;
4. executes:

```bash
databricks bundle validate -t dev
```

A Bundle configuration that does not validate therefore fails before being merged into `main`.

## Continuous Deployment to DEV

Workflow:

```text
.github/workflows/cd.yml
```

Primary trigger:

```text
push → main
```

After validated code is merged into `main`, GitHub Actions automatically executes:

```bash
databricks bundle validate -t dev
databricks bundle deploy -t dev
```

The deployment flow is therefore:

```text
Pull Request
    ↓
CI PASS
    ↓
Merge to main
    ↓
CD starts automatically
    ↓
Validate DEV
    ↓
Deploy DEV
```

This keeps DEV synchronized with the stable state of `main`.

## Controlled PROD Deployment

Workflow:

```text
.github/workflows/deploy-prod.yml
```

PROD uses:

```text
workflow_dispatch
```

and therefore requires an explicit manual trigger.

The workflow executes:

```bash
databricks bundle validate -t prod
databricks bundle deploy -t prod
```

PROD is intentionally not deployed automatically on every merge to `main`.

The promotion flow is:

```text
main
  ↓
DEV deployment successful
  ↓
manual PROD trigger
  ↓
Validate PROD
  ↓
Deploy PROD
```

This separates continuous DEV deployment from controlled production promotion.

## Complete CI/CD Flow

```text
Developer
    ↓
Feature / Fix Branch
    ↓
Commit + Push
    ↓
Pull Request to main
    ↓
┌─────────────────────────┐
│ GitHub Actions - CI     │
│ bundle validate -t dev  │
└────────────┬────────────┘
             ↓
          CI PASS
             ↓
        Merge to main
             ↓
┌─────────────────────────┐
│ GitHub Actions - CD     │
│ validate DEV            │
│ deploy DEV              │
└────────────┬────────────┘
             ↓
        DEV deployed
             ↓
      Manual promotion
             ↓
┌─────────────────────────┐
│ PROD Deployment         │
│ validate PROD           │
│ deploy PROD             │
└────────────┬────────────┘
             ↓
        PROD deployed
```

## Authentication

GitHub Actions authenticates against the Databricks workspace using repository secrets:

```text
DATABRICKS_HOST
DATABRICKS_TOKEN
```

Secrets are stored in GitHub and are not committed to the repository.

The workflow injects them only during execution.

## Validation Commands

Validate DEV:

```bash
databricks bundle validate -t dev
```

Validate PROD:

```bash
databricks bundle validate -t prod
```

Inspect DEV resources:

```bash
databricks bundle summary -t dev
```

Inspect PROD resources:

```bash
databricks bundle summary -t prod
```

## Deployment Commands

Manual DEV deployment:

```bash
databricks bundle deploy -t dev
```

Manual PROD deployment:

```bash
databricks bundle deploy -t prod
```

Normal deployments are handled through GitHub Actions.

## Pipeline Execution

The complete DEV pipeline can be executed with:

```bash
databricks bundle run -t dev retail_analytics_pipeline
```

The deployed Lakeflow Job orchestrates:

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

## Deployment Validation

The implementation has been validated end-to-end.

Confirmed behavior:

- Bundle validation succeeds in DEV;
- Bundle validation succeeds in PROD;
- Pull Request CI validation succeeds;
- merge to `main` automatically deploys DEV;
- DEV deployment succeeds after environment separation;
- PROD deployment can be triggered independently;
- existing PROD schemas are preserved;
- controlled PROD deployment succeeds.

## Design Decision: Existing PROD Schemas

During the first production deployment, the Bundle attempted to create schemas that already existed in `retail_analytics`.

The deployment correctly failed with `SCHEMA_ALREADY_EXISTS`.

The architecture was adjusted so that:

```text
DEV  → Bundle manages schemas
PROD → Bundle reuses existing schemas
```

This preserves existing production data structures while maintaining reproducible provisioning for DEV.

The corrected deployment was subsequently validated successfully in both environments.

## Final Deployment Model

The resulting delivery model is:

```text
Git
  +
GitHub
  +
GitHub Actions
  +
Databricks Asset Bundles
  +
Unity Catalog
  +
Lakeflow Jobs
```

This provides version control, environment isolation, automated validation, continuous DEV deployment and controlled PROD promotion for the complete Retail Analytics & ML platform.
