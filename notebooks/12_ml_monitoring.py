# Databricks notebook source
# MAGIC %md
# MAGIC # 00. CONFIGURACIÓN Y LECTURA DE DATOS
# MAGIC
# MAGIC ## Objetivo
# MAGIC
# MAGIC Construir la capa de **ML Monitoring** del modelo de Sales Forecasting desarrollado para Retail Analytics.
# MAGIC
# MAGIC Hasta este punto, el proyecto dispone de un flujo automatizado capaz de:
# MAGIC
# MAGIC **Gold → Feature Engineering → Model Registry → Inference → Delta Forecast**
# MAGIC
# MAGIC El objetivo de este notebook es añadir una capa adicional de observabilidad que permita evaluar el comportamiento del modelo una vez que las predicciones puedan compararse con ventas reales.
# MAGIC
# MAGIC ## Estrategia de Monitoring
# MAGIC
# MAGIC Se monitorizarán cuatro dimensiones principales:
# MAGIC
# MAGIC ### 1. Forecast Quality
# MAGIC
# MAGIC Validación técnica de las predicciones generadas:
# MAGIC
# MAGIC - predicciones NULL
# MAGIC - predicciones negativas
# MAGIC - duplicados
# MAGIC - cobertura de tiendas
# MAGIC - cobertura temporal
# MAGIC
# MAGIC ### 2. Model Performance
# MAGIC
# MAGIC Cuando exista el dato real correspondiente al forecast se calcularán:
# MAGIC
# MAGIC - MAE
# MAGIC - RMSE
# MAGIC - WAPE
# MAGIC
# MAGIC Esto permitirá detectar degradación del rendimiento respecto al modelo originalmente validado.
# MAGIC
# MAGIC ### 3. Store-Level Performance
# MAGIC
# MAGIC El error se analizará también por tienda para identificar establecimientos donde el modelo presente sistemáticamente peor comportamiento.
# MAGIC
# MAGIC ### 4. Feature Drift
# MAGIC
# MAGIC Se comparará la distribución reciente de las principales features con la distribución histórica utilizada como referencia.
# MAGIC
# MAGIC Esto permitirá detectar cambios en los patrones de datos que puedan afectar al rendimiento del modelo.
# MAGIC
# MAGIC ## Arquitectura
# MAGIC
# MAGIC **Gold Actuals**
# MAGIC
# MAGIC ↓
# MAGIC
# MAGIC **Forecast Predictions**
# MAGIC
# MAGIC ↓
# MAGIC
# MAGIC **Forecast vs Actual**
# MAGIC
# MAGIC ↓
# MAGIC
# MAGIC **Performance Metrics**
# MAGIC
# MAGIC ↓
# MAGIC
# MAGIC **Feature Drift**
# MAGIC
# MAGIC ↓
# MAGIC
# MAGIC **Quality Gates**
# MAGIC
# MAGIC ↓
# MAGIC
# MAGIC **Monitoring Tables**
# MAGIC
# MAGIC ## Consideración temporal
# MAGIC
# MAGIC Una predicción solo puede evaluarse cuando el dato real correspondiente a su `forecast_date` está disponible en Gold.
# MAGIC
# MAGIC Por tanto:
# MAGIC
# MAGIC `forecast_date <= MAX(fecha disponible en Gold)`
# MAGIC
# MAGIC será la condición necesaria para evaluar el rendimiento real del modelo.
# MAGIC
# MAGIC Los forecasts todavía no observados permanecerán pendientes de evaluación.

# COMMAND ----------

# Catálogo recibido desde Databricks Asset Bundles.
# DEV  -> retail_analytics_dev
# PROD -> retail_analytics

dbutils.widgets.text("catalog", "retail_analytics")
CATALOG = dbutils.widgets.get("catalog")

spark.sql(f"USE CATALOG `{CATALOG}`")
spark.sql("USE SCHEMA `5_ml`")

print(f"Environment catalog: {CATALOG}")
print("Environment schema: 5_ml")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 00.01 CONFIGURACIÓN DE FUENTES DE MONITORING
# MAGIC
# MAGIC Se configuran las principales fuentes necesarias para el proceso de ML Monitoring.
# MAGIC
# MAGIC El proceso utilizará:
# MAGIC
# MAGIC - **Gold Fact Sales:** contiene las ventas reales observadas.
# MAGIC - **Forecast Predictions:** contiene las predicciones generadas por el modelo.
# MAGIC - **Feature Table:** contiene las features utilizadas durante inference.
# MAGIC
# MAGIC Estas tres fuentes permitirán posteriormente comparar:
# MAGIC
# MAGIC **Predicción → Venta real → Features utilizadas**
# MAGIC
# MAGIC Antes de iniciar el proceso se valida:
# MAGIC
# MAGIC - existencia de las tablas
# MAGIC - disponibilidad de datos
# MAGIC - rango temporal disponible
# MAGIC - número de tiendas
# MAGIC - forecasts disponibles
# MAGIC - forecasts que ya pueden ser evaluados con datos reales

# COMMAND ----------

# DBTITLE 1,00.01 CONFIGURACIÓN DE FUENTES DE MONITORING
# ============================================================
# 00.01 CONFIGURACIÓN DE FUENTES DE MONITORING
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Configurar y validar todas las fuentes necesarias para
# ejecutar el proceso de ML Monitoring.
#
# En este bloque también se obtiene dinámicamente el número
# esperado de tiendas desde dim_store.
#
# De esta forma evitamos utilizar hardcodes como:
#
#     EXPECTED_STORES = 31
#
# en los bloques posteriores.
#
# ============================================================


from pyspark.sql import functions as F


# ============================================================
# 00.01.01 CONFIGURACIÓN DE TABLAS
# ============================================================

FACT_SALES_TABLE = (
    f"{CATALOG}.3_gold.fact_sales"
)


DIM_DATE_TABLE = (
    f"{CATALOG}.3_gold.dim_date"
)


DIM_STORE_TABLE = (
    f"{CATALOG}.3_gold.dim_store"
)


FORECAST_TABLE = (
    f"{CATALOG}.5_ml.sales_forecast_predictions"
)


FEATURE_TABLE = (
    f"{CATALOG}.5_ml.daily_store_features"
)


# ============================================================
# 00.01.02 VALIDACIÓN DE EXISTENCIA
# ============================================================

required_tables = [

    FACT_SALES_TABLE,
    DIM_DATE_TABLE,
    DIM_STORE_TABLE,
    FORECAST_TABLE,
    FEATURE_TABLE

]


missing_tables = [

    table_name

    for table_name in required_tables

    if not spark.catalog.tableExists(
        table_name
    )

]


if missing_tables:

    raise RuntimeError(

        "Faltan tablas necesarias para ML Monitoring: "
        + ", ".join(
            missing_tables
        )

    )


print(
    "OK - Todas las tablas necesarias para Monitoring "
    "están disponibles"
)


# ============================================================
# 00.01.03 LECTURA DE DATOS
# ============================================================

fact_sales_df = (

    spark.table(
        FACT_SALES_TABLE
    )

)


dim_date_df = (

    spark.table(
        DIM_DATE_TABLE
    )

    .select(
        "date_key",
        "date"
    )

)


dim_store_df = (

    spark.table(
        DIM_STORE_TABLE
    )

)


forecast_monitor_df = (

    spark.table(
        FORECAST_TABLE
    )

)


feature_monitor_df = (

    spark.table(
        FEATURE_TABLE
    )

)


# ============================================================
# 00.01.04 NÚMERO ESPERADO DE TIENDAS
# ============================================================
#
# La dimensión de tiendas actúa como fuente de verdad para
# determinar la cobertura esperada.
#
# Esto evita hardcodes en Monitoring.
#
# ============================================================

EXPECTED_STORES = (

    dim_store_df

    .select(
        "store_id"
    )

    .distinct()

    .count()

)


if EXPECTED_STORES == 0:

    raise RuntimeError(

        "dim_store no contiene tiendas. "
        "No es posible inicializar ML Monitoring."

    )


print(
    f"OK - Tiendas esperadas obtenidas dinámicamente: "
    f"{EXPECTED_STORES}"
)


# ============================================================
# 00.01.05 GOLD CON FECHA REAL
# ============================================================
#
# fact_sales utiliza date_key como clave temporal.
#
# Para trabajar con la fecha real realizamos JOIN con dim_date.
#
# ============================================================

fact_sales_with_date_df = (

    fact_sales_df

    .join(

        dim_date_df,

        on="date_key",

        how="left"

    )

)


# ============================================================
# 00.01.06 VALIDACIÓN DEL JOIN TEMPORAL
# ============================================================

missing_gold_dates = (

    fact_sales_with_date_df

    .filter(
        F.col("date").isNull()
    )

    .count()

)


if missing_gold_dates > 0:

    raise RuntimeError(

        "Existen registros de fact_sales sin correspondencia "
        f"en dim_date: {missing_gold_dates:,}"

    )


# ============================================================
# 00.01.07 ESTADO DE GOLD
# ============================================================

gold_status = (

    fact_sales_with_date_df

    .agg(

        F.min(
            "date"
        ).alias(
            "min_date"
        ),

        F.max(
            "date"
        ).alias(
            "max_date"
        ),

        F.count(
            "*"
        ).alias(
            "records"
        ),

        F.countDistinct(
            "store_id"
        ).alias(
            "stores"
        )

    )

    .first()

)


# ============================================================
# 00.01.08 ESTADO DEL FORECAST
# ============================================================

forecast_status = (

    forecast_monitor_df

    .agg(

        F.min(
            "forecast_date"
        ).alias(
            "min_date"
        ),

        F.max(
            "forecast_date"
        ).alias(
            "max_date"
        ),

        F.count(
            "*"
        ).alias(
            "records"
        ),

        F.countDistinct(
            "forecast_date"
        ).alias(
            "forecast_dates"
        ),

        F.countDistinct(
            "store_id"
        ).alias(
            "stores"
        ),

        F.countDistinct(
            "model_version"
        ).alias(
            "model_versions"
        )

    )

    .first()

)


# ============================================================
# 00.01.09 ESTADO DE FEATURE TABLE
# ============================================================

feature_status = (

    feature_monitor_df

    .agg(

        F.min(
            "sale_date"
        ).alias(
            "min_date"
        ),

        F.max(
            "sale_date"
        ).alias(
            "max_date"
        ),

        F.count(
            "*"
        ).alias(
            "records"
        ),

        F.countDistinct(
            "store_id"
        ).alias(
            "stores"
        )

    )

    .first()

)


# ============================================================
# 00.01.10 VALIDACIONES DE COBERTURA
# ============================================================

if gold_status["stores"] != EXPECTED_STORES:

    raise RuntimeError(

        "La cobertura de tiendas en Gold no coincide "
        "con dim_store. "
        f"Esperadas: {EXPECTED_STORES} | "
        f"Gold: {gold_status['stores']}"

    )


if feature_status["stores"] != EXPECTED_STORES:

    raise RuntimeError(

        "La cobertura de tiendas en Feature Table no coincide "
        "con dim_store. "
        f"Esperadas: {EXPECTED_STORES} | "
        f"Feature Table: {feature_status['stores']}"

    )


# ============================================================
# 00.01.11 FORECASTS EVALUABLES
# ============================================================
#
# Un forecast únicamente puede evaluarse cuando su fecha
# ya se encuentra disponible en Gold.
#
# Condición:
#
# forecast_date <= MAX(fecha Gold)
#
# IMPORTANTE:
#
# Aquí pueden coexistir diferentes model_version.
# La separación por versión se realizará en los bloques
# posteriores cuando calculemos las métricas.
#
# ============================================================

max_gold_date = (
    gold_status[
        "max_date"
    ]
)


evaluable_forecasts_df = (

    forecast_monitor_df

    .filter(

        F.col(
            "forecast_date"
        )

        <=

        F.lit(
            max_gold_date
        )

    )

)


evaluable_records = (
    evaluable_forecasts_df.count()
)


evaluable_dates = (

    evaluable_forecasts_df

    .select(
        "forecast_date"
    )

    .distinct()

    .count()

)


evaluable_versions = (

    evaluable_forecasts_df

    .select(
        "model_version"
    )

    .distinct()

    .count()

)


# ============================================================
# 00.01.12 RESULTADO
# ============================================================

print()
print("=" * 80)
print("CONFIGURACIÓN ML MONITORING")
print("=" * 80)


print()
print("TABLAS")
print("-" * 80)


print(
    f"Gold Fact:      "
    f"{FACT_SALES_TABLE}"
)


print(
    f"Dim Date:       "
    f"{DIM_DATE_TABLE}"
)


print(
    f"Dim Store:      "
    f"{DIM_STORE_TABLE}"
)


print(
    f"Forecast:       "
    f"{FORECAST_TABLE}"
)


print(
    f"Feature Table:  "
    f"{FEATURE_TABLE}"
)


# ============================================================
# ESTADO GOLD
# ============================================================

print()
print("-" * 80)
print("ESTADO GOLD")
print("-" * 80)


print(
    f"Registros:      "
    f"{gold_status['records']:,}"
)


print(
    f"Tiendas:        "
    f"{gold_status['stores']} "
    f"/ {EXPECTED_STORES}"
)


print(
    f"Periodo:        "
    f"{gold_status['min_date']} "
    f"-> "
    f"{gold_status['max_date']}"
)


# ============================================================
# ESTADO FORECAST
# ============================================================

print()
print("-" * 80)
print("ESTADO FORECAST")
print("-" * 80)


print(
    f"Registros:      "
    f"{forecast_status['records']:,}"
)


print(
    f"Tiendas:        "
    f"{forecast_status['stores']}"
)


print(
    f"Fechas:         "
    f"{forecast_status['forecast_dates']}"
)


print(
    f"Periodo:        "
    f"{forecast_status['min_date']} "
    f"-> "
    f"{forecast_status['max_date']}"
)


print(
    f"Versiones:      "
    f"{forecast_status['model_versions']}"
)


# ============================================================
# ESTADO FEATURE TABLE
# ============================================================

print()
print("-" * 80)
print("ESTADO FEATURE TABLE")
print("-" * 80)


print(
    f"Registros:      "
    f"{feature_status['records']:,}"
)


print(
    f"Tiendas:        "
    f"{feature_status['stores']} "
    f"/ {EXPECTED_STORES}"
)


print(
    f"Periodo:        "
    f"{feature_status['min_date']} "
    f"-> "
    f"{feature_status['max_date']}"
)


# ============================================================
# FORECASTS EVALUABLES
# ============================================================

print()
print("-" * 80)
print("FORECASTS EVALUABLES")
print("-" * 80)


print(
    f"Última fecha real Gold: "
    f"{max_gold_date}"
)


print(
    f"Registros evaluables:   "
    f"{evaluable_records:,}"
)


print(
    f"Fechas evaluables:      "
    f"{evaluable_dates}"
)


print(
    f"Versiones evaluables:   "
    f"{evaluable_versions}"
)


# ============================================================
# 00.01.13 INTERPRETACIÓN DEL ESTADO
# ============================================================

if evaluable_records == 0:

    print()

    print(
        "INFO - No existen todavía forecasts con dato real "
        "disponible para evaluar."
    )

    print(
        "El Monitoring de performance permanecerá pendiente "
        "hasta que Gold alcance alguna forecast_date."
    )


else:

    print()

    print(
        "OK - Existen forecasts disponibles para evaluar "
        "contra ventas reales."
    )


# ============================================================
# 00.01.14 CIERRE
# ============================================================

print()
print("=" * 80)

print(
    "OK - Fuentes de ML Monitoring configuradas correctamente"
)

print(
    f"OK - Cobertura esperada dinámica: "
    f"{EXPECTED_STORES} tiendas"
)

print("=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC # 01. MONITORING DE CALIDAD DEL FORECAST
# MAGIC
# MAGIC ## Objetivo
# MAGIC
# MAGIC Validar la calidad técnica de las predicciones generadas antes de utilizarlas para análisis de performance.
# MAGIC
# MAGIC Estas comprobaciones son independientes de la disponibilidad del dato real y pueden ejecutarse inmediatamente después del proceso de inference.
# MAGIC
# MAGIC Se validará:
# MAGIC
# MAGIC - ausencia de predicciones NULL
# MAGIC - ausencia de predicciones negativas
# MAGIC - unicidad por `forecast_date + store_id`
# MAGIC - cobertura completa de tiendas
# MAGIC - existencia de versión de modelo
# MAGIC - existencia de timestamp de predicción
# MAGIC
# MAGIC Estas validaciones constituyen la primera capa de control del proceso de ML Monitoring.

# COMMAND ----------

# DBTITLE 1,01.01 QUALITY CHECKS DEL FORECAST
# ============================================================
# 01.01 QUALITY CHECKS DEL FORECAST
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Validar la calidad técnica de las predicciones almacenadas
# antes de utilizarlas para análisis de performance.
#
# La granularidad lógica del histórico de forecasts es:
#
#     forecast_date + store_id + model_version
#
# Esto permite conservar diferentes versiones del modelo
# para una misma fecha y tienda sin considerarlas duplicados.
#
# EXPECTED_STORES ya se obtiene dinámicamente en 00.01
# a partir de dim_store.
#
# ============================================================


from pyspark.sql import functions as F


# ============================================================
# 01.01.01 VALIDACIONES PREVIAS
# ============================================================

required_objects = [

    "forecast_monitor_df",
    "EXPECTED_STORES"

]


missing_objects = [

    object_name

    for object_name in required_objects

    if object_name not in globals()

]


if missing_objects:

    raise RuntimeError(

        "Faltan objetos necesarios para validar el forecast: "
        f"{missing_objects}"

    )


required_columns = [

    "forecast_date",
    "store_id",
    "predicted_net_sales",
    "model_name",
    "model_version",
    "prediction_timestamp"

]


missing_columns = [

    column_name

    for column_name in required_columns

    if column_name not in forecast_monitor_df.columns

]


if missing_columns:

    raise RuntimeError(

        "Faltan columnas necesarias en la tabla de forecasts: "
        f"{missing_columns}"

    )


# ============================================================
# 01.01.02 MÉTRICAS GENERALES DE CALIDAD
# ============================================================

forecast_quality = (

    forecast_monitor_df

    .agg(

        F.count(
            "*"
        ).alias(
            "records"
        ),

        F.countDistinct(
            "forecast_date"
        ).alias(
            "forecast_dates"
        ),

        F.countDistinct(
            "store_id"
        ).alias(
            "stores"
        ),

        F.countDistinct(
            "model_version"
        ).alias(
            "model_versions"
        ),

        F.sum(

            F.when(

                F.col(
                    "predicted_net_sales"
                ).isNull(),

                1

            ).otherwise(
                0
            )

        ).alias(
            "null_predictions"
        ),

        F.sum(

            F.when(

                F.isnan(
                    F.col(
                        "predicted_net_sales"
                    )
                ),

                1

            ).otherwise(
                0
            )

        ).alias(
            "nan_predictions"
        ),

        F.sum(

            F.when(

                F.col(
                    "predicted_net_sales"
                ) < 0,

                1

            ).otherwise(
                0
            )

        ).alias(
            "negative_predictions"
        ),

        F.sum(

            F.when(

                F.col(
                    "forecast_date"
                ).isNull(),

                1

            ).otherwise(
                0
            )

        ).alias(
            "null_forecast_date"
        ),

        F.sum(

            F.when(

                F.col(
                    "store_id"
                ).isNull(),

                1

            ).otherwise(
                0
            )

        ).alias(
            "null_store_id"
        ),

        F.sum(

            F.when(

                F.col(
                    "model_name"
                ).isNull(),

                1

            ).otherwise(
                0
            )

        ).alias(
            "null_model_name"
        ),

        F.sum(

            F.when(

                F.col(
                    "model_version"
                ).isNull(),

                1

            ).otherwise(
                0
            )

        ).alias(
            "null_model_version"
        ),

        F.sum(

            F.when(

                F.col(
                    "prediction_timestamp"
                ).isNull(),

                1

            ).otherwise(
                0
            )

        ).alias(
            "null_timestamp"
        )

    )

    .first()

)


# ============================================================
# 01.01.03 DUPLICADOS LÓGICOS
# ============================================================
#
# IMPORTANTE
# ------------------------------------------------------------
#
# Dos registros NO son duplicados si pertenecen a diferentes
# model_version.
#
# Clave lógica:
#
# forecast_date + store_id + model_version
#
# ============================================================

duplicate_forecasts_df = (

    forecast_monitor_df

    .groupBy(

        "forecast_date",
        "store_id",
        "model_version"

    )

    .count()

    .filter(
        F.col("count") > 1
    )

)


duplicate_forecasts = (
    duplicate_forecasts_df.count()
)


# ============================================================
# 01.01.04 COBERTURA POR FECHA Y VERSIÓN
# ============================================================
#
# La cobertura también debe evaluarse por versión.
#
# Ejemplo válido:
#
# 2026-08-01 | Version 1 | 31 tiendas
# 2026-09-01 | Version 2 | 31 tiendas
#
# No tendría sentido mezclar ambas versiones en una única
# comprobación por forecast_date.
#
# ============================================================

forecast_coverage_df = (

    forecast_monitor_df

    .groupBy(

        "forecast_date",
        "model_version"

    )

    .agg(

        F.count(
            "*"
        ).alias(
            "predictions"
        ),

        F.countDistinct(
            "store_id"
        ).alias(
            "stores"
        )

    )

    .withColumn(

        "expected_stores",

        F.lit(
            EXPECTED_STORES
        )

    )

    .withColumn(

        "missing_stores",

        F.col(
            "expected_stores"
        )

        -

        F.col(
            "stores"
        )

    )

    .orderBy(

        "forecast_date",
        "model_version"

    )

)


# ============================================================
# 01.01.05 FECHAS / VERSIONES CON COBERTURA INCOMPLETA
# ============================================================

incomplete_forecast_groups_df = (

    forecast_coverage_df

    .filter(

        F.col(
            "stores"
        )

        !=

        F.col(
            "expected_stores"
        )

    )

)


incomplete_forecast_groups = (
    incomplete_forecast_groups_df.count()
)


# ============================================================
# 01.01.06 QUALITY ERRORS
# ============================================================

quality_errors = []


if forecast_quality["records"] == 0:

    quality_errors.append(
        "La tabla de forecasts está vacía"
    )


if forecast_quality["null_predictions"] > 0:

    quality_errors.append(
        "Existen predicciones NULL"
    )


if forecast_quality["nan_predictions"] > 0:

    quality_errors.append(
        "Existen predicciones NaN"
    )


if forecast_quality["negative_predictions"] > 0:

    quality_errors.append(
        "Existen predicciones negativas"
    )


if forecast_quality["null_forecast_date"] > 0:

    quality_errors.append(
        "Existen registros sin forecast_date"
    )


if forecast_quality["null_store_id"] > 0:

    quality_errors.append(
        "Existen registros sin store_id"
    )


if forecast_quality["null_model_name"] > 0:

    quality_errors.append(
        "Existen registros sin model_name"
    )


if forecast_quality["null_model_version"] > 0:

    quality_errors.append(
        "Existen registros sin model_version"
    )


if forecast_quality["null_timestamp"] > 0:

    quality_errors.append(
        "Existen registros sin prediction_timestamp"
    )


if duplicate_forecasts > 0:

    quality_errors.append(

        "Existen forecasts duplicados según la clave "
        "forecast_date + store_id + model_version"

    )


if incomplete_forecast_groups > 0:

    quality_errors.append(

        "Existen combinaciones forecast_date + model_version "
        "con cobertura incompleta de tiendas"

    )


# ============================================================
# 01.01.07 RESULTADO
# ============================================================

print()
print("=" * 80)
print("FORECAST QUALITY MONITORING")
print("=" * 80)


print(
    f"Registros:                 "
    f"{forecast_quality['records']:,}"
)


print(
    f"Fechas forecast:            "
    f"{forecast_quality['forecast_dates']}"
)


print(
    f"Tiendas distintas:          "
    f"{forecast_quality['stores']}"
)


print(
    f"Versiones modelo:           "
    f"{forecast_quality['model_versions']}"
)


print(
    f"Tiendas esperadas:          "
    f"{EXPECTED_STORES}"
)


print()
print("-" * 80)
print("CALIDAD TÉCNICA")
print("-" * 80)


print(
    f"Predicciones NULL:          "
    f"{forecast_quality['null_predictions']}"
)


print(
    f"Predicciones NaN:           "
    f"{forecast_quality['nan_predictions']}"
)


print(
    f"Predicciones negativas:     "
    f"{forecast_quality['negative_predictions']}"
)


print(
    f"Forecast Date NULL:         "
    f"{forecast_quality['null_forecast_date']}"
)


print(
    f"Store ID NULL:              "
    f"{forecast_quality['null_store_id']}"
)


print(
    f"Model Name NULL:            "
    f"{forecast_quality['null_model_name']}"
)


print(
    f"Model Version NULL:         "
    f"{forecast_quality['null_model_version']}"
)


print(
    f"Prediction Timestamp NULL:  "
    f"{forecast_quality['null_timestamp']}"
)


print(
    f"Duplicados lógicos:         "
    f"{duplicate_forecasts}"
)


print(
    f"Coberturas incompletas:     "
    f"{incomplete_forecast_groups}"
)


# ============================================================
# 01.01.08 COBERTURA POR FECHA Y VERSIÓN
# ============================================================

print()
print("-" * 80)
print("COBERTURA POR FECHA + MODEL VERSION")
print("-" * 80)


display(
    forecast_coverage_df
)


# ============================================================
# 01.01.09 VALIDACIÓN FINAL
# ============================================================

if quality_errors:

    print()
    print("=" * 80)
    print("ERROR - FORECAST QUALITY GATE FAILED")
    print("=" * 80)

    for error in quality_errors:

        print(
            f" - {error}"
        )

    raise RuntimeError(
        "Forecast Quality Gate failed"
    )


print()
print("=" * 80)
print("VALIDACIÓN FINAL")
print("=" * 80)


print(
    "OK - Sin predicciones NULL"
)


print(
    "OK - Sin predicciones NaN"
)


print(
    "OK - Sin predicciones negativas"
)


print(
    "OK - Claves obligatorias completas"
)


print(
    "OK - Sin duplicados lógicos"
)


print(
    "OK - Cobertura completa por forecast_date + model_version"
)


print(
    f"OK - {EXPECTED_STORES} tiendas esperadas"
)


print()
print(
    "OK - FORECAST QUALITY GATE PASSED"
)

# COMMAND ----------

# MAGIC %md
# MAGIC # 02. MODEL PERFORMANCE MONITORING
# MAGIC
# MAGIC ## Objetivo
# MAGIC
# MAGIC Evaluar el rendimiento real del modelo comparando las predicciones almacenadas con las ventas posteriormente observadas en Gold.
# MAGIC
# MAGIC Una predicción únicamente puede evaluarse cuando:
# MAGIC
# MAGIC `forecast_date <= MAX(fecha disponible en Gold)`
# MAGIC
# MAGIC Para cada forecast evaluable se reconstruirá la venta real diaria por tienda y se calcularán:
# MAGIC
# MAGIC - **Absolute Error**
# MAGIC - **Squared Error**
# MAGIC - **Absolute Percentage Error**
# MAGIC - **MAE**
# MAGIC - **RMSE**
# MAGIC - **WAPE**
# MAGIC
# MAGIC Si todavía no existen forecasts evaluables, el proceso devolverá estado **PENDING** en lugar de generar métricas artificiales.

# COMMAND ----------

# DBTITLE 1,02.01 FORECAST VS ACTUAL
# ============================================================
# 02.01 FORECAST VS ACTUAL
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Comparar cada forecast evaluable contra la venta real
# observada posteriormente en Gold.
#
# Granularidad lógica del forecast:
#
#     forecast_date + store_id + model_version
#
# IMPORTANTE
# ------------------------------------------------------------
#
# Si una fecha ya está disponible en Gold pero una tienda no
# tiene registros de venta ese día, interpretamos:
#
#     actual_net_sales = 0
#
# Esto es coherente con la regularización temporal utilizada
# durante Feature Engineering.
#
# ============================================================


from pyspark.sql import functions as F


# ============================================================
# 02.01.01 CONSTRUCCIÓN DE ACTUALS DIARIOS POR TIENDA
# ============================================================
#
# fact_sales tiene granularidad de línea de ticket.
#
# Se agrega:
#
#     fecha + tienda
#
# para obtener la venta diaria real.
#
# ============================================================

actual_store_day_df = (

    fact_sales_with_date_df

    .groupBy(

        F.col(
            "date"
        ).alias(
            "actual_date"
        ),

        "store_id"

    )

    .agg(

        F.sum(
            "net_amount"
        ).alias(
            "actual_net_sales"
        )

    )

)


# ============================================================
# 02.01.02 SELECCIÓN DE FORECASTS EVALUABLES
# ============================================================
#
# Una predicción puede evaluarse únicamente cuando:
#
#     forecast_date <= max_gold_date
#
# ============================================================

evaluable_forecasts_df = (

    forecast_monitor_df

    .filter(

        F.col(
            "forecast_date"
        )

        <=

        F.lit(
            max_gold_date
        )

    )

)


# ============================================================
# 02.01.03 VALIDACIÓN DE DUPLICADOS EN FORECASTS EVALUABLES
# ============================================================

evaluable_duplicates = (

    evaluable_forecasts_df

    .groupBy(

        "forecast_date",
        "store_id",
        "model_version"

    )

    .count()

    .filter(
        F.col("count") > 1
    )

    .count()

)


if evaluable_duplicates > 0:

    raise RuntimeError(

        "Existen forecasts evaluables duplicados según "
        "forecast_date + store_id + model_version."

    )


# ============================================================
# 02.01.04 FORECAST VS ACTUAL
# ============================================================
#
# LEFT JOIN porque el forecast es la entidad que queremos
# evaluar.
#
# Si una tienda no presenta movimientos de venta en una fecha
# ya disponible en Gold:
#
#     actual_net_sales = 0
#
# ============================================================

forecast_vs_actual_df = (

    evaluable_forecasts_df
    .alias(
        "f"
    )

    .join(

        actual_store_day_df
        .alias(
            "a"
        ),

        (
            F.col(
                "f.forecast_date"
            )
            ==
            F.col(
                "a.actual_date"
            )
        )

        &

        (
            F.col(
                "f.store_id"
            )
            ==
            F.col(
                "a.store_id"
            )
        ),

        "left"

    )

    .select(

        F.col(
            "f.forecast_date"
        ).alias(
            "forecast_date"
        ),

        F.col(
            "f.store_id"
        ).alias(
            "store_id"
        ),

        F.col(
            "f.predicted_net_sales"
        ).cast(
            "double"
        ).alias(
            "predicted_net_sales"
        ),

        F.coalesce(

            F.col(
                "a.actual_net_sales"
            ),

            F.lit(
                0.0
            )

        ).cast(
            "double"
        ).alias(
            "actual_net_sales"
        ),

        F.col(
            "f.model_name"
        ).alias(
            "model_name"
        ),

        F.col(
            "f.model_version"
        ).alias(
            "model_version"
        ),

        F.col(
            "f.prediction_timestamp"
        ).alias(
            "prediction_timestamp"
        )

    )

)


# ============================================================
# 02.01.05 CÁLCULO DE ERRORES
# ============================================================

forecast_vs_actual_df = (

    forecast_vs_actual_df


    # --------------------------------------------------------
    # Error firmado
    #
    # positivo  -> sobrepredicción
    # negativo  -> infrapredicción
    # --------------------------------------------------------

    .withColumn(

        "error",

        F.col(
            "predicted_net_sales"
        )

        -

        F.col(
            "actual_net_sales"
        )

    )


    # --------------------------------------------------------
    # Error absoluto
    # --------------------------------------------------------

    .withColumn(

        "absolute_error",

        F.abs(
            F.col(
                "error"
            )
        )

    )


    # --------------------------------------------------------
    # Error cuadrático
    # --------------------------------------------------------

    .withColumn(

        "squared_error",

        F.pow(

            F.col(
                "error"
            ),

            2

        )

    )


    # --------------------------------------------------------
    # Absolute Percentage Error
    #
    # No se calcula cuando actual = 0 para evitar división
    # entre cero.
    # --------------------------------------------------------

    .withColumn(

        "absolute_percentage_error",

        F.when(

            F.col(
                "actual_net_sales"
            ) != 0,

            (
                F.col(
                    "absolute_error"
                )

                /

                F.col(
                    "actual_net_sales"
                )

            )

            * 100

        )

        .otherwise(
            F.lit(None).cast("double")
        )

    )

)


# ============================================================
# 02.01.06 ESTADO DE EVALUACIÓN
# ============================================================

evaluation_status = (

    forecast_vs_actual_df

    .agg(

        F.count(
            "*"
        ).alias(
            "forecasts"
        ),

        F.countDistinct(
            "forecast_date"
        ).alias(
            "forecast_dates"
        ),

        F.countDistinct(
            "store_id"
        ).alias(
            "stores"
        ),

        F.countDistinct(
            "model_version"
        ).alias(
            "model_versions"
        ),

        F.sum(

            F.when(

                F.col(
                    "actual_net_sales"
                ).isNull(),

                1

            ).otherwise(
                0
            )

        ).alias(
            "null_actuals"
        )

    )

    .first()

)


forecasts_to_evaluate = int(
    evaluation_status[
        "forecasts"
    ]
)


actuals_available = (

    forecasts_to_evaluate

    -

    int(
        evaluation_status[
            "null_actuals"
        ] or 0
    )

)


# ============================================================
# 02.01.07 VALIDACIÓN DE RESULTADOS
# ============================================================

comparison_duplicates = (

    forecast_vs_actual_df

    .groupBy(

        "forecast_date",
        "store_id",
        "model_version"

    )

    .count()

    .filter(
        F.col("count") > 1
    )

    .count()

)


null_predictions = (

    forecast_vs_actual_df

    .filter(

        F.col(
            "predicted_net_sales"
        ).isNull()

    )

    .count()

)


null_actuals = (

    forecast_vs_actual_df

    .filter(

        F.col(
            "actual_net_sales"
        ).isNull()

    )

    .count()

)


# ============================================================
# 02.01.08 RESULTADO
# ============================================================

print()
print("=" * 80)
print("MODEL PERFORMANCE MONITORING")
print("=" * 80)


print(
    f"Última fecha disponible Gold: "
    f"{max_gold_date}"
)


print(
    f"Forecasts evaluables:         "
    f"{forecasts_to_evaluate:,}"
)


print(
    f"Fechas evaluables:            "
    f"{evaluation_status['forecast_dates']}"
)


print(
    f"Tiendas evaluables:           "
    f"{evaluation_status['stores']}"
)


print(
    f"Versiones evaluables:         "
    f"{evaluation_status['model_versions']}"
)


print(
    f"Actuals disponibles:          "
    f"{actuals_available:,}"
)


print(
    f"Actuals NULL:                 "
    f"{null_actuals}"
)


print(
    f"Duplicados comparación:       "
    f"{comparison_duplicates}"
)


# ============================================================
# 02.01.09 CONTROL DEL ESTADO
# ============================================================

if forecasts_to_evaluate == 0:

    print()
    print("-" * 80)
    print("STATUS: PENDING")
    print("-" * 80)

    print(
        "Todavía no existen forecasts cuya fecha "
        "esté disponible en Gold."
    )

    print(
        "No se calculan métricas de performance."
    )


else:

    # --------------------------------------------------------
    # VALIDACIONES ESTRICTAS
    # --------------------------------------------------------

    if comparison_duplicates > 0:

        raise RuntimeError(

            "Se han generado duplicados durante la comparación "
            "Forecast vs Actual."

        )


    if null_predictions > 0:

        raise RuntimeError(

            "Existen predicciones NULL entre los forecasts "
            "evaluables."

        )


    if null_actuals > 0:

        raise RuntimeError(

            "Existen actuals NULL después de construir "
            "Forecast vs Actual."

        )


    print()
    print("=" * 80)
    print("VALIDACIÓN FINAL")
    print("=" * 80)


    print(
        "OK - Todos los forecasts evaluables disponen de actual"
    )


    print(
        "OK - Días sin ventas interpretados como actual = 0"
    )


    print(
        "OK - Sin predicciones NULL"
    )


    print(
        "OK - Sin actuals NULL"
    )


    print(
        "OK - Sin duplicados"
    )


    print(
        "OK - model_version preservada"
    )


    print()
    print(
        "OK - FORECAST VS ACTUAL READY FOR PERFORMANCE METRICS"
    )


# ============================================================
# 02.01.10 INSPECCIÓN
# ============================================================

if forecasts_to_evaluate > 0:

    display(

        forecast_vs_actual_df

        .orderBy(

            "forecast_date",
            "model_version",
            "store_id"

        )

    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 02.02 CÁLCULO DE MÉTRICAS DE PERFORMANCE
# MAGIC
# MAGIC Una vez disponibles los datos reales, se calculan las principales métricas de rendimiento del modelo.
# MAGIC
# MAGIC ### Métricas
# MAGIC
# MAGIC - **MAE (Mean Absolute Error):** error absoluto medio.
# MAGIC - **RMSE (Root Mean Squared Error):** penaliza especialmente errores elevados.
# MAGIC - **WAPE (Weighted Absolute Percentage Error):** relaciona el error absoluto total con las ventas reales totales.
# MAGIC
# MAGIC Las métricas se calculan únicamente sobre forecasts que disponen de un valor real observado.
# MAGIC
# MAGIC Esto permite monitorizar el rendimiento efectivo del modelo en producción sin mezclar predicciones todavía pendientes de evaluación.

# COMMAND ----------

# DBTITLE 1,02.02 CÁLCULO DE MÉTRICAS DE PERFORMANCE
# ============================================================
# 02.02 CÁLCULO DE MÉTRICAS DE PERFORMANCE
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Calcular las métricas de rendimiento real del modelo sobre
# los forecasts que ya disponen de actual observado.
#
# Las métricas se calculan por:
#
#     forecast_date + model_name + model_version
#
# De esta forma diferentes versiones del modelo pueden
# evaluarse de forma independiente.
#
# ============================================================


from pyspark.sql import functions as F


# ============================================================
# 02.02.01 REGISTROS CON ACTUAL DISPONIBLE
# ============================================================

evaluated_forecasts_df = (

    forecast_vs_actual_df

    .filter(
        F.col("actual_net_sales").isNotNull()
    )

)


evaluated_records = (
    evaluated_forecasts_df.count()
)


# ============================================================
# 02.02.02 CONTROL DE DISPONIBILIDAD
# ============================================================

print()
print("=" * 80)
print("PERFORMANCE METRICS")
print("=" * 80)


print(
    f"Forecasts evaluados: "
    f"{evaluated_records:,}"
)


# ============================================================
# 02.02.03 CÁLCULO DE MÉTRICAS
# ============================================================

if evaluated_records == 0:

    performance_metrics_df = None

    print()
    print("-" * 80)
    print("STATUS: PENDING")
    print("-" * 80)

    print(
        "No existen todavía forecasts con actual disponible."
    )

    print(
        "MAE, RMSE, WAPE y Bias permanecen pendientes."
    )


else:

    performance_metrics_df = (

        evaluated_forecasts_df

        .groupBy(

            "forecast_date",
            "model_name",
            "model_version"

        )

        .agg(

            # ------------------------------------------------
            # Número de predicciones evaluadas
            # ------------------------------------------------

            F.count(
                "*"
            ).alias(
                "evaluated_predictions"
            ),


            # ------------------------------------------------
            # MAE
            # ------------------------------------------------

            F.avg(
                "absolute_error"
            ).alias(
                "mae"
            ),


            # ------------------------------------------------
            # RMSE
            # ------------------------------------------------

            F.sqrt(

                F.avg(
                    "squared_error"
                )

            ).alias(
                "rmse"
            ),


            # ------------------------------------------------
            # Error absoluto agregado
            # ------------------------------------------------

            F.sum(
                "absolute_error"
            ).alias(
                "total_absolute_error"
            ),


            # ------------------------------------------------
            # Ventas reales agregadas
            # ------------------------------------------------

            F.sum(
                "actual_net_sales"
            ).alias(
                "actual_net_sales"
            ),


            # ------------------------------------------------
            # Ventas predichas agregadas
            # ------------------------------------------------

            F.sum(
                "predicted_net_sales"
            ).alias(
                "predicted_net_sales"
            )

        )


        # ====================================================
        # WAPE
        # ====================================================
        #
        # WAPE =
        #
        # SUM(|error|)
        # -------------------------- * 100
        # SUM(actual_net_sales)
        #
        # Si las ventas reales agregadas son 0,
        # WAPE no es matemáticamente evaluable.
        #
        # ====================================================

        .withColumn(

            "wape",

            F.when(

                F.col(
                    "actual_net_sales"
                ) != 0,

                (
                    F.col(
                        "total_absolute_error"
                    )

                    /

                    F.col(
                        "actual_net_sales"
                    )

                )

                * 100

            )

            .otherwise(
                F.lit(None).cast("double")
            )

        )


        # ====================================================
        # FORECAST BIAS %
        # ====================================================
        #
        # Bias positivo  -> sobrepredicción
        # Bias negativo  -> infrapredicción
        #
        # ====================================================

        .withColumn(

            "forecast_bias_pct",

            F.when(

                F.col(
                    "actual_net_sales"
                ) != 0,

                (
                    (
                        F.col(
                            "predicted_net_sales"
                        )

                        -

                        F.col(
                            "actual_net_sales"
                        )
                    )

                    /

                    F.col(
                        "actual_net_sales"
                    )

                )

                * 100

            )

            .otherwise(
                F.lit(None).cast("double")
            )

        )


        # ====================================================
        # ORDEN
        # ====================================================

        .orderBy(

            "forecast_date",
            "model_version"

        )

    )


    # ========================================================
    # 02.02.04 VALIDACIONES
    # ========================================================

    metric_rows = (
        performance_metrics_df.count()
    )


    null_mae = (

        performance_metrics_df

        .filter(
            F.col("mae").isNull()
        )

        .count()

    )


    null_rmse = (

        performance_metrics_df

        .filter(
            F.col("rmse").isNull()
        )

        .count()

    )


    invalid_prediction_counts = (

        performance_metrics_df

        .filter(

            F.col(
                "evaluated_predictions"
            ) <= 0

        )

        .count()

    )


    # ========================================================
    # 02.02.05 RESULTADO
    # ========================================================

    print()
    print("-" * 80)
    print("RESULTADO")
    print("-" * 80)


    print(
        f"Grupos evaluados:          "
        f"{metric_rows}"
    )


    print(
        f"MAE NULL:                  "
        f"{null_mae}"
    )


    print(
        f"RMSE NULL:                 "
        f"{null_rmse}"
    )


    print(
        f"Grupos sin predicciones:   "
        f"{invalid_prediction_counts}"
    )


    # ========================================================
    # 02.02.06 VALIDACIÓN FINAL
    # ========================================================

    if null_mae > 0:

        raise RuntimeError(
            "Existen grupos con MAE NULL."
        )


    if null_rmse > 0:

        raise RuntimeError(
            "Existen grupos con RMSE NULL."
        )


    if invalid_prediction_counts > 0:

        raise RuntimeError(

            "Existen grupos de performance "
            "sin predicciones evaluadas."

        )


    print()
    print("=" * 80)
    print("VALIDACIÓN FINAL")
    print("=" * 80)


    print(
        "OK - MAE calculado"
    )


    print(
        "OK - RMSE calculado"
    )


    print(
        "OK - WAPE protegido ante actual total = 0"
    )


    print(
        "OK - Forecast Bias protegido ante actual total = 0"
    )


    print(
        "OK - Métricas separadas por model_version"
    )


    print()
    print(
        "OK - PERFORMANCE METRICS READY"
    )


    # ========================================================
    # 02.02.07 INSPECCIÓN
    # ========================================================

    display(
        performance_metrics_df
    )

# COMMAND ----------

# MAGIC %md
# MAGIC # 03. STORE-LEVEL PERFORMANCE MONITORING
# MAGIC
# MAGIC ## Objetivo
# MAGIC
# MAGIC Analizar el rendimiento del modelo a nivel de tienda.
# MAGIC
# MAGIC Las métricas globales pueden ocultar comportamientos anómalos en establecimientos concretos. Por ello, una vez disponibles los datos reales, se calcularán métricas específicas para cada `store_id`.
# MAGIC
# MAGIC Se monitorizarán:
# MAGIC
# MAGIC - número de forecasts evaluados
# MAGIC - MAE por tienda
# MAGIC - RMSE por tienda
# MAGIC - WAPE por tienda
# MAGIC - Bias del forecast
# MAGIC - ventas reales y predichas
# MAGIC
# MAGIC Este análisis permitirá identificar tiendas donde el modelo presente errores significativamente superiores al comportamiento global.

# COMMAND ----------

# DBTITLE 1,03.01 STORE-LEVEL PERFORMANCE
# ============================================================
# 03.01 STORE-LEVEL PERFORMANCE
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Analizar el rendimiento del modelo a nivel de tienda.
#
# Las métricas se calculan por:
#
#     store_id + model_name + model_version
#
# Esto permite comparar el comportamiento de diferentes
# versiones del modelo por establecimiento.
#
# ============================================================


from pyspark.sql import functions as F


# ============================================================
# 03.01.01 COMPROBACIÓN DE FORECASTS EVALUADOS
# ============================================================

print()
print("=" * 80)
print("STORE-LEVEL PERFORMANCE MONITORING")
print("=" * 80)

print(
    f"Forecasts evaluados: {evaluated_records:,}"
)


# ============================================================
# 03.01.02 CONTROL DE DISPONIBILIDAD
# ============================================================

if evaluated_records == 0:

    store_performance_df = None

    print()
    print("-" * 80)
    print("STATUS: PENDING")
    print("-" * 80)

    print(
        "No existen todavía forecasts con actual disponible."
    )

    print(
        "El análisis de performance por tienda "
        "permanece pendiente."
    )


else:

    # ========================================================
    # 03.01.03 CÁLCULO BASE DE MÉTRICAS POR TIENDA
    # ========================================================

    store_performance_df = (

        evaluated_forecasts_df

        .groupBy(

            "store_id",
            "model_name",
            "model_version"

        )

        .agg(

            # ------------------------------------------------
            # Número de forecasts evaluados
            # ------------------------------------------------

            F.count(
                "*"
            ).alias(
                "evaluated_forecasts"
            ),


            # ------------------------------------------------
            # MAE
            # ------------------------------------------------

            F.avg(
                "absolute_error"
            ).alias(
                "mae"
            ),


            # ------------------------------------------------
            # RMSE
            # ------------------------------------------------

            F.sqrt(

                F.avg(
                    "squared_error"
                )

            ).alias(
                "rmse"
            ),


            # ------------------------------------------------
            # Error absoluto acumulado
            # ------------------------------------------------

            F.sum(
                "absolute_error"
            ).alias(
                "total_absolute_error"
            ),


            # ------------------------------------------------
            # Ventas reales acumuladas
            # ------------------------------------------------

            F.sum(
                "actual_net_sales"
            ).alias(
                "actual_net_sales"
            ),


            # ------------------------------------------------
            # Ventas predichas acumuladas
            # ------------------------------------------------

            F.sum(
                "predicted_net_sales"
            ).alias(
                "predicted_net_sales"
            )

        )

    )


    # ========================================================
    # 03.01.04 WAPE POR TIENDA
    # ========================================================
    #
    # WAPE =
    #
    # SUM(|error|)
    # --------------------- * 100
    # SUM(actual)
    #
    # Si actual_net_sales = 0, WAPE no es evaluable.
    #
    # ========================================================

    store_performance_df = (

        store_performance_df

        .withColumn(

            "wape",

            F.when(

                F.col(
                    "actual_net_sales"
                ) != 0,

                (
                    F.col(
                        "total_absolute_error"
                    )

                    /

                    F.col(
                        "actual_net_sales"
                    )

                )

                * 100

            )

            .otherwise(
                F.lit(None).cast("double")
            )

        )

    )


    # ========================================================
    # 03.01.05 BIAS POR TIENDA
    # ========================================================
    #
    # Bias positivo  -> sobrepredicción
    # Bias negativo  -> infrapredicción
    #
    # ========================================================

    store_performance_df = (

        store_performance_df

        .withColumn(

            "forecast_bias_pct",

            F.when(

                F.col(
                    "actual_net_sales"
                ) != 0,

                (
                    (
                        F.col(
                            "predicted_net_sales"
                        )

                        -

                        F.col(
                            "actual_net_sales"
                        )
                    )

                    /

                    F.col(
                        "actual_net_sales"
                    )

                )

                * 100

            )

            .otherwise(
                F.lit(None).cast("double")
            )

        )

    )


    # ========================================================
    # 03.01.06 PERFORMANCE STATUS
    # ========================================================
    #
    # Thresholds exploratorios actuales:
    #
    # WAPE <= 20  -> GOOD
    # WAPE <= 35  -> WARNING
    # WAPE > 35   -> CRITICAL
    #
    # Si WAPE es NULL porque actual_net_sales = 0:
    #
    # NOT_EVALUABLE
    #
    # ========================================================

    store_performance_df = (

        store_performance_df

        .withColumn(

            "performance_status",

            F.when(

                F.col(
                    "wape"
                ).isNull(),

                "NOT_EVALUABLE"

            )

            .when(

                F.col(
                    "wape"
                ) <= 20,

                "GOOD"

            )

            .when(

                F.col(
                    "wape"
                ) <= 35,

                "WARNING"

            )

            .otherwise(
                "CRITICAL"
            )

        )

    )


    # ========================================================
    # 03.01.07 ORDEN
    # ========================================================

    store_performance_df = (

        store_performance_df

        .orderBy(

            F.desc_nulls_last(
                "wape"
            ),

            "store_id",
            "model_version"

        )

    )


    # ========================================================
    # 03.01.08 VALIDACIONES
    # ========================================================

    store_metric_rows = (
        store_performance_df.count()
    )


    distinct_stores_evaluated = (

        store_performance_df

        .select(
            "store_id"
        )

        .distinct()

        .count()

    )


    duplicate_store_metrics = (

        store_performance_df

        .groupBy(

            "store_id",
            "model_name",
            "model_version"

        )

        .count()

        .filter(
            F.col("count") > 1
        )

        .count()

    )


    null_mae_store = (

        store_performance_df

        .filter(
            F.col("mae").isNull()
        )

        .count()

    )


    null_rmse_store = (

        store_performance_df

        .filter(
            F.col("rmse").isNull()
        )

        .count()

    )


    not_evaluable_stores = (

        store_performance_df

        .filter(
            F.col("performance_status") == "NOT_EVALUABLE"
        )

        .count()

    )


    # ========================================================
    # 03.01.09 RESULTADO
    # ========================================================

    print()
    print("-" * 80)
    print("RESULTADO")
    print("-" * 80)


    print(
        f"Grupos tienda/modelo:       "
        f"{store_metric_rows}"
    )


    print(
        f"Tiendas evaluadas:          "
        f"{distinct_stores_evaluated}"
    )


    print(
        f"Duplicados métricas:        "
        f"{duplicate_store_metrics}"
    )


    print(
        f"MAE NULL:                   "
        f"{null_mae_store}"
    )


    print(
        f"RMSE NULL:                  "
        f"{null_rmse_store}"
    )


    print(
        f"Tiendas NOT_EVALUABLE:      "
        f"{not_evaluable_stores}"
    )


    # ========================================================
    # 03.01.10 VALIDACIÓN FINAL
    # ========================================================

    if duplicate_store_metrics > 0:

        raise RuntimeError(

            "Existen métricas duplicadas por "
            "store_id + model_name + model_version."

        )


    if null_mae_store > 0:

        raise RuntimeError(
            "Existen tiendas con MAE NULL."
        )


    if null_rmse_store > 0:

        raise RuntimeError(
            "Existen tiendas con RMSE NULL."
        )


    print()
    print("=" * 80)
    print("VALIDACIÓN FINAL")
    print("=" * 80)


    print(
        "OK - Métricas calculadas por store_id + model_version"
    )


    print(
        "OK - WAPE protegido ante actual total = 0"
    )


    print(
        "OK - Forecast Bias protegido ante actual total = 0"
    )


    print(
        "OK - NOT_EVALUABLE gestionado correctamente"
    )


    print(
        "OK - Sin duplicados"
    )


    print()
    print(
        "OK - STORE-LEVEL PERFORMANCE READY"
    )


    # ========================================================
    # 03.01.11 INSPECCIÓN
    # ========================================================

    display(
        store_performance_df
    )

# COMMAND ----------

# MAGIC %md
# MAGIC # 04. FEATURE DRIFT MONITORING
# MAGIC
# MAGIC ## Objetivo
# MAGIC
# MAGIC Detectar cambios en la distribución de las principales variables utilizadas por el modelo.
# MAGIC
# MAGIC El Feature Drift puede indicar que los patrones actuales de ventas están cambiando respecto a los datos históricos sobre los que se desarrolló el modelo.
# MAGIC
# MAGIC Se compararán dos ventanas:
# MAGIC
# MAGIC **Reference Window:** histórico anterior a los últimos 30 días.
# MAGIC
# MAGIC **Current Window:** últimos 30 días disponibles.
# MAGIC
# MAGIC Se analizarán principalmente las features relacionadas con el comportamiento reciente de ventas:
# MAGIC
# MAGIC - lag_1
# MAGIC - lag_7
# MAGIC - lag_14
# MAGIC - lag_28
# MAGIC - rolling_mean_7
# MAGIC - rolling_mean_28
# MAGIC - rolling_std_7
# MAGIC - lag1_minus_lag7
# MAGIC - lag1_vs_mean7
# MAGIC
# MAGIC Esta primera implementación utilizará cambios relativos en media y desviación estándar como indicadores exploratorios de drift.
# MAGIC
# MAGIC Estos indicadores no constituyen todavía un test estadístico formal de Data Drift.

# COMMAND ----------

# DBTITLE 1,04.01 PREPARACIÓN DE VENTANAS PARA FEATURE DRIFT
# ============================================================
# 04.01 PREPARACIÓN DE VENTANAS PARA FEATURE DRIFT
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Preparar las ventanas Reference y Current utilizadas para
# analizar Feature Drift.
#
# Current Window:
#     últimos 30 días disponibles en Feature Table.
#
# Reference Window:
#     todo el histórico anterior a Current Window.
#
# Las validaciones utilizan EXPECTED_STORES obtenido
# dinámicamente en 00.01.
#
# ============================================================


from datetime import timedelta
from pyspark.sql import functions as F


# ============================================================
# 04.01.01 CONFIGURACIÓN
# ============================================================

CURRENT_WINDOW_DAYS = 30


# ============================================================
# 04.01.02 RANGO TEMPORAL DE FEATURE TABLE
# ============================================================

feature_date_status = (

    feature_monitor_df

    .agg(

        F.min(
            "sale_date"
        ).alias(
            "min_date"
        ),

        F.max(
            "sale_date"
        ).alias(
            "max_date"
        )

    )

    .first()

)


min_feature_date = (
    feature_date_status[
        "min_date"
    ]
)


max_feature_date = (
    feature_date_status[
        "max_date"
    ]
)


if (
    min_feature_date is None
    or max_feature_date is None
):

    raise RuntimeError(
        "La Feature Table no contiene rango temporal válido."
    )


# ============================================================
# 04.01.03 CURRENT WINDOW
# ============================================================
#
# Últimos 30 días incluyendo max_feature_date.
#
# ============================================================

current_start_date = (

    max_feature_date

    -

    timedelta(
        days=CURRENT_WINDOW_DAYS - 1
    )

)


current_end_date = (
    max_feature_date
)


# ============================================================
# 04.01.04 REFERENCE WINDOW
# ============================================================

reference_start_date = (
    min_feature_date
)


reference_end_date = (

    current_start_date

    -

    timedelta(
        days=1
    )

)


# ============================================================
# 04.01.05 CURRENT DATASET
# ============================================================

current_features_df = (

    feature_monitor_df

    .filter(

        F.col(
            "sale_date"
        )
        >=
        F.lit(
            current_start_date
        )

    )

    .filter(

        F.col(
            "sale_date"
        )
        <=
        F.lit(
            current_end_date
        )

    )

)


# ============================================================
# 04.01.06 REFERENCE DATASET
# ============================================================

reference_features_df = (

    feature_monitor_df

    .filter(

        F.col(
            "sale_date"
        )
        >=
        F.lit(
            reference_start_date
        )

    )

    .filter(

        F.col(
            "sale_date"
        )
        <=
        F.lit(
            reference_end_date
        )

    )

)


# ============================================================
# 04.01.07 MÉTRICAS DE LAS VENTANAS
# ============================================================

current_rows = (
    current_features_df.count()
)


reference_rows = (
    reference_features_df.count()
)


current_stores = (

    current_features_df

    .select(
        "store_id"
    )

    .distinct()

    .count()

)


reference_stores = (

    reference_features_df

    .select(
        "store_id"
    )

    .distinct()

    .count()

)


current_dates = (

    current_features_df

    .select(
        "sale_date"
    )

    .distinct()

    .count()

)


reference_dates = (

    reference_features_df

    .select(
        "sale_date"
    )

    .distinct()

    .count()

)


# ============================================================
# 04.01.08 DUPLICADOS
# ============================================================

current_duplicates = (

    current_features_df

    .groupBy(
        "sale_date",
        "store_id"
    )

    .count()

    .filter(
        F.col("count") > 1
    )

    .count()

)


reference_duplicates = (

    reference_features_df

    .groupBy(
        "sale_date",
        "store_id"
    )

    .count()

    .filter(
        F.col("count") > 1
    )

    .count()

)


# ============================================================
# 04.01.09 COBERTURA ESPERADA DE CURRENT WINDOW
# ============================================================
#
# Como la Feature Table está regularizada:
#
#     30 días × número de tiendas
#
# debe ser exactamente el número de filas de Current Window.
#
# ============================================================

expected_current_rows = (

    CURRENT_WINDOW_DAYS

    *

    EXPECTED_STORES

)


# ============================================================
# 04.01.10 RESULTADO
# ============================================================

print("=" * 80)
print("FEATURE DRIFT - VENTANAS DE ANÁLISIS")
print("=" * 80)


print(
    f"Feature Table:              "
    f"{min_feature_date} -> {max_feature_date}"
)


print()
print("-" * 80)
print("REFERENCE WINDOW")
print("-" * 80)


print(
    f"Periodo:                    "
    f"{reference_start_date} -> {reference_end_date}"
)


print(
    f"Registros:                  "
    f"{reference_rows:,}"
)


print(
    f"Días:                       "
    f"{reference_dates:,}"
)


print(
    f"Tiendas:                    "
    f"{reference_stores} / {EXPECTED_STORES}"
)


print(
    f"Duplicados:                 "
    f"{reference_duplicates}"
)


print()
print("-" * 80)
print("CURRENT WINDOW")
print("-" * 80)


print(
    f"Periodo:                    "
    f"{current_start_date} -> {current_end_date}"
)


print(
    f"Registros:                  "
    f"{current_rows:,}"
)


print(
    f"Registros esperados:        "
    f"{expected_current_rows:,}"
)


print(
    f"Días:                       "
    f"{current_dates}"
)


print(
    f"Tiendas:                    "
    f"{current_stores} / {EXPECTED_STORES}"
)


print(
    f"Duplicados:                 "
    f"{current_duplicates}"
)


# ============================================================
# 04.01.11 QUALITY GATES
# ============================================================

if reference_rows == 0:

    raise RuntimeError(
        "Reference Window no contiene datos."
    )


if current_rows == 0:

    raise RuntimeError(
        "Current Window no contiene datos."
    )


if reference_stores != EXPECTED_STORES:

    raise RuntimeError(

        "Reference Window no contiene la cobertura completa "
        f"de tiendas. Esperadas: {EXPECTED_STORES} | "
        f"Encontradas: {reference_stores}"

    )


if current_stores != EXPECTED_STORES:

    raise RuntimeError(

        "Current Window no contiene la cobertura completa "
        f"de tiendas. Esperadas: {EXPECTED_STORES} | "
        f"Encontradas: {current_stores}"

    )


if current_dates != CURRENT_WINDOW_DAYS:

    raise RuntimeError(

        "Current Window no contiene exactamente "
        f"{CURRENT_WINDOW_DAYS} días. "
        f"Encontrados: {current_dates}"

    )


if current_rows != expected_current_rows:

    raise RuntimeError(

        "Current Window no contiene el número esperado "
        "de registros. "
        f"Esperados: {expected_current_rows:,} | "
        f"Encontrados: {current_rows:,}"

    )


if reference_duplicates > 0:

    raise RuntimeError(

        "Existen duplicados sale_date + store_id "
        "en Reference Window."

    )


if current_duplicates > 0:

    raise RuntimeError(

        "Existen duplicados sale_date + store_id "
        "en Current Window."

    )


# ============================================================
# 04.01.12 VALIDACIÓN FINAL
# ============================================================

print()
print("=" * 80)
print("VALIDACIÓN FINAL")
print("=" * 80)


print(
    "OK - Reference Window disponible"
)


print(
    "OK - Current Window disponible"
)


print(
    f"OK - Current Window = {CURRENT_WINDOW_DAYS} días"
)


print(
    f"OK - Cobertura completa de {EXPECTED_STORES} tiendas"
)


print(
    f"OK - Current Window = {expected_current_rows:,} registros"
)


print(
    "OK - Sin duplicados"
)


print()
print(
    "OK - VENTANAS PREPARADAS PARA FEATURE DRIFT"
)

# COMMAND ----------

# DBTITLE 1,04.02 CÁLCULO DE INDICADORES DE FEATURE DRIFT
# ============================================================
# 04.02 CÁLCULO DE INDICADORES DE FEATURE DRIFT
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Comparar estadísticas descriptivas entre:
#
# - Reference Window
# - Current Window
#
# para detectar cambios relevantes en las features utilizadas
# por Model Version 2.
#
# ============================================================


from pyspark.sql import functions as F


# ============================================================
# 04.02.01 FEATURES A MONITORIZAR
# ============================================================

drift_features = [

    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",

    "rolling_mean_7",
    "rolling_mean_28",
    "rolling_std_7",

    "lag1_minus_lag7",
    "lag1_vs_mean7"

]


# ============================================================
# 04.02.02 VALIDAMOS QUE LAS FEATURES EXISTEN
# ============================================================

missing_drift_features = [

    feature_name

    for feature_name in drift_features

    if feature_name not in feature_monitor_df.columns

]


if missing_drift_features:

    raise RuntimeError(

        "Faltan features necesarias para calcular Feature Drift: "
        f"{missing_drift_features}"

    )


# ============================================================
# 04.02.03 FUNCIÓN PARA CALCULAR ESTADÍSTICAS
# ============================================================

def calculate_feature_stats(df, feature):

    result = (

        df

        .agg(

            F.avg(
                feature
            ).alias(
                "mean"
            ),

            F.stddev_samp(
                feature
            ).alias(
                "std"
            ),

            F.count(
                feature
            ).alias(
                "valid_records"
            )

        )

        .first()

    )


    mean_value = (
        float(result["mean"])
        if result["mean"] is not None
        else None
    )


    std_value = (
        float(result["std"])
        if result["std"] is not None
        else None
    )


    valid_records = int(
        result["valid_records"]
        or 0
    )


    return {

        "mean": mean_value,
        "std": std_value,
        "valid_records": valid_records

    }


# ============================================================
# 04.02.04 CALCULAMOS DRIFT PARA CADA FEATURE
# ============================================================

drift_results = []


for feature in drift_features:

    reference_stats = calculate_feature_stats(
        reference_features_df,
        feature
    )


    current_stats = calculate_feature_stats(
        current_features_df,
        feature
    )


    # --------------------------------------------------------
    # CAMBIO RELATIVO DE LA MEDIA
    # --------------------------------------------------------
    #
    # Si la media de referencia es:
    #
    # - NULL
    # - 0
    #
    # el porcentaje relativo no es evaluable.
    #
    # --------------------------------------------------------

    if (
        reference_stats["mean"] is not None
        and
        current_stats["mean"] is not None
        and
        reference_stats["mean"] != 0
    ):

        mean_change_pct = (

            (
                current_stats["mean"]
                -
                reference_stats["mean"]
            )

            /

            abs(
                reference_stats["mean"]
            )

        ) * 100


    else:

        mean_change_pct = None


    # --------------------------------------------------------
    # CAMBIO RELATIVO DE DESVIACIÓN ESTÁNDAR
    # --------------------------------------------------------

    if (
        reference_stats["std"] is not None
        and
        current_stats["std"] is not None
        and
        reference_stats["std"] != 0
    ):

        std_change_pct = (

            (
                current_stats["std"]
                -
                reference_stats["std"]
            )

            /

            abs(
                reference_stats["std"]
            )

        ) * 100


    else:

        std_change_pct = None


    # --------------------------------------------------------
    # GUARDAMOS RESULTADO
    # --------------------------------------------------------

    drift_results.append(

        (

            feature,

            reference_stats["valid_records"],
            current_stats["valid_records"],

            reference_stats["mean"],
            current_stats["mean"],
            mean_change_pct,

            reference_stats["std"],
            current_stats["std"],
            std_change_pct

        )

    )


# ============================================================
# 04.02.05 DATAFRAME DE RESULTADOS
# ============================================================

drift_schema = [

    "feature",

    "reference_records",
    "current_records",

    "reference_mean",
    "current_mean",
    "mean_change_pct",

    "reference_std",
    "current_std",
    "std_change_pct"

]


feature_drift_df = (

    spark.createDataFrame(

        drift_results,
        drift_schema

    )

)


# ============================================================
# 04.02.06 VALIDACIONES
# ============================================================

drift_feature_count = (
    feature_drift_df.count()
)


null_reference_stats = (

    feature_drift_df

    .filter(

        F.col(
            "reference_mean"
        ).isNull()

        |

        F.col(
            "reference_std"
        ).isNull()

    )

    .count()

)


null_current_stats = (

    feature_drift_df

    .filter(

        F.col(
            "current_mean"
        ).isNull()

        |

        F.col(
            "current_std"
        ).isNull()

    )

    .count()

)


invalid_reference_counts = (

    feature_drift_df

    .filter(
        F.col("reference_records") <= 0
    )

    .count()

)


invalid_current_counts = (

    feature_drift_df

    .filter(
        F.col("current_records") <= 0
    )

    .count()

)


# ============================================================
# 04.02.07 DATAFRAME PARA VISUALIZACIÓN
# ============================================================

feature_drift_display_df = (

    feature_drift_df

    .select(

        "feature",

        "reference_records",
        "current_records",

        F.round(
            "reference_mean",
            2
        ).alias(
            "reference_mean"
        ),

        F.round(
            "current_mean",
            2
        ).alias(
            "current_mean"
        ),

        F.round(
            "mean_change_pct",
            2
        ).alias(
            "mean_change_pct"
        ),

        F.round(
            "reference_std",
            2
        ).alias(
            "reference_std"
        ),

        F.round(
            "current_std",
            2
        ).alias(
            "current_std"
        ),

        F.round(
            "std_change_pct",
            2
        ).alias(
            "std_change_pct"
        )

    )

)


# ============================================================
# 04.02.08 RESULTADO
# ============================================================

print("=" * 80)
print("FEATURE DRIFT MONITORING")
print("=" * 80)


print(
    f"Features monitorizadas:      "
    f"{len(drift_features)}"
)


print(
    f"Reference records:           "
    f"{reference_rows:,}"
)


print(
    f"Current records:             "
    f"{current_rows:,}"
)


print(
    f"Features calculadas:         "
    f"{drift_feature_count}"
)


print(
    f"Stats NULL Reference:        "
    f"{null_reference_stats}"
)


print(
    f"Stats NULL Current:          "
    f"{null_current_stats}"
)


print(
    f"Reference sin registros:     "
    f"{invalid_reference_counts}"
)


print(
    f"Current sin registros:       "
    f"{invalid_current_counts}"
)


# ============================================================
# 04.02.09 QUALITY GATES
# ============================================================

if drift_feature_count != len(drift_features):

    raise RuntimeError(

        "No se han calculado estadísticas para "
        "todas las features de drift."

    )


if null_reference_stats > 0:

    raise RuntimeError(

        "Existen features sin estadísticas válidas "
        "en Reference Window."

    )


if null_current_stats > 0:

    raise RuntimeError(

        "Existen features sin estadísticas válidas "
        "en Current Window."

    )


if invalid_reference_counts > 0:

    raise RuntimeError(

        "Existen features sin registros válidos "
        "en Reference Window."

    )


if invalid_current_counts > 0:

    raise RuntimeError(

        "Existen features sin registros válidos "
        "en Current Window."

    )


# ============================================================
# 04.02.10 VALIDACIÓN FINAL
# ============================================================

print()
print("=" * 80)
print("VALIDACIÓN FINAL")
print("=" * 80)


print(
    "OK - Todas las features disponibles"
)


print(
    "OK - Estadísticas Reference calculadas"
)


print(
    "OK - Estadísticas Current calculadas"
)


print(
    "OK - Sin estadísticas NULL"
)


print(
    "OK - lag1_minus_lag7 incluida"
)


print()
print(
    "OK - FEATURE DRIFT DESCRIPTIVE METRICS READY"
)


# ============================================================
# 04.02.11 INSPECCIÓN
# ============================================================

display(

    feature_drift_display_df

    .orderBy(

        F.desc_nulls_last(

            F.abs(
                F.col(
                    "mean_change_pct"
                )
            )

        )

    )

)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 04.03 POPULATION STABILITY INDEX (PSI)
# MAGIC
# MAGIC Para evaluar cambios en la distribución completa de las features se utiliza **Population Stability Index (PSI)**.
# MAGIC
# MAGIC A diferencia de comparar únicamente medias y desviaciones estándar, PSI analiza cómo cambia la distribución de cada variable entre:
# MAGIC
# MAGIC - **Reference Window:** histórico de referencia.
# MAGIC - **Current Window:** últimos 30 días.
# MAGIC
# MAGIC ### Interpretación operativa inicial
# MAGIC
# MAGIC - `PSI < 0.10` → **STABLE**
# MAGIC - `0.10 <= PSI < 0.25` → **WARNING**
# MAGIC - `PSI >= 0.25` → **DRIFT**
# MAGIC
# MAGIC Estos thresholds se utilizan como criterios iniciales de monitoring y podrán ajustarse posteriormente según el comportamiento observado del modelo y los datos.

# COMMAND ----------

# DBTITLE 1,04.03 POPULATION STABILITY INDEX (PSI)
# ============================================================
# 04.03 POPULATION STABILITY INDEX (PSI)
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Medir cambios en la distribución completa de las features
# entre:
#
# - Reference Window
# - Current Window
#
# Interpretación operativa:
#
# PSI < 0.10          -> STABLE
# 0.10 <= PSI < 0.25  -> WARNING
# PSI >= 0.25         -> DRIFT
#
# ============================================================


import numpy as np

from pyspark.sql import functions as F
from pyspark.ml.feature import Bucketizer


# ============================================================
# 04.03.01 CONFIGURACIÓN
# ============================================================

PSI_BINS = 10

EPSILON = 1e-6

PSI_WARNING_THRESHOLD = 0.10

PSI_DRIFT_THRESHOLD = 0.25


psi_results = []


# ============================================================
# 04.03.02 FUNCIÓN PSI
# ============================================================

def calculate_psi(
    reference_df,
    current_df,
    feature,
    bins=10
):

    # ========================================================
    # A. DATASETS VÁLIDOS
    # ========================================================
    #
    # Eliminamos NULL y NaN antes de calcular la distribución.
    #
    # ========================================================

    reference_valid_df = (

        reference_df

        .select(
            F.col(feature).cast("double").alias(feature)
        )

        .filter(
            F.col(feature).isNotNull()
        )

        .filter(
            ~F.isnan(
                F.col(feature)
            )
        )

    )


    current_valid_df = (

        current_df

        .select(
            F.col(feature).cast("double").alias(feature)
        )

        .filter(
            F.col(feature).isNotNull()
        )

        .filter(
            ~F.isnan(
                F.col(feature)
            )
        )

    )


    # ========================================================
    # B. NÚMERO DE REGISTROS VÁLIDOS
    # ========================================================

    reference_count = (
        reference_valid_df.count()
    )


    current_count = (
        current_valid_df.count()
    )


    # Si alguna ventana no contiene registros válidos,
    # PSI no puede calcularse.

    if (
        reference_count == 0
        or
        current_count == 0
    ):

        return {

            "psi": None,
            "bins_used": 0,
            "reference_records": reference_count,
            "current_records": current_count

        }


    # ========================================================
    # C. QUANTILES DE REFERENCE
    # ========================================================
    #
    # Los bins se construyen utilizando exclusivamente
    # Reference Window.
    #
    # Esto evita que Current influya en los límites.
    #
    # ========================================================

    probabilities = [

        i / bins

        for i in range(
            bins + 1
        )

    ]


    quantiles = (

        reference_valid_df

        .approxQuantile(

            feature,
            probabilities,
            0.001

        )

    )


    # ========================================================
    # D. ELIMINAMOS QUANTILES DUPLICADOS
    # ========================================================
    #
    # Algunas variables pueden tener poca variabilidad.
    #
    # En ese caso varios percentiles pueden generar
    # exactamente el mismo límite.
    #
    # ========================================================

    quantiles = sorted(
        set(quantiles)
    )


    if len(quantiles) < 2:

        return {

            "psi": None,
            "bins_used": 0,
            "reference_records": reference_count,
            "current_records": current_count

        }


    # ========================================================
    # E. CONSTRUCCIÓN DE BINS
    # ========================================================
    #
    # Extendemos los extremos hasta -inf / +inf para cubrir
    # cualquier valor observado en Current.
    #
    # ========================================================

    boundaries = (

        [-float("inf")]

        +

        quantiles[1:-1]

        +

        [float("inf")]

    )


    bins_used = (
        len(boundaries) - 1
    )


    # ========================================================
    # F. BUCKETIZER
    # ========================================================
    #
    # En lugar de ejecutar un count() independiente para cada
    # bin, clasificamos todos los registros de una vez.
    #
    # ========================================================

    bucketizer = Bucketizer(

        splits=boundaries,

        inputCol=feature,

        outputCol="_psi_bin",

        handleInvalid="keep"

    )


    reference_binned_df = (

        bucketizer

        .transform(
            reference_valid_df
        )

    )


    current_binned_df = (

        bucketizer

        .transform(
            current_valid_df
        )

    )


    # ========================================================
    # G. DISTRIBUCIÓN REFERENCE
    # ========================================================

    reference_distribution = {

        int(row["_psi_bin"]):
            int(row["count"])

        for row in (

            reference_binned_df

            .groupBy(
                "_psi_bin"
            )

            .count()

            .collect()

        )

    }


    # ========================================================
    # H. DISTRIBUCIÓN CURRENT
    # ========================================================

    current_distribution = {

        int(row["_psi_bin"]):
            int(row["count"])

        for row in (

            current_binned_df

            .groupBy(
                "_psi_bin"
            )

            .count()

            .collect()

        )

    }


    # ========================================================
    # I. CÁLCULO PSI
    # ========================================================

    psi_value = 0.0


    for bin_index in range(
        bins_used
    ):

        reference_bin_count = (

            reference_distribution
            .get(
                bin_index,
                0
            )

        )


        current_bin_count = (

            current_distribution
            .get(
                bin_index,
                0
            )

        )


        reference_pct = (

            reference_bin_count
            /
            reference_count

        )


        current_pct = (

            current_bin_count
            /
            current_count

        )


        # ----------------------------------------------------
        # Evitamos log(0)
        # ----------------------------------------------------

        reference_pct = max(

            reference_pct,
            EPSILON

        )


        current_pct = max(

            current_pct,
            EPSILON

        )


        psi_value += (

            (
                current_pct
                -
                reference_pct
            )

            *

            np.log(

                current_pct
                /
                reference_pct

            )

        )


    return {

        "psi": float(
            psi_value
        ),

        "bins_used": bins_used,

        "reference_records": reference_count,

        "current_records": current_count

    }


# ============================================================
# 04.03.03 PSI POR FEATURE
# ============================================================

for feature in drift_features:

    psi_result = calculate_psi(

        reference_features_df,
        current_features_df,
        feature,
        PSI_BINS

    )


    psi = (
        psi_result[
            "psi"
        ]
    )


    # ========================================================
    # CLASIFICACIÓN
    # ========================================================

    if psi is None:

        status = (
            "NOT_EVALUABLE"
        )


    elif psi < PSI_WARNING_THRESHOLD:

        status = (
            "STABLE"
        )


    elif psi < PSI_DRIFT_THRESHOLD:

        status = (
            "WARNING"
        )


    else:

        status = (
            "DRIFT"
        )


    psi_results.append(

        (

            feature,

            psi,

            status,

            int(
                psi_result[
                    "bins_used"
                ]
            ),

            int(
                psi_result[
                    "reference_records"
                ]
            ),

            int(
                psi_result[
                    "current_records"
                ]
            )

        )

    )


# ============================================================
# 04.03.04 DATAFRAME DE RESULTADOS
# ============================================================

psi_schema = [

    "feature",

    "psi",

    "drift_status",

    "bins_used",

    "reference_records",

    "current_records"

]


psi_df = (

    spark.createDataFrame(

        psi_results,
        psi_schema

    )

)


# ============================================================
# 04.03.05 DATAFRAME PARA VISUALIZACIÓN
# ============================================================

psi_display_df = (

    psi_df

    .select(

        "feature",

        F.round(

            F.col(
                "psi"
            ),

            4

        ).alias(
            "psi"
        ),

        "drift_status",

        "bins_used",

        "reference_records",

        "current_records"

    )

    .orderBy(

        F.desc_nulls_last(
            "psi"
        )

    )

)


# ============================================================
# 04.03.06 RESUMEN DE ESTADOS
# ============================================================

stable_features = (

    psi_df

    .filter(
        F.col("drift_status") == "STABLE"
    )

    .count()

)


warning_features = (

    psi_df

    .filter(
        F.col("drift_status") == "WARNING"
    )

    .count()

)


drifted_features = (

    psi_df

    .filter(
        F.col("drift_status") == "DRIFT"
    )

    .count()

)


not_evaluable_features = (

    psi_df

    .filter(
        F.col("drift_status") == "NOT_EVALUABLE"
    )

    .count()

)


# ============================================================
# 04.03.07 VALIDACIONES
# ============================================================

psi_feature_count = (
    psi_df.count()
)


invalid_reference_records = (

    psi_df

    .filter(
        F.col("reference_records") <= 0
    )

    .count()

)


invalid_current_records = (

    psi_df

    .filter(
        F.col("current_records") <= 0
    )

    .count()

)


# ============================================================
# 04.03.08 RESULTADO
# ============================================================

print("=" * 80)
print("PSI FEATURE DRIFT MONITORING")
print("=" * 80)


print(
    f"Features analizadas:       "
    f"{psi_feature_count}"
)


print(
    f"STABLE:                    "
    f"{stable_features}"
)


print(
    f"WARNING:                   "
    f"{warning_features}"
)


print(
    f"DRIFT:                     "
    f"{drifted_features}"
)


print(
    f"NOT_EVALUABLE:             "
    f"{not_evaluable_features}"
)


print()
print("-" * 80)


print(
    f"Warning threshold:         "
    f"{PSI_WARNING_THRESHOLD}"
)


print(
    f"Drift threshold:           "
    f"{PSI_DRIFT_THRESHOLD}"
)


# ============================================================
# 04.03.09 QUALITY GATES TÉCNICOS
# ============================================================

if psi_feature_count != len(drift_features):

    raise RuntimeError(

        "No se ha generado resultado PSI "
        "para todas las features."

    )


if invalid_reference_records > 0:

    raise RuntimeError(

        "Existen features sin registros válidos "
        "en Reference Window."

    )


if invalid_current_records > 0:

    raise RuntimeError(

        "Existen features sin registros válidos "
        "en Current Window."

    )


# ============================================================
# 04.03.10 VALIDACIÓN FINAL
# ============================================================

print()
print("=" * 80)
print("VALIDACIÓN FINAL")
print("=" * 80)


print(
    "OK - PSI calculado para todas las features técnicamente evaluables"
)


print(
    "OK - Bins construidos exclusivamente desde Reference Window"
)


print(
    "OK - Valores fuera del rango histórico cubiertos con +/- infinito"
)


print(
    "OK - Divisiones y log(0) protegidos"
)


print(
    "OK - Features con baja variabilidad gestionadas como NOT_EVALUABLE"
)


print(
    "OK - Thresholds PSI aplicados correctamente"
)


print()
print(
    "OK - PSI FEATURE DRIFT READY"
)


# ============================================================
# 04.03.11 INSPECCIÓN
# ============================================================

display(
    psi_display_df
)

# COMMAND ----------

# MAGIC %md
# MAGIC # 05. ML MONITORING QUALITY GATES
# MAGIC
# MAGIC ## Objetivo
# MAGIC
# MAGIC Consolidar las distintas validaciones del sistema de ML Monitoring en un conjunto de Quality Gates.
# MAGIC
# MAGIC Se evaluarán tres dimensiones:
# MAGIC
# MAGIC ### Forecast Quality Gate
# MAGIC
# MAGIC Comprueba la calidad técnica de las predicciones:
# MAGIC
# MAGIC - NULLs
# MAGIC - valores negativos
# MAGIC - duplicados
# MAGIC - cobertura de tiendas
# MAGIC - trazabilidad del modelo
# MAGIC
# MAGIC ### Feature Drift Gate
# MAGIC
# MAGIC Evalúa cambios en la distribución de las features mediante PSI:
# MAGIC
# MAGIC - PSI < 0.10 → STABLE
# MAGIC - 0.10 <= PSI < 0.25 → WARNING
# MAGIC - PSI >= 0.25 → DRIFT
# MAGIC
# MAGIC ### Model Performance Gate
# MAGIC
# MAGIC Evalúa MAE, RMSE y WAPE cuando existen datos reales disponibles.
# MAGIC
# MAGIC Mientras los forecasts todavía no puedan compararse con datos reales, este control permanecerá en estado `PENDING`.
# MAGIC
# MAGIC ## Estados
# MAGIC
# MAGIC El resultado global podrá ser:
# MAGIC
# MAGIC - **HEALTHY**
# MAGIC - **WARNING**
# MAGIC - **CRITICAL**
# MAGIC - **PENDING_PERFORMANCE**
# MAGIC
# MAGIC De esta forma, la ausencia temporal de actuals no se interpreta erróneamente como un fallo del modelo.

# COMMAND ----------

# DBTITLE 1,05.01 ML MONITORING QUALITY GATES
# ============================================================
# 05.01 ML MONITORING QUALITY GATES
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Consolidar los distintos controles de ML Monitoring en
# estados funcionales.
#
# IMPORTANTE
# ------------------------------------------------------------
#
# Un estado funcional:
#
#     WARNING
#     CRITICAL
#
# NO representa necesariamente un error técnico.
#
# El objetivo es:
#
# - detectar la condición
# - persistirla
# - visualizarla
# - generar alertas posteriormente
#
# pero permitir que el proceso técnico de Monitoring termine
# correctamente siempre que no exista una excepción real.
#
# ============================================================


# ============================================================
# 05.01.01 FORECAST QUALITY GATE
# ============================================================

if len(quality_errors) == 0:

    forecast_quality_status = (
        "PASSED"
    )

else:

    forecast_quality_status = (
        "FAILED"
    )


# ============================================================
# 05.01.02 FEATURE DRIFT GATE
# ============================================================
#
# Prioridad:
#
# DRIFT          -> CRITICAL
# WARNING        -> WARNING
# NOT_EVALUABLE  -> WARNING
# resto          -> STABLE
#
# NOT_EVALUABLE se trata como WARNING porque no implica
# necesariamente drift, pero sí requiere revisión.
#
# ============================================================

if drifted_features > 0:

    feature_drift_status = (
        "CRITICAL"
    )


elif warning_features > 0:

    feature_drift_status = (
        "WARNING"
    )


elif not_evaluable_features > 0:

    feature_drift_status = (
        "WARNING"
    )


else:

    feature_drift_status = (
        "STABLE"
    )


# ============================================================
# 05.01.03 MODEL PERFORMANCE GATE
# ============================================================
#
# Todavía no utilizamos thresholds de MAE/RMSE/WAPE para
# clasificar degradación.
#
# Motivo:
#
# necesitamos acumular más forecasts evaluados en producción
# antes de establecer un baseline operativo estable.
#
# ============================================================

if evaluated_records == 0:

    model_performance_status = (
        "PENDING"
    )

else:

    model_performance_status = (
        "AVAILABLE"
    )


# ============================================================
# 05.01.04 ESTADO GLOBAL
# ============================================================
#
# Prioridad:
#
# 1. Forecast Quality FAILED
# 2. Feature Drift CRITICAL
# 3. Feature Drift WARNING
# 4. Performance PENDING
# 5. HEALTHY
#
# ============================================================

if forecast_quality_status == "FAILED":

    monitoring_status = (
        "CRITICAL"
    )


elif feature_drift_status == "CRITICAL":

    monitoring_status = (
        "CRITICAL"
    )


elif feature_drift_status == "WARNING":

    monitoring_status = (
        "WARNING"
    )


elif model_performance_status == "PENDING":

    monitoring_status = (
        "PENDING_PERFORMANCE"
    )


else:

    monitoring_status = (
        "HEALTHY"
    )


# ============================================================
# 05.01.05 VALIDACIÓN DE ESTADOS
# ============================================================

valid_forecast_quality_statuses = [
    "PASSED",
    "FAILED"
]


valid_feature_drift_statuses = [
    "STABLE",
    "WARNING",
    "CRITICAL"
]


valid_model_performance_statuses = [
    "PENDING",
    "AVAILABLE"
]


valid_monitoring_statuses = [
    "HEALTHY",
    "WARNING",
    "CRITICAL",
    "PENDING_PERFORMANCE"
]


if (
    forecast_quality_status
    not in valid_forecast_quality_statuses
):

    raise RuntimeError(
        "Estado de Forecast Quality no válido."
    )


if (
    feature_drift_status
    not in valid_feature_drift_statuses
):

    raise RuntimeError(
        "Estado de Feature Drift no válido."
    )


if (
    model_performance_status
    not in valid_model_performance_statuses
):

    raise RuntimeError(
        "Estado de Model Performance no válido."
    )


if (
    monitoring_status
    not in valid_monitoring_statuses
):

    raise RuntimeError(
        "Estado global de Monitoring no válido."
    )


# ============================================================
# 05.01.06 RESUMEN
# ============================================================

print()
print("=" * 80)
print("ML MONITORING - QUALITY GATES")
print("=" * 80)


print()
print(
    f"Forecast Quality:          "
    f"{forecast_quality_status}"
)


print(
    f"Feature Drift:             "
    f"{feature_drift_status}"
)


print(
    f"Model Performance:         "
    f"{model_performance_status}"
)


print()
print("-" * 80)
print("FEATURE DRIFT DETAIL")
print("-" * 80)


print(
    f"Stable features:           "
    f"{stable_features}"
)


print(
    f"Warning features:          "
    f"{warning_features}"
)


print(
    f"Drifted features:          "
    f"{drifted_features}"
)


print(
    f"Not evaluable features:    "
    f"{not_evaluable_features}"
)


print()
print("=" * 80)

print(
    f"GLOBAL STATUS: "
    f"{monitoring_status}"
)

print("=" * 80)


# ============================================================
# 05.01.07 INTERPRETACIÓN
# ============================================================

if monitoring_status == "HEALTHY":

    print()

    print(
        "OK - El sistema de ML cumple los controles "
        "funcionales definidos."
    )


elif monitoring_status == "PENDING_PERFORMANCE":

    print()

    print(
        "OK - Forecast Quality y Feature Drift "
        "cumplen los controles."
    )

    print(
        "INFO - Model Performance permanece pendiente "
        "hasta disponer de actuals."
    )


elif monitoring_status == "WARNING":

    print()

    print(
        "WARNING - Se han detectado señales de cambio "
        "que requieren seguimiento."
    )

    print(
        "INFO - El estado WARNING es funcional y no "
        "provoca un fallo técnico del Job."
    )


elif monitoring_status == "CRITICAL":

    print()

    print(
        "CRITICAL - El sistema ha detectado una condición "
        "funcional crítica de Monitoring."
    )

    print(
        "INFO - El estado CRITICAL será persistido y podrá "
        "generar alertas, pero no provoca por sí mismo "
        "un fallo técnico del Lakeflow Job."
    )


# ============================================================
# 05.01.08 CIERRE
# ============================================================

print()
print("-" * 80)

print(
    "OK - ML Monitoring Quality Gates evaluados correctamente"
)


# COMMAND ----------

# MAGIC %md
# MAGIC # 06. PERSISTENCIA DE MÉTRICAS DE MONITORING
# MAGIC
# MAGIC ## Objetivo
# MAGIC
# MAGIC Persistir los resultados obtenidos durante cada ejecución del proceso de ML Monitoring.
# MAGIC
# MAGIC Hasta este punto, las métricas calculadas existen únicamente durante la ejecución del notebook. Para disponer de observabilidad histórica es necesario almacenarlas de forma persistente.
# MAGIC
# MAGIC Se crearán dos tablas Delta:
# MAGIC
# MAGIC ### `ml_monitoring_runs`
# MAGIC
# MAGIC Almacena el estado general de cada ejecución:
# MAGIC
# MAGIC - timestamp de ejecución
# MAGIC - última fecha disponible en Gold
# MAGIC - número de forecasts
# MAGIC - forecasts evaluados
# MAGIC - Forecast Quality Status
# MAGIC - Feature Drift Status
# MAGIC - Model Performance Status
# MAGIC - Global Monitoring Status
# MAGIC - número de features estables, en warning y con drift
# MAGIC
# MAGIC ### `feature_drift_monitoring`
# MAGIC
# MAGIC Almacena el detalle del PSI calculado para cada feature:
# MAGIC
# MAGIC - timestamp de ejecución
# MAGIC - ventanas Reference y Current
# MAGIC - feature
# MAGIC - PSI
# MAGIC - Drift Status
# MAGIC
# MAGIC Esto permitirá construir posteriormente históricos, dashboards y alertas sobre la evolución del modelo.

# COMMAND ----------

# DBTITLE 1,06.01 PREPARACIÓN DE DATASETS DE MONITORING
# ============================================================
# 06.01 PREPARACIÓN DE DATASETS DE MONITORING
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Preparar los datasets que posteriormente serán persistidos
# como histórico de ML Monitoring.
#
# Se generan dos datasets:
#
# 1. monitoring_run_df
#
#    Resumen global del estado del sistema.
#
# 2. feature_drift_history_df
#
#    Detalle del PSI por feature para la ventana evaluada.
#
# ============================================================


from pyspark.sql import functions as F


# ============================================================
# 06.01.01 CONFIGURACIÓN
# ============================================================

MONITORING_RUNS_TABLE = (
    f"{CATALOG}.5_ml.ml_monitoring_runs"
)


FEATURE_DRIFT_MONITORING_TABLE = (
    f"{CATALOG}.5_ml.feature_drift_monitoring"
)


# ============================================================
# 06.01.02 VALIDACIONES PREVIAS
# ============================================================

required_objects = [

    "max_gold_date",

    "forecast_quality",
    "evaluated_records",

    "forecast_quality_status",
    "feature_drift_status",
    "model_performance_status",
    "monitoring_status",

    "stable_features",
    "warning_features",
    "drifted_features",
    "not_evaluable_features",

    "psi_df",

    "reference_start_date",
    "reference_end_date",
    "current_start_date",
    "current_end_date"

]


missing_objects = [

    object_name

    for object_name in required_objects

    if object_name not in globals()

]


if missing_objects:

    raise RuntimeError(

        "Faltan objetos necesarios para preparar "
        f"los datasets de Monitoring: {missing_objects}"

    )


# ============================================================
# 06.01.03 TIMESTAMP DE EJECUCIÓN
# ============================================================

monitoring_timestamp = (

    spark.sql(
        "SELECT current_timestamp() AS ts"
    )

    .first()[
        "ts"
    ]

)


# ============================================================
# 06.01.04 RESUMEN GLOBAL DE LA EJECUCIÓN
# ============================================================
#
# Una fila representa el estado lógico del sistema de
# Monitoring para la ejecución actual.
#
# ============================================================

monitoring_run_data = [

    (

        monitoring_timestamp,

        max_gold_date,

        int(
            forecast_quality[
                "records"
            ]
        ),

        int(
            evaluated_records
        ),

        forecast_quality_status,

        feature_drift_status,

        model_performance_status,

        monitoring_status,

        int(
            stable_features
        ),

        int(
            warning_features
        ),

        int(
            drifted_features
        ),

        int(
            not_evaluable_features
        )

    )

]


monitoring_run_columns = [

    "monitoring_timestamp",

    "max_gold_date",

    "forecast_records",

    "evaluated_forecasts",

    "forecast_quality_status",

    "feature_drift_status",

    "model_performance_status",

    "global_status",

    "stable_features",

    "warning_features",

    "drifted_features",

    "not_evaluable_features"

]


monitoring_run_df = (

    spark.createDataFrame(

        monitoring_run_data,
        monitoring_run_columns

    )

)


# ============================================================
# 06.01.05 DETALLE DEL FEATURE DRIFT
# ============================================================
#
# psi_df contiene:
#
# - feature
# - psi
# - drift_status
# - bins_used
# - reference_records
# - current_records
#
# Añadimos además la información temporal de las ventanas.
#
# ============================================================

feature_drift_history_df = (

    psi_df


    # --------------------------------------------------------
    # Timestamp de ejecución
    # --------------------------------------------------------

    .withColumn(

        "monitoring_timestamp",

        F.lit(
            monitoring_timestamp
        )

    )


    # --------------------------------------------------------
    # Reference Window
    # --------------------------------------------------------

    .withColumn(

        "reference_start_date",

        F.lit(
            reference_start_date
        )

    )

    .withColumn(

        "reference_end_date",

        F.lit(
            reference_end_date
        )

    )


    # --------------------------------------------------------
    # Current Window
    # --------------------------------------------------------

    .withColumn(

        "current_start_date",

        F.lit(
            current_start_date
        )

    )

    .withColumn(

        "current_end_date",

        F.lit(
            current_end_date
        )

    )


    # --------------------------------------------------------
    # Selección final
    # --------------------------------------------------------

    .select(

        "monitoring_timestamp",

        "reference_start_date",
        "reference_end_date",

        "current_start_date",
        "current_end_date",

        "feature",

        "psi",
        "drift_status",

        "bins_used",

        "reference_records",
        "current_records"

    )

)


# ============================================================
# 06.01.06 VALIDACIÓN DEL DATASET GLOBAL
# ============================================================

monitoring_run_rows = (
    monitoring_run_df.count()
)


if monitoring_run_rows != 1:

    raise RuntimeError(

        "monitoring_run_df debe contener exactamente "
        f"1 registro. Actual: {monitoring_run_rows}"

    )


# ============================================================
# 06.01.07 VALIDACIÓN DEL DATASET DE DRIFT
# ============================================================

feature_drift_rows = (
    feature_drift_history_df.count()
)


expected_drift_rows = (
    len(
        drift_features
    )
)


if feature_drift_rows != expected_drift_rows:

    raise RuntimeError(

        "El histórico de Feature Drift no contiene "
        "un registro por feature. "
        f"Esperados: {expected_drift_rows} | "
        f"Actuales: {feature_drift_rows}"

    )


# ============================================================
# 06.01.08 DUPLICADOS EN EL DATASET DE DRIFT
# ============================================================

feature_drift_duplicates = (

    feature_drift_history_df

    .groupBy(

        "current_end_date",
        "feature"

    )

    .count()

    .filter(
        F.col("count") > 1
    )

    .count()

)


if feature_drift_duplicates > 0:

    raise RuntimeError(

        "Existen duplicados en feature_drift_history_df "
        "por current_end_date + feature."

    )


# ============================================================
# 06.01.09 VALIDACIÓN DE ESTADOS PSI
# ============================================================

invalid_drift_status = (

    feature_drift_history_df

    .filter(

        ~F.col(
            "drift_status"
        ).isin(

            "STABLE",
            "WARNING",
            "DRIFT",
            "NOT_EVALUABLE"

        )

    )

    .count()

)


if invalid_drift_status > 0:

    raise RuntimeError(

        "Se han detectado estados PSI no válidos."

    )


# ============================================================
# 06.01.10 VALIDACIÓN DE CONSISTENCIA DEL RESUMEN PSI
# ============================================================

psi_summary_total = (

    stable_features
    +
    warning_features
    +
    drifted_features
    +
    not_evaluable_features

)


if psi_summary_total != expected_drift_rows:

    raise RuntimeError(

        "El resumen de estados PSI no coincide con "
        "el número de features monitorizadas. "
        f"Resumen: {psi_summary_total} | "
        f"Features: {expected_drift_rows}"

    )


# ============================================================
# 06.01.11 RESULTADO
# ============================================================

print()
print("=" * 80)
print("DATASETS DE MONITORING PREPARADOS")
print("=" * 80)


print(
    f"Monitoring Run:             "
    f"{monitoring_run_rows} registro"
)


print(
    f"Feature Drift:              "
    f"{feature_drift_rows} registros"
)


print(
    f"Global Status:              "
    f"{monitoring_status}"
)


print(
    f"Monitoring Timestamp:       "
    f"{monitoring_timestamp}"
)


print()
print("-" * 80)
print("FEATURE DRIFT SUMMARY")
print("-" * 80)


print(
    f"STABLE:                     "
    f"{stable_features}"
)


print(
    f"WARNING:                    "
    f"{warning_features}"
)


print(
    f"DRIFT:                      "
    f"{drifted_features}"
)


print(
    f"NOT_EVALUABLE:              "
    f"{not_evaluable_features}"
)


print()
print("-" * 80)
print("VENTANAS")
print("-" * 80)


print(
    f"Reference:                  "
    f"{reference_start_date} -> {reference_end_date}"
)


print(
    f"Current:                    "
    f"{current_start_date} -> {current_end_date}"
)


# ============================================================
# 06.01.12 VALIDACIÓN FINAL
# ============================================================

print()
print("=" * 80)
print("VALIDACIÓN FINAL")
print("=" * 80)


print(
    "OK - Dataset global preparado"
)


print(
    "OK - Dataset de Feature Drift preparado"
)


print(
    "OK - Un registro por feature"
)


print(
    "OK - Sin duplicados"
)


print(
    "OK - Estados PSI válidos"
)


print(
    "OK - Resumen PSI consistente"
)


print(
    "OK - Ventanas temporales preservadas"
)


print()
print(
    "OK - DATASETS READY FOR DELTA PERSISTENCE"
)


# ============================================================
# 06.01.13 INSPECCIÓN
# ============================================================

display(
    monitoring_run_df
)


display(

    feature_drift_history_df

    .orderBy(
        F.desc_nulls_last(
            "psi"
        )
    )

)

# COMMAND ----------

# DBTITLE 1,06.02 PERSISTENCIA DEL HISTÓRICO DE ML MONITORING
# ============================================================
# 06.02 PERSISTENCIA DEL HISTÓRICO DE ML MONITORING
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Persistir de forma idempotente:
#
# 1. Estado global de cada ejecución lógica de Monitoring.
# 2. Resultado PSI de cada feature y ventana temporal.
#
# IMPORTANTE
# ------------------------------------------------------------
#
# Las tablas pueden contener histórico generado con versiones
# anteriores del notebook.
#
# Por tanto:
#
# - preservamos el histórico existente
# - evolucionamos el schema cuando sea necesario
# - utilizamos claves lógicas completas
# - validamos específicamente la ejecución actual
#
# ============================================================


from delta.tables import DeltaTable
from pyspark.sql import functions as F


# ============================================================
# 06.02.01 FUNCIÓN PARA EVOLUCIÓN DE SCHEMA
# ============================================================

def add_missing_columns(
    table_name,
    source_df
):

    target_df = spark.table(
        table_name
    )

    target_columns = set(
        target_df.columns
    )


    missing_fields = [

        field

        for field in source_df.schema.fields

        if field.name not in target_columns

    ]


    if missing_fields:

        print()
        print(
            f"INFO - Evolución de schema requerida: "
            f"{table_name}"
        )


        for field in missing_fields:

            column_name = (
                field.name
            )

            column_type = (
                field.dataType.simpleString()
            )


            spark.sql(
                f"""
                ALTER TABLE {table_name}
                ADD COLUMNS (
                    `{column_name}` {column_type}
                )
                """
            )


            print(
                f"OK - Columna añadida: "
                f"{column_name} ({column_type})"
            )

    else:

        print(
            f"OK - Schema ya compatible: "
            f"{table_name}"
        )


# ============================================================
# 06.02.02 PERSISTENCIA DEL RESUMEN GLOBAL
# ============================================================
#
# CLAVE LÓGICA
# ------------------------------------------------------------
#
# max_gold_date
#
# Si Monitoring vuelve a ejecutarse sobre el mismo estado de
# Gold, actualizamos ese registro.
#
# ============================================================

if not spark.catalog.tableExists(
    MONITORING_RUNS_TABLE
):

    (
        monitoring_run_df

        .write

        .format("delta")

        .mode("overwrite")

        .saveAsTable(
            MONITORING_RUNS_TABLE
        )
    )


    print(
        f"OK - Tabla creada: "
        f"{MONITORING_RUNS_TABLE}"
    )


else:

    add_missing_columns(
        MONITORING_RUNS_TABLE,
        monitoring_run_df
    )


    monitoring_delta = (

        DeltaTable.forName(
            spark,
            MONITORING_RUNS_TABLE
        )

    )


    (
        monitoring_delta

        .alias("target")

        .merge(

            monitoring_run_df.alias(
                "source"
            ),

            """
            target.max_gold_date =
            source.max_gold_date
            """

        )

        .whenMatchedUpdateAll()

        .whenNotMatchedInsertAll()

        .execute()
    )


    print(
        f"OK - Tabla actualizada: "
        f"{MONITORING_RUNS_TABLE}"
    )


# ============================================================
# 06.02.03 PERSISTENCIA DEL FEATURE DRIFT
# ============================================================
#
# CLAVE LÓGICA
# ------------------------------------------------------------
#
# La evaluación PSI queda identificada por:
#
# reference_start_date
# reference_end_date
# current_start_date
# current_end_date
# feature
#
# Esto identifica inequívocamente:
#
#     una feature
#     +
#     una comparación temporal concreta
#
# ============================================================

if not spark.catalog.tableExists(
    FEATURE_DRIFT_MONITORING_TABLE
):

    (
        feature_drift_history_df

        .write

        .format("delta")

        .mode("overwrite")

        .saveAsTable(
            FEATURE_DRIFT_MONITORING_TABLE
        )
    )


    print(
        f"OK - Tabla creada: "
        f"{FEATURE_DRIFT_MONITORING_TABLE}"
    )


else:

    add_missing_columns(
        FEATURE_DRIFT_MONITORING_TABLE,
        feature_drift_history_df
    )


    drift_delta = (

        DeltaTable.forName(
            spark,
            FEATURE_DRIFT_MONITORING_TABLE
        )

    )


    (
        drift_delta

        .alias("target")

        .merge(

            feature_drift_history_df.alias(
                "source"
            ),

            """
            target.reference_start_date =
                source.reference_start_date

            AND target.reference_end_date =
                source.reference_end_date

            AND target.current_start_date =
                source.current_start_date

            AND target.current_end_date =
                source.current_end_date

            AND target.feature =
                source.feature
            """

        )

        .whenMatchedUpdateAll()

        .whenNotMatchedInsertAll()

        .execute()
    )


    print(
        f"OK - Tabla actualizada: "
        f"{FEATURE_DRIFT_MONITORING_TABLE}"
    )


# ============================================================
# 06.02.04 RECARGA DESDE DELTA
# ============================================================

monitoring_saved_df = (

    spark.table(
        MONITORING_RUNS_TABLE
    )

)


drift_saved_df = (

    spark.table(
        FEATURE_DRIFT_MONITORING_TABLE
    )

)


# ============================================================
# 06.02.05 DUPLICADOS DEL HISTÓRICO GLOBAL
# ============================================================

monitoring_duplicates = (

    monitoring_saved_df

    .groupBy(
        "max_gold_date"
    )

    .count()

    .filter(
        F.col("count") > 1
    )

    .count()

)


# ============================================================
# 06.02.06 DUPLICADOS DEL NUEVO FORMATO DE FEATURE DRIFT
# ============================================================
#
# Los registros legacy pueden tener NULL en
# reference_start_date.
#
# Por eso validamos duplicados del nuevo formato únicamente
# sobre registros que disponen de las ventanas completas.
#
# ============================================================

new_format_drift_df = (

    drift_saved_df

    .filter(
        F.col(
            "reference_start_date"
        ).isNotNull()
    )

)


drift_duplicates = (

    new_format_drift_df

    .groupBy(

        "reference_start_date",
        "reference_end_date",

        "current_start_date",
        "current_end_date",

        "feature"

    )

    .count()

    .filter(
        F.col("count") > 1
    )

    .count()

)


# ============================================================
# 06.02.07 VALIDACIÓN DEL MONITORING RUN ACTUAL
# ============================================================

current_monitoring_df = (

    monitoring_saved_df

    .filter(
        F.col("max_gold_date")
        ==
        F.lit(max_gold_date)
    )

)


current_monitoring_records = (
    current_monitoring_df.count()
)


# ============================================================
# 06.02.08 VALIDACIÓN DEL FEATURE DRIFT ACTUAL
# ============================================================
#
# Aquí utilizamos EXACTAMENTE la misma ventana que hemos
# calculado durante esta ejecución.
#
# No contamos simplemente current_end_date porque podrían
# existir registros históricos asociados a otra ventana.
#
# ============================================================

current_drift_saved_df = (

    drift_saved_df

    .filter(
        F.col("reference_start_date")
        ==
        F.lit(reference_start_date)
    )

    .filter(
        F.col("reference_end_date")
        ==
        F.lit(reference_end_date)
    )

    .filter(
        F.col("current_start_date")
        ==
        F.lit(current_start_date)
    )

    .filter(
        F.col("current_end_date")
        ==
        F.lit(current_end_date)
    )

)


current_drift_records = (
    current_drift_saved_df.count()
)


current_drift_features = (

    current_drift_saved_df

    .select(
        "feature"
    )

    .distinct()

    .count()

)


expected_current_drift_records = (
    len(
        drift_features
    )
)


# ============================================================
# 06.02.09 VALIDACIÓN DE GLOBAL STATUS
# ============================================================

current_status_row = (

    current_monitoring_df

    .select(
        "global_status"
    )

    .first()

)


persisted_global_status = (

    current_status_row[
        "global_status"
    ]

    if current_status_row is not None

    else None

)


# ============================================================
# 06.02.10 VALIDACIÓN DE ESTADOS PSI ACTUALES
# ============================================================

invalid_current_drift_status = (

    current_drift_saved_df

    .filter(

        ~F.col(
            "drift_status"
        ).isin(

            "STABLE",
            "WARNING",
            "DRIFT",
            "NOT_EVALUABLE"

        )

    )

    .count()

)


# ============================================================
# 06.02.11 VALIDACIÓN DE SCHEMA
# ============================================================

monitoring_required_columns = set(
    monitoring_run_df.columns
)


monitoring_persisted_columns = set(
    monitoring_saved_df.columns
)


missing_monitoring_columns = (

    monitoring_required_columns
    -
    monitoring_persisted_columns

)


drift_required_columns = set(
    feature_drift_history_df.columns
)


drift_persisted_columns = set(
    drift_saved_df.columns
)


missing_drift_columns = (

    drift_required_columns
    -
    drift_persisted_columns

)


# ============================================================
# 06.02.12 RESULTADO
# ============================================================

monitoring_total_records = (
    monitoring_saved_df.count()
)


drift_total_records = (
    drift_saved_df.count()
)


print()
print("=" * 80)
print("VALIDACIÓN HISTÓRICO ML MONITORING")
print("=" * 80)


print(
    f"Monitoring Runs históricos:     "
    f"{monitoring_total_records:,}"
)


print(
    f"Feature Drift históricos:       "
    f"{drift_total_records:,}"
)


print()
print("-" * 80)
print("EJECUCIÓN ACTUAL")
print("-" * 80)


print(
    f"Max Gold Date:                   "
    f"{max_gold_date}"
)


print(
    f"Monitoring Run actual:           "
    f"{current_monitoring_records}"
)


print(
    f"Reference Window:                "
    f"{reference_start_date} -> "
    f"{reference_end_date}"
)


print(
    f"Current Window:                  "
    f"{current_start_date} -> "
    f"{current_end_date}"
)


print(
    f"Feature Drift actual:            "
    f"{current_drift_records} / "
    f"{expected_current_drift_records}"
)


print(
    f"Features distintas actuales:     "
    f"{current_drift_features}"
)


print(
    f"Global Status calculado:         "
    f"{monitoring_status}"
)


print(
    f"Global Status persistido:        "
    f"{persisted_global_status}"
)


print()
print("-" * 80)
print("CALIDAD DEL HISTÓRICO")
print("-" * 80)


print(
    f"Duplicados Monitoring Runs:      "
    f"{monitoring_duplicates}"
)


print(
    f"Duplicados Drift nuevo formato:  "
    f"{drift_duplicates}"
)


print(
    f"Estados Drift inválidos actuales:"
    f" {invalid_current_drift_status}"
)


print(
    f"Columnas Monitoring faltantes:   "
    f"{len(missing_monitoring_columns)}"
)


print(
    f"Columnas Drift faltantes:        "
    f"{len(missing_drift_columns)}"
)


# ============================================================
# 06.02.13 QUALITY GATES
# ============================================================

if monitoring_duplicates > 0:

    raise RuntimeError(

        "Existen duplicados por max_gold_date "
        "en ml_monitoring_runs."

    )


if drift_duplicates > 0:

    raise RuntimeError(

        "Existen duplicados en el histórico "
        "de Feature Drift del nuevo formato."

    )


if current_monitoring_records != 1:

    raise RuntimeError(

        "La ejecución actual debe tener exactamente "
        "un registro en ml_monitoring_runs."

    )


if (
    current_drift_records
    !=
    expected_current_drift_records
):

    raise RuntimeError(

        "La ventana actual no contiene exactamente "
        "un registro PSI por feature."

    )


if (
    current_drift_features
    !=
    expected_current_drift_records
):

    raise RuntimeError(

        "La ventana actual no contiene exactamente "
        "una fila por feature."

    )


if invalid_current_drift_status > 0:

    raise RuntimeError(

        "Existen estados PSI inválidos en "
        "la ejecución actual."

    )


if (
    persisted_global_status
    !=
    monitoring_status
):

    raise RuntimeError(

        "El Global Status persistido no coincide "
        "con el calculado."

    )


if missing_monitoring_columns:

    raise RuntimeError(

        "Faltan columnas en ml_monitoring_runs: "
        f"{sorted(missing_monitoring_columns)}"

    )


if missing_drift_columns:

    raise RuntimeError(

        "Faltan columnas en feature_drift_monitoring: "
        f"{sorted(missing_drift_columns)}"

    )


# ============================================================
# 06.02.14 VALIDACIÓN FINAL
# ============================================================

print()
print("=" * 80)
print("VALIDACIÓN FINAL")
print("=" * 80)


print(
    "OK - Histórico global persistido"
)


print(
    "OK - Histórico de Feature Drift persistido"
)


print(
    "OK - MERGE idempotente"
)


print(
    "OK - Ejecución actual identificada por ventana completa"
)


print(
    "OK - Un registro PSI por feature"
)


print(
    "OK - Sin duplicados en el nuevo formato"
)


print(
    "OK - Schema compatible"
)


print(
    "OK - Global Status preservado"
)


print()
print(
    "OK - HISTÓRICO DE ML MONITORING "
    "PERSISTIDO CORRECTAMENTE EN DELTA"
)


# ============================================================
# 06.02.15 VISUALIZACIÓN
# ============================================================

display(

    monitoring_saved_df

    .orderBy(
        F.desc(
            "max_gold_date"
        )
    )

)


display(

    current_drift_saved_df

    .orderBy(
        F.desc_nulls_last(
            "psi"
        )
    )

)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # 07. AUTOMATIZACIÓN DEL ML MONITORING
# MAGIC
# MAGIC ## 07.01 Preparación para Lakeflow Jobs
# MAGIC
# MAGIC El notebook de **ML Monitoring** se integra como una tarea posterior a los procesos de Feature Engineering e inference.
# MAGIC
# MAGIC La secuencia objetivo será:
# MAGIC
# MAGIC **Gold Modeling**
# MAGIC
# MAGIC ↓
# MAGIC
# MAGIC **ML Feature Refresh**
# MAGIC
# MAGIC ↓
# MAGIC
# MAGIC **Sales Forecast Inference**
# MAGIC
# MAGIC ↓
# MAGIC
# MAGIC **ML Monitoring**
# MAGIC
# MAGIC El proceso de Monitoring ejecutará automáticamente:
# MAGIC
# MAGIC - validación técnica del forecast
# MAGIC - detección de forecasts evaluables
# MAGIC - comparación Forecast vs Actual
# MAGIC - cálculo de métricas de performance cuando existan actuals
# MAGIC - monitoring de performance por tienda
# MAGIC - Feature Drift
# MAGIC - Population Stability Index (PSI)
# MAGIC - Quality Gates
# MAGIC - persistencia histórica de los resultados en Delta
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Comportamiento esperado
# MAGIC
# MAGIC El proceso distingue entre **errores técnicos** y **estados funcionales de Monitoring**.
# MAGIC
# MAGIC ### Error técnico
# MAGIC
# MAGIC Representa una situación que impide garantizar la integridad o correcta ejecución del proceso.
# MAGIC
# MAGIC Ejemplos:
# MAGIC
# MAGIC - tablas necesarias inexistentes
# MAGIC - predicciones NULL o NaN
# MAGIC - predicciones negativas
# MAGIC - forecasts duplicados
# MAGIC - cobertura incompleta de tiendas
# MAGIC - inconsistencias en las claves del forecast
# MAGIC - errores de persistencia
# MAGIC - inconsistencias en los datasets de Monitoring
# MAGIC - errores de ejecución
# MAGIC
# MAGIC En estos casos el notebook debe lanzar una excepción y la task de Lakeflow Jobs debe finalizar como:
# MAGIC
# MAGIC `FAILED`
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Performance pendiente
# MAGIC
# MAGIC Un forecast únicamente puede evaluarse cuando su fecha ya dispone de ventas reales en Gold.
# MAGIC
# MAGIC Si todavía no existen actuals:
# MAGIC
# MAGIC `Model Performance = PENDING`
# MAGIC
# MAGIC Esto **no representa un error técnico**.
# MAGIC
# MAGIC El proceso de Monitoring puede finalizar correctamente y esperar a futuras ejecuciones para calcular las métricas de performance.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Feature Drift
# MAGIC
# MAGIC El cambio en la distribución de las features se monitoriza mediante **Population Stability Index (PSI)**.
# MAGIC
# MAGIC Los resultados se clasifican inicialmente como:
# MAGIC
# MAGIC - `PSI < 0.10` → `STABLE`
# MAGIC - `0.10 <= PSI < 0.25` → `WARNING`
# MAGIC - `PSI >= 0.25` → `DRIFT`
# MAGIC - feature no evaluable técnicamente → `NOT_EVALUABLE`
# MAGIC
# MAGIC A partir de estos resultados se determina el estado funcional de Feature Drift:
# MAGIC
# MAGIC - sin señales relevantes → `STABLE`
# MAGIC - alguna feature en WARNING → `WARNING`
# MAGIC - alguna feature en DRIFT → `CRITICAL`
# MAGIC
# MAGIC Un estado `WARNING` o `CRITICAL` representa una **señal funcional de Monitoring**, no un fallo técnico del pipeline.
# MAGIC
# MAGIC Por tanto, estos estados deben:
# MAGIC
# MAGIC - persistirse en las tablas históricas de Monitoring
# MAGIC - quedar disponibles para dashboards y análisis
# MAGIC - poder utilizarse posteriormente para generar alertas
# MAGIC
# MAGIC pero **no deben provocar por sí solos que la task de Lakeflow Jobs finalice como FAILED**.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Estado global de Monitoring
# MAGIC
# MAGIC El proceso puede generar los siguientes estados:
# MAGIC
# MAGIC - `HEALTHY`
# MAGIC - `WARNING`
# MAGIC - `CRITICAL`
# MAGIC - `PENDING_PERFORMANCE`
# MAGIC
# MAGIC Estos estados describen la situación funcional del sistema de ML y quedan almacenados históricamente.
# MAGIC
# MAGIC La ejecución de Lakeflow Jobs únicamente debe finalizar como `FAILED` cuando exista un **error técnico real** que impida garantizar la integridad del proceso.
# MAGIC
# MAGIC De esta forma se separan claramente:
# MAGIC
# MAGIC **Estado técnico del pipeline**
# MAGIC
# MAGIC de
# MAGIC
# MAGIC **Estado funcional del modelo en producción**

# COMMAND ----------

# DBTITLE 1,07.01 VALIDACIÓN FINAL PARA LAKEFLOW JOBS
# ============================================================
# 07.01 VALIDACIÓN FINAL PARA LAKEFLOW JOBS
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Validar el resultado final del proceso de ML Monitoring
# utilizando exclusivamente información persistida en Delta.
#
# Este bloque debe poder ejecutarse de forma independiente
# del estado en memoria del notebook.
#
# IMPORTANTE
# ------------------------------------------------------------
#
# Un estado funcional:
#
#     WARNING
#     CRITICAL
#     PENDING_PERFORMANCE
#
# NO debe provocar por sí mismo un fallo técnico del Job.
#
# Un fallo técnico se reserva para problemas como:
#
# - tablas inexistentes
# - histórico inconsistente
# - estados desconocidos
# - duplicados
# - ejecución persistida incompleta
# - problemas de lectura/escritura
#
# ============================================================


from pyspark.sql import functions as F


# ============================================================
# 07.01.01 CONFIGURACIÓN
# ============================================================

MONITORING_RUNS_TABLE = (
    f"{CATALOG}.5_ml.ml_monitoring_runs"
)


FEATURE_DRIFT_MONITORING_TABLE = (
    f"{CATALOG}.5_ml.feature_drift_monitoring"
)


# ============================================================
# 07.01.02 VALIDACIÓN DE EXISTENCIA
# ============================================================

required_tables = [

    MONITORING_RUNS_TABLE,
    FEATURE_DRIFT_MONITORING_TABLE

]


missing_tables = [

    table_name

    for table_name in required_tables

    if not spark.catalog.tableExists(
        table_name
    )

]


if missing_tables:

    raise RuntimeError(

        "Faltan tablas necesarias para validar ML Monitoring: "
        + ", ".join(
            missing_tables
        )

    )


# ============================================================
# 07.01.03 LECTURA DEL HISTÓRICO
# ============================================================

monitoring_saved_df = (

    spark.table(
        MONITORING_RUNS_TABLE
    )

)


drift_saved_df = (

    spark.table(
        FEATURE_DRIFT_MONITORING_TABLE
    )

)


# ============================================================
# 07.01.04 VALIDACIÓN DE HISTÓRICO NO VACÍO
# ============================================================

if monitoring_saved_df.limit(1).count() == 0:

    raise RuntimeError(

        "No existen ejecuciones registradas "
        "en ml_monitoring_runs."

    )


if drift_saved_df.limit(1).count() == 0:

    raise RuntimeError(

        "No existen registros históricos "
        "en feature_drift_monitoring."

    )


# ============================================================
# 07.01.05 RECUPERAMOS LA ÚLTIMA EJECUCIÓN GLOBAL
# ============================================================
#
# Utilizamos monitoring_timestamp únicamente para identificar
# el run global más reciente.
#
# ============================================================

latest_monitoring_run = (

    monitoring_saved_df

    .orderBy(
        F.desc(
            "monitoring_timestamp"
        )
    )

    .first()

)


if latest_monitoring_run is None:

    raise RuntimeError(

        "No ha sido posible recuperar "
        "la última ejecución de Monitoring."

    )


# ============================================================
# 07.01.06 ESTADOS DE LA ÚLTIMA EJECUCIÓN
# ============================================================

monitoring_timestamp = (
    latest_monitoring_run[
        "monitoring_timestamp"
    ]
)


max_gold_date = (
    latest_monitoring_run[
        "max_gold_date"
    ]
)


monitoring_status = (
    latest_monitoring_run[
        "global_status"
    ]
)


forecast_quality_status = (
    latest_monitoring_run[
        "forecast_quality_status"
    ]
)


feature_drift_status = (
    latest_monitoring_run[
        "feature_drift_status"
    ]
)


model_performance_status = (
    latest_monitoring_run[
        "model_performance_status"
    ]
)


stable_features_expected = int(
    latest_monitoring_run[
        "stable_features"
    ]
    or 0
)


warning_features_expected = int(
    latest_monitoring_run[
        "warning_features"
    ]
    or 0
)


drifted_features_expected = int(
    latest_monitoring_run[
        "drifted_features"
    ]
    or 0
)


not_evaluable_features_expected = int(
    latest_monitoring_run[
        "not_evaluable_features"
    ]
    or 0
)


# ============================================================
# 07.01.07 RECUPERAMOS LA VENTANA DE DRIFT MÁS RECIENTE
# ============================================================
#
# Ya NO utilizamos monitoring_timestamp para identificar
# el detalle de drift.
#
# La identidad lógica del drift está formada por:
#
# reference_start_date
# reference_end_date
# current_start_date
# current_end_date
#
# ============================================================

latest_drift_window = (

    drift_saved_df

    .filter(
        F.col(
            "reference_start_date"
        ).isNotNull()
    )

    .select(

        "reference_start_date",
        "reference_end_date",
        "current_start_date",
        "current_end_date"

    )

    .distinct()

    .orderBy(
        F.desc(
            "current_end_date"
        )
    )

    .first()

)


if latest_drift_window is None:

    raise RuntimeError(

        "No existe ninguna ejecución de Feature Drift "
        "con el nuevo formato de ventanas."

    )


reference_start_date = (
    latest_drift_window[
        "reference_start_date"
    ]
)


reference_end_date = (
    latest_drift_window[
        "reference_end_date"
    ]
)


current_start_date = (
    latest_drift_window[
        "current_start_date"
    ]
)


current_end_date = (
    latest_drift_window[
        "current_end_date"
    ]
)


# ============================================================
# 07.01.08 DETALLE DE FEATURE DRIFT ACTUAL
# ============================================================

latest_drift_df = (

    drift_saved_df

    .filter(
        F.col(
            "reference_start_date"
        )
        ==
        F.lit(
            reference_start_date
        )
    )

    .filter(
        F.col(
            "reference_end_date"
        )
        ==
        F.lit(
            reference_end_date
        )
    )

    .filter(
        F.col(
            "current_start_date"
        )
        ==
        F.lit(
            current_start_date
        )
    )

    .filter(
        F.col(
            "current_end_date"
        )
        ==
        F.lit(
            current_end_date
        )
    )

)


# ============================================================
# 07.01.09 RESUMEN REAL DEL FEATURE DRIFT
# ============================================================

stable_features = (

    latest_drift_df

    .filter(
        F.col(
            "drift_status"
        ) == "STABLE"
    )

    .count()

)


warning_features = (

    latest_drift_df

    .filter(
        F.col(
            "drift_status"
        ) == "WARNING"
    )

    .count()

)


drifted_features = (

    latest_drift_df

    .filter(
        F.col(
            "drift_status"
        ) == "DRIFT"
    )

    .count()

)


not_evaluable_features = (

    latest_drift_df

    .filter(
        F.col(
            "drift_status"
        ) == "NOT_EVALUABLE"
    )

    .count()

)


total_drift_features = (

    stable_features
    +
    warning_features
    +
    drifted_features
    +
    not_evaluable_features

)


expected_total_drift_features = (

    stable_features_expected
    +
    warning_features_expected
    +
    drifted_features_expected
    +
    not_evaluable_features_expected

)


# ============================================================
# 07.01.10 VALIDACIÓN DE DUPLICADOS
# ============================================================

latest_drift_duplicates = (

    latest_drift_df

    .groupBy(
        "feature"
    )

    .count()

    .filter(
        F.col(
            "count"
        ) > 1
    )

    .count()

)


# ============================================================
# 07.01.11 VALIDACIÓN DE ESTADOS
# ============================================================

valid_global_statuses = [

    "HEALTHY",
    "WARNING",
    "CRITICAL",
    "PENDING_PERFORMANCE"

]


valid_forecast_quality_statuses = [

    "PASSED",
    "FAILED"

]


valid_feature_drift_statuses = [

    "STABLE",
    "WARNING",
    "CRITICAL"

]


valid_model_performance_statuses = [

    "PENDING",
    "AVAILABLE"

]


if monitoring_status not in valid_global_statuses:

    raise RuntimeError(

        f"Estado global de Monitoring no reconocido: "
        f"{monitoring_status}"

    )


if (
    forecast_quality_status
    not in valid_forecast_quality_statuses
):

    raise RuntimeError(

        "Estado de Forecast Quality no reconocido: "
        f"{forecast_quality_status}"

    )


if (
    feature_drift_status
    not in valid_feature_drift_statuses
):

    raise RuntimeError(

        "Estado de Feature Drift no reconocido: "
        f"{feature_drift_status}"

    )


if (
    model_performance_status
    not in valid_model_performance_statuses
):

    raise RuntimeError(

        "Estado de Model Performance no reconocido: "
        f"{model_performance_status}"

    )


# ============================================================
# 07.01.12 VALIDACIÓN DE CONSISTENCIA
# ============================================================

if latest_drift_duplicates > 0:

    raise RuntimeError(

        "La última ventana de Feature Drift "
        "contiene features duplicadas."

    )


if (
    total_drift_features
    !=
    expected_total_drift_features
):

    raise RuntimeError(

        "El número de features persistido en el detalle "
        "no coincide con el resumen del Monitoring Run. "
        f"Detalle: {total_drift_features} | "
        f"Resumen: {expected_total_drift_features}"

    )


if (
    stable_features
    !=
    stable_features_expected
):

    raise RuntimeError(

        "El número de features STABLE "
        "no coincide con el resumen persistido."

    )


if (
    warning_features
    !=
    warning_features_expected
):

    raise RuntimeError(

        "El número de features WARNING "
        "no coincide con el resumen persistido."

    )


if (
    drifted_features
    !=
    drifted_features_expected
):

    raise RuntimeError(

        "El número de features DRIFT "
        "no coincide con el resumen persistido."

    )


if (
    not_evaluable_features
    !=
    not_evaluable_features_expected
):

    raise RuntimeError(

        "El número de features NOT_EVALUABLE "
        "no coincide con el resumen persistido."

    )


# ============================================================
# 07.01.13 RESULTADO
# ============================================================

print("=" * 80)
print("ML MONITORING - JOB VALIDATION")
print("=" * 80)


print()
print(
    f"Monitoring Timestamp:        "
    f"{monitoring_timestamp}"
)


print(
    f"Última fecha Gold:            "
    f"{max_gold_date}"
)


print()
print("-" * 80)
print("ESTADOS")
print("-" * 80)


print(
    f"Global Status:                "
    f"{monitoring_status}"
)


print(
    f"Forecast Quality:             "
    f"{forecast_quality_status}"
)


print(
    f"Feature Drift:                "
    f"{feature_drift_status}"
)


print(
    f"Model Performance:            "
    f"{model_performance_status}"
)


print()
print("-" * 80)
print("FEATURE DRIFT WINDOW")
print("-" * 80)


print(
    f"Reference:                    "
    f"{reference_start_date} -> "
    f"{reference_end_date}"
)


print(
    f"Current:                      "
    f"{current_start_date} -> "
    f"{current_end_date}"
)


print()
print("-" * 80)
print("FEATURE DRIFT DETAIL")
print("-" * 80)


print(
    f"Stable Features:              "
    f"{stable_features}"
)


print(
    f"Warning Features:             "
    f"{warning_features}"
)


print(
    f"Drifted Features:             "
    f"{drifted_features}"
)


print(
    f"Not Evaluable Features:       "
    f"{not_evaluable_features}"
)


print(
    f"Total Features:               "
    f"{total_drift_features}"
)


print(
    f"Duplicados:                   "
    f"{latest_drift_duplicates}"
)


# ============================================================
# 07.01.14 INTERPRETACIÓN FUNCIONAL
# ============================================================

print()
print("=" * 80)
print("INTERPRETACIÓN DEL ESTADO")
print("=" * 80)


if monitoring_status == "CRITICAL":

    print()

    print(
        "CRITICAL - ML Monitoring ha detectado "
        "una condición funcional crítica."
    )

    print(
        "El pipeline ha finalizado técnicamente correctamente."
    )

    print(
        "La condición queda persistida en Delta "
        "para dashboards y alertas."
    )


elif monitoring_status == "WARNING":

    print()

    print(
        "WARNING - ML Monitoring ha detectado señales "
        "que requieren seguimiento."
    )

    print(
        "El pipeline ha finalizado técnicamente correctamente."
    )


elif monitoring_status == "PENDING_PERFORMANCE":

    print()

    print(
        "INFO - Model Performance permanece pendiente "
        "hasta disponer de actuals suficientes."
    )

    print(
        "El pipeline ha finalizado técnicamente correctamente."
    )


elif monitoring_status == "HEALTHY":

    print()

    print(
        "OK - Modelo y datos dentro de "
        "los controles funcionales definidos."
    )


# ============================================================
# 07.01.15 CIERRE TÉCNICO
# ============================================================

print()
print("=" * 80)


print(
    "OK - Persistencia de Monitoring validada"
)


print(
    "OK - Estados persistidos válidos"
)


print(
    "OK - Detalle y resumen de Feature Drift consistentes"
)


print(
    "OK - Estado funcional separado del estado técnico"
)


print()
print(
    "OK - ML MONITORING FINALIZADO "
    "TÉCNICAMENTE CORRECTAMENTE"
)


print("=" * 80)

# COMMAND ----------

# DBTITLE 1,08.01 VIEW - ML MONITORING OVERVIEW
# MAGIC %sql
# MAGIC
# MAGIC -- ============================================================
# MAGIC -- 08.01 VIEW - ML MONITORING OVERVIEW
# MAGIC -- ============================================================
# MAGIC --
# MAGIC -- OBJETIVO
# MAGIC -- ------------------------------------------------------------
# MAGIC --
# MAGIC -- Crear una capa de consumo simplificada para el dashboard
# MAGIC -- de ML Monitoring.
# MAGIC --
# MAGIC -- Una fila representa una ejecución lógica del Monitoring.
# MAGIC --
# MAGIC -- Incluye:
# MAGIC --
# MAGIC -- - estado de Forecast Quality
# MAGIC -- - estado de Feature Drift
# MAGIC -- - disponibilidad de Model Performance
# MAGIC -- - estado global
# MAGIC -- - resumen de features según PSI
# MAGIC --
# MAGIC -- ============================================================
# MAGIC
# MAGIC
# MAGIC CREATE OR REPLACE VIEW
# MAGIC vw_ml_monitoring_overview
# MAGIC AS
# MAGIC
# MAGIC
# MAGIC SELECT
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- IDENTIFICACIÓN DE LA EJECUCIÓN
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     monitoring_timestamp,
# MAGIC
# MAGIC     max_gold_date,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- COBERTURA DEL FORECAST
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     forecast_records,
# MAGIC
# MAGIC     evaluated_forecasts,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- ESTADOS DE MONITORING
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     forecast_quality_status,
# MAGIC
# MAGIC     feature_drift_status,
# MAGIC
# MAGIC     model_performance_status,
# MAGIC
# MAGIC     global_status,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- RESUMEN FEATURE DRIFT / PSI
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     stable_features,
# MAGIC
# MAGIC     warning_features,
# MAGIC
# MAGIC     drifted_features,
# MAGIC
# MAGIC     not_evaluable_features
# MAGIC
# MAGIC
# MAGIC FROM
# MAGIC     ml_monitoring_runs;

# COMMAND ----------

# DBTITLE 1,VALIDACIÓN
# MAGIC %sql
# MAGIC
# MAGIC -- ============================================================
# MAGIC -- VALIDACIÓN
# MAGIC -- ============================================================
# MAGIC
# MAGIC SELECT *
# MAGIC FROM vw_ml_monitoring_overview
# MAGIC ORDER BY monitoring_timestamp DESC;

# COMMAND ----------

# DBTITLE 1,08.02 VIEW - FEATURE DRIFT HISTORY
# MAGIC %sql
# MAGIC
# MAGIC -- ============================================================
# MAGIC -- 08.02 VIEW - FEATURE DRIFT HISTORY
# MAGIC -- ============================================================
# MAGIC --
# MAGIC -- OBJETIVO
# MAGIC -- ------------------------------------------------------------
# MAGIC --
# MAGIC -- Crear una capa de consumo para analizar la evolución
# MAGIC -- histórica del Feature Drift mediante PSI.
# MAGIC --
# MAGIC -- Una fila representa:
# MAGIC --
# MAGIC -- una feature
# MAGIC -- +
# MAGIC -- una ventana temporal de Monitoring
# MAGIC --
# MAGIC -- ============================================================
# MAGIC
# MAGIC
# MAGIC CREATE OR REPLACE VIEW
# MAGIC vw_feature_drift_history
# MAGIC AS
# MAGIC
# MAGIC
# MAGIC SELECT
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- IDENTIFICACIÓN DE LA EJECUCIÓN
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     monitoring_timestamp,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- REFERENCE WINDOW
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     reference_start_date,
# MAGIC
# MAGIC     reference_end_date,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- CURRENT WINDOW
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     current_start_date,
# MAGIC
# MAGIC     current_end_date,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- FEATURE
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     feature,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- PSI
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     psi,
# MAGIC
# MAGIC     drift_status,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- CLASIFICACIÓN PSI
# MAGIC     -- ========================================================
# MAGIC     --
# MAGIC     -- La clasificación se vuelve a exponer explícitamente
# MAGIC     -- para facilitar el consumo desde dashboards.
# MAGIC     --
# MAGIC     -- IMPORTANTE:
# MAGIC     --
# MAGIC     -- PSI NULL representa una feature NOT_EVALUABLE.
# MAGIC     --
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     CASE
# MAGIC
# MAGIC         WHEN psi IS NULL
# MAGIC             THEN 'NOT_EVALUABLE'
# MAGIC
# MAGIC         WHEN psi < 0.10
# MAGIC             THEN 'STABLE'
# MAGIC
# MAGIC         WHEN psi < 0.25
# MAGIC             THEN 'WARNING'
# MAGIC
# MAGIC         ELSE 'DRIFT'
# MAGIC
# MAGIC     END AS psi_status,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- TRAZABILIDAD DEL CÁLCULO
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     bins_used,
# MAGIC
# MAGIC     reference_records,
# MAGIC
# MAGIC     current_records,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- THRESHOLDS
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     0.10 AS warning_threshold,
# MAGIC
# MAGIC     0.25 AS drift_threshold
# MAGIC
# MAGIC
# MAGIC FROM
# MAGIC     feature_drift_monitoring;

# COMMAND ----------

# DBTITLE 1,VALIDACIÓN
# MAGIC %sql
# MAGIC
# MAGIC SELECT *
# MAGIC FROM vw_feature_drift_history
# MAGIC ORDER BY current_end_date DESC, psi DESC;

# COMMAND ----------

# DBTITLE 1,08.03 VIEW - FORECAST MONITORING
# MAGIC %sql
# MAGIC
# MAGIC -- ============================================================
# MAGIC -- 08.03 VIEW - FORECAST MONITORING
# MAGIC -- ============================================================
# MAGIC --
# MAGIC -- OBJETIVO
# MAGIC -- ------------------------------------------------------------
# MAGIC --
# MAGIC -- Crear una capa de consumo para analizar:
# MAGIC --
# MAGIC -- - forecasts generados
# MAGIC -- - actuals observados
# MAGIC -- - error del forecast
# MAGIC -- - estado de evaluación
# MAGIC -- - trazabilidad del modelo
# MAGIC --
# MAGIC -- IMPORTANTE
# MAGIC -- ------------------------------------------------------------
# MAGIC --
# MAGIC -- Una predicción es evaluable cuando:
# MAGIC --
# MAGIC --     forecast_date <= MAX(fecha disponible en Gold)
# MAGIC --
# MAGIC -- Si una fecha ya está disponible en Gold pero una tienda
# MAGIC -- no presenta ventas:
# MAGIC --
# MAGIC --     actual_net_sales = 0
# MAGIC --
# MAGIC -- Esto mantiene la misma semántica utilizada en el proceso
# MAGIC -- de Feature Engineering y Model Performance Monitoring.
# MAGIC --
# MAGIC -- ============================================================
# MAGIC
# MAGIC
# MAGIC CREATE OR REPLACE VIEW
# MAGIC vw_forecast_monitoring
# MAGIC AS
# MAGIC
# MAGIC
# MAGIC -- ============================================================
# MAGIC -- 1. ACTUALS DIARIOS POR TIENDA
# MAGIC -- ============================================================
# MAGIC
# MAGIC WITH actuals AS (
# MAGIC
# MAGIC     SELECT
# MAGIC
# MAGIC         d.date AS actual_date,
# MAGIC
# MAGIC         f.store_id,
# MAGIC
# MAGIC         SUM(
# MAGIC             f.net_amount
# MAGIC         ) AS actual_net_sales
# MAGIC
# MAGIC     FROM
# MAGIC         `3_gold`.fact_sales f
# MAGIC
# MAGIC     INNER JOIN
# MAGIC         `3_gold`.dim_date d
# MAGIC
# MAGIC         ON
# MAGIC             f.date_key = d.date_key
# MAGIC
# MAGIC     GROUP BY
# MAGIC
# MAGIC         d.date,
# MAGIC         f.store_id
# MAGIC
# MAGIC ),
# MAGIC
# MAGIC
# MAGIC -- ============================================================
# MAGIC -- 2. ÚLTIMA FECHA REAL DISPONIBLE EN GOLD
# MAGIC -- ============================================================
# MAGIC
# MAGIC gold_status AS (
# MAGIC
# MAGIC     SELECT
# MAGIC
# MAGIC         MAX(
# MAGIC             d.date
# MAGIC         ) AS max_gold_date
# MAGIC
# MAGIC     FROM
# MAGIC         `3_gold`.fact_sales f
# MAGIC
# MAGIC     INNER JOIN
# MAGIC         `3_gold`.dim_date d
# MAGIC
# MAGIC         ON
# MAGIC             f.date_key = d.date_key
# MAGIC
# MAGIC ),
# MAGIC
# MAGIC
# MAGIC -- ============================================================
# MAGIC -- 3. BASE FORECAST + ACTUAL
# MAGIC -- ============================================================
# MAGIC
# MAGIC forecast_base AS (
# MAGIC
# MAGIC     SELECT
# MAGIC
# MAGIC         -- ----------------------------------------------------
# MAGIC         -- IDENTIFICACIÓN
# MAGIC         -- ----------------------------------------------------
# MAGIC
# MAGIC         p.forecast_date,
# MAGIC
# MAGIC         p.store_id,
# MAGIC
# MAGIC
# MAGIC         -- ----------------------------------------------------
# MAGIC         -- PREDICCIÓN
# MAGIC         -- ----------------------------------------------------
# MAGIC
# MAGIC         p.predicted_net_sales,
# MAGIC
# MAGIC
# MAGIC         -- ----------------------------------------------------
# MAGIC         -- ACTUAL
# MAGIC         -- ----------------------------------------------------
# MAGIC         --
# MAGIC         -- Si la fecha ya existe en Gold:
# MAGIC         --
# MAGIC         --     ventas encontradas     -> actual
# MAGIC         --     sin ventas de tienda   -> 0
# MAGIC         --
# MAGIC         -- Si la fecha todavía no existe en Gold:
# MAGIC         --
# MAGIC         --     NULL
# MAGIC         --
# MAGIC         -- ----------------------------------------------------
# MAGIC
# MAGIC         CASE
# MAGIC
# MAGIC             WHEN p.forecast_date <= g.max_gold_date
# MAGIC
# MAGIC             THEN COALESCE(
# MAGIC                 a.actual_net_sales,
# MAGIC                 0.0
# MAGIC             )
# MAGIC
# MAGIC             ELSE NULL
# MAGIC
# MAGIC         END AS actual_net_sales,
# MAGIC
# MAGIC
# MAGIC         -- ----------------------------------------------------
# MAGIC         -- TRAZABILIDAD DEL MODELO
# MAGIC         -- ----------------------------------------------------
# MAGIC
# MAGIC         p.model_name,
# MAGIC
# MAGIC         p.model_version,
# MAGIC
# MAGIC         p.mlflow_run_id,
# MAGIC
# MAGIC         p.prediction_timestamp,
# MAGIC
# MAGIC
# MAGIC         -- ----------------------------------------------------
# MAGIC         -- FECHA MÁXIMA GOLD
# MAGIC         -- ----------------------------------------------------
# MAGIC
# MAGIC         g.max_gold_date
# MAGIC
# MAGIC
# MAGIC     FROM
# MAGIC         sales_forecast_predictions p
# MAGIC
# MAGIC
# MAGIC     CROSS JOIN
# MAGIC         gold_status g
# MAGIC
# MAGIC
# MAGIC     LEFT JOIN
# MAGIC         actuals a
# MAGIC
# MAGIC         ON
# MAGIC             p.forecast_date = a.actual_date
# MAGIC
# MAGIC         AND
# MAGIC             p.store_id = a.store_id
# MAGIC
# MAGIC )
# MAGIC
# MAGIC
# MAGIC -- ============================================================
# MAGIC -- 4. VIEW FINAL
# MAGIC -- ============================================================
# MAGIC
# MAGIC SELECT
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- IDENTIFICACIÓN
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     forecast_date,
# MAGIC
# MAGIC     store_id,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- FORECAST VS ACTUAL
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     predicted_net_sales,
# MAGIC
# MAGIC     actual_net_sales,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- TRAZABILIDAD
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     model_name,
# MAGIC
# MAGIC     model_version,
# MAGIC
# MAGIC     mlflow_run_id,
# MAGIC
# MAGIC     prediction_timestamp,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- ERROR
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     CASE
# MAGIC
# MAGIC         WHEN actual_net_sales IS NOT NULL
# MAGIC
# MAGIC         THEN
# MAGIC             predicted_net_sales
# MAGIC             -
# MAGIC             actual_net_sales
# MAGIC
# MAGIC     END AS error,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- ABSOLUTE ERROR
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     CASE
# MAGIC
# MAGIC         WHEN actual_net_sales IS NOT NULL
# MAGIC
# MAGIC         THEN ABS(
# MAGIC             predicted_net_sales
# MAGIC             -
# MAGIC             actual_net_sales
# MAGIC         )
# MAGIC
# MAGIC     END AS absolute_error,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- ABSOLUTE PERCENTAGE ERROR
# MAGIC     -- ========================================================
# MAGIC     --
# MAGIC     -- Si actual = 0 no calculamos porcentaje.
# MAGIC     --
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     CASE
# MAGIC
# MAGIC         WHEN actual_net_sales IS NOT NULL
# MAGIC              AND actual_net_sales <> 0
# MAGIC
# MAGIC         THEN
# MAGIC
# MAGIC             ABS(
# MAGIC                 predicted_net_sales
# MAGIC                 -
# MAGIC                 actual_net_sales
# MAGIC             )
# MAGIC
# MAGIC             /
# MAGIC
# MAGIC             actual_net_sales
# MAGIC
# MAGIC             * 100
# MAGIC
# MAGIC     END AS absolute_percentage_error,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- EVALUATION STATUS
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     CASE
# MAGIC
# MAGIC         WHEN forecast_date <= max_gold_date
# MAGIC
# MAGIC             THEN 'EVALUATED'
# MAGIC
# MAGIC         ELSE 'PENDING'
# MAGIC
# MAGIC     END AS evaluation_status
# MAGIC
# MAGIC
# MAGIC FROM
# MAGIC     forecast_base;

# COMMAND ----------

# DBTITLE 1,VALIDACIÓN
# MAGIC %sql
# MAGIC
# MAGIC SELECT *
# MAGIC FROM vw_forecast_monitoring
# MAGIC ORDER BY forecast_date DESC, store_id;
