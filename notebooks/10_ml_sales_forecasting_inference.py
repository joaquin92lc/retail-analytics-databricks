# Databricks notebook source
# MAGIC %md
# MAGIC # 00. CONFIGURACIÓN Y CARGA DEL MODELO
# MAGIC
# MAGIC ## Objetivo
# MAGIC
# MAGIC Construir el proceso de **Inference / Scoring** del modelo de Sales Forecasting desarrollado, evaluado y registrado previamente.
# MAGIC
# MAGIC A diferencia de los notebooks anteriores, en este proceso:
# MAGIC
# MAGIC - no se entrena ningún modelo
# MAGIC - no se realiza tuning
# MAGIC - no se utiliza el dataset **Test** para evaluar o ajustar el modelo
# MAGIC - el conjunto **OOT permanece aislado**
# MAGIC - el modelo se consume directamente desde **Unity Catalog**
# MAGIC
# MAGIC Modelo utilizado:
# MAGIC
# MAGIC `retail_analytics.5_ml.sales_forecasting_random_forest`
# MAGIC
# MAGIC Versión utilizada:
# MAGIC
# MAGIC `Version 2`
# MAGIC
# MAGIC La **Version 2** corresponde al modelo final **RandomForestRegressor con Spark ML**, registrado y validado previamente mediante **MLflow + Unity Catalog**.
# MAGIC
# MAGIC El objetivo final será generar predicciones de ventas por **tienda y día** y persistir los resultados en **Delta**, manteniendo la trazabilidad de la versión del modelo utilizada y permitiendo su posterior automatización mediante **Lakeflow Jobs**.

# COMMAND ----------

# DBTITLE 1,00.01 CONFIGURACIÓN DEL PROCESO DE INFERENCE
# ============================================================
# 00.01 CONFIGURACIÓN DEL PROCESO DE INFERENCE
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Configurar el proceso de inference y cargar desde
# Unity Catalog el modelo final de Sales Forecasting.
#
# IMPORTANTE
# ------------------------------------------------------------
#
# - NO se entrena ningún modelo.
# - NO se realiza tuning.
# - NO se utiliza TEST.
# - NO se utiliza OOT en este bloque.
# - El modelo se carga directamente desde Unity Catalog.
# - El modelo registrado utiliza Spark ML.
#
# En Databricks Serverless, Spark ML necesita un Unity
# Catalog Volume temporal para poder cargar el modelo.
#
# ============================================================


import os

import mlflow
import mlflow.spark

from mlflow import MlflowClient
from pyspark.sql import functions as F


# ============================================================
# 00.01.01 CONFIGURACIÓN DE UNITY CATALOG
# ============================================================

mlflow.set_registry_uri(
    "databricks-uc"
)


client = MlflowClient()


# ============================================================
# 00.01.02 IDENTIFICACIÓN DEL MODELO
# ============================================================

MODEL_NAME = (
    "retail_analytics."
    "5_ml."
    "sales_forecasting_random_forest"
)


MODEL_VERSION = "2"


MODEL_URI = (
    f"models:/{MODEL_NAME}/{MODEL_VERSION}"
)


# ============================================================
# 00.01.03 RUTA TEMPORAL PARA SPARK ML EN SERVERLESS
# ============================================================
#
# Spark ML necesita una ruta dentro de Unity Catalog Volumes
# para determinadas operaciones de save/load en Serverless.
#
# Reutilizamos la misma ruta utilizada durante el registro.
#
# ============================================================

MLFLOW_DFS_TMP = (
    "/Volumes/retail_analytics/0_landing/"
    "autoloader_metadata/mlflow_tmp"
)


dbutils.fs.mkdirs(
    MLFLOW_DFS_TMP
)


os.environ[
    "MLFLOW_DFS_TMP"
] = MLFLOW_DFS_TMP


# ============================================================
# 00.01.04 VALIDACIÓN DE LA MODEL VERSION
# ============================================================

model_version_info = (

    client.get_model_version(

        name=
            MODEL_NAME,

        version=
            MODEL_VERSION

    )

)


if str(
    model_version_info.version
) != MODEL_VERSION:

    raise RuntimeError(

        "La Model Version recuperada no coincide "
        "con la versión solicitada."

    )


if (
    model_version_info.status
    !=
    "READY"
):

    raise RuntimeError(

        "La Model Version no se encuentra en estado READY. "
        f"Estado actual: {model_version_info.status}"

    )


# ============================================================
# 00.01.05 INFORMACIÓN DE TRAZABILIDAD
# ============================================================

MODEL_RUN_ID = (
    model_version_info.run_id
)


print("=" * 80)
print("CONFIGURACIÓN DEL PROCESO DE INFERENCE")
print("=" * 80)


print(
    f"Modelo:      "
    f"{MODEL_NAME}"
)


print(
    f"Versión:     "
    f"{MODEL_VERSION}"
)


print(
    f"Model URI:   "
    f"{MODEL_URI}"
)


print(
    f"Run ID:      "
    f"{MODEL_RUN_ID}"
)


print(
    f"Estado:      "
    f"{model_version_info.status}"
)


print(
    "Framework:   Spark ML"
)


print(
    f"MLflow DFS temp: "
    f"{MLFLOW_DFS_TMP}"
)


# ============================================================
# 00.01.06 CARGA DEL MODELO DESDE UNITY CATALOG
# ============================================================
#
# El modelo fue registrado mediante mlflow.spark.
#
# Por tanto, debe cargarse mediante mlflow.spark.
#
# En Serverless indicamos explícitamente dfs_tmpdir.
#
# ============================================================

inference_model = (

    mlflow.spark.load_model(

        MODEL_URI,

        dfs_tmpdir=
            MLFLOW_DFS_TMP

    )

)


# ============================================================
# 00.01.07 VALIDACIÓN DEL MODELO CARGADO
# ============================================================

if inference_model is None:

    raise RuntimeError(

        "No se ha podido cargar el modelo "
        "desde Unity Catalog."

    )


# ============================================================
# 00.01.08 REFERENCIA DEL MODELO
# ============================================================

inference_model_reference = {

    "model_name":
        MODEL_NAME,

    "model_version":
        MODEL_VERSION,

    "model_uri":
        MODEL_URI,

    "run_id":
        MODEL_RUN_ID,

    "framework":
        "Spark ML",

    "algorithm":
        "RandomForestRegressor"

}


# ============================================================
# 00.01.09 VALIDACIÓN FINAL
# ============================================================

print()
print("=" * 80)
print("VALIDACIÓN FINAL")
print("=" * 80)


print(
    "OK - Registry configurado para Unity Catalog"
)


print(
    "OK - UC Volume temporal configurado"
)


print(
    "OK - Registered Model localizado"
)


print(
    f"OK - Model Version "
    f"{MODEL_VERSION} localizada"
)


print(
    "OK - Model Version en estado READY"
)


print(
    "OK - Run ID recuperado"
)


print(
    "OK - Modelo Spark ML cargado desde Unity Catalog"
)


print(
    "OK - inference_model disponible"
)


print(
    "OK - inference_model_reference preparado"
)


print(
    "OK - No se ha realizado entrenamiento"
)


print(
    "OK - TEST no utilizado"
)


print(
    "OK - OOT no utilizado"
)


print()
print(
    "Configuración del proceso de inference completada."
)

# COMMAND ----------

# MAGIC %md
# MAGIC # 01. PREPARACIÓN DEL DATASET DE INFERENCE
# MAGIC
# MAGIC ## Objetivo
# MAGIC
# MAGIC Construir el dataset de entrada necesario para generar una predicción de ventas para el **siguiente día disponible** a partir del histórico conocido.
# MAGIC
# MAGIC El modelo final utiliza las siguientes features:
# MAGIC
# MAGIC ### Variables temporales
# MAGIC
# MAGIC - `year`
# MAGIC - `month`
# MAGIC - `day`
# MAGIC - `day_of_week`
# MAGIC - `week_of_year`
# MAGIC - `is_weekend`
# MAGIC
# MAGIC ### Variables históricas
# MAGIC
# MAGIC - `lag_1`
# MAGIC - `lag_7`
# MAGIC - `lag_14`
# MAGIC - `lag_28`
# MAGIC - `rolling_mean_7`
# MAGIC - `rolling_mean_28`
# MAGIC - `rolling_std_7`
# MAGIC - `lag1_minus_lag7`
# MAGIC - `lag1_vs_mean7`
# MAGIC
# MAGIC La feature antigua `diff_7` ya no forma parte del modelo. Fue sustituida por:
# MAGIC
# MAGIC `lag1_minus_lag7 = lag_1 - lag_7`
# MAGIC
# MAGIC ## Estrategia
# MAGIC
# MAGIC Para cada tienda:
# MAGIC
# MAGIC 1. identificamos la última fecha disponible
# MAGIC 2. definimos el día siguiente como fecha objetivo
# MAGIC 3. recuperamos el histórico necesario
# MAGIC 4. construimos las mismas features utilizadas durante el entrenamiento
# MAGIC 5. generamos una fila de inference por tienda
# MAGIC 6. convertimos el resultado a un **Spark DataFrame** compatible directamente con el modelo registrado
# MAGIC
# MAGIC El resultado esperado será:
# MAGIC
# MAGIC **31 registros = 31 tiendas**
# MAGIC
# MAGIC Granularidad:
# MAGIC
# MAGIC **1 fila = 1 tienda + 1 fecha objetivo**
# MAGIC
# MAGIC En esta fase realizamos un forecast **one-step ahead**, es decir, únicamente para el siguiente día disponible.
# MAGIC
# MAGIC El modelo utilizado será la **Version 2** cargada previamente desde Unity Catalog.

# COMMAND ----------

# DBTITLE 1,01.01 PREPARACIÓN DEL DATASET DE INFERENCE
# ============================================================
# 01.01 PREPARACIÓN DEL DATASET DE INFERENCE
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Construir las features necesarias para predecir las ventas
# del siguiente día disponible para cada tienda.
#
# El resultado final será un Spark DataFrame compatible
# directamente con el PipelineModel Spark ML cargado desde
# Unity Catalog.
#
# IMPORTANTE
# ------------------------------------------------------------
#
# - Se utiliza únicamente información histórica conocida.
# - No utilizamos TEST.
# - No utilizamos OOT como dataset de evaluación.
# - No se entrena ningún modelo.
# - Se replica exactamente la estructura de features usada
#   durante el entrenamiento.
#
# ============================================================


import pandas as pd
import numpy as np

from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    DoubleType,
    DateType
)


# ============================================================
# 01.01.01 CONFIGURACIÓN
# ============================================================

FEATURE_TABLE = (
    "retail_analytics.5_ml.daily_store_features"
)


EXPECTED_STORES = 31


# ------------------------------------------------------------
# FEATURES EXACTAS UTILIZADAS POR EL MODELO FINAL
# ------------------------------------------------------------

NUMERIC_FEATURES = [

    "year",
    "month",
    "day",
    "day_of_week",
    "week_of_year",
    "is_weekend",

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


MODEL_FEATURES = [

    "store_id",

    *NUMERIC_FEATURES

]


# ============================================================
# 01.01.02 VALIDAMOS LA TABLA DE FEATURES
# ============================================================

if not spark.catalog.tableExists(
    FEATURE_TABLE
):

    raise RuntimeError(

        "No existe la tabla de features: "
        f"{FEATURE_TABLE}"

    )


ml_features_df = (
    spark.table(
        FEATURE_TABLE
    )
)


# ============================================================
# 01.01.03 ÚLTIMA FECHA DISPONIBLE
# ============================================================

last_available_date = (

    ml_features_df

    .agg(

        F.max(
            "sale_date"
        ).alias(
            "last_date"
        )

    )

    .first()[
        "last_date"
    ]

)


if last_available_date is None:

    raise RuntimeError(
        "No existe ninguna fecha disponible."
    )


# ============================================================
# 01.01.04 FECHA OBJETIVO
# ============================================================

target_date = (

    pd.Timestamp(
        last_available_date
    )

    +

    pd.Timedelta(
        days=1
    )

)


print("=" * 80)
print("FECHA DE INFERENCE")
print("=" * 80)


print(
    f"Última fecha disponible: "
    f"{last_available_date}"
)


print(
    f"Fecha objetivo:          "
    f"{target_date.date()}"
)


# ============================================================
# 01.01.05 HISTÓRICO NECESARIO
# ============================================================
#
# Necesitamos como mínimo 28 observaciones anteriores para:
#
# - lag_28
# - rolling_mean_28
#
# Recuperamos 40 días naturales para disponer de margen.
#
# ============================================================

history_start_date = (

    target_date

    -

    pd.Timedelta(
        days=40
    )

)


recent_history_df = (

    ml_features_df

    .filter(

        F.col("sale_date")
        >=
        F.lit(
            history_start_date.date()
        )

    )

    .select(

        "sale_date",
        "store_id",
        "net_sales"

    )

)


# ============================================================
# 01.01.06 PASAMOS EL HISTÓRICO RECIENTE A PANDAS
# ============================================================
#
# Solo manejamos aproximadamente:
#
# 31 tiendas × 40 días
#
# por lo que el volumen es mínimo.
#
# ============================================================

recent_history_pd = (

    recent_history_df

    .toPandas()

)


recent_history_pd[
    "sale_date"
] = pd.to_datetime(

    recent_history_pd[
        "sale_date"
    ]

)


recent_history_pd[
    "net_sales"
] = (

    recent_history_pd[
        "net_sales"
    ]

    .astype(
        float
    )

)


# ============================================================
# 01.01.07 VALIDAMOS COBERTURA DE TIENDAS
# ============================================================

available_stores = (

    recent_history_pd[
        "store_id"
    ]

    .nunique()

)


if available_stores != EXPECTED_STORES:

    raise RuntimeError(

        "Número de tiendas inesperado en el histórico reciente. "
        f"Esperadas: {EXPECTED_STORES} | "
        f"Disponibles: {available_stores}"

    )


print()
print(
    f"OK - Tiendas disponibles: "
    f"{available_stores}"
)


# ============================================================
# 01.01.08 CONSTRUCCIÓN DE FEATURES
# ============================================================

inference_rows = []


stores_without_history = []


for store_id, store_history in (

    recent_history_pd

    .groupby(
        "store_id"
    )

):


    # --------------------------------------------------------
    # ORDEN CRONOLÓGICO
    # --------------------------------------------------------

    store_history = (

        store_history

        .sort_values(
            "sale_date"
        )

        .reset_index(
            drop=True
        )

    )


    # --------------------------------------------------------
    # VALIDAMOS HISTÓRICO SUFICIENTE
    # --------------------------------------------------------

    if len(store_history) < 28:

        stores_without_history.append(
            str(store_id)
        )

        continue


    sales = (

        store_history[
            "net_sales"
        ]

        .to_numpy(
            dtype=float
        )

    )


    # ========================================================
    # LAGS
    # ========================================================

    lag_1 = float(
        sales[-1]
    )


    lag_7 = float(
        sales[-7]
    )


    lag_14 = float(
        sales[-14]
    )


    lag_28 = float(
        sales[-28]
    )


    # ========================================================
    # ROLLING FEATURES
    # ========================================================
    #
    # Replicamos la lógica utilizada durante entrenamiento:
    #
    # rolling_mean_7  -> 7 observaciones anteriores
    # rolling_mean_28 -> 28 observaciones anteriores
    #
    # La fila objetivo todavía no existe, por lo que no hay
    # riesgo de leakage.
    #
    # ========================================================

    rolling_mean_7 = float(
        sales[-7:].mean()
    )


    rolling_mean_28 = float(
        sales[-28:].mean()
    )


    rolling_std_7 = float(

        sales[-7:].std(
            ddof=1
        )

    )


    # ========================================================
    # FEATURES DERIVADAS
    # ========================================================

    lag1_minus_lag7 = (

        lag_1
        -
        lag_7

    )


    lag1_vs_mean7 = (

        lag_1
        /
        rolling_mean_7

        if rolling_mean_7 != 0

        else 0.0

    )


    # ========================================================
    # FEATURES TEMPORALES
    # ========================================================

    year = int(
        target_date.year
    )


    month = int(
        target_date.month
    )


    day = int(
        target_date.day
    )


    # --------------------------------------------------------
    # Spark dayofweek:
    #
    # Sunday    = 1
    # Monday    = 2
    # ...
    # Saturday  = 7
    #
    # Pandas weekday():
    #
    # Monday    = 0
    # ...
    # Sunday    = 6
    #
    # Conversión:
    #
    # ((weekday + 1) % 7) + 1
    #
    # --------------------------------------------------------

    pandas_weekday = int(
        target_date.weekday()
    )


    day_of_week = int(

        (
            (
                pandas_weekday
                +
                1
            )
            %
            7
        )
        +
        1

    )


    week_of_year = int(
        target_date.isocalendar().week
    )


    is_weekend = (

        1

        if day_of_week in [
            1,
            7
        ]

        else 0

    )


    # ========================================================
    # REGISTRO FINAL
    # ========================================================

    inference_rows.append({

        "forecast_date":
            target_date.date(),

        "store_id":
            str(
                store_id
            ),

        "year":
            year,

        "month":
            month,

        "day":
            day,

        "day_of_week":
            day_of_week,

        "week_of_year":
            week_of_year,

        "is_weekend":
            int(
                is_weekend
            ),

        "lag_1":
            lag_1,

        "lag_7":
            lag_7,

        "lag_14":
            lag_14,

        "lag_28":
            lag_28,

        "rolling_mean_7":
            rolling_mean_7,

        "rolling_mean_28":
            rolling_mean_28,

        "rolling_std_7":
            rolling_std_7,

        "lag1_minus_lag7":
            float(
                lag1_minus_lag7
            ),

        "lag1_vs_mean7":
            float(
                lag1_vs_mean7
            )

    })


# ============================================================
# 01.01.09 VALIDACIÓN DEL HISTÓRICO
# ============================================================

if stores_without_history:

    raise RuntimeError(

        "Existen tiendas sin histórico suficiente: "
        f"{stores_without_history}"

    )


# ============================================================
# 01.01.10 DATAFRAME PANDAS
# ============================================================
#
# Lo conservamos porque algunos sanity checks posteriores
# pueden resultar cómodos en Pandas.
#
# ============================================================

inference_pd = pd.DataFrame(
    inference_rows
)


# ============================================================
# 01.01.11 VALIDACIÓN ESTRUCTURAL
# ============================================================

if len(inference_pd) != EXPECTED_STORES:

    raise RuntimeError(

        "El número de registros de inference "
        "no coincide con el número esperado de tiendas. "
        f"Esperados: {EXPECTED_STORES} | "
        f"Generados: {len(inference_pd)}"

    )


if (
    inference_pd[
        "store_id"
    ]
    .nunique()
    !=
    EXPECTED_STORES
):

    raise RuntimeError(

        "El número de tiendas únicas "
        "del dataset de inference es incorrecto."

    )


# ============================================================
# 01.01.12 VALIDAMOS NULLS / NAN
# ============================================================

null_count = int(

    inference_pd[
        MODEL_FEATURES
    ]

    .isna()

    .sum()

    .sum()

)


if null_count != 0:

    raise RuntimeError(

        "Se han detectado NULL/NaN "
        "en las features de inference."

    )


# ============================================================
# 01.01.13 SCHEMA EXPLÍCITO PARA SPARK
# ============================================================

inference_schema = StructType([

    StructField(
        "forecast_date",
        DateType(),
        False
    ),

    StructField(
        "store_id",
        StringType(),
        False
    ),

    StructField(
        "year",
        IntegerType(),
        False
    ),

    StructField(
        "month",
        IntegerType(),
        False
    ),

    StructField(
        "day",
        IntegerType(),
        False
    ),

    StructField(
        "day_of_week",
        IntegerType(),
        False
    ),

    StructField(
        "week_of_year",
        IntegerType(),
        False
    ),

    StructField(
        "is_weekend",
        IntegerType(),
        False
    ),

    StructField(
        "lag_1",
        DoubleType(),
        False
    ),

    StructField(
        "lag_7",
        DoubleType(),
        False
    ),

    StructField(
        "lag_14",
        DoubleType(),
        False
    ),

    StructField(
        "lag_28",
        DoubleType(),
        False
    ),

    StructField(
        "rolling_mean_7",
        DoubleType(),
        False
    ),

    StructField(
        "rolling_mean_28",
        DoubleType(),
        False
    ),

    StructField(
        "rolling_std_7",
        DoubleType(),
        False
    ),

    StructField(
        "lag1_minus_lag7",
        DoubleType(),
        False
    ),

    StructField(
        "lag1_vs_mean7",
        DoubleType(),
        False
    )

])


# ============================================================
# 01.01.14 CONVERSIÓN A SPARK DATAFRAME
# ============================================================

inference_df = spark.createDataFrame(

    inference_pd.to_dict(
        orient="records"
    ),

    schema=
        inference_schema

)


# ============================================================
# 01.01.15 VALIDACIÓN SPARK
# ============================================================

inference_count = (
    inference_df.count()
)


inference_stores = (

    inference_df

    .select(
        "store_id"
    )

    .distinct()

    .count()

)


if inference_count != EXPECTED_STORES:

    raise RuntimeError(

        "El Spark DataFrame de inference "
        "no contiene 31 registros."

    )


if inference_stores != EXPECTED_STORES:

    raise RuntimeError(

        "El Spark DataFrame de inference "
        "no contiene 31 tiendas únicas."

    )


# ============================================================
# 01.01.16 RESULTADO
# ============================================================

print()
print("=" * 80)
print("DATASET DE INFERENCE")
print("=" * 80)


print(
    f"Última fecha real:    "
    f"{last_available_date}"
)


print(
    f"Fecha forecast:       "
    f"{target_date.date()}"
)


print(
    f"Registros generados:  "
    f"{inference_count}"
)


print(
    f"Tiendas únicas:       "
    f"{inference_stores}"
)


print(
    f"Features del modelo:  "
    f"{len(MODEL_FEATURES)}"
)


print(
    f"NULL / NaN:           "
    f"{null_count}"
)


# ============================================================
# 01.01.17 VALIDACIÓN FINAL
# ============================================================

print()
print("=" * 80)
print("VALIDACIÓN FINAL")
print("=" * 80)


print(
    "OK - Histórico cargado"
)


print(
    "OK - Fecha objetivo calculada"
)


print(
    "OK - Histórico mínimo disponible para todas las tiendas"
)


print(
    "OK - 31 registros generados"
)


print(
    "OK - 31 tiendas únicas"
)


print(
    "OK - Features temporales generadas"
)


print(
    "OK - Lags generados"
)


print(
    "OK - Rolling features generadas"
)


print(
    "OK - lag1_minus_lag7 generado"
)


print(
    "OK - Sin NULL / NaN"
)


print(
    "OK - Spark DataFrame preparado"
)


print(
    "OK - Dataset compatible con Spark ML"
)


print(
    "OK - No se ha realizado entrenamiento"
)


print()
print(
    "Dataset de inference preparado correctamente."
)


# ============================================================
# 01.01.18 VISUALIZACIÓN
# ============================================================

display(

    inference_df

    .orderBy(
        "store_id"
    )

)

# COMMAND ----------

# MAGIC %md
# MAGIC # 02. GENERACIÓN DE PREDICCIONES
# MAGIC
# MAGIC ## Objetivo
# MAGIC
# MAGIC Utilizar la **Version 2** del modelo registrada en Unity Catalog para generar el forecast de ventas del siguiente día disponible.
# MAGIC
# MAGIC El modelo cargado es un **PipelineModel de Spark ML**, por lo que las predicciones se generan directamente sobre el Spark DataFrame preparado en el bloque anterior.
# MAGIC
# MAGIC El dataset de inference contiene:
# MAGIC
# MAGIC - una fila por tienda
# MAGIC - todas las features utilizadas durante el entrenamiento
# MAGIC - la fecha objetivo del forecast
# MAGIC
# MAGIC Granularidad:
# MAGIC
# MAGIC **1 fila = 1 tienda + 1 fecha de forecast**
# MAGIC
# MAGIC El resultado incluirá:
# MAGIC
# MAGIC - fecha de forecast
# MAGIC - tienda
# MAGIC - predicción de `net_sales`
# MAGIC - versión del modelo utilizada
# MAGIC
# MAGIC No se realiza entrenamiento ni ajuste del modelo durante esta fase.

# COMMAND ----------

# DBTITLE 1,02.01 GENERACIÓN DEL FORECAST
# ============================================================
# 02.01 GENERACIÓN DEL FORECAST
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Generar las predicciones utilizando el PipelineModel
# Spark ML cargado desde Unity Catalog.
#
# Entrada:
#
#       inference_df
#
# Salida:
#
#       forecast_df
#
# Granularidad:
#
#       1 fila = 1 tienda + 1 fecha forecast
#
# ============================================================


from pyspark.sql import functions as F


# ============================================================
# 02.01.01 VALIDACIONES PREVIAS
# ============================================================

if "inference_model" not in globals():

    raise RuntimeError(
        "No existe inference_model. "
        "Ejecuta primero el bloque 00.01."
    )


if "inference_df" not in globals():

    raise RuntimeError(
        "No existe inference_df. "
        "Ejecuta primero el bloque 01.01."
    )


if "MODEL_VERSION" not in globals():

    raise RuntimeError(
        "No existe MODEL_VERSION."
    )


# ============================================================
# 02.01.02 GENERACIÓN DE PREDICCIONES
# ============================================================

prediction_df = (

    inference_model

    .transform(
        inference_df
    )

)


# ============================================================
# 02.01.03 DATASET FINAL DE FORECAST
# ============================================================

forecast_df = (

    prediction_df

    .select(

        F.col(
            "forecast_date"
        ),

        F.col(
            "store_id"
        ),

        F.col(
            "prediction"
        )
        .cast("double")
        .alias(
            "predicted_net_sales"
        )

    )

)


# ============================================================
# 02.01.04 AÑADIMOS TRAZABILIDAD DEL MODELO
# ============================================================

forecast_df = (

    forecast_df

    .withColumn(

        "model_name",

        F.lit(
            MODEL_NAME
        )

    )

    .withColumn(

        "model_version",

        F.lit(
            int(MODEL_VERSION)
        )

    )

)


# ============================================================
# 02.01.05 VALIDACIONES
# ============================================================

forecast_rows = (
    forecast_df.count()
)


forecast_stores = (

    forecast_df

    .select(
        "store_id"
    )

    .distinct()

    .count()

)


forecast_dates = (

    forecast_df

    .select(
        "forecast_date"
    )

    .distinct()

    .count()

)


prediction_quality = (

    forecast_df

    .agg(

        F.sum(

            F.col(
                "predicted_net_sales"
            )
            .isNull()
            .cast("int")

        ).alias(
            "null_predictions"
        ),

        F.sum(

            F.isnan(
                F.col(
                    "predicted_net_sales"
                )
            )
            .cast("int")

        ).alias(
            "nan_predictions"
        ),

        F.sum(

            (
                F.col(
                    "predicted_net_sales"
                )
                <
                0
            )
            .cast("int")

        ).alias(
            "negative_predictions"
        ),

        F.min(
            "predicted_net_sales"
        ).alias(
            "min_prediction"
        ),

        F.max(
            "predicted_net_sales"
        ).alias(
            "max_prediction"
        ),

        F.avg(
            "predicted_net_sales"
        ).alias(
            "avg_prediction"
        )

    )

    .first()

)


null_predictions = int(
    prediction_quality[
        "null_predictions"
    ] or 0
)


nan_predictions = int(
    prediction_quality[
        "nan_predictions"
    ] or 0
)


negative_predictions = int(
    prediction_quality[
        "negative_predictions"
    ] or 0
)


min_prediction = float(
    prediction_quality[
        "min_prediction"
    ]
)


max_prediction = float(
    prediction_quality[
        "max_prediction"
    ]
)


avg_prediction = float(
    prediction_quality[
        "avg_prediction"
    ]
)


# ============================================================
# 02.01.06 RESULTADOS
# ============================================================

print("=" * 80)
print("FORECAST GENERADO")
print("=" * 80)


print(
    f"Fecha forecast:         "
    f"{target_date.date()}"
)


print(
    f"Predicciones:           "
    f"{forecast_rows}"
)


print(
    f"Tiendas:                "
    f"{forecast_stores}"
)


print(
    f"Fechas forecast:        "
    f"{forecast_dates}"
)


print(
    f"Model Version:          "
    f"{MODEL_VERSION}"
)


print(
    f"Predicciones NULL:      "
    f"{null_predictions}"
)


print(
    f"Predicciones NaN:       "
    f"{nan_predictions}"
)


print(
    f"Predicciones negativas: "
    f"{negative_predictions}"
)


print(
    f"Predicción mínima:      "
    f"{min_prediction:,.2f}"
)


print(
    f"Predicción máxima:      "
    f"{max_prediction:,.2f}"
)


print(
    f"Predicción media:       "
    f"{avg_prediction:,.2f}"
)


# ============================================================
# 02.01.07 VALIDACIÓN FINAL
# ============================================================

if forecast_rows != EXPECTED_STORES:

    raise RuntimeError(

        "El número de predicciones no coincide "
        "con el número esperado de tiendas."

    )


if forecast_stores != EXPECTED_STORES:

    raise RuntimeError(

        "El número de tiendas únicas "
        "del forecast es incorrecto."

    )


if forecast_dates != 1:

    raise RuntimeError(

        "El forecast contiene más de una fecha objetivo."

    )


if null_predictions != 0:

    raise RuntimeError(
        "Existen predicciones NULL."
    )


if nan_predictions != 0:

    raise RuntimeError(
        "Existen predicciones NaN."
    )


if negative_predictions != 0:

    raise RuntimeError(
        "Existen predicciones negativas."
    )


print()
print("=" * 80)
print("VALIDACIÓN FINAL")
print("=" * 80)


print(
    "OK - Forecast generado con Spark ML"
)


print(
    "OK - 31 predicciones generadas"
)


print(
    "OK - 31 tiendas únicas"
)


print(
    "OK - Una única fecha de forecast"
)


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
    f"OK - Model Version utilizada: "
    f"{MODEL_VERSION}"
)


print(
    "OK - No se ha realizado entrenamiento"
)


print()
print(
    "Generación del forecast completada."
)


# ============================================================
# 02.01.08 VISUALIZACIÓN
# ============================================================

display(

    forecast_df

    .orderBy(
        "store_id"
    )

)

# COMMAND ----------

# MAGIC %md
# MAGIC # 03. VALIDACIÓN Y SANITY CHECKS DEL FORECAST
# MAGIC
# MAGIC ## Objetivo
# MAGIC
# MAGIC Validar la coherencia de las predicciones generadas antes de persistirlas.
# MAGIC
# MAGIC Aunque el proceso de inference haya generado correctamente una predicción para cada tienda, necesitamos comprobar que los resultados presentan un comportamiento razonable respecto al histórico reciente.
# MAGIC
# MAGIC Se analizará:
# MAGIC
# MAGIC - distribución de las predicciones
# MAGIC - dispersión entre tiendas
# MAGIC - comparación con `lag_1`
# MAGIC - comparación con `rolling_mean_7`
# MAGIC - comparación con `rolling_mean_28`
# MAGIC - desviaciones relativas
# MAGIC - tiendas con mayores cambios respecto a su comportamiento reciente
# MAGIC - valores NULL
# MAGIC - valores NaN
# MAGIC - duplicados
# MAGIC - predicciones negativas
# MAGIC
# MAGIC Estos controles no miden la precisión real del forecast, porque las ventas reales del día objetivo todavía no están disponibles.
# MAGIC
# MAGIC Su finalidad es detectar comportamientos anómalos antes de persistir el forecast.

# COMMAND ----------

# DBTITLE 1,03.01 SANITY CHECKS DEL FORECAST
# ============================================================
# 03.01 SANITY CHECKS DEL FORECAST
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Validar técnicamente y funcionalmente el forecast antes
# de persistirlo.
#
# Trabajamos completamente en Spark.
#
# ============================================================


from pyspark.sql import functions as F


# ============================================================
# 03.01.01 VALIDACIONES PREVIAS
# ============================================================

if "forecast_df" not in globals():

    raise RuntimeError(
        "No existe forecast_df. "
        "Ejecuta primero el bloque 02.01."
    )


if "inference_df" not in globals():

    raise RuntimeError(
        "No existe inference_df. "
        "Ejecuta primero el bloque 01.01."
    )


# ============================================================
# 03.01.02 UNIMOS FORECAST + FEATURES DE INFERENCE
# ============================================================

forecast_validation_df = (

    forecast_df

    .alias("forecast")

    .join(

        inference_df

        .select(

            "store_id",
            "forecast_date",
            "lag_1",
            "rolling_mean_7",
            "rolling_mean_28"

        )

        .alias("features"),

        on=[
            "store_id",
            "forecast_date"
        ],

        how=
            "left"

    )

)


# ============================================================
# 03.01.03 DIFERENCIAS CONTRA HISTÓRICO
# ============================================================

forecast_validation_df = (

    forecast_validation_df

    .withColumn(

        "diff_vs_lag1",

        F.col("predicted_net_sales")
        -
        F.col("lag_1")

    )

    .withColumn(

        "diff_pct_vs_lag1",

        F.when(

            F.col("lag_1") != 0,

            (
                F.col("predicted_net_sales")
                -
                F.col("lag_1")
            )
            /
            F.col("lag_1")
            *
            100

        )

        .otherwise(
            F.lit(None)
        )

    )

    .withColumn(

        "diff_vs_mean7",

        F.col("predicted_net_sales")
        -
        F.col("rolling_mean_7")

    )

    .withColumn(

        "diff_pct_vs_mean7",

        F.when(

            F.col("rolling_mean_7") != 0,

            (
                F.col("predicted_net_sales")
                -
                F.col("rolling_mean_7")
            )
            /
            F.col("rolling_mean_7")
            *
            100

        )

        .otherwise(
            F.lit(None)
        )

    )

    .withColumn(

        "diff_vs_mean28",

        F.col("predicted_net_sales")
        -
        F.col("rolling_mean_28")

    )

    .withColumn(

        "diff_pct_vs_mean28",

        F.when(

            F.col("rolling_mean_28") != 0,

            (
                F.col("predicted_net_sales")
                -
                F.col("rolling_mean_28")
            )
            /
            F.col("rolling_mean_28")
            *
            100

        )

        .otherwise(
            F.lit(None)
        )

    )

)


# ============================================================
# 03.01.04 ESTADÍSTICAS DEL FORECAST
# ============================================================

forecast_stats = (

    forecast_validation_df

    .agg(

        F.count("*").alias(
            "rows"
        ),

        F.countDistinct(
            "store_id"
        ).alias(
            "stores"
        ),

        F.avg(
            "predicted_net_sales"
        ).alias(
            "mean"
        ),

        F.expr(
            "percentile_approx(predicted_net_sales, 0.5)"
        ).alias(
            "median"
        ),

        F.stddev_samp(
            "predicted_net_sales"
        ).alias(
            "std"
        ),

        F.min(
            "predicted_net_sales"
        ).alias(
            "min"
        ),

        F.max(
            "predicted_net_sales"
        ).alias(
            "max"
        )

    )

    .first()

)


forecast_mean = float(
    forecast_stats["mean"]
)


forecast_median = float(
    forecast_stats["median"]
)


forecast_std = float(
    forecast_stats["std"]
)


forecast_min = float(
    forecast_stats["min"]
)


forecast_max = float(
    forecast_stats["max"]
)


forecast_cv = (

    forecast_std
    /
    forecast_mean
    *
    100

    if forecast_mean != 0

    else 0.0

)


# ============================================================
# 03.01.05 COMPARACIÓN MEDIA CONTRA HISTÓRICO
# ============================================================

comparison_stats = (

    forecast_validation_df

    .agg(

        F.avg(
            F.abs(
                F.col("diff_pct_vs_lag1")
            )
        ).alias(
            "mean_abs_diff_lag1"
        ),

        F.avg(
            F.abs(
                F.col("diff_pct_vs_mean7")
            )
        ).alias(
            "mean_abs_diff_mean7"
        ),

        F.avg(
            F.abs(
                F.col("diff_pct_vs_mean28")
            )
        ).alias(
            "mean_abs_diff_mean28"
        )

    )

    .first()

)


mean_abs_diff_lag1 = float(
    comparison_stats[
        "mean_abs_diff_lag1"
    ] or 0
)


mean_abs_diff_mean7 = float(
    comparison_stats[
        "mean_abs_diff_mean7"
    ] or 0
)


mean_abs_diff_mean28 = float(
    comparison_stats[
        "mean_abs_diff_mean28"
    ] or 0
)


# ============================================================
# 03.01.06 VALIDACIONES TÉCNICAS
# ============================================================

technical_stats = (

    forecast_validation_df

    .agg(

        F.sum(

            F.col("predicted_net_sales")
            .isNull()
            .cast("int")

        ).alias(
            "null_predictions"
        ),

        F.sum(

            F.isnan(
                F.col("predicted_net_sales")
            )
            .cast("int")

        ).alias(
            "nan_predictions"
        ),

        F.sum(

            (
                F.col("predicted_net_sales")
                <
                0
            )
            .cast("int")

        ).alias(
            "negative_predictions"
        )

    )

    .first()

)


null_predictions = int(
    technical_stats[
        "null_predictions"
    ] or 0
)


nan_predictions = int(
    technical_stats[
        "nan_predictions"
    ] or 0
)


negative_predictions = int(
    technical_stats[
        "negative_predictions"
    ] or 0
)


duplicate_count = (

    forecast_validation_df

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


# ============================================================
# 03.01.07 RESULTADOS
# ============================================================

print("=" * 80)
print("SANITY CHECKS DEL FORECAST")
print("=" * 80)


print(
    f"Fecha forecast: "
    f"{target_date.date()}"
)


print(
    f"Tiendas:        "
    f"{forecast_stats['stores']}"
)


print()
print("-" * 80)
print("DISTRIBUCIÓN DEL FORECAST")
print("-" * 80)


print(
    f"Media:           "
    f"{forecast_mean:,.2f}"
)


print(
    f"Mediana:         "
    f"{forecast_median:,.2f}"
)


print(
    f"Desv. estándar:  "
    f"{forecast_std:,.2f}"
)


print(
    f"Mínimo:          "
    f"{forecast_min:,.2f}"
)


print(
    f"Máximo:          "
    f"{forecast_max:,.2f}"
)


print(
    f"Coef. variación: "
    f"{forecast_cv:,.2f}%"
)


print()
print("-" * 80)
print("COMPARACIÓN CON HISTÓRICO")
print("-" * 80)


print(
    f"Diferencia media absoluta vs lag_1:   "
    f"{mean_abs_diff_lag1:.2f}%"
)


print(
    f"Diferencia media absoluta vs mean_7:  "
    f"{mean_abs_diff_mean7:.2f}%"
)


print(
    f"Diferencia media absoluta vs mean_28: "
    f"{mean_abs_diff_mean28:.2f}%"
)


print()
print("-" * 80)
print("VALIDACIONES TÉCNICAS")
print("-" * 80)


print(
    f"NULL predictions:     "
    f"{null_predictions}"
)


print(
    f"NaN predictions:      "
    f"{nan_predictions}"
)


print(
    f"Duplicados:           "
    f"{duplicate_count}"
)


print(
    f"Predicciones negativas: "
    f"{negative_predictions}"
)


# ============================================================
# 03.01.08 TOP 10 MAYORES DESVIACIONES
# ============================================================

largest_changes_df = (

    forecast_validation_df

    .withColumn(

        "abs_diff_pct_vs_mean7",

        F.abs(
            F.col(
                "diff_pct_vs_mean7"
            )
        )

    )

    .orderBy(

        F.desc(
            "abs_diff_pct_vs_mean7"
        )

    )

    .limit(
        10
    )

)


print()
print("-" * 80)
print("TOP 10 MAYORES DESVIACIONES VS MEDIA 7 DÍAS")
print("-" * 80)


display(

    largest_changes_df

    .select(

        "store_id",
        "predicted_net_sales",
        "lag_1",
        "rolling_mean_7",
        "rolling_mean_28",
        "diff_pct_vs_lag1",
        "diff_pct_vs_mean7",
        "diff_pct_vs_mean28"

    )

)


# ============================================================
# 03.01.09 VALIDACIÓN FINAL
# ============================================================

if null_predictions != 0:

    raise RuntimeError(
        "Existen predicciones NULL."
    )


if nan_predictions != 0:

    raise RuntimeError(
        "Existen predicciones NaN."
    )


if duplicate_count != 0:

    raise RuntimeError(
        "Existen duplicados en el forecast."
    )


if negative_predictions != 0:

    raise RuntimeError(
        "Existen predicciones negativas."
    )


if int(
    forecast_stats["stores"]
) != EXPECTED_STORES:

    raise RuntimeError(

        "La cobertura de tiendas "
        "del forecast es incorrecta."

    )


# ============================================================
# 03.01.10 RESULTADO
# ============================================================

print()
print("=" * 80)
print("VALIDACIÓN FINAL")
print("=" * 80)


print(
    "OK - Distribución del forecast calculada"
)


print(
    "OK - Comparación contra lag_1 realizada"
)


print(
    "OK - Comparación contra rolling_mean_7 realizada"
)


print(
    "OK - Comparación contra rolling_mean_28 realizada"
)


print(
    "OK - 31 tiendas cubiertas"
)


print(
    "OK - Sin predicciones NULL"
)


print(
    "OK - Sin predicciones NaN"
)


print(
    "OK - Sin duplicados"
)


print(
    "OK - Sin predicciones negativas"
)


print()
print(
    "Forecast técnicamente válido para persistencia."
)

# COMMAND ----------

# MAGIC %md
# MAGIC # 04. PERSISTENCIA DEL FORECAST EN DELTA
# MAGIC
# MAGIC ## Objetivo
# MAGIC
# MAGIC Persistir las predicciones generadas en una tabla **Delta** para mantener un histórico de forecasts.
# MAGIC
# MAGIC La tabla almacenará sucesivas ejecuciones del modelo con granularidad:
# MAGIC
# MAGIC **1 fila = fecha de forecast + tienda + versión del modelo**
# MAGIC
# MAGIC Además de la predicción, se almacenará información de trazabilidad:
# MAGIC
# MAGIC - fecha de forecast
# MAGIC - tienda
# MAGIC - ventas predichas
# MAGIC - nombre completo del modelo
# MAGIC - versión del modelo utilizada
# MAGIC - Run ID de MLflow asociada
# MAGIC - fecha y hora de generación
# MAGIC
# MAGIC El forecast ya ha sido generado mediante **Spark ML**, por lo que la persistencia se realizará directamente desde el Spark DataFrame `forecast_df`.
# MAGIC
# MAGIC No se realizará ninguna conversión intermedia desde Pandas.
# MAGIC
# MAGIC La persistencia utilizará posteriormente una operación **MERGE** para garantizar idempotencia ante reintentos o ejecuciones repetidas.

# COMMAND ----------

# DBTITLE 1,04.01 PREPARACIÓN DEL DATASET DE SALIDA
# ============================================================
# 04.01 PREPARACIÓN DEL DATASET DE SALIDA
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Preparar el Spark DataFrame final que será persistido
# posteriormente en la tabla Delta de forecasts.
#
# Partimos directamente de:
#
#       forecast_df
#
# generado previamente con Spark ML.
#
# Añadiremos información de trazabilidad:
#
# - nombre del modelo
# - versión del modelo
# - Run ID de MLflow
# - timestamp de generación
#
# IMPORTANTE
# ------------------------------------------------------------
#
# - NO utilizamos forecast_pd.
# - NO realizamos conversión Pandas -> Spark.
# - MODEL_VERSION procede de la configuración central.
# - No se hardcodea Version 1.
#
# ============================================================


from pyspark.sql import functions as F


# ============================================================
# 04.01.01 VALIDACIONES PREVIAS
# ============================================================

required_objects = [

    "forecast_df",
    "MODEL_NAME",
    "MODEL_VERSION",
    "MODEL_RUN_ID",
    "EXPECTED_STORES"

]


missing_objects = [

    object_name

    for object_name in required_objects

    if object_name not in globals()

]


if missing_objects:

    raise RuntimeError(

        "Faltan objetos necesarios para preparar "
        "el dataset de salida: "
        f"{missing_objects}"

    )


# ============================================================
# 04.01.02 INFORMACIÓN DE TRAZABILIDAD
# ============================================================

prediction_timestamp = (
    F.current_timestamp()
)


# ============================================================
# 04.01.03 DATASET FINAL DE SALIDA
# ============================================================

forecast_output_df = (

    forecast_df

    .select(

        F.col(
            "forecast_date"
        ).cast(
            "date"
        ).alias(
            "forecast_date"
        ),

        F.col(
            "store_id"
        ).cast(
            "string"
        ).alias(
            "store_id"
        ),

        F.col(
            "predicted_net_sales"
        ).cast(
            "double"
        ).alias(
            "predicted_net_sales"
        )

    )

    .withColumn(

        "model_name",

        F.lit(
            MODEL_NAME
        )

    )

    .withColumn(

        "model_version",

        F.lit(
            int(
                MODEL_VERSION
            )
        )

    )

    .withColumn(

        "mlflow_run_id",

        F.lit(
            MODEL_RUN_ID
        )

    )

    .withColumn(

        "prediction_timestamp",

        prediction_timestamp

    )

)


# ============================================================
# 04.01.04 ORDEN FINAL DE COLUMNAS
# ============================================================

forecast_output_df = (

    forecast_output_df

    .select(

        "forecast_date",

        "store_id",

        "predicted_net_sales",

        "model_name",

        "model_version",

        "mlflow_run_id",

        "prediction_timestamp"

    )

)


# ============================================================
# 04.01.05 VALIDACIONES ESTRUCTURALES
# ============================================================

forecast_count = (
    forecast_output_df.count()
)


forecast_stores = (

    forecast_output_df

    .select(
        "store_id"
    )

    .distinct()

    .count()

)


forecast_dates = (

    forecast_output_df

    .select(
        "forecast_date"
    )

    .distinct()

    .count()

)


model_versions = (

    forecast_output_df

    .select(
        "model_version"
    )

    .distinct()

    .collect()

)


run_ids = (

    forecast_output_df

    .select(
        "mlflow_run_id"
    )

    .distinct()

    .collect()

)


# ============================================================
# 04.01.06 VALIDACIÓN DE CALIDAD
# ============================================================

quality_stats = (

    forecast_output_df

    .agg(

        F.sum(

            F.col(
                "forecast_date"
            )
            .isNull()
            .cast("int")

        ).alias(
            "forecast_date_nulls"
        ),

        F.sum(

            F.col(
                "store_id"
            )
            .isNull()
            .cast("int")

        ).alias(
            "store_id_nulls"
        ),

        F.sum(

            F.col(
                "predicted_net_sales"
            )
            .isNull()
            .cast("int")

        ).alias(
            "prediction_nulls"
        ),

        F.sum(

            F.col(
                "model_name"
            )
            .isNull()
            .cast("int")

        ).alias(
            "model_name_nulls"
        ),

        F.sum(

            F.col(
                "model_version"
            )
            .isNull()
            .cast("int")

        ).alias(
            "model_version_nulls"
        ),

        F.sum(

            F.col(
                "mlflow_run_id"
            )
            .isNull()
            .cast("int")

        ).alias(
            "run_id_nulls"
        ),

        F.sum(

            F.col(
                "prediction_timestamp"
            )
            .isNull()
            .cast("int")

        ).alias(
            "timestamp_nulls"
        )

    )

    .first()

)


total_nulls = sum(

    int(
        quality_stats[column] or 0
    )

    for column in quality_stats.__fields__

)


# ============================================================
# 04.01.07 DUPLICADOS
# ============================================================

duplicate_count = (

    forecast_output_df

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


# ============================================================
# 04.01.08 RESULTADOS
# ============================================================

print("=" * 80)
print("DATASET FINAL DE FORECAST")
print("=" * 80)


print(
    f"Fecha forecast: "
    f"{target_date.date()}"
)


print(
    f"Registros:      "
    f"{forecast_count}"
)


print(
    f"Tiendas:        "
    f"{forecast_stores}"
)


print(
    f"Fechas:         "
    f"{forecast_dates}"
)


print(
    f"Model Version:  "
    f"{MODEL_VERSION}"
)


print(
    f"Run ID:         "
    f"{MODEL_RUN_ID}"
)


print(
    f"Duplicados:     "
    f"{duplicate_count}"
)


print(
    f"NULLs:          "
    f"{total_nulls}"
)


# ============================================================
# 04.01.09 VALIDACIÓN FINAL
# ============================================================

if forecast_count != EXPECTED_STORES:

    raise RuntimeError(

        "El número de registros del forecast "
        "no coincide con las tiendas esperadas."

    )


if forecast_stores != EXPECTED_STORES:

    raise RuntimeError(

        "El número de tiendas únicas "
        "del forecast es incorrecto."

    )


if forecast_dates != 1:

    raise RuntimeError(

        "El dataset contiene más de una fecha de forecast."

    )


if len(model_versions) != 1:

    raise RuntimeError(

        "El dataset contiene más de una Model Version."

    )


if int(
    model_versions[0]["model_version"]
) != int(
    MODEL_VERSION
):

    raise RuntimeError(

        "La Model Version persistida no coincide "
        "con la utilizada en inference."

    )


if len(run_ids) != 1:

    raise RuntimeError(

        "El dataset contiene más de un MLflow Run ID."

    )


if (
    run_ids[0]["mlflow_run_id"]
    !=
    MODEL_RUN_ID
):

    raise RuntimeError(

        "El MLflow Run ID no coincide "
        "con el modelo cargado."

    )


if duplicate_count != 0:

    raise RuntimeError(

        "Se han detectado duplicados "
        "en el dataset de salida."

    )


if total_nulls != 0:

    raise RuntimeError(

        "Se han detectado NULLs "
        "en el dataset de salida."

    )


print()
print("=" * 80)
print("VALIDACIÓN FINAL")
print("=" * 80)


print(
    "OK - Dataset preparado directamente desde forecast_df"
)


print(
    "OK - 31 registros"
)


print(
    "OK - 31 tiendas"
)


print(
    "OK - Una única fecha de forecast"
)


print(
    f"OK - Model Version correcta: "
    f"{MODEL_VERSION}"
)


print(
    "OK - MLflow Run ID incluido"
)


print(
    "OK - Sin duplicados"
)


print(
    "OK - Sin NULLs"
)


print(
    "OK - Timestamp de generación incluido"
)


print()
print(
    "Dataset preparado para persistencia en Delta."
)


# ============================================================
# 04.01.10 VISUALIZACIÓN
# ============================================================

display(

    forecast_output_df

    .orderBy(
        "store_id"
    )

)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 04.02 Persistencia del forecast en Delta
# MAGIC
# MAGIC ### Objetivo
# MAGIC
# MAGIC Persistir el forecast generado en la tabla Delta:
# MAGIC
# MAGIC `retail_analytics.5_ml.sales_forecast_predictions`
# MAGIC
# MAGIC La tabla mantiene el histórico de predicciones generado por las distintas versiones del modelo.
# MAGIC
# MAGIC La clave lógica será:
# MAGIC
# MAGIC **forecast_date + store_id + model_version**
# MAGIC
# MAGIC Esto permite conservar, por ejemplo, forecasts realizados con **Version 1** y **Version 2** sin sobrescribir el histórico entre versiones.
# MAGIC
# MAGIC La persistencia se realizará mediante `MERGE`:
# MAGIC
# MAGIC - si ya existe una predicción para la misma fecha, tienda y versión, se actualizará
# MAGIC - si no existe, se insertará
# MAGIC
# MAGIC De esta forma el proceso será **idempotente** y podrá reejecutarse desde Lakeflow Jobs sin generar duplicados.
# MAGIC
# MAGIC Cada registro conservará además:
# MAGIC
# MAGIC - fecha del forecast
# MAGIC - tienda
# MAGIC - ventas predichas
# MAGIC - modelo utilizado
# MAGIC - versión del modelo
# MAGIC - Run ID de MLflow
# MAGIC - timestamp de generación

# COMMAND ----------

# DBTITLE 1,04.02 PERSISTENCIA DEL FORECAST EN DELTA
# ============================================================
# 04.02 PERSISTENCIA DEL FORECAST EN DELTA
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Persistir forecast_output_df en una tabla Delta histórica
# mediante MERGE.
#
# Clave lógica:
#
#   forecast_date + store_id + model_version
#
# Esto permite:
#
# - mantener forecasts generados por diferentes versiones
# - reejecutar el proceso sin generar duplicados
# - actualizar una predicción si se repite exactamente
#   la misma fecha + tienda + versión
#
# IMPORTANTE
# ------------------------------------------------------------
#
# - NO eliminamos históricos de Version 1.
# - Version 2 se añade como histórico independiente.
# - Se incluye mlflow_run_id para trazabilidad.
#
# ============================================================


from delta.tables import DeltaTable

from pyspark.sql import functions as F


# ============================================================
# 04.02.01 CONFIGURACIÓN
# ============================================================

FORECAST_TABLE = (
    "retail_analytics.5_ml.sales_forecast_predictions"
)


# ============================================================
# 04.02.02 VALIDACIONES PREVIAS
# ============================================================

if "forecast_output_df" not in globals():

    raise RuntimeError(

        "No existe forecast_output_df. "
        "Ejecuta primero el bloque 04.01."

    )


if "MODEL_VERSION" not in globals():

    raise RuntimeError(
        "No existe MODEL_VERSION."
    )


if "EXPECTED_STORES" not in globals():

    raise RuntimeError(
        "No existe EXPECTED_STORES."
    )


source_rows = (
    forecast_output_df.count()
)


source_stores = (

    forecast_output_df

    .select(
        "store_id"
    )

    .distinct()

    .count()

)


if source_rows != EXPECTED_STORES:

    raise RuntimeError(

        "El dataset que se intenta persistir "
        "no contiene el número esperado de registros."

    )


if source_stores != EXPECTED_STORES:

    raise RuntimeError(

        "El dataset que se intenta persistir "
        "no contiene el número esperado de tiendas."

    )


# ============================================================
# 04.02.03 COMPROBAMOS SI EXISTE LA TABLA
# ============================================================

table_exists = (
    spark.catalog.tableExists(
        FORECAST_TABLE
    )
)


print("=" * 80)
print("PERSISTENCIA DEL FORECAST EN DELTA")
print("=" * 80)


print(
    f"Tabla:          "
    f"{FORECAST_TABLE}"
)


print(
    f"Fecha forecast: "
    f"{target_date.date()}"
)


print(
    f"Model Version:  "
    f"{MODEL_VERSION}"
)


print(
    f"Registros:      "
    f"{source_rows}"
)


# ============================================================
# 04.02.04 PRIMERA EJECUCIÓN
# ============================================================
#
# Si la tabla todavía no existe:
#
# - la creamos directamente desde forecast_output_df
#
# ============================================================

if not table_exists:

    (

        forecast_output_df

        .write

        .format(
            "delta"
        )

        .mode(
            "overwrite"
        )

        .saveAsTable(
            FORECAST_TABLE
        )

    )


    print()
    print(
        "OK - Tabla de forecast creada"
    )


# ============================================================
# 04.02.05 TABLA YA EXISTENTE
# ============================================================

else:


    # ========================================================
    # 04.02.05.01 COMPATIBILIDAD DE SCHEMA
    # ========================================================
    #
    # Las primeras versiones de la tabla podían no contener
    # mlflow_run_id.
    #
    # No queremos recrear la tabla ni perder históricos.
    #
    # Si falta la columna, la añadimos.
    #
    # ========================================================

    existing_columns = set(

        spark.table(
            FORECAST_TABLE
        ).columns

    )


    if (
        "mlflow_run_id"
        not in existing_columns
    ):

        spark.sql(
            f"""
            ALTER TABLE {FORECAST_TABLE}
            ADD COLUMNS (
                mlflow_run_id STRING
            )
            """
        )


        print()
        print(
            "OK - Columna mlflow_run_id añadida "
            "a la tabla histórica"
        )


    # ========================================================
    # 04.02.05.02 MERGE
    # ========================================================

    forecast_delta = DeltaTable.forName(

        spark,

        FORECAST_TABLE

    )


    (

        forecast_delta

        .alias(
            "target"
        )

        .merge(

            forecast_output_df.alias(
                "source"
            ),

            """
            target.forecast_date = source.forecast_date
            AND target.store_id = source.store_id
            AND target.model_version = source.model_version
            """

        )

        .whenMatchedUpdate(

            set={

                "predicted_net_sales":
                    "source.predicted_net_sales",

                "model_name":
                    "source.model_name",

                "mlflow_run_id":
                    "source.mlflow_run_id",

                "prediction_timestamp":
                    "source.prediction_timestamp"

            }

        )

        .whenNotMatchedInsert(

            values={

                "forecast_date":
                    "source.forecast_date",

                "store_id":
                    "source.store_id",

                "predicted_net_sales":
                    "source.predicted_net_sales",

                "model_name":
                    "source.model_name",

                "model_version":
                    "source.model_version",

                "mlflow_run_id":
                    "source.mlflow_run_id",

                "prediction_timestamp":
                    "source.prediction_timestamp"

            }

        )

        .execute()

    )


    print()
    print(
        "OK - Forecast persistido mediante MERGE"
    )


# ============================================================
# 04.02.06 RECUPERAMOS EL FORECAST ACTUAL
# ============================================================

forecast_saved_df = (
    spark.table(
        FORECAST_TABLE
    )
)


current_forecast_df = (

    forecast_saved_df

    .filter(

        F.col(
            "forecast_date"
        )
        ==
        F.lit(
            target_date.date()
        )

    )

    .filter(

        F.col(
            "model_version"
        )
        ==
        F.lit(
            int(
                MODEL_VERSION
            )
        )

    )

)


# ============================================================
# 04.02.07 VALIDACIONES DEL FORECAST PERSISTIDO
# ============================================================

current_count = (
    current_forecast_df.count()
)


current_stores = (

    current_forecast_df

    .select(
        "store_id"
    )

    .distinct()

    .count()

)


duplicate_count = (

    current_forecast_df

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


current_versions = (

    current_forecast_df

    .select(
        "model_version"
    )

    .distinct()

    .collect()

)


current_run_ids = (

    current_forecast_df

    .select(
        "mlflow_run_id"
    )

    .distinct()

    .collect()

)


# ============================================================
# 04.02.08 VALIDACIÓN DEL HISTÓRICO
# ============================================================
#
# Queremos comprobar también cuántas versiones diferentes
# existen en la tabla completa.
#
# Version 1 puede seguir apareciendo y es correcto.
#
# ============================================================

historical_versions_df = (

    forecast_saved_df

    .groupBy(
        "model_version"
    )

    .agg(

        F.count(
            "*"
        ).alias(
            "rows"
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
        )

    )

    .orderBy(
        "model_version"
    )

)


# ============================================================
# 04.02.09 RESULTADOS
# ============================================================

print()
print("=" * 80)
print("VALIDACIÓN DEL FORECAST PERSISTIDO")
print("=" * 80)


print(
    f"Fecha:       "
    f"{target_date.date()}"
)


print(
    f"Versión:     "
    f"{MODEL_VERSION}"
)


print(
    f"Registros:   "
    f"{current_count}"
)


print(
    f"Tiendas:     "
    f"{current_stores}"
)


print(
    f"Duplicados:  "
    f"{duplicate_count}"
)


# ============================================================
# 04.02.10 VALIDACIÓN FINAL
# ============================================================

if current_count != EXPECTED_STORES:

    raise RuntimeError(

        "El número de forecasts persistidos "
        "no coincide con el esperado."

    )


if current_stores != EXPECTED_STORES:

    raise RuntimeError(

        "La cobertura de tiendas persistida "
        "no coincide con la esperada."

    )


if duplicate_count != 0:

    raise RuntimeError(

        "Se han detectado duplicados "
        "en el forecast persistido."

    )


if len(
    current_versions
) != 1:

    raise RuntimeError(

        "Se ha detectado más de una versión "
        "en el forecast actual."

    )


if int(
    current_versions[0][
        "model_version"
    ]
) != int(
    MODEL_VERSION
):

    raise RuntimeError(

        "La versión persistida no coincide "
        "con la utilizada en inference."

    )


if len(
    current_run_ids
) != 1:

    raise RuntimeError(

        "Existe más de un MLflow Run ID "
        "para el forecast actual."

    )


if (
    current_run_ids[0][
        "mlflow_run_id"
    ]
    !=
    MODEL_RUN_ID
):

    raise RuntimeError(

        "El MLflow Run ID persistido "
        "no coincide con el modelo utilizado."

    )


print()
print("=" * 80)
print("VALIDACIÓN FINAL")
print("=" * 80)


print(
    "OK - Forecast persistido en Delta"
)


print(
    "OK - MERGE idempotente aplicado"
)


print(
    "OK - 31 registros persistidos"
)


print(
    "OK - 31 tiendas"
)


print(
    "OK - Sin duplicados"
)


print(
    f"OK - Model Version correcta: "
    f"{MODEL_VERSION}"
)


print(
    "OK - MLflow Run ID correcto"
)


print(
    "OK - Históricos de otras versiones preservados"
)


print()
print(
    "Persistencia del forecast completada."
)


# ============================================================
# 04.02.11 HISTÓRICO POR VERSIÓN
# ============================================================

print()
print("-" * 80)
print("HISTÓRICO POR MODEL VERSION")
print("-" * 80)


display(
    historical_versions_df
)


# ============================================================
# 04.02.12 FORECAST ACTUAL
# ============================================================

display(

    current_forecast_df

    .orderBy(
        "store_id"
    )

)

# COMMAND ----------

# MAGIC %md
# MAGIC # 05. VALIDACIÓN DE LA TABLA DE SCORING
# MAGIC
# MAGIC ## Objetivo
# MAGIC
# MAGIC Validar la integridad de la tabla Delta utilizada como histórico de predicciones:
# MAGIC
# MAGIC `retail_analytics.5_ml.sales_forecast_predictions`
# MAGIC
# MAGIC La tabla puede contener forecasts generados por distintas versiones del modelo, por lo que las validaciones deben preservar correctamente el histórico.
# MAGIC
# MAGIC Se comprobará:
# MAGIC
# MAGIC - número total de forecasts almacenados
# MAGIC - fechas de forecast disponibles
# MAGIC - tiendas disponibles
# MAGIC - versiones del modelo utilizadas
# MAGIC - unicidad de `forecast_date + store_id + model_version`
# MAGIC - cobertura de tiendas por fecha y versión
# MAGIC - valores NULL en columnas obligatorias
# MAGIC - trazabilidad mediante `mlflow_run_id`
# MAGIC - predicciones negativas
# MAGIC - estadísticas de las predicciones
# MAGIC - consistencia específica de la Version 2 actualmente utilizada
# MAGIC
# MAGIC Las versiones antiguas se conservarán como histórico.
# MAGIC
# MAGIC La columna `mlflow_run_id` fue incorporada posteriormente al diseño de la tabla, por lo que los registros históricos anteriores pueden no disponer de este dato. Esta ausencia no se considerará un error para versiones antiguas.
# MAGIC
# MAGIC El objetivo es garantizar que la tabla de scoring mantiene una estructura consistente y trazable antes de continuar con la automatización del proceso de inference.

# COMMAND ----------

# DBTITLE 1,05.01 VALIDACIÓN GLOBAL DE LA TABLA DE SCORING
# ============================================================
# 05.01 VALIDACIÓN GLOBAL DE LA TABLA DE SCORING
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Validar la integridad global del histórico de forecasts.
#
# La tabla puede contener distintas Model Versions.
#
# Validaremos:
#
# - estructura
# - duplicados
# - NULLs obligatorios
# - predicciones negativas
# - cobertura por fecha y versión
# - trazabilidad de Version 2
#
# IMPORTANTE
# ------------------------------------------------------------
#
# mlflow_run_id fue incorporado después de Version 1.
#
# Por tanto:
#
# - NULL mlflow_run_id en históricos Version 1 -> permitido
# - NULL mlflow_run_id en Version 2           -> ERROR
#
# ============================================================


from pyspark.sql import functions as F


# ============================================================
# 05.01.01 CONFIGURACIÓN
# ============================================================

FORECAST_TABLE = (
    "retail_analytics.5_ml.sales_forecast_predictions"
)


# ============================================================
# 05.01.02 VALIDACIONES PREVIAS
# ============================================================

if not spark.catalog.tableExists(
    FORECAST_TABLE
):

    raise RuntimeError(

        "No existe la tabla de scoring: "
        f"{FORECAST_TABLE}"

    )


required_columns = [

    "forecast_date",
    "store_id",
    "predicted_net_sales",
    "model_name",
    "model_version",
    "mlflow_run_id",
    "prediction_timestamp"

]


forecast_history_df = (
    spark.table(
        FORECAST_TABLE
    )
)


available_columns = set(
    forecast_history_df.columns
)


missing_columns = [

    column

    for column in required_columns

    if column not in available_columns

]


if missing_columns:

    raise RuntimeError(

        "Faltan columnas obligatorias en la tabla "
        f"de scoring: {missing_columns}"

    )


# ============================================================
# 05.01.03 MÉTRICAS GENERALES
# ============================================================

general_stats = (

    forecast_history_df

    .agg(

        F.count("*").alias(
            "rows"
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

        F.min(
            "forecast_date"
        ).alias(
            "min_forecast_date"
        ),

        F.max(
            "forecast_date"
        ).alias(
            "max_forecast_date"
        )

    )

    .first()

)


total_rows = int(
    general_stats["rows"]
)


total_dates = int(
    general_stats["forecast_dates"]
)


total_stores = int(
    general_stats["stores"]
)


total_model_versions = int(
    general_stats["model_versions"]
)


min_forecast_date = (
    general_stats["min_forecast_date"]
)


max_forecast_date = (
    general_stats["max_forecast_date"]
)


# ============================================================
# 05.01.04 DUPLICADOS
# ============================================================
#
# Clave lógica:
#
# forecast_date + store_id + model_version
#
# ============================================================

duplicate_keys_df = (

    forecast_history_df

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


duplicate_count = (
    duplicate_keys_df.count()
)


# ============================================================
# 05.01.05 NULLS EN COLUMNAS OBLIGATORIAS
# ============================================================
#
# mlflow_run_id NO se incluye aquí porque puede ser NULL
# legítimamente en históricos anteriores.
#
# ============================================================

mandatory_columns = [

    "forecast_date",
    "store_id",
    "predicted_net_sales",
    "model_name",
    "model_version",
    "prediction_timestamp"

]


mandatory_null_expressions = [

    F.sum(

        F.col(column)
        .isNull()
        .cast("int")

    ).alias(
        column
    )

    for column in mandatory_columns

]


mandatory_null_stats = (

    forecast_history_df

    .agg(
        *mandatory_null_expressions
    )

    .first()

)


mandatory_nulls = {

    column:
        int(
            mandatory_null_stats[column] or 0
        )

    for column in mandatory_columns

}


total_mandatory_nulls = sum(
    mandatory_nulls.values()
)


# ============================================================
# 05.01.06 NULLS DE MLFLOW_RUN_ID
# ============================================================

mlflow_run_id_nulls_total = (

    forecast_history_df

    .filter(
        F.col("mlflow_run_id").isNull()
    )

    .count()

)


# ------------------------------------------------------------
# Para Version 2, mlflow_run_id es obligatorio.
# ------------------------------------------------------------

current_version_run_id_nulls = (

    forecast_history_df

    .filter(

        F.col("model_version")
        ==
        F.lit(
            int(MODEL_VERSION)
        )

    )

    .filter(
        F.col("mlflow_run_id").isNull()
    )

    .count()

)


# ============================================================
# 05.01.07 PREDICCIONES NEGATIVAS / NAN
# ============================================================

prediction_quality = (

    forecast_history_df

    .agg(

        F.sum(

            (
                F.col(
                    "predicted_net_sales"
                )
                <
                0
            )
            .cast("int")

        ).alias(
            "negative_predictions"
        ),

        F.sum(

            F.isnan(
                F.col(
                    "predicted_net_sales"
                )
            )
            .cast("int")

        ).alias(
            "nan_predictions"
        )

    )

    .first()

)


negative_predictions = int(
    prediction_quality[
        "negative_predictions"
    ] or 0
)


nan_predictions = int(
    prediction_quality[
        "nan_predictions"
    ] or 0
)


# ============================================================
# 05.01.08 COBERTURA POR FECHA Y MODEL VERSION
# ============================================================

coverage_df = (

    forecast_history_df

    .groupBy(

        "forecast_date",
        "model_version"

    )

    .agg(

        F.count("*").alias(
            "predictions"
        ),

        F.countDistinct(
            "store_id"
        ).alias(
            "stores"
        ),

        F.countDistinct(
            "mlflow_run_id"
        ).alias(
            "mlflow_runs"
        ),

        F.round(

            F.avg(
                "predicted_net_sales"
            ),

            2

        ).alias(
            "avg_predicted_sales"
        ),

        F.round(

            F.min(
                "predicted_net_sales"
            ),

            2

        ).alias(
            "min_predicted_sales"
        ),

        F.round(

            F.max(
                "predicted_net_sales"
            ),

            2

        ).alias(
            "max_predicted_sales"
        )

    )

    .orderBy(

        F.desc(
            "forecast_date"
        ),

        F.desc(
            "model_version"
        )

    )

)


# ============================================================
# 05.01.09 RESUMEN POR MODEL VERSION
# ============================================================

versions_summary_df = (

    forecast_history_df

    .groupBy(
        "model_version"
    )

    .agg(

        F.count("*").alias(
            "rows"
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

        F.sum(

            F.col(
                "mlflow_run_id"
            )
            .isNull()
            .cast("int")

        ).alias(
            "missing_mlflow_run_id"
        ),

        F.min(
            "forecast_date"
        ).alias(
            "min_date"
        ),

        F.max(
            "forecast_date"
        ).alias(
            "max_date"
        )

    )

    .orderBy(
        "model_version"
    )

)


# ============================================================
# 05.01.10 VALIDACIÓN ESPECÍFICA DE LA VERSIÓN ACTUAL
# ============================================================

current_version_df = (

    forecast_history_df

    .filter(

        F.col(
            "model_version"
        )
        ==
        F.lit(
            int(
                MODEL_VERSION
            )
        )

    )

)


current_version_rows = (
    current_version_df.count()
)


current_version_dates = (

    current_version_df

    .select(
        "forecast_date"
    )

    .distinct()

    .count()

)


current_version_stores = (

    current_version_df

    .select(
        "store_id"
    )

    .distinct()

    .count()

)


current_version_run_ids = (

    current_version_df

    .select(
        "mlflow_run_id"
    )

    .distinct()

    .collect()

)


# ============================================================
# 05.01.11 RESULTADOS GENERALES
# ============================================================

print("=" * 80)
print("VALIDACIÓN GLOBAL DE LA TABLA DE SCORING")
print("=" * 80)


print(
    f"Tabla:                "
    f"{FORECAST_TABLE}"
)


print(
    f"Registros:            "
    f"{total_rows:,}"
)


print(
    f"Fechas forecast:      "
    f"{total_dates:,}"
)


print(
    f"Periodo:              "
    f"{min_forecast_date} "
    f"-> "
    f"{max_forecast_date}"
)


print(
    f"Tiendas históricas:   "
    f"{total_stores:,}"
)


print(
    f"Versiones modelo:     "
    f"{total_model_versions:,}"
)


print()
print("-" * 80)
print("CALIDAD GLOBAL")
print("-" * 80)


print(
    f"Duplicados:                 "
    f"{duplicate_count:,}"
)


print(
    f"NULLs obligatorios:         "
    f"{total_mandatory_nulls:,}"
)


print(
    f"NULL mlflow_run_id totales: "
    f"{mlflow_run_id_nulls_total:,}"
)


print(
    f"NULL Run ID Version "
    f"{MODEL_VERSION}:        "
    f"{current_version_run_id_nulls:,}"
)


print(
    f"Predicciones NaN:           "
    f"{nan_predictions:,}"
)


print(
    f"Predicciones negativas:     "
    f"{negative_predictions:,}"
)


# ============================================================
# 05.01.12 VERSION ACTUAL
# ============================================================

print()
print("-" * 80)
print(f"MODEL VERSION {MODEL_VERSION}")
print("-" * 80)


print(
    f"Registros:      "
    f"{current_version_rows:,}"
)


print(
    f"Fechas:         "
    f"{current_version_dates:,}"
)


print(
    f"Tiendas:        "
    f"{current_version_stores:,}"
)


print(
    f"MLflow Run ID:  "
    f"{MODEL_RUN_ID}"
)


# ============================================================
# 05.01.13 VALIDACIONES ESTRICTAS
# ============================================================

if total_rows == 0:

    raise RuntimeError(
        "La tabla de scoring está vacía."
    )


if duplicate_count != 0:

    raise RuntimeError(

        "Existen duplicados según la clave lógica "
        "forecast_date + store_id + model_version."

    )


if total_mandatory_nulls != 0:

    raise RuntimeError(

        "Existen NULLs en columnas obligatorias "
        "de la tabla de scoring."

    )


if nan_predictions != 0:

    raise RuntimeError(

        "Existen predicciones NaN "
        "en la tabla de scoring."

    )


if negative_predictions != 0:

    raise RuntimeError(

        "Existen predicciones negativas "
        "en la tabla de scoring."

    )


if current_version_rows == 0:

    raise RuntimeError(

        f"No existen forecasts para "
        f"Model Version {MODEL_VERSION}."

    )


if current_version_run_id_nulls != 0:

    raise RuntimeError(

        f"Existen forecasts de Version "
        f"{MODEL_VERSION} sin mlflow_run_id."

    )


if len(
    current_version_run_ids
) != 1:

    raise RuntimeError(

        f"Version {MODEL_VERSION} contiene "
        "más de un MLflow Run ID."

    )


if (
    current_version_run_ids[0][
        "mlflow_run_id"
    ]
    !=
    MODEL_RUN_ID
):

    raise RuntimeError(

        "El Run ID registrado en la tabla "
        "no coincide con el modelo utilizado."

    )


# ============================================================
# 05.01.14 VALIDACIÓN FINAL
# ============================================================

print()
print("=" * 80)
print("VALIDACIÓN FINAL")
print("=" * 80)


print(
    "OK - Tabla de scoring disponible"
)


print(
    "OK - Schema esperado disponible"
)


print(
    "OK - Sin duplicados"
)


print(
    "OK - Sin NULLs en columnas obligatorias"
)


if mlflow_run_id_nulls_total > 0:

    print(

        "INFO - Existen históricos antiguos sin "
        "mlflow_run_id; se conservan correctamente"

    )

else:

    print(
        "OK - Todos los registros contienen mlflow_run_id"
    )


print(
    f"OK - Version {MODEL_VERSION} "
    f"tiene trazabilidad MLflow completa"
)


print(
    "OK - Sin predicciones NaN"
)


print(
    "OK - Sin predicciones negativas"
)


print(
    "OK - Histórico multiversión preservado"
)


print()
print(
    "Tabla de scoring validada correctamente."
)


# ============================================================
# 05.01.15 RESUMEN POR MODEL VERSION
# ============================================================

print()
print("-" * 80)
print("RESUMEN POR MODEL VERSION")
print("-" * 80)


display(
    versions_summary_df
)


# ============================================================
# 05.01.16 COBERTURA POR FECHA Y VERSIÓN
# ============================================================

print()
print("-" * 80)
print("COBERTURA POR FECHA Y MODEL VERSION")
print("-" * 80)


display(
    coverage_df
)

# COMMAND ----------

# MAGIC %md
# MAGIC # 06. AUTOMATIZACIÓN DEL PROCESO DE INFERENCE
# MAGIC
# MAGIC ## Objetivo
# MAGIC
# MAGIC Preparar el proceso de **Inference / Scoring** para su ejecución automática mediante **Lakeflow Jobs**.
# MAGIC
# MAGIC Hasta este punto hemos construido y validado manualmente todo el flujo:
# MAGIC
# MAGIC **Modelo registrado → Preparación de features → Inference → Sanity Checks → Persistencia Delta → Validación**
# MAGIC
# MAGIC Ahora prepararemos el notebook para que pueda ejecutarse de forma repetible sin modificar código entre ejecuciones.
# MAGIC
# MAGIC El proceso automatizado deberá:
# MAGIC
# MAGIC 1. identificar el modelo registrado en **Unity Catalog**
# MAGIC 2. utilizar la **Model Version** configurada para producción
# MAGIC 3. cargar el histórico de features disponible
# MAGIC 4. determinar automáticamente la siguiente fecha de forecast
# MAGIC 5. generar una predicción por tienda
# MAGIC 6. validar técnicamente las predicciones
# MAGIC 7. persistir el resultado mediante `MERGE`
# MAGIC 8. mantener trazabilidad mediante `model_version` y `mlflow_run_id`
# MAGIC 9. garantizar la idempotencia ante reejecuciones
# MAGIC
# MAGIC ### Principio de diseño
# MAGIC
# MAGIC El notebook no realizará:
# MAGIC
# MAGIC - entrenamiento
# MAGIC - tuning
# MAGIC - selección de hiperparámetros
# MAGIC - evaluación sobre TEST
# MAGIC - evaluación sobre OOT
# MAGIC
# MAGIC El modelo se consumirá directamente desde **Unity Catalog**.
# MAGIC
# MAGIC La ejecución quedará preparada para integrarse posteriormente como una tarea dentro del pipeline de **Lakeflow Jobs**.

# COMMAND ----------

# DBTITLE 1,06.01 PARÁMETROS DEL PROCESO DE INFERENCE
# ============================================================
# 06.01 PARÁMETROS DEL PROCESO DE INFERENCE
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Centralizar y validar los parámetros necesarios para
# automatizar el proceso de inference.
#
# Reutilizamos la configuración ya definida anteriormente:
#
# - MODEL_NAME
# - MODEL_VERSION
# - MODEL_URI
# - MODEL_RUN_ID
# - FEATURE_TABLE
# - FORECAST_TABLE
#
# Evitamos volver a hardcodear estos valores.
#
# ============================================================


from pyspark.sql import functions as F


# ============================================================
# 06.01.01 VALIDACIÓN DE VARIABLES DEL PROCESO
# ============================================================

required_variables = [

    "MODEL_NAME",
    "MODEL_VERSION",
    "MODEL_URI",
    "MODEL_RUN_ID",
    "FEATURE_TABLE",
    "FORECAST_TABLE"

]


missing_variables = [

    variable_name

    for variable_name in required_variables

    if variable_name not in globals()

]


if missing_variables:

    raise RuntimeError(

        "Faltan variables necesarias para automatizar "
        f"el proceso de inference: {missing_variables}"

    )


# ============================================================
# 06.01.02 VALIDACIÓN DE LA FEATURE TABLE
# ============================================================

if not spark.catalog.tableExists(
    FEATURE_TABLE
):

    raise RuntimeError(

        "No existe la tabla de features ML: "
        f"{FEATURE_TABLE}"

    )


feature_source_df = (

    spark.table(
        FEATURE_TABLE
    )

)


# ============================================================
# 06.01.03 NÚMERO DE TIENDAS
# ============================================================
#
# No hardcodeamos 31.
#
# Lo obtenemos directamente de la Feature Table.
#
# ============================================================

EXPECTED_STORES = (

    feature_source_df

    .select(
        "store_id"
    )

    .distinct()

    .count()

)


if EXPECTED_STORES <= 0:

    raise RuntimeError(

        "No se han encontrado tiendas "
        "en la Feature Table."

    )


# ============================================================
# 06.01.04 ÚLTIMA FECHA DISPONIBLE
# ============================================================

latest_feature_date = (

    feature_source_df

    .agg(

        F.max(
            "sale_date"
        ).alias(
            "latest_date"
        )

    )

    .first()[
        "latest_date"
    ]

)


if latest_feature_date is None:

    raise RuntimeError(

        "No se ha podido determinar la última fecha "
        "disponible en la Feature Table."

    )


# ============================================================
# 06.01.05 FECHA OBJETIVO ESPERADA
# ============================================================
#
# Forecast one-step ahead:
#
# última fecha disponible + 1 día
#
# ============================================================

expected_forecast_date = (

    feature_source_df

    .select(

        F.date_add(

            F.lit(
                latest_feature_date
            ),

            1

        ).alias(
            "forecast_date"
        )

    )

    .first()[
        "forecast_date"
    ]

)


# ============================================================
# 06.01.06 VALIDACIÓN DE CONSISTENCIA CON EL FORECAST ACTUAL
# ============================================================
#
# Si target_date ya fue calculada durante esta ejecución,
# debe coincidir con expected_forecast_date.
#
# ============================================================

if "target_date" in globals():

    if (
        target_date.date()
        !=
        expected_forecast_date
    ):

        raise RuntimeError(

            "La fecha de forecast calculada durante "
            "la ejecución no coincide con la esperada.\n"
            f"Esperada: {expected_forecast_date}\n"
            f"Actual:   {target_date.date()}"

        )


# ============================================================
# 06.01.07 VALIDACIÓN DEL MODELO
# ============================================================

if int(
    MODEL_VERSION
) <= 0:

    raise RuntimeError(

        "MODEL_VERSION debe ser mayor que 0."

    )


if not MODEL_RUN_ID:

    raise RuntimeError(

        "MODEL_RUN_ID no puede estar vacío."

    )


expected_model_uri = (

    f"models:/{MODEL_NAME}/{MODEL_VERSION}"

)


if MODEL_URI != expected_model_uri:

    raise RuntimeError(

        "MODEL_URI no coincide con MODEL_NAME "
        "y MODEL_VERSION.\n"
        f"Esperado: {expected_model_uri}\n"
        f"Actual:   {MODEL_URI}"

    )


# ============================================================
# 06.01.08 VALIDACIÓN DE LA TABLA DE SCORING
# ============================================================

if not spark.catalog.tableExists(
    FORECAST_TABLE
):

    raise RuntimeError(

        "No existe la tabla de scoring: "
        f"{FORECAST_TABLE}"

    )


# ============================================================
# 06.01.09 RESUMEN DEL PROCESO
# ============================================================

print("=" * 80)
print("PARÁMETROS DEL PROCESO DE INFERENCE")
print("=" * 80)


print(
    f"Modelo:                "
    f"{MODEL_NAME}"
)


print(
    f"Model Version:         "
    f"{MODEL_VERSION}"
)


print(
    f"Model URI:             "
    f"{MODEL_URI}"
)


print(
    f"MLflow Run ID:         "
    f"{MODEL_RUN_ID}"
)


print(
    f"Feature Table:         "
    f"{FEATURE_TABLE}"
)


print(
    f"Forecast Table:        "
    f"{FORECAST_TABLE}"
)


print(
    f"Tiendas esperadas:     "
    f"{EXPECTED_STORES}"
)


print(
    f"Última fecha features: "
    f"{latest_feature_date}"
)


print(
    f"Próximo forecast:      "
    f"{expected_forecast_date}"
)


# ============================================================
# 06.01.10 VALIDACIÓN FINAL
# ============================================================

print()
print("=" * 80)
print("VALIDACIÓN FINAL")
print("=" * 80)


print(
    "OK - Configuración del modelo disponible"
)


print(
    f"OK - Model Version configurada: "
    f"{MODEL_VERSION}"
)


print(
    "OK - Model URI consistente"
)


print(
    "OK - MLflow Run ID disponible"
)


print(
    "OK - Feature Table disponible"
)


print(
    "OK - Tabla de scoring disponible"
)


print(
    f"OK - {EXPECTED_STORES} tiendas detectadas"
)


print(
    f"OK - Última fecha disponible: "
    f"{latest_feature_date}"
)


print(
    f"OK - Próxima fecha de forecast: "
    f"{expected_forecast_date}"
)


if "target_date" in globals():

    print(
        "OK - Fecha objetivo de la ejecución "
        "coincide con la fecha esperada"
    )


print(
    "OK - No se ha realizado entrenamiento"
)


print(
    "OK - TEST no utilizado"
)


print(
    "OK - OOT no utilizado"
)


print()
print(
    "Parámetros del proceso de inference "
    "validados correctamente."
)

# COMMAND ----------

# DBTITLE 1,06.02 VALIDACIONES PREVIAS A LA EJECUCIÓN
# ============================================================
# 06.02 VALIDACIONES PREVIAS A LA EJECUCIÓN
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Comprobar que el proceso de inference puede ejecutarse
# de forma segura antes de generar nuevas predicciones.
#
# Validamos:
#
# - existencia de Feature Table
# - existencia de Forecast Table
# - cobertura de tiendas
# - histórico mínimo por tienda
# - última fecha disponible
# - disponibilidad de la Model Version
# - estado READY en Unity Catalog
# - trazabilidad con MLflow
# - carga real del modelo Spark ML
#
# Si alguna validación falla, lanzamos una excepción para
# que Lakeflow Jobs marque correctamente la ejecución.
#
# ============================================================


import os

import mlflow
import mlflow.spark

from mlflow import MlflowClient

from pyspark.sql import functions as F


# ============================================================
# 06.02.01 VALIDACIONES DE VARIABLES
# ============================================================

required_variables = [

    "FEATURE_TABLE",
    "FORECAST_TABLE",

    "MODEL_NAME",
    "MODEL_VERSION",
    "MODEL_URI",
    "MODEL_RUN_ID",

    "EXPECTED_STORES",
    "MLFLOW_DFS_TMP"

]


missing_variables = [

    variable_name

    for variable_name in required_variables

    if variable_name not in globals()

]


if missing_variables:

    raise RuntimeError(

        "Faltan variables necesarias para ejecutar "
        f"las validaciones previas: {missing_variables}"

    )


# ============================================================
# 06.02.02 CONFIGURACIÓN DE MLFLOW / UNITY CATALOG
# ============================================================

mlflow.set_registry_uri(
    "databricks-uc"
)


client = MlflowClient()


# ============================================================
# 06.02.03 FEATURE TABLE
# ============================================================

if not spark.catalog.tableExists(
    FEATURE_TABLE
):

    raise RuntimeError(

        "No existe la Feature Table: "
        f"{FEATURE_TABLE}"

    )


features_df = (

    spark.table(
        FEATURE_TABLE
    )

)


print(
    f"OK - Feature Table disponible: "
    f"{FEATURE_TABLE}"
)


# ============================================================
# 06.02.04 FORECAST TABLE
# ============================================================

if not spark.catalog.tableExists(
    FORECAST_TABLE
):

    raise RuntimeError(

        "No existe la tabla de scoring: "
        f"{FORECAST_TABLE}"

    )


print(
    f"OK - Forecast Table disponible: "
    f"{FORECAST_TABLE}"
)


# ============================================================
# 06.02.05 COBERTURA DE TIENDAS
# ============================================================

available_stores = (

    features_df

    .select(
        "store_id"
    )

    .distinct()

    .count()

)


if available_stores != EXPECTED_STORES:

    raise RuntimeError(

        "Número de tiendas inesperado. "
        f"Esperadas: {EXPECTED_STORES} | "
        f"Disponibles: {available_stores}"

    )


print(
    f"OK - Cobertura de tiendas correcta: "
    f"{available_stores}"
)


# ============================================================
# 06.02.06 ÚLTIMA FECHA DISPONIBLE
# ============================================================

last_available_date_check = (

    features_df

    .agg(

        F.max(
            "sale_date"
        ).alias(
            "last_date"
        )

    )

    .first()[
        "last_date"
    ]

)


if last_available_date_check is None:

    raise RuntimeError(

        "No existen fechas disponibles "
        "en la Feature Table."

    )


print(
    f"OK - Última fecha disponible: "
    f"{last_available_date_check}"
)


# ============================================================
# 06.02.07 HISTÓRICO MÍNIMO POR TIENDA
# ============================================================
#
# El modelo utiliza:
#
# - lag_28
# - rolling_mean_28
#
# Por tanto necesitamos al menos 28 observaciones históricas
# por tienda.
#
# ============================================================

store_history_df = (

    features_df

    .groupBy(
        "store_id"
    )

    .agg(

        F.count("*").alias(
            "history_rows"
        )

    )

)


stores_without_history_df = (

    store_history_df

    .filter(
        F.col("history_rows") < 28
    )

)


stores_without_history = (
    stores_without_history_df.count()
)


if stores_without_history > 0:

    display(
        stores_without_history_df
    )

    raise RuntimeError(

        f"{stores_without_history} tiendas "
        "no tienen al menos 28 observaciones históricas."

    )


print(
    "OK - Todas las tiendas tienen histórico suficiente"
)


# ============================================================
# 06.02.08 VALIDAMOS MODEL VERSION EN UNITY CATALOG
# ============================================================

model_version_info = (

    client.get_model_version(

        name=
            MODEL_NAME,

        version=
            MODEL_VERSION

    )

)


if str(
    model_version_info.version
) != str(
    MODEL_VERSION
):

    raise RuntimeError(

        "La Model Version localizada no coincide "
        "con MODEL_VERSION."

    )


if model_version_info.status != "READY":

    raise RuntimeError(

        "La Model Version no está READY. "
        f"Estado actual: {model_version_info.status}"

    )


print(
    f"OK - Model Version {MODEL_VERSION} disponible"
)


print(
    "OK - Model Version en estado READY"
)


# ============================================================
# 06.02.09 VALIDAMOS RUN ID
# ============================================================

registry_run_id = (
    model_version_info.run_id
)


if registry_run_id != MODEL_RUN_ID:

    raise RuntimeError(

        "La Model Version no pertenece al MLflow Run ID "
        "esperado.\n"
        f"Esperado: {MODEL_RUN_ID}\n"
        f"Actual:   {registry_run_id}"

    )


print(
    "OK - Model Version asociada al MLflow Run ID correcto"
)


# ============================================================
# 06.02.10 VALIDAMOS MODEL URI
# ============================================================

expected_model_uri = (

    f"models:/{MODEL_NAME}/{MODEL_VERSION}"

)


if MODEL_URI != expected_model_uri:

    raise RuntimeError(

        "MODEL_URI no coincide con la configuración esperada.\n"
        f"Esperado: {expected_model_uri}\n"
        f"Actual:   {MODEL_URI}"

    )


print(
    "OK - Model URI consistente"
)


# ============================================================
# 06.02.11 UC VOLUME TEMPORAL
# ============================================================

dbutils.fs.mkdirs(
    MLFLOW_DFS_TMP
)


os.environ[
    "MLFLOW_DFS_TMP"
] = MLFLOW_DFS_TMP


print(
    f"OK - UC Volume temporal disponible: "
    f"{MLFLOW_DFS_TMP}"
)


# ============================================================
# 06.02.12 CARGA REAL DEL MODELO
# ============================================================
#
# No basta con comprobar que la Model Version existe.
#
# Intentamos cargarla realmente para detectar:
#
# - problemas de permisos
# - problemas de artefactos
# - incompatibilidades de Spark ML
#
# ============================================================

try:

    validation_model = (

        mlflow.spark.load_model(

            MODEL_URI,

            dfs_tmpdir=
                MLFLOW_DFS_TMP

        )

    )


except Exception as error:

    raise RuntimeError(

        "No se puede cargar el modelo Spark ML "
        "desde Unity Catalog: "
        f"{MODEL_URI}"

    ) from error


if validation_model is None:

    raise RuntimeError(

        "La carga del modelo no ha devuelto "
        "un objeto válido."

    )


print(
    f"OK - Modelo Spark ML cargado correctamente: "
    f"{MODEL_URI}"
)


# ============================================================
# 06.02.13 VALIDACIÓN DE FECHA OBJETIVO
# ============================================================

expected_date_check = (

    features_df

    .select(

        F.date_add(

            F.lit(
                last_available_date_check
            ),

            1

        ).alias(
            "expected_date"
        )

    )

    .first()[
        "expected_date"
    ]

)


if "target_date" in globals():

    if (
        target_date.date()
        !=
        expected_date_check
    ):

        raise RuntimeError(

            "La fecha objetivo calculada durante inference "
            "no coincide con la fecha esperada.\n"
            f"Esperada: {expected_date_check}\n"
            f"Actual:   {target_date.date()}"

        )


print(
    f"OK - Próxima fecha de forecast: "
    f"{expected_date_check}"
)


# ============================================================
# 06.02.14 VALIDACIÓN DE FORECAST ACTUAL
# ============================================================
#
# Si forecast_output_df ya existe en la sesión, comprobamos
# que corresponde a la configuración actual.
#
# ============================================================

if "forecast_output_df" in globals():

    forecast_current_stats = (

        forecast_output_df

        .agg(

            F.count("*").alias(
                "rows"
            ),

            F.countDistinct(
                "store_id"
            ).alias(
                "stores"
            ),

            F.countDistinct(
                "forecast_date"
            ).alias(
                "dates"
            ),

            F.countDistinct(
                "model_version"
            ).alias(
                "versions"
            ),

            F.countDistinct(
                "mlflow_run_id"
            ).alias(
                "run_ids"
            )

        )

        .first()

    )


    if int(
        forecast_current_stats["rows"]
    ) != EXPECTED_STORES:

        raise RuntimeError(

            "El forecast actual no contiene "
            "el número esperado de filas."

        )


    if int(
        forecast_current_stats["stores"]
    ) != EXPECTED_STORES:

        raise RuntimeError(

            "El forecast actual no contiene "
            "la cobertura esperada de tiendas."

        )


    if int(
        forecast_current_stats["dates"]
    ) != 1:

        raise RuntimeError(

            "El forecast actual contiene "
            "más de una fecha."

        )


    if int(
        forecast_current_stats["versions"]
    ) != 1:

        raise RuntimeError(

            "El forecast actual contiene "
            "más de una Model Version."

        )


    if int(
        forecast_current_stats["run_ids"]
    ) != 1:

        raise RuntimeError(

            "El forecast actual contiene "
            "más de un MLflow Run ID."

        )


    print(
        "OK - Forecast actual consistente "
        "con la configuración del proceso"
    )


# ============================================================
# 06.02.15 VALIDACIÓN FINAL
# ============================================================

print()
print("=" * 80)
print("VALIDACIONES PREVIAS A LA EJECUCIÓN")
print("=" * 80)


print(
    "OK - Feature Table disponible"
)


print(
    "OK - Forecast Table disponible"
)


print(
    f"OK - {available_stores} tiendas disponibles"
)


print(
    "OK - Histórico mínimo suficiente"
)


print(
    f"OK - Última fecha disponible: "
    f"{last_available_date_check}"
)


print(
    f"OK - Model Version {MODEL_VERSION} disponible"
)


print(
    "OK - Model Version READY"
)


print(
    "OK - MLflow Run ID validado"
)


print(
    "OK - Model URI validado"
)


print(
    "OK - UC Volume temporal disponible"
)


print(
    "OK - Modelo Spark ML cargable desde Unity Catalog"
)


print(
    f"OK - Próximo forecast: "
    f"{expected_date_check}"
)


print(
    "OK - TEST no utilizado"
)


print(
    "OK - OOT no utilizado"
)


print()
print(
    "Todas las condiciones necesarias para ejecutar "
    "inference son correctas."
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 06.03 Validación de idempotencia
# MAGIC
# MAGIC El proceso de inference utiliza una operación `MERGE` sobre la tabla Delta de forecasts.
# MAGIC
# MAGIC La clave lógica utilizada es:
# MAGIC
# MAGIC **forecast_date + store_id + model_version**
# MAGIC
# MAGIC Esto permite ejecutar nuevamente el proceso para una misma fecha y versión del modelo sin generar registros duplicados.
# MAGIC
# MAGIC Si una predicción ya existe, se actualiza. Si no existe, se inserta.
# MAGIC
# MAGIC Esta característica permite que el pipeline pueda ser reejecutado de forma segura desde Lakeflow Jobs ante errores o reintentos.

# COMMAND ----------

# DBTITLE 1,06.03 VALIDACIÓN DE IDEMPOTENCIA
# ============================================================
# 06.03 VALIDACIÓN DE IDEMPOTENCIA
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Validar que el histórico de forecasts mantiene una salida
# idempotente.
#
# Clave lógica:
#
#   forecast_date + store_id + model_version
#
# Además validamos que el forecast actual:
#
# - contiene todas las tiendas esperadas
# - no tiene duplicados
# - corresponde a la Model Version actual
# - corresponde al MLflow Run ID actual
#
# ============================================================


from pyspark.sql import functions as F


# ============================================================
# 06.03.01 VALIDACIONES PREVIAS
# ============================================================

required_variables = [

    "FORECAST_TABLE",
    "MODEL_VERSION",
    "MODEL_RUN_ID",
    "EXPECTED_STORES",
    "target_date"

]


missing_variables = [

    variable_name

    for variable_name in required_variables

    if variable_name not in globals()

]


if missing_variables:

    raise RuntimeError(

        "Faltan variables necesarias para validar "
        f"la idempotencia: {missing_variables}"

    )


if not spark.catalog.tableExists(
    FORECAST_TABLE
):

    raise RuntimeError(

        "No existe la tabla de forecast: "
        f"{FORECAST_TABLE}"

    )


# ============================================================
# 06.03.02 LEEMOS LA TABLA DE FORECAST
# ============================================================

forecast_check_df = (

    spark.table(
        FORECAST_TABLE
    )

)


# ============================================================
# 06.03.03 DUPLICADOS GLOBALES
# ============================================================
#
# La clave lógica de la tabla es:
#
# forecast_date + store_id + model_version
#
# ============================================================

duplicate_keys_df = (

    forecast_check_df

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


duplicate_keys = (
    duplicate_keys_df.count()
)


# ============================================================
# 06.03.04 FORECAST ACTUAL
# ============================================================

current_forecast_df = (

    forecast_check_df

    .filter(

        F.col("forecast_date")
        ==
        F.lit(
            target_date.date()
        )

    )

    .filter(

        F.col("model_version")
        ==
        F.lit(
            int(
                MODEL_VERSION
            )
        )

    )

)


# ============================================================
# 06.03.05 MÉTRICAS DEL FORECAST ACTUAL
# ============================================================

current_stats = (

    current_forecast_df

    .agg(

        F.count("*").alias(
            "rows"
        ),

        F.countDistinct(
            "store_id"
        ).alias(
            "stores"
        ),

        F.countDistinct(
            "model_version"
        ).alias(
            "versions"
        ),

        F.countDistinct(
            "mlflow_run_id"
        ).alias(
            "run_ids"
        )

    )

    .first()

)


current_rows = int(
    current_stats["rows"]
)


current_stores = int(
    current_stats["stores"]
)


current_versions = int(
    current_stats["versions"]
)


current_run_ids = int(
    current_stats["run_ids"]
)


# ============================================================
# 06.03.06 RUN ID DEL FORECAST ACTUAL
# ============================================================

current_run_id_values = (

    current_forecast_df

    .select(
        "mlflow_run_id"
    )

    .distinct()

    .collect()

)


current_run_id = (

    current_run_id_values[0][
        "mlflow_run_id"
    ]

    if len(
        current_run_id_values
    ) == 1

    else None

)


# ============================================================
# 06.03.07 RESULTADOS
# ============================================================

print("=" * 80)
print("VALIDACIÓN DE IDEMPOTENCIA")
print("=" * 80)


print(
    f"Fecha forecast:      "
    f"{target_date.date()}"
)


print(
    f"Model Version:       "
    f"{MODEL_VERSION}"
)


print(
    f"Registros fecha:     "
    f"{current_rows}"
)


print(
    f"Tiendas fecha:       "
    f"{current_stores}"
)


print(
    f"Versiones detectadas:"
    f" {current_versions}"
)


print(
    f"MLflow Run IDs:      "
    f"{current_run_ids}"
)


print(
    f"Claves duplicadas:   "
    f"{duplicate_keys}"
)


# ============================================================
# 06.03.08 VALIDACIÓN FINAL
# ============================================================

if duplicate_keys != 0:

    raise RuntimeError(

        "Se han detectado claves duplicadas "
        "en la tabla de forecast."

    )


if current_rows != EXPECTED_STORES:

    raise RuntimeError(

        "El forecast actual no contiene "
        "el número esperado de registros."

    )


if current_stores != EXPECTED_STORES:

    raise RuntimeError(

        "El forecast actual no contiene "
        "el número esperado de tiendas."

    )


if current_versions != 1:

    raise RuntimeError(

        "El forecast actual contiene "
        "más de una Model Version."

    )


if current_run_ids != 1:

    raise RuntimeError(

        "El forecast actual contiene "
        "más de un MLflow Run ID."

    )


if current_run_id != MODEL_RUN_ID:

    raise RuntimeError(

        "El MLflow Run ID del forecast actual "
        "no coincide con el modelo utilizado.\n"
        f"Esperado: {MODEL_RUN_ID}\n"
        f"Actual:   {current_run_id}"

    )


# ============================================================
# 06.03.09 RESULTADO
# ============================================================

print()
print("=" * 80)
print("VALIDACIÓN FINAL")
print("=" * 80)


print(
    "OK - Sin duplicados según la clave lógica"
)


print(
    "OK - Cobertura completa de tiendas"
)


print(
    f"OK - Model Version correcta: "
    f"{MODEL_VERSION}"
)


print(
    "OK - MLflow Run ID correcto"
)


print(
    "OK - El MERGE mantiene una salida idempotente"
)


print()
print(
    "Validación de idempotencia completada correctamente."
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 06.04 Automatización con Lakeflow Jobs
# MAGIC
# MAGIC El pipeline de inference está preparado para ejecutarse de forma automatizada mediante **Lakeflow Jobs**.
# MAGIC
# MAGIC El notebook incorpora:
# MAGIC
# MAGIC - carga de la **Version 2** del modelo desde Unity Catalog
# MAGIC - carga del modelo mediante **Spark ML**
# MAGIC - validación previa del modelo y de las fuentes
# MAGIC - preparación dinámica del dataset de inference
# MAGIC - generación del forecast
# MAGIC - sanity checks
# MAGIC - persistencia mediante `MERGE` en Delta
# MAGIC - trazabilidad mediante `model_version` y `mlflow_run_id`
# MAGIC - validación global de la tabla de scoring
# MAGIC - control de idempotencia
# MAGIC - errores controlados mediante excepciones
# MAGIC
# MAGIC Esto permite utilizar el notebook como una tarea dentro de un Lakeflow Job y realizar reintentos sin generar registros duplicados.
# MAGIC
# MAGIC El flujo automatizado será:
# MAGIC
# MAGIC **Datos actualizados → Feature Engineering → Inference → Validaciones → MERGE Delta**
# MAGIC
# MAGIC El proceso de inference debe ejecutarse únicamente cuando la **Feature Table** haya sido actualizada con nuevos datos.
# MAGIC
# MAGIC La persistencia utiliza como clave lógica:
# MAGIC
# MAGIC **forecast_date + store_id + model_version**
# MAGIC
# MAGIC por lo que una reejecución para la misma fecha y versión actualiza las predicciones existentes en lugar de insertar duplicados.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 06.05 ESTRATEGIA DE EJECUCIÓN
# MAGIC
# MAGIC ### Objetivo
# MAGIC
# MAGIC Definir cómo debe ejecutarse el proceso de inference dentro de la arquitectura completa del proyecto.
# MAGIC
# MAGIC El forecast no debe ejecutarse simplemente por horario.
# MAGIC
# MAGIC El modelo utiliza información histórica como:
# MAGIC
# MAGIC - `lag_1`
# MAGIC - `lag_7`
# MAGIC - `lag_14`
# MAGIC - `lag_28`
# MAGIC - `rolling_mean_7`
# MAGIC - `rolling_mean_28`
# MAGIC - `rolling_std_7`
# MAGIC - `lag1_minus_lag7`
# MAGIC - `lag1_vs_mean7`
# MAGIC
# MAGIC Por tanto, una nueva ejecución solo tiene sentido cuando el histórico de ventas y la **Feature Table** han sido actualizados.
# MAGIC
# MAGIC ### Flujo objetivo
# MAGIC
# MAGIC **Nueva llegada de datos**
# MAGIC
# MAGIC ↓
# MAGIC
# MAGIC **Ingesta Bronze**
# MAGIC
# MAGIC ↓
# MAGIC
# MAGIC **Transformación Silver**
# MAGIC
# MAGIC ↓
# MAGIC
# MAGIC **Modelo Gold**
# MAGIC
# MAGIC ↓
# MAGIC
# MAGIC **Actualización de la Feature Table**
# MAGIC
# MAGIC ↓
# MAGIC
# MAGIC **Sales Forecast Inference**
# MAGIC
# MAGIC ↓
# MAGIC
# MAGIC **Sanity Checks y validaciones**
# MAGIC
# MAGIC ↓
# MAGIC
# MAGIC **MERGE en Delta**
# MAGIC
# MAGIC ↓
# MAGIC
# MAGIC **Validación de la tabla de scoring**
# MAGIC
# MAGIC ### Estrategia recomendada
# MAGIC
# MAGIC La tarea de inference debe ejecutarse después de una actualización correcta de:
# MAGIC
# MAGIC `retail_analytics.5_ml.daily_store_features`
# MAGIC
# MAGIC El proceso utiliza la **Version 2** del modelo:
# MAGIC
# MAGIC `retail_analytics.5_ml.sales_forecasting_random_forest`
# MAGIC
# MAGIC La fecha de forecast se calcula dinámicamente:
# MAGIC
# MAGIC **última fecha disponible + 1 día**
# MAGIC
# MAGIC Por tanto, no es necesario introducir manualmente la fecha objetivo.
# MAGIC
# MAGIC No se recomienda ejecutar el notebook de inference de forma independiente si la Feature Table no ha cambiado, ya que volvería a generar el forecast para la misma fecha.
# MAGIC
# MAGIC La persistencia mediante `MERGE` garantiza idempotencia ante reintentos o ejecuciones repetidas.
# MAGIC
# MAGIC El proceso queda preparado para integrarse dentro del pipeline principal mediante **Lakeflow Jobs**, después de la tarea de actualización de features.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 06.06 VALIDACIÓN Y CIERRE DE LA AUTOMATIZACIÓN
# MAGIC
# MAGIC ### Validación del proceso de inference
# MAGIC
# MAGIC El notebook de inference ha sido preparado para ejecutarse de forma independiente y automatizable mediante **Lakeflow Jobs**.
# MAGIC
# MAGIC El proceso completo permite:
# MAGIC
# MAGIC - recuperar la **Version 2** del modelo desde Unity Catalog
# MAGIC - cargar el modelo mediante **Spark ML**
# MAGIC - validar la configuración y trazabilidad del modelo
# MAGIC - recuperar los datos desde la Feature Table
# MAGIC - construir dinámicamente las features necesarias para inference
# MAGIC - calcular automáticamente la siguiente fecha de forecast
# MAGIC - generar una predicción por tienda
# MAGIC - ejecutar sanity checks sobre las predicciones
# MAGIC - persistir los resultados mediante `MERGE` en Delta
# MAGIC - mantener trazabilidad mediante `model_version` y `mlflow_run_id`
# MAGIC - validar la integridad de la tabla de scoring
# MAGIC - garantizar idempotencia ante reintentos
# MAGIC
# MAGIC El proceso no realiza entrenamiento, tuning ni utiliza los conjuntos **TEST** u **OOT**.
# MAGIC
# MAGIC ### Arquitectura final de Machine Learning
# MAGIC
# MAGIC **Gold**
# MAGIC
# MAGIC ↓
# MAGIC
# MAGIC **Feature Engineering**
# MAGIC
# MAGIC ↓
# MAGIC
# MAGIC **Delta Feature Table**
# MAGIC
# MAGIC ↓
# MAGIC
# MAGIC **Spark ML**
# MAGIC
# MAGIC ↓
# MAGIC
# MAGIC **MLflow Tracking**
# MAGIC
# MAGIC ↓
# MAGIC
# MAGIC **Unity Catalog Model Registry**
# MAGIC
# MAGIC ↓
# MAGIC
# MAGIC **Model Version 2**
# MAGIC
# MAGIC ↓
# MAGIC
# MAGIC **Inference / Scoring**
# MAGIC
# MAGIC ↓
# MAGIC
# MAGIC **Lakeflow Jobs**
# MAGIC
# MAGIC ↓
# MAGIC
# MAGIC **Delta Forecast History**
# MAGIC
# MAGIC ### Modelo operacional
# MAGIC
# MAGIC Modelo registrado:
# MAGIC
# MAGIC `retail_analytics.5_ml.sales_forecasting_random_forest`
# MAGIC
# MAGIC Versión utilizada:
# MAGIC
# MAGIC **Version 2**
# MAGIC
# MAGIC Feature Table:
# MAGIC
# MAGIC `retail_analytics.5_ml.daily_store_features`
# MAGIC
# MAGIC Tabla histórica de forecasts:
# MAGIC
# MAGIC `retail_analytics.5_ml.sales_forecast_predictions`
# MAGIC
# MAGIC La granularidad del forecast es:
# MAGIC
# MAGIC **1 fila = forecast_date + store_id + model_version**
# MAGIC
# MAGIC La persistencia mediante `MERGE` permite reejecutar el proceso para una misma fecha y versión sin generar duplicados.
# MAGIC
# MAGIC ### Estado final
# MAGIC
# MAGIC El proceso de inference queda preparado para su integración dentro del pipeline completo de **Lakeflow Jobs**.
# MAGIC
# MAGIC El flujo objetivo es:
# MAGIC
# MAGIC **Actualización de datos → Medallion → Feature Refresh → Inference → Validaciones → Persistencia Delta**
# MAGIC
# MAGIC De esta forma, el forecasting puede ejecutarse después de que nuevos datos hayan actualizado correctamente la Feature Table, manteniendo un proceso reproducible, trazable e idempotente.