# Databricks notebook source
# DBTITLE 1,00. CONFIGURACIÓN Y LECTURA DEL DATASET ML
# ============================================================
# 00. CONFIGURACIÓN Y LECTURA DEL DATASET ML
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Cargar y validar el dataset de Machine Learning ya
# materializado en Delta.
#
# Este notebook se dedicará exclusivamente a:
#
# - entrenamiento final
# - evaluación sobre TEST
# - registro con MLflow
# - Model Registry en Unity Catalog
#
# El Feature Engineering NO se reconstruye aquí.
#
# Fuente:
#
# retail_analytics.5_ml.daily_store_features
#
# ============================================================


from pyspark.sql import functions as F


# ============================================================
# 00.01 CONFIGURACIÓN
# ============================================================

ML_FEATURE_TABLE = (
    "retail_analytics.5_ml.daily_store_features"
)


TARGET_COLUMN = (
    "net_sales"
)


IDENTIFIER_COLUMNS = [

    "sale_date",
    "store_id"

]


NUMERIC_FEATURES = [

    # --------------------------------------------------------
    # VARIABLES TEMPORALES
    # --------------------------------------------------------

    "year",
    "month",
    "day",
    "day_of_week",
    "week_of_year",
    "is_weekend",

    # --------------------------------------------------------
    # LAGS
    # --------------------------------------------------------

    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",

    # --------------------------------------------------------
    # ROLLING FEATURES
    # --------------------------------------------------------

    "rolling_mean_7",
    "rolling_mean_28",
    "rolling_std_7",

    # --------------------------------------------------------
    # FEATURES DERIVADAS
    # --------------------------------------------------------

    "lag1_minus_lag7",
    "lag1_vs_mean7"

]


CATEGORICAL_FEATURES = [

    "store_id"

]


MODEL_FEATURES = (

    NUMERIC_FEATURES
    +
    CATEGORICAL_FEATURES

)


REQUIRED_COLUMNS = (

    IDENTIFIER_COLUMNS
    +
    NUMERIC_FEATURES
    +
    [TARGET_COLUMN]

)


# ============================================================
# 00.02 VALIDAMOS QUE LA TABLA EXISTE
# ============================================================

print("=" * 80)
print("CARGA Y VALIDACIÓN DEL DATASET ML")
print("=" * 80)


if not spark.catalog.tableExists(
    ML_FEATURE_TABLE
):

    raise RuntimeError(

        "No existe la tabla ML requerida: "
        f"{ML_FEATURE_TABLE}"

    )


print(
    f"OK - Tabla encontrada: "
    f"{ML_FEATURE_TABLE}"
)


# ============================================================
# 00.03 LECTURA DEL DATASET
# ============================================================

ml_model_df = (

    spark

    .table(
        ML_FEATURE_TABLE
    )

)


# ============================================================
# 00.04 VALIDACIÓN DE COLUMNAS
# ============================================================

available_columns = set(
    ml_model_df.columns
)


missing_columns = [

    column

    for column in REQUIRED_COLUMNS

    if column not in available_columns

]


if missing_columns:

    raise RuntimeError(

        "Faltan columnas necesarias en el dataset ML: "
        f"{missing_columns}"

    )


print(
    "OK - Todas las columnas necesarias están disponibles"
)


# ============================================================
# 00.05 ESTADÍSTICAS GENERALES
# ============================================================

dataset_stats = (

    ml_model_df

    .agg(

        F.count("*").alias(
            "rows"
        ),

        F.countDistinct(
            "store_id"
        ).alias(
            "stores"
        ),

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


dataset_rows = int(
    dataset_stats["rows"]
)


dataset_stores = int(
    dataset_stats["stores"]
)


dataset_min_date = (
    dataset_stats["min_date"]
)


dataset_max_date = (
    dataset_stats["max_date"]
)


print()
print("=" * 80)
print("RESUMEN DEL DATASET ML")
print("=" * 80)


print(
    f"Registros:        "
    f"{dataset_rows:,}"
)


print(
    f"Tiendas:          "
    f"{dataset_stores:,}"
)


print(
    f"Fecha mínima:     "
    f"{dataset_min_date}"
)


print(
    f"Fecha máxima:     "
    f"{dataset_max_date}"
)


# ============================================================
# 00.06 DUPLICADOS POR GRANULARIDAD DEL MODELO
# ============================================================
#
# El dataset tiene granularidad:
#
#       sale_date + store_id
#
# Por tanto debe existir como máximo una fila por tienda/día.
#
# ============================================================

duplicate_keys_df = (

    ml_model_df

    .groupBy(
        "sale_date",
        "store_id"
    )

    .count()

    .filter(
        F.col("count") > 1
    )

)


duplicate_keys = (
    duplicate_keys_df.count()
)


print()
print("=" * 80)
print("VALIDACIÓN DE GRANULARIDAD")
print("=" * 80)


print(
    f"Duplicados sale_date + store_id: "
    f"{duplicate_keys:,}"
)


# ============================================================
# 00.07 NULLS EN COLUMNAS CRÍTICAS
# ============================================================

critical_null_stats = (

    ml_model_df

    .agg(

        F.sum(
            F.col("sale_date").isNull().cast("int")
        ).alias(
            "sale_date_nulls"
        ),

        F.sum(
            F.col("store_id").isNull().cast("int")
        ).alias(
            "store_id_nulls"
        ),

        F.sum(
            F.col(TARGET_COLUMN).isNull().cast("int")
        ).alias(
            "target_nulls"
        )

    )

    .first()

)


sale_date_nulls = int(
    critical_null_stats["sale_date_nulls"] or 0
)


store_id_nulls = int(
    critical_null_stats["store_id_nulls"] or 0
)


target_nulls = int(
    critical_null_stats["target_nulls"] or 0
)


print()
print("=" * 80)
print("NULLS EN COLUMNAS CRÍTICAS")
print("=" * 80)


print(
    f"sale_date: "
    f"{sale_date_nulls:,}"
)


print(
    f"store_id:  "
    f"{store_id_nulls:,}"
)


print(
    f"net_sales: "
    f"{target_nulls:,}"
)


# ============================================================
# 00.08 NULLS / NAN EN FEATURES
# ============================================================

feature_quality_rows = []


for feature in NUMERIC_FEATURES:

    feature_dtype = dict(
        ml_model_df.dtypes
    )[feature]


    null_count = (

        ml_model_df

        .filter(
            F.col(feature).isNull()
        )

        .count()

    )


    if feature_dtype in (
        "float",
        "double"
    ):

        nan_count = (

            ml_model_df

            .filter(
                F.isnan(
                    F.col(feature)
                )
            )

            .count()

        )

    else:

        nan_count = 0


    feature_quality_rows.append(

        (

            feature,
            feature_dtype,
            int(null_count),
            int(nan_count)

        )

    )


feature_quality_df = spark.createDataFrame(

    feature_quality_rows,

    [

        "feature",
        "data_type",
        "null_count",
        "nan_count"

    ]

)


print()
print("=" * 80)
print("CALIDAD DE FEATURES")
print("=" * 80)


display(
    feature_quality_df
)


# ============================================================
# 00.09 VALIDACIÓN FINAL
# ============================================================

total_feature_nulls = sum(

    row[2]

    for row in feature_quality_rows

)


total_feature_nans = sum(

    row[3]

    for row in feature_quality_rows

)


print()
print("=" * 80)
print("VALIDACIÓN FINAL")
print("=" * 80)


if dataset_rows == 0:

    raise RuntimeError(
        "El dataset ML está vacío."
    )


if duplicate_keys != 0:

    raise RuntimeError(

        "Se han detectado duplicados en "
        "sale_date + store_id."

    )


if sale_date_nulls != 0:

    raise RuntimeError(
        "Existen NULLs en sale_date."
    )


if store_id_nulls != 0:

    raise RuntimeError(
        "Existen NULLs en store_id."
    )


if target_nulls != 0:

    raise RuntimeError(
        "Existen NULLs en net_sales."
    )


if total_feature_nulls != 0:

    raise RuntimeError(

        "Existen NULLs en las features numéricas."

    )


if total_feature_nans != 0:

    raise RuntimeError(

        "Existen NaN en las features numéricas."

    )


print(
    f"OK - Dataset ML cargado: "
    f"{dataset_rows:,} registros"
)


print(
    f"OK - Tiendas disponibles: "
    f"{dataset_stores}"
)


print(
    f"OK - Periodo: "
    f"{dataset_min_date} -> {dataset_max_date}"
)


print(
    "OK - Sin duplicados por sale_date + store_id"
)


print(
    "OK - Sin NULLs en columnas críticas"
)


print(
    "OK - Sin NULLs en features"
)


print(
    "OK - Sin NaN en features"
)


print()
print(
    "Dataset ML preparado para el entrenamiento final."
)


# ============================================================
# 00.10 MUESTRA
# ============================================================

display(

    ml_model_df

    .orderBy(
        "sale_date",
        "store_id"
    )

    .limit(
        50
    )

)

# COMMAND ----------

# DBTITLE 1,01. SPLIT TEMPORAL FINAL
# ============================================================
# 01. SPLIT TEMPORAL FINAL
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Reconstruir los cuatro periodos temporales utilizados
# durante el desarrollo del modelo:
#
# TRAIN
#     2024-01-29 -> 2025-12-31
#
# VALIDATION
#     2026-01-01 -> 2026-04-30
#
# TEST
#     2026-05-01 -> 2026-07-31
#
# OOT
#     2026-08-01 -> 2026-08-31
#
# IMPORTANTE
# ------------------------------------------------------------
#
# - No utilizamos randomSplit.
# - Respetamos estrictamente el orden temporal.
# - TEST solo se utilizará para la evaluación final.
# - OOT permanece fuera del entrenamiento y evaluación final.
#
# ============================================================


from pyspark.sql import functions as F


# ============================================================
# 01.01 FECHAS DE CORTE
# ============================================================

TRAIN_END = (
    "2025-12-31"
)


VALIDATION_START = (
    "2026-01-01"
)

VALIDATION_END = (
    "2026-04-30"
)


TEST_START = (
    "2026-05-01"
)

TEST_END = (
    "2026-07-31"
)


OOT_START = (
    "2026-08-01"
)

OOT_END = (
    "2026-08-31"
)


# ============================================================
# 01.02 TRAIN
# ============================================================

train_df = (

    ml_model_df

    .filter(

        F.col("sale_date")
        <=
        F.lit(TRAIN_END)

    )

)


# ============================================================
# 01.03 VALIDATION
# ============================================================

validation_df = (

    ml_model_df

    .filter(

        (
            F.col("sale_date")
            >=
            F.lit(VALIDATION_START)
        )

        &

        (
            F.col("sale_date")
            <=
            F.lit(VALIDATION_END)
        )

    )

)


# ============================================================
# 01.04 TEST
# ============================================================

test_df = (

    ml_model_df

    .filter(

        (
            F.col("sale_date")
            >=
            F.lit(TEST_START)
        )

        &

        (
            F.col("sale_date")
            <=
            F.lit(TEST_END)
        )

    )

)


# ============================================================
# 01.05 OUT-OF-TIME
# ============================================================

oot_df = (

    ml_model_df

    .filter(

        (
            F.col("sale_date")
            >=
            F.lit(OOT_START)
        )

        &

        (
            F.col("sale_date")
            <=
            F.lit(OOT_END)
        )

    )

)


# ============================================================
# 01.06 FUNCIÓN DE RESUMEN
# ============================================================

def summarize_period(
    dataframe,
    period_name
):

    stats = (

        dataframe

        .agg(

            F.count("*").alias(
                "rows"
            ),

            F.countDistinct(
                "store_id"
            ).alias(
                "stores"
            ),

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


    return {

        "period":
            period_name,

        "rows":
            int(
                stats["rows"]
            ),

        "stores":
            int(
                stats["stores"]
            ),

        "min_date":
            stats["min_date"],

        "max_date":
            stats["max_date"]

    }


# ============================================================
# 01.07 RESUMEN DE PERIODOS
# ============================================================

train_stats = summarize_period(
    train_df,
    "TRAIN"
)


validation_stats = summarize_period(
    validation_df,
    "VALIDATION"
)


test_stats = summarize_period(
    test_df,
    "TEST"
)


oot_stats = summarize_period(
    oot_df,
    "OOT"
)


print("=" * 80)
print("SPLIT TEMPORAL FINAL")
print("=" * 80)


for stats in [

    train_stats,
    validation_stats,
    test_stats,
    oot_stats

]:

    print()

    print(
        f"{stats['period']}"
    )

    print(
        f"  Registros: "
        f"{stats['rows']:,}"
    )

    print(
        f"  Tiendas:   "
        f"{stats['stores']}"
    )

    print(
        f"  Periodo:   "
        f"{stats['min_date']} "
        f"-> "
        f"{stats['max_date']}"
    )


# ============================================================
# 01.08 VALIDACIÓN DE CONSERVACIÓN DE REGISTROS
# ============================================================

split_total_rows = (

    train_stats["rows"]
    +
    validation_stats["rows"]
    +
    test_stats["rows"]
    +
    oot_stats["rows"]

)


original_total_rows = (
    ml_model_df.count()
)


row_difference = (

    split_total_rows
    -
    original_total_rows

)


print()
print("=" * 80)
print("CONSERVACIÓN DE REGISTROS")
print("=" * 80)


print(
    f"Dataset original: "
    f"{original_total_rows:,}"
)


print(
    f"Suma de splits:   "
    f"{split_total_rows:,}"
)


print(
    f"Diferencia:       "
    f"{row_difference:,}"
)


# ============================================================
# 01.09 VALIDACIÓN DE SOLAPAMIENTOS
# ============================================================

train_validation_overlap = (

    train_df

    .select(
        "sale_date",
        "store_id"
    )

    .intersect(

        validation_df.select(
            "sale_date",
            "store_id"
        )

    )

    .count()

)


train_test_overlap = (

    train_df

    .select(
        "sale_date",
        "store_id"
    )

    .intersect(

        test_df.select(
            "sale_date",
            "store_id"
        )

    )

    .count()

)


train_oot_overlap = (

    train_df

    .select(
        "sale_date",
        "store_id"
    )

    .intersect(

        oot_df.select(
            "sale_date",
            "store_id"
        )

    )

    .count()

)


validation_test_overlap = (

    validation_df

    .select(
        "sale_date",
        "store_id"
    )

    .intersect(

        test_df.select(
            "sale_date",
            "store_id"
        )

    )

    .count()

)


validation_oot_overlap = (

    validation_df

    .select(
        "sale_date",
        "store_id"
    )

    .intersect(

        oot_df.select(
            "sale_date",
            "store_id"
        )

    )

    .count()

)


test_oot_overlap = (

    test_df

    .select(
        "sale_date",
        "store_id"
    )

    .intersect(

        oot_df.select(
            "sale_date",
            "store_id"
        )

    )

    .count()

)


print()
print("=" * 80)
print("VALIDACIÓN DE SOLAPAMIENTOS")
print("=" * 80)


print(
    f"TRAIN vs VALIDATION:      "
    f"{train_validation_overlap}"
)


print(
    f"TRAIN vs TEST:            "
    f"{train_test_overlap}"
)


print(
    f"TRAIN vs OOT:             "
    f"{train_oot_overlap}"
)


print(
    f"VALIDATION vs TEST:       "
    f"{validation_test_overlap}"
)


print(
    f"VALIDATION vs OOT:        "
    f"{validation_oot_overlap}"
)


print(
    f"TEST vs OOT:              "
    f"{test_oot_overlap}"
)


# ============================================================
# 01.10 VALIDACIÓN FINAL
# ============================================================

print()
print("=" * 80)
print("VALIDACIÓN FINAL")
print("=" * 80)


if row_difference != 0:

    raise RuntimeError(

        "La suma de TRAIN + VALIDATION + TEST + OOT "
        "no coincide con el dataset completo."

    )


if train_stats["rows"] == 0:

    raise RuntimeError(
        "TRAIN está vacío."
    )


if validation_stats["rows"] == 0:

    raise RuntimeError(
        "VALIDATION está vacío."
    )


if test_stats["rows"] == 0:

    raise RuntimeError(
        "TEST está vacío."
    )


if oot_stats["rows"] == 0:

    raise RuntimeError(
        "OOT está vacío."
    )


overlaps = (

    train_validation_overlap
    +
    train_test_overlap
    +
    train_oot_overlap
    +
    validation_test_overlap
    +
    validation_oot_overlap
    +
    test_oot_overlap

)


if overlaps != 0:

    raise RuntimeError(

        "Se han detectado solapamientos "
        "entre los splits temporales."

    )


if not (
    train_stats["max_date"]
    <
    validation_stats["min_date"]
    <
    validation_stats["max_date"]
    <
    test_stats["min_date"]
    <
    test_stats["max_date"]
    <
    oot_stats["min_date"]
):

    raise RuntimeError(

        "El orden temporal de los datasets "
        "no es correcto."

    )


print(
    "OK - TRAIN definido correctamente"
)


print(
    "OK - VALIDATION definido correctamente"
)


print(
    "OK - TEST definido correctamente"
)


print(
    "OK - OOT definido correctamente"
)


print(
    "OK - Todos los registros están asignados"
)


print(
    "OK - Sin solapamientos temporales"
)


print(
    "OK - Orden temporal correcto"
)


print(
    "OK - TEST permanece aislado"
)


print(
    "OK - OOT permanece aislado"
)


print()
print(
    "Split temporal final validado correctamente."
)

# COMMAND ----------

# DBTITLE 1,02. PREPARACIÓN Y ENTRENAMIENTO DEL MODELO FINAL
# # ============================================================
# 02. PREPARACIÓN Y ENTRENAMIENTO DEL MODELO FINAL
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Entrenar el modelo final utilizando:
#
#       TRAIN + VALIDATION
#
# La configuración ya fue seleccionada previamente en el
# notebook de Hyperparameter Tuning.
#
# Configuración seleccionada:
#
#       RF_01_BASE
#
#       numTrees = 60
#       maxDepth = 8
#       minInstancesPerNode = 3
#       featureSubsetStrategy = auto
#       seed = 42
#
# IMPORTANTE
# ------------------------------------------------------------
#
# - TEST permanece completamente aislado.
# - OOT permanece completamente aislado.
# - No se realiza ningún tuning adicional.
# - No utilizamos TEST para tomar decisiones.
#
# ============================================================


from pyspark.sql import functions as F

from pyspark.ml import Pipeline

from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler
)

from pyspark.ml.regression import (
    RandomForestRegressor
)


# ============================================================
# 02.01 CONFIGURACIÓN DEL MODELO FINAL
# ============================================================

FINAL_MODEL_CONFIG = {

    "config_name":
        "RF_01_BASE",

    "numTrees":
        60,

    "maxDepth":
        8,

    "minInstancesPerNode":
        3,

    "featureSubsetStrategy":
        "auto",

    "seed":
        42

}


print("=" * 80)
print("CONFIGURACIÓN DEL MODELO FINAL")
print("=" * 80)


for parameter_name, parameter_value in FINAL_MODEL_CONFIG.items():

    print(
        f"{parameter_name}: "
        f"{parameter_value}"
    )


# ============================================================
# 02.02 FEATURES DEL MODELO
# ============================================================

numeric_features = [

    # --------------------------------------------------------
    # VARIABLES TEMPORALES
    # --------------------------------------------------------

    "year",
    "month",
    "day",
    "day_of_week",
    "week_of_year",
    "is_weekend",

    # --------------------------------------------------------
    # LAGS
    # --------------------------------------------------------

    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",

    # --------------------------------------------------------
    # ROLLING FEATURES
    # --------------------------------------------------------

    "rolling_mean_7",
    "rolling_mean_28",
    "rolling_std_7",

    # --------------------------------------------------------
    # FEATURES DERIVADAS
    # --------------------------------------------------------

    "lag1_minus_lag7",
    "lag1_vs_mean7"

]


categorical_features = [

    "store_id"

]


target_column = (
    "net_sales"
)


required_model_columns = (

    ["sale_date"]
    +
    categorical_features
    +
    numeric_features
    +
    [target_column]

)


# ============================================================
# 02.03 VALIDACIÓN DE COLUMNAS
# ============================================================

available_columns = set(
    ml_model_df.columns
)


missing_columns = [

    column

    for column in required_model_columns

    if column not in available_columns

]


if missing_columns:

    raise RuntimeError(

        "Faltan columnas necesarias para entrenar el modelo: "
        f"{missing_columns}"

    )


print()
print(
    "OK - Todas las columnas necesarias están disponibles"
)


# ============================================================
# 02.04 DATASET FINAL DE ENTRENAMIENTO
# ============================================================
#
# El modelo final puede incorporar Validation porque:
#
# - Validation ya se utilizó para seleccionar hiperparámetros.
# - La fase de tuning está cerrada.
#
# Por tanto:
#
#       TRAIN_FINAL = TRAIN + VALIDATION
#
# ============================================================

train_final_df = (

    train_df

    .unionByName(
        validation_df
    )

)


# ============================================================
# 02.05 VALIDACIÓN DEL DATASET FINAL
# ============================================================

train_rows = (
    train_df.count()
)


validation_rows = (
    validation_df.count()
)


train_final_rows = (
    train_final_df.count()
)


expected_train_final_rows = (

    train_rows
    +
    validation_rows

)


train_final_stats = (

    train_final_df

    .agg(

        F.countDistinct(
            "store_id"
        ).alias(
            "stores"
        ),

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


train_final_stores = int(
    train_final_stats["stores"]
)


train_final_min_date = (
    train_final_stats["min_date"]
)


train_final_max_date = (
    train_final_stats["max_date"]
)


print()
print("=" * 80)
print("DATASET FINAL DE ENTRENAMIENTO")
print("=" * 80)


print(
    f"TRAIN:              "
    f"{train_rows:,}"
)


print(
    f"VALIDATION:         "
    f"{validation_rows:,}"
)


print(
    f"TRAIN + VALIDATION: "
    f"{train_final_rows:,}"
)


print(
    f"Tiendas:            "
    f"{train_final_stores}"
)


print(
    f"Periodo:            "
    f"{train_final_min_date} "
    f"-> "
    f"{train_final_max_date}"
)


if train_final_rows != expected_train_final_rows:

    raise RuntimeError(

        "El dataset final de entrenamiento no contiene "
        "exactamente TRAIN + VALIDATION."

    )


# ============================================================
# 02.06 VALIDACIÓN DE DUPLICADOS
# ============================================================

train_final_duplicates = (

    train_final_df

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


print(
    f"Duplicados sale_date + store_id: "
    f"{train_final_duplicates:,}"
)


if train_final_duplicates != 0:

    raise RuntimeError(

        "Se han detectado duplicados en el dataset final "
        "de entrenamiento."

    )


# ============================================================
# 02.07 VALIDACIÓN DE NULLS
# ============================================================

columns_to_validate = (

    categorical_features
    +
    numeric_features
    +
    [target_column]

)


null_expressions = [

    F.sum(
        F.col(column)
        .isNull()
        .cast("int")
    ).alias(
        column
    )

    for column in columns_to_validate

]


null_stats = (

    train_final_df

    .agg(
        *null_expressions
    )

    .first()

)


columns_with_nulls = {

    column:
        int(
            null_stats[column] or 0
        )

    for column in columns_to_validate

    if int(
        null_stats[column] or 0
    ) > 0

}


if columns_with_nulls:

    raise RuntimeError(

        "Se han detectado NULLs en el dataset final: "
        f"{columns_with_nulls}"

    )


print(
    "OK - Dataset final sin NULLs"
)


# ============================================================
# 02.08 PREPROCESSING
# ============================================================

store_indexer = StringIndexer(

    inputCol=
        "store_id",

    outputCol=
        "store_id_index",

    handleInvalid=
        "keep"

)


store_encoder = OneHotEncoder(

    inputCol=
        "store_id_index",

    outputCol=
        "store_id_encoded",

    handleInvalid=
        "keep"

)


assembler = VectorAssembler(

    inputCols=(

        numeric_features

        +

        [
            "store_id_encoded"
        ]

    ),

    outputCol=
        "features",

    handleInvalid=
        "error"

)


# ============================================================
# 02.09 RANDOM FOREST FINAL
# ============================================================

final_rf = RandomForestRegressor(

    featuresCol=
        "features",

    labelCol=
        target_column,

    predictionCol=
        "prediction",

    numTrees=
        FINAL_MODEL_CONFIG[
            "numTrees"
        ],

    maxDepth=
        FINAL_MODEL_CONFIG[
            "maxDepth"
        ],

    minInstancesPerNode=
        FINAL_MODEL_CONFIG[
            "minInstancesPerNode"
        ],

    featureSubsetStrategy=
        FINAL_MODEL_CONFIG[
            "featureSubsetStrategy"
        ],

    seed=
        FINAL_MODEL_CONFIG[
            "seed"
        ]

)


# ============================================================
# 02.10 PIPELINE FINAL
# ============================================================

final_rf_pipeline = Pipeline(

    stages=[

        store_indexer,
        store_encoder,
        assembler,
        final_rf

    ]

)


# ============================================================
# 02.11 ENTRENAMIENTO
# ============================================================

print()
print("=" * 80)
print("ENTRENAMIENTO DEL MODELO FINAL")
print("=" * 80)


final_rf_model = (

    final_rf_pipeline

    .fit(
        train_final_df
    )

)


print(
    "OK - Modelo final entrenado correctamente"
)


print(
    f"OK - Entrenado con "
    f"{train_final_rows:,} registros"
)


print(
    f"OK - Periodo entrenamiento: "
    f"{train_final_min_date} "
    f"-> "
    f"{train_final_max_date}"
)


# ============================================================
# 02.12 VALIDACIÓN DE AISLAMIENTO
# ============================================================

train_final_max_date_check = (

    train_final_df

    .agg(
        F.max("sale_date")
    )

    .first()[0]

)


test_min_date_check = (

    test_df

    .agg(
        F.min("sale_date")
    )

    .first()[0]

)


oot_min_date_check = (

    oot_df

    .agg(
        F.min("sale_date")
    )

    .first()[0]

)


if not (
    train_final_max_date_check
    <
    test_min_date_check
    <
    oot_min_date_check
):

    raise RuntimeError(

        "El aislamiento temporal entre "
        "TRAIN_FINAL, TEST y OOT no es correcto."

    )


# ============================================================
# 02.13 VALIDACIÓN FINAL
# ============================================================

print()
print("=" * 80)
print("VALIDACIÓN FINAL")
print("=" * 80)


print(
    f"OK - Configuración utilizada: "
    f"{FINAL_MODEL_CONFIG['config_name']}"
)


print(
    f"OK - TRAIN + VALIDATION: "
    f"{train_final_rows:,} registros"
)


print(
    "OK - Sin duplicados"
)


print(
    "OK - Sin NULLs"
)


print(
    "OK - Pipeline final creado"
)


print(
    "OK - Random Forest final entrenado"
)


print(
    "OK - TEST no utilizado durante entrenamiento"
)


print(
    "OK - OOT no utilizado durante entrenamiento"
)


print()
print(
    "Modelo final preparado para evaluación sobre TEST."
)

# COMMAND ----------

# MAGIC %md
# MAGIC # 03. EVALUACIÓN FINAL SOBRE TEST
# MAGIC
# MAGIC ## Objetivo
# MAGIC
# MAGIC Evaluar el modelo **Random Forest** seleccionado sobre el conjunto **Test**, que hasta este momento no ha participado en:
# MAGIC
# MAGIC - entrenamiento
# MAGIC - tuning
# MAGIC - selección de hiperparámetros
# MAGIC - comparación entre modelos
# MAGIC
# MAGIC La configuración seleccionada es:
# MAGIC
# MAGIC - `numTrees = 60`
# MAGIC - `maxDepth = 8`
# MAGIC - `minInstancesPerNode = 3`
# MAGIC - `featureSubsetStrategy = auto`
# MAGIC - `seed = 42`
# MAGIC
# MAGIC El modelo final ya ha sido reentrenado utilizando:
# MAGIC
# MAGIC **TRAIN + VALIDATION**
# MAGIC
# MAGIC Periodo de entrenamiento:
# MAGIC
# MAGIC **29/01/2024 → 30/04/2026**
# MAGIC
# MAGIC Registros utilizados:
# MAGIC
# MAGIC **25.513**
# MAGIC
# MAGIC ## Importancia del conjunto Test
# MAGIC
# MAGIC Test representa un periodo futuro completamente independiente respecto a los datos utilizados para desarrollar y seleccionar el modelo.
# MAGIC
# MAGIC Nos permite responder a la pregunta:
# MAGIC
# MAGIC **¿Cómo se comporta el modelo sobre un periodo futuro que no hemos utilizado para tomar ninguna decisión?**
# MAGIC
# MAGIC Periodo Test:
# MAGIC
# MAGIC **01/05/2026 → 31/07/2026**
# MAGIC
# MAGIC Registros:
# MAGIC
# MAGIC **2.852**
# MAGIC
# MAGIC A partir de este punto:
# MAGIC
# MAGIC - Test se utilizará únicamente para la evaluación final.
# MAGIC - No se realizarán ajustes adicionales del modelo utilizando Test.
# MAGIC - OOT seguirá completamente aislado.
# MAGIC
# MAGIC Periodo OOT reservado:
# MAGIC
# MAGIC **01/08/2026 → 31/08/2026**

# COMMAND ----------

# DBTITLE 1,03.01 EVALUACIÓN FINAL SOBRE TEST
# ============================================================
# 03.01 EVALUACIÓN FINAL SOBRE TEST
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Evaluar por primera y única vez el modelo final sobre TEST.
#
# El modelo ya fue:
#
# - seleccionado utilizando VALIDATION
# - reentrenado utilizando TRAIN + VALIDATION
#
# TEST no ha participado en ninguna decisión del modelo.
#
# Métricas:
#
# - MAE
# - RMSE
# - WAPE
#
# OOT continúa completamente aislado.
#
# ============================================================


from pyspark.sql import functions as F

from pyspark.ml.evaluation import (
    RegressionEvaluator
)


# ============================================================
# 03.01.01 VALIDACIONES PREVIAS
# ============================================================

if "final_rf_model" not in globals():

    raise RuntimeError(

        "No existe final_rf_model. "
        "Ejecuta primero el bloque 02."

    )


if "test_df" not in globals():

    raise RuntimeError(

        "No existe test_df. "
        "Ejecuta primero el bloque 01."

    )


test_rows = (
    test_df.count()
)


if test_rows == 0:

    raise RuntimeError(
        "El dataset TEST está vacío."
    )


print("=" * 80)
print("EVALUACIÓN FINAL SOBRE TEST")
print("=" * 80)


print(
    f"Registros TEST: "
    f"{test_rows:,}"
)


# ============================================================
# 03.01.02 PREDICCIONES
# ============================================================

test_predictions = (

    final_rf_model

    .transform(
        test_df
    )

    .select(

        "sale_date",
        "store_id",
        "net_sales",
        "prediction"

    )

)


prediction_rows = (
    test_predictions.count()
)


if prediction_rows != test_rows:

    raise RuntimeError(

        "El número de predicciones no coincide "
        "con el número de registros de TEST."

    )


print(
    f"Predicciones generadas: "
    f"{prediction_rows:,}"
)


# ============================================================
# 03.01.03 VALIDACIÓN DE PREDICCIONES
# ============================================================

prediction_quality = (

    test_predictions

    .agg(

        F.sum(

            F.col("prediction")
            .isNull()
            .cast("int")

        ).alias(
            "prediction_nulls"
        ),

        F.sum(

            F.isnan(
                F.col("prediction")
            )
            .cast("int")

        ).alias(
            "prediction_nans"
        ),

        F.min(
            "prediction"
        ).alias(
            "min_prediction"
        ),

        F.max(
            "prediction"
        ).alias(
            "max_prediction"
        )

    )

    .first()

)


prediction_nulls = int(
    prediction_quality[
        "prediction_nulls"
    ] or 0
)


prediction_nans = int(
    prediction_quality[
        "prediction_nans"
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


print()
print(
    f"NULL predictions: "
    f"{prediction_nulls:,}"
)


print(
    f"NaN predictions:  "
    f"{prediction_nans:,}"
)


print(
    f"Predicción mínima: "
    f"{min_prediction:,.2f}"
)


print(
    f"Predicción máxima: "
    f"{max_prediction:,.2f}"
)


if prediction_nulls != 0:

    raise RuntimeError(
        "Existen predicciones NULL."
    )


if prediction_nans != 0:

    raise RuntimeError(
        "Existen predicciones NaN."
    )


# ============================================================
# 03.01.04 MAE
# ============================================================

mae_evaluator = RegressionEvaluator(

    labelCol=
        "net_sales",

    predictionCol=
        "prediction",

    metricName=
        "mae"

)


test_mae = (

    mae_evaluator

    .evaluate(
        test_predictions
    )

)


# ============================================================
# 03.01.05 RMSE
# ============================================================

rmse_evaluator = RegressionEvaluator(

    labelCol=
        "net_sales",

    predictionCol=
        "prediction",

    metricName=
        "rmse"

)


test_rmse = (

    rmse_evaluator

    .evaluate(
        test_predictions
    )

)


# ============================================================
# 03.01.06 WAPE
# ============================================================

wape_stats = (

    test_predictions

    .agg(

        F.sum(

            F.abs(

                F.col("net_sales")
                -
                F.col("prediction")

            )

        ).alias(
            "absolute_error"
        ),

        F.sum(

            F.abs(
                F.col("net_sales")
            )

        ).alias(
            "actual_total"
        )

    )

    .first()

)


absolute_error = float(
    wape_stats[
        "absolute_error"
    ]
)


actual_total = float(
    wape_stats[
        "actual_total"
    ]
)


if actual_total == 0:

    raise RuntimeError(

        "No se puede calcular WAPE porque "
        "el total real de ventas es 0."

    )


test_wape = (

    absolute_error
    /
    actual_total
    *
    100.0

)


# ============================================================
# 03.01.07 RESULTADOS
# ============================================================

print()
print("=" * 80)
print("MÉTRICAS FINALES SOBRE TEST")
print("=" * 80)


print(
    f"MAE:   "
    f"{test_mae:,.2f}"
)


print(
    f"RMSE:  "
    f"{test_rmse:,.2f}"
)


print(
    f"WAPE:  "
    f"{test_wape:.4f}%"
)


# ============================================================
# 03.01.08 GUARDAMOS MÉTRICAS
# ============================================================

final_test_metrics = {

    "mae":
        float(
            test_mae
        ),

    "rmse":
        float(
            test_rmse
        ),

    "wape":
        float(
            test_wape
        )

}


# ============================================================
# 03.01.09 VALIDACIÓN TEMPORAL
# ============================================================

test_date_stats = (

    test_predictions

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


test_min_date = (
    test_date_stats[
        "min_date"
    ]
)


test_max_date = (
    test_date_stats[
        "max_date"
    ]
)


print()
print(
    f"Periodo evaluado: "
    f"{test_min_date} "
    f"-> "
    f"{test_max_date}"
)


# ============================================================
# 03.01.10 VALIDACIÓN FINAL
# ============================================================

print()
print("=" * 80)
print("VALIDACIÓN FINAL")
print("=" * 80)


if str(test_min_date) != TEST_START:

    raise RuntimeError(

        "La fecha inicial de TEST "
        "no coincide con la esperada."

    )


if str(test_max_date) != TEST_END:

    raise RuntimeError(

        "La fecha final de TEST "
        "no coincide con la esperada."

    )


print(
    "OK - TEST cargado correctamente"
)


print(
    f"OK - "
    f"{prediction_rows:,} predicciones generadas"
)


print(
    "OK - Sin predicciones NULL"
)


print(
    "OK - Sin predicciones NaN"
)


print(
    "OK - MAE calculado"
)


print(
    "OK - RMSE calculado"
)


print(
    "OK - WAPE calculado"
)


print(
    "OK - TEST utilizado únicamente para evaluación"
)


print(
    "OK - OOT continúa aislado"
)


print()
print(
    "Evaluación final sobre TEST completada."
)


# ============================================================
# 03.01.11 MUESTRA DE PREDICCIONES
# ============================================================

display(

    test_predictions

    .orderBy(
        "sale_date",
        "store_id"
    )

    .limit(
        50
    )

)

# COMMAND ----------

# DBTITLE 1,03.02 VALIDACIÓN DE GENERALIZACIÓN
# ============================================================
# 03.02 VALIDACIÓN DE GENERALIZACIÓN
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Comparar el rendimiento observado durante la selección
# del modelo en VALIDATION con el rendimiento final obtenido
# sobre TEST.
#
# Esto nos permite comprobar si existe una degradación
# significativa al avanzar temporalmente hacia datos futuros.
#
# IMPORTANTE
# ------------------------------------------------------------
#
# - VALIDATION se utilizó durante la selección del modelo.
# - TEST se utiliza únicamente para evaluación final.
# - No se realizarán ajustes utilizando TEST.
# - OOT continúa completamente aislado.
#
# ============================================================


# ============================================================
# 03.02.01 MÉTRICAS DE VALIDATION
# ============================================================
#
# Estas son las métricas obtenidas por RF_01_BASE durante
# el Hyperparameter Tuning.
#
# ============================================================

validation_mae = (
    2142.99
)


validation_rmse = (
    2730.90
)


validation_wape = (
    35.4789
)


# ============================================================
# 03.02.02 VALIDAMOS MÉTRICAS TEST
# ============================================================

if "final_test_metrics" not in globals():

    raise RuntimeError(

        "No existe final_test_metrics. "
        "Ejecuta primero el bloque 03.01."

    )


test_mae = float(
    final_test_metrics[
        "mae"
    ]
)


test_rmse = float(
    final_test_metrics[
        "rmse"
    ]
)


test_wape = float(
    final_test_metrics[
        "wape"
    ]
)


# ============================================================
# 03.02.03 DIFERENCIAS VALIDATION VS TEST
# ============================================================

mae_difference = (

    test_mae
    -
    validation_mae

)


rmse_difference = (

    test_rmse
    -
    validation_rmse

)


wape_difference_pp = (

    test_wape
    -
    validation_wape

)


if validation_wape == 0:

    wape_relative_change = (
        0.0
    )

else:

    wape_relative_change = (

        wape_difference_pp
        /
        validation_wape
        *
        100.0

    )


# ============================================================
# 03.02.04 RESULTADOS
# ============================================================

print("=" * 80)
print("VALIDATION VS TEST")
print("=" * 80)


print()
print(
    f"{'Dataset':15} | "
    f"{'MAE':>12} | "
    f"{'RMSE':>12} | "
    f"{'WAPE':>10}"
)


print("-" * 60)


print(

    f"{'VALIDATION':15} | "
    f"{validation_mae:>12,.2f} | "
    f"{validation_rmse:>12,.2f} | "
    f"{validation_wape:>9.4f}%"

)


print(

    f"{'TEST':15} | "
    f"{test_mae:>12,.2f} | "
    f"{test_rmse:>12,.2f} | "
    f"{test_wape:>9.4f}%"

)


# ============================================================
# 03.02.05 CAMBIO DE MÉTRICAS
# ============================================================

print()
print("=" * 80)
print("CAMBIO DE RENDIMIENTO")
print("=" * 80)


print(
    f"Diferencia MAE:   "
    f"{mae_difference:+,.2f}"
)


print(
    f"Diferencia RMSE:  "
    f"{rmse_difference:+,.2f}"
)


print(
    f"Diferencia WAPE:  "
    f"{wape_difference_pp:+.4f} "
    f"puntos porcentuales"
)


print(
    f"Cambio relativo WAPE: "
    f"{wape_relative_change:+.4f}%"
)


# ============================================================
# 03.02.06 CRITERIO DE ESTABILIDAD
# ============================================================
#
# Consideramos una degradación de WAPE inferior a
# 1 punto porcentual como una variación razonable para
# este proyecto.
#
# Este umbral NO se utiliza para reajustar el modelo.
#
# Solo sirve como indicador descriptivo de estabilidad.
#
# ============================================================

MAX_ACCEPTABLE_WAPE_DEGRADATION_PP = (
    1.0
)


if wape_difference_pp <= 0:

    generalization_status = (
        "MEJORA"
    )


elif (
    wape_difference_pp
    <=
    MAX_ACCEPTABLE_WAPE_DEGRADATION_PP
):

    generalization_status = (
        "ESTABLE"
    )


else:

    generalization_status = (
        "DEGRADACIÓN"
    )


# ============================================================
# 03.02.07 CONCLUSIÓN
# ============================================================

print()
print("=" * 80)
print("GENERALIZACIÓN DEL MODELO")
print("=" * 80)


print(
    f"Estado: "
    f"{generalization_status}"
)


if generalization_status == "MEJORA":

    print(

        "El modelo obtiene un WAPE inferior en TEST "
        "respecto a VALIDATION."

    )


elif generalization_status == "ESTABLE":

    print(

        "El rendimiento en TEST permanece próximo al "
        "observado durante VALIDATION."

    )


else:

    print(

        "Se observa una degradación superior al umbral "
        "descriptivo definido."

    )


# ============================================================
# 03.02.08 RESUMEN PARA MLOPS
# ============================================================

generalization_metrics = {

    "validation_mae":
        float(
            validation_mae
        ),

    "validation_rmse":
        float(
            validation_rmse
        ),

    "validation_wape":
        float(
            validation_wape
        ),

    "test_mae":
        float(
            test_mae
        ),

    "test_rmse":
        float(
            test_rmse
        ),

    "test_wape":
        float(
            test_wape
        ),

    "wape_difference_pp":
        float(
            wape_difference_pp
        ),

    "wape_relative_change":
        float(
            wape_relative_change
        )

}


# ============================================================
# 03.02.09 VALIDACIÓN FINAL
# ============================================================

print()
print("=" * 80)
print("VALIDACIÓN FINAL")
print("=" * 80)


print(
    "OK - Métricas de VALIDATION disponibles"
)


print(
    "OK - Métricas de TEST disponibles"
)


print(
    "OK - Comparación temporal realizada"
)


print(
    f"OK - Diferencia WAPE: "
    f"{wape_difference_pp:+.4f} pp"
)


print(
    f"OK - Estado de generalización: "
    f"{generalization_status}"
)


print(
    "OK - TEST no se utilizará para nuevos ajustes"
)


print(
    "OK - OOT continúa aislado"
)


print(
    "OK - generalization_metrics preparado para MLflow"
)


print()
print(
    "Validación de generalización completada."
)

# COMMAND ----------

# MAGIC %md
# MAGIC # 03.03 CONCLUSIONES DE LA EVALUACIÓN FINAL
# MAGIC
# MAGIC ## Modelo seleccionado
# MAGIC
# MAGIC El modelo final es un **Random Forest Regressor de Spark ML**.
# MAGIC
# MAGIC Configuración:
# MAGIC
# MAGIC - `numTrees = 60`
# MAGIC - `maxDepth = 8`
# MAGIC - `minInstancesPerNode = 3`
# MAGIC - `featureSubsetStrategy = auto`
# MAGIC - `seed = 42`
# MAGIC
# MAGIC La configuración seleccionada corresponde a:
# MAGIC
# MAGIC **RF_01_BASE**
# MAGIC
# MAGIC Aunque durante el Hyperparameter Tuning la configuración `RF_04_DEPTH_10` obtuvo un WAPE ligeramente inferior, la mejora relativa frente al modelo base fue únicamente del **0,0268 %**.
# MAGIC
# MAGIC Al no superar el umbral de materialidad definido del **0,5 %**, se mantuvo el modelo base por simplicidad y menor complejidad.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Estrategia de entrenamiento final
# MAGIC
# MAGIC Una vez cerrada la fase de selección de hiperparámetros, el modelo final fue reentrenado utilizando:
# MAGIC
# MAGIC **TRAIN + VALIDATION**
# MAGIC
# MAGIC Periodo de entrenamiento:
# MAGIC
# MAGIC **29/01/2024 → 30/04/2026**
# MAGIC
# MAGIC Registros utilizados:
# MAGIC
# MAGIC **25.513**
# MAGIC
# MAGIC El conjunto **TEST** permaneció completamente aislado hasta finalizar:
# MAGIC
# MAGIC - desarrollo del modelo
# MAGIC - comparación de algoritmos
# MAGIC - Hyperparameter Tuning
# MAGIC - selección de hiperparámetros
# MAGIC - entrenamiento final
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Rendimiento sobre Validation
# MAGIC
# MAGIC Resultados obtenidos durante la fase de selección:
# MAGIC
# MAGIC - **MAE:** 2.142,99
# MAGIC - **RMSE:** 2.730,90
# MAGIC - **WAPE:** 35,4789 %
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Rendimiento final sobre Test
# MAGIC
# MAGIC Periodo Test:
# MAGIC
# MAGIC **01/05/2026 → 31/07/2026**
# MAGIC
# MAGIC Registros evaluados:
# MAGIC
# MAGIC **2.852**
# MAGIC
# MAGIC Resultados:
# MAGIC
# MAGIC - **MAE:** 2.126,96
# MAGIC - **RMSE:** 2.710,35
# MAGIC - **WAPE:** 35,7727 %
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Validation vs Test
# MAGIC
# MAGIC La diferencia observada entre ambos periodos es:
# MAGIC
# MAGIC - **MAE:** -16,03
# MAGIC - **RMSE:** -20,55
# MAGIC - **WAPE:** +0,2938 puntos porcentuales
# MAGIC - **Cambio relativo WAPE:** +0,8282 %
# MAGIC
# MAGIC El modelo mejora ligeramente en MAE y RMSE sobre Test, mientras que el WAPE presenta una pequeña degradación.
# MAGIC
# MAGIC La variación se mantiene por debajo del umbral descriptivo de **1 punto porcentual de WAPE**, por lo que el comportamiento del modelo se considera:
# MAGIC
# MAGIC **ESTABLE**
# MAGIC
# MAGIC No se observa una degradación significativa al evaluar el modelo sobre datos futuros.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Decisiones metodológicas
# MAGIC
# MAGIC La metodología utilizada mantiene una separación temporal estricta:
# MAGIC
# MAGIC **TRAIN → VALIDATION → TEST → OOT**
# MAGIC
# MAGIC - TRAIN se utilizó para entrenamiento inicial.
# MAGIC - VALIDATION se utilizó para comparar y seleccionar configuraciones.
# MAGIC - TRAIN + VALIDATION se utilizaron para entrenar el modelo final.
# MAGIC - TEST se utilizó exclusivamente para la evaluación final.
# MAGIC - OOT permanece completamente aislado.
# MAGIC
# MAGIC No se realizarán modificaciones adicionales del modelo basadas en los resultados obtenidos sobre TEST.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Estado del modelo
# MAGIC
# MAGIC La fase de:
# MAGIC
# MAGIC - desarrollo
# MAGIC - comparación
# MAGIC - tuning
# MAGIC - selección
# MAGIC - entrenamiento final
# MAGIC - evaluación sobre Test
# MAGIC
# MAGIC queda completada.
# MAGIC
# MAGIC El siguiente paso es incorporar el modelo final al flujo de **MLOps con MLflow**, registrando:
# MAGIC
# MAGIC - hiperparámetros
# MAGIC - métricas de Validation
# MAGIC - métricas de Test
# MAGIC - modelo entrenado
# MAGIC - metadata
# MAGIC - trazabilidad de la ejecución

# COMMAND ----------

# MAGIC %md
# MAGIC # 04. MLFLOW Y MLOPS
# MAGIC
# MAGIC ## Objetivo
# MAGIC
# MAGIC Una vez seleccionado, entrenado y evaluado el modelo final, incorporamos **MLflow** para gestionar su trazabilidad y ciclo de vida.
# MAGIC
# MAGIC El modelo final es un **Random Forest Regressor de Spark ML**.
# MAGIC
# MAGIC Configuración seleccionada:
# MAGIC
# MAGIC - `numTrees = 60`
# MAGIC - `maxDepth = 8`
# MAGIC - `minInstancesPerNode = 3`
# MAGIC - `featureSubsetStrategy = auto`
# MAGIC - `seed = 42`
# MAGIC
# MAGIC El modelo definitivo fue entrenado utilizando:
# MAGIC
# MAGIC **TRAIN + VALIDATION**
# MAGIC
# MAGIC Periodo de entrenamiento:
# MAGIC
# MAGIC **29/01/2024 → 30/04/2026**
# MAGIC
# MAGIC Registros utilizados:
# MAGIC
# MAGIC **25.513**
# MAGIC
# MAGIC ## Rendimiento final
# MAGIC
# MAGIC ### Validation
# MAGIC
# MAGIC - **MAE:** 2.142,99
# MAGIC - **RMSE:** 2.730,90
# MAGIC - **WAPE:** 35,4789 %
# MAGIC
# MAGIC ### Test
# MAGIC
# MAGIC - **MAE:** 2.126,96
# MAGIC - **RMSE:** 2.710,35
# MAGIC - **WAPE:** 35,7727 %
# MAGIC
# MAGIC El comportamiento entre Validation y Test se considera **estable**.
# MAGIC
# MAGIC ## Flujo MLOps
# MAGIC
# MAGIC El flujo será:
# MAGIC
# MAGIC **Delta Feature Table → Entrenamiento Spark ML → MLflow Tracking → Model Registry → Validación**
# MAGIC
# MAGIC MLflow permitirá registrar de forma estructurada:
# MAGIC
# MAGIC - hiperparámetros
# MAGIC - métricas de Validation
# MAGIC - métricas de Test
# MAGIC - modelo entrenado
# MAGIC - metadata
# MAGIC - artefactos
# MAGIC - trazabilidad de cada ejecución
# MAGIC
# MAGIC El conjunto **OOT permanece aislado** y no participa en esta fase.

# COMMAND ----------

import mlflow
import mlflow.sklearn

# COMMAND ----------

# DBTITLE 1,04.01 CONFIGURACIÓN DE MLFLOW
# ============================================================
# 04.01 CONFIGURACIÓN DE MLFLOW
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Preparar MLflow Tracking para registrar el modelo final
# de forecasting entrenado con Spark ML.
#
# En los siguientes bloques registraremos:
#
# - hiperparámetros
# - métricas de VALIDATION
# - métricas de TEST
# - modelo final
# - metadata
# - artefactos
#
# IMPORTANTE
# ------------------------------------------------------------
#
# El modelo final ya está entrenado y disponible en:
#
#       final_rf_model
#
# TEST ya fue utilizado exclusivamente para evaluación final.
#
# OOT continúa completamente aislado.
#
# ============================================================


import mlflow
import mlflow.spark


# ============================================================
# 04.01.01 VALIDACIONES PREVIAS
# ============================================================

if "final_rf_model" not in globals():

    raise RuntimeError(

        "No existe final_rf_model. "
        "Ejecuta primero el bloque 02."

    )


if "generalization_metrics" not in globals():

    raise RuntimeError(

        "No existe generalization_metrics. "
        "Ejecuta primero el bloque 03.02."

    )


# ============================================================
# 04.01.02 TRACKING URI
# ============================================================

tracking_uri = (
    mlflow.get_tracking_uri()
)


print("=" * 80)
print("CONFIGURACIÓN DE MLFLOW")
print("=" * 80)


print(
    f"Tracking URI: "
    f"{tracking_uri}"
)


# ============================================================
# 04.01.03 REGISTRY URI
# ============================================================
#
# En Databricks utilizaremos Unity Catalog como Model Registry.
#
# ============================================================

mlflow.set_registry_uri(
    "databricks-uc"
)


registry_uri = (
    mlflow.get_registry_uri()
)


print(
    f"Registry URI: "
    f"{registry_uri}"
)


# ============================================================
# 04.01.04 VERSIÓN DE MLFLOW
# ============================================================

print(
    f"MLflow version: "
    f"{mlflow.__version__}"
)


# ============================================================
# 04.01.05 INFORMACIÓN DEL MODELO
# ============================================================

MODEL_FRAMEWORK = (
    "Spark ML"
)


MODEL_ALGORITHM = (
    "RandomForestRegressor"
)


MODEL_CONFIG_NAME = (
    "RF_01_BASE"
)


print()
print("=" * 80)
print("MODELO A REGISTRAR")
print("=" * 80)


print(
    f"Framework:       "
    f"{MODEL_FRAMEWORK}"
)


print(
    f"Algoritmo:       "
    f"{MODEL_ALGORITHM}"
)


print(
    f"Configuración:   "
    f"{MODEL_CONFIG_NAME}"
)


# ============================================================
# 04.01.06 CONFIGURACIÓN DEL EXPERIMENTO
# ============================================================
#
# En Databricks, si no establecemos explícitamente otro
# experimento, las Runs quedarán asociadas al experimento
# del notebook.
#
# No necesitamos crear un experimento nuevo para este proyecto.
#
# ============================================================

active_run = (
    mlflow.active_run()
)


if active_run is not None:

    active_experiment_id = (
        active_run.info.experiment_id
    )

else:

    active_experiment_id = (
        None
    )


print()
print("=" * 80)
print("EXPERIMENTO")
print("=" * 80)


if active_experiment_id is not None:

    print(
        f"Experiment ID activo: "
        f"{active_experiment_id}"
    )

else:

    print(
        "No existe una Run activa actualmente."
    )


print(
    "Las nuevas Runs quedarán asociadas "
    "al experimento del notebook."
)


# ============================================================
# 04.01.07 VALIDACIÓN DE MÉTRICAS DISPONIBLES
# ============================================================

required_metric_keys = [

    "validation_mae",
    "validation_rmse",
    "validation_wape",

    "test_mae",
    "test_rmse",
    "test_wape",

    "wape_difference_pp",
    "wape_relative_change"

]


missing_metric_keys = [

    key

    for key in required_metric_keys

    if key not in generalization_metrics

]


if missing_metric_keys:

    raise RuntimeError(

        "Faltan métricas necesarias para MLflow: "
        f"{missing_metric_keys}"

    )


print()
print(
    "OK - Métricas de VALIDATION disponibles"
)


print(
    "OK - Métricas de TEST disponibles"
)


print(
    "OK - Métricas de generalización disponibles"
)


# ============================================================
# 04.01.08 VALIDACIÓN FINAL
# ============================================================

print()
print("=" * 80)
print("VALIDACIÓN FINAL")
print("=" * 80)


print(
    "OK - MLflow importado correctamente"
)


print(
    "OK - Integración mlflow.spark disponible"
)


print(
    "OK - Tracking URI disponible"
)


print(
    "OK - Registry configurado para Unity Catalog"
)


print(
    "OK - Modelo final disponible"
)


print(
    "OK - Métricas disponibles para registro"
)


print(
    "OK - OOT continúa aislado"
)


print()
print(
    "MLflow preparado para registrar el modelo final."
)

# COMMAND ----------

# DBTITLE 1,04.02 REGISTRO DE PARÁMETROS Y MÉTRICAS EN MLFLOW
# ============================================================
# 04.02 REGISTRO DE PARÁMETROS Y MÉTRICAS EN MLFLOW
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Crear una nueva Run de MLflow y registrar:
#
# - configuración del modelo final
# - métricas de VALIDATION
# - métricas de TEST
# - métricas de generalización
# - tags descriptivos del proyecto
#
# IMPORTANTE
# ------------------------------------------------------------
#
# En este bloque todavía NO registramos el modelo como artefacto.
#
# Eso se hará en el siguiente bloque.
#
# ============================================================


import mlflow


# ============================================================
# 04.02.01 VALIDACIONES PREVIAS
# ============================================================

if "FINAL_MODEL_CONFIG" not in globals():

    raise RuntimeError(

        "No existe FINAL_MODEL_CONFIG. "
        "Ejecuta primero el bloque 02."

    )


if "generalization_metrics" not in globals():

    raise RuntimeError(

        "No existe generalization_metrics. "
        "Ejecuta primero el bloque 03.02."

    )


# ============================================================
# 04.02.02 PARÁMETROS DEL MODELO
# ============================================================

model_params = {

    "model_type":
        "RandomForestRegressor",

    "framework":
        "Spark ML",

    "config_name":
        FINAL_MODEL_CONFIG[
            "config_name"
        ],

    "numTrees":
        int(
            FINAL_MODEL_CONFIG[
                "numTrees"
            ]
        ),

    "maxDepth":
        int(
            FINAL_MODEL_CONFIG[
                "maxDepth"
            ]
        ),

    "minInstancesPerNode":
        int(
            FINAL_MODEL_CONFIG[
                "minInstancesPerNode"
            ]
        ),

    "featureSubsetStrategy":
        FINAL_MODEL_CONFIG[
            "featureSubsetStrategy"
        ],

    "seed":
        int(
            FINAL_MODEL_CONFIG[
                "seed"
            ]
        ),

    "training_strategy":
        "TRAIN + VALIDATION",

    "training_rows":
        int(
            train_final_rows
        ),

    "training_start_date":
        str(
            train_final_min_date
        ),

    "training_end_date":
        str(
            train_final_max_date
        ),

    "forecast_granularity":
        "daily_store",

    "target":
        "net_sales"

}


# ============================================================
# 04.02.03 MÉTRICAS
# ============================================================

mlflow_metrics = {

    "validation_mae":
        float(
            generalization_metrics[
                "validation_mae"
            ]
        ),

    "validation_rmse":
        float(
            generalization_metrics[
                "validation_rmse"
            ]
        ),

    "validation_wape":
        float(
            generalization_metrics[
                "validation_wape"
            ]
        ),

    "test_mae":
        float(
            generalization_metrics[
                "test_mae"
            ]
        ),

    "test_rmse":
        float(
            generalization_metrics[
                "test_rmse"
            ]
        ),

    "test_wape":
        float(
            generalization_metrics[
                "test_wape"
            ]
        ),

    "wape_difference_pp":
        float(
            generalization_metrics[
                "wape_difference_pp"
            ]
        ),

    "wape_relative_change":
        float(
            generalization_metrics[
                "wape_relative_change"
            ]
        )

}


# ============================================================
# 04.02.04 CREAMOS LA RUN
# ============================================================

print("=" * 80)
print("REGISTRO DE PARÁMETROS Y MÉTRICAS")
print("=" * 80)


with mlflow.start_run(

    run_name=
        "retail_sales_forecasting_rf_spark_final"

) as run:


    # ========================================================
    # PARÁMETROS
    # ========================================================

    mlflow.log_params(
        model_params
    )


    # ========================================================
    # MÉTRICAS
    # ========================================================

    mlflow.log_metrics(
        mlflow_metrics
    )


    # ========================================================
    # TAGS
    # ========================================================

    mlflow.set_tags({

        "project":
            "Retail Analytics",

        "use_case":
            "Sales Forecasting",

        "framework":
            "Spark ML",

        "algorithm":
            "RandomForestRegressor",

        "data_platform":
            "Databricks",

        "model_stage":
            "final_candidate",

        "forecast_level":
            "store_day",

        "training_strategy":
            "train_plus_validation",

        "test_usage":
            "final_evaluation_only",

        "oot_status":
            "isolated"

    })


    # ========================================================
    # IDENTIFICADORES DE LA RUN
    # ========================================================

    mlflow_run_id = (
        run.info.run_id
    )


    mlflow_experiment_id = (
        run.info.experiment_id
    )


# ============================================================
# 04.02.05 RESULTADOS
# ============================================================

print()
print(
    f"Run ID: "
    f"{mlflow_run_id}"
)


print(
    f"Experiment ID: "
    f"{mlflow_experiment_id}"
)


print()
print("=" * 80)
print("PARÁMETROS REGISTRADOS")
print("=" * 80)


for parameter_name, parameter_value in model_params.items():

    print(
        f"{parameter_name}: "
        f"{parameter_value}"
    )


print()
print("=" * 80)
print("MÉTRICAS REGISTRADAS")
print("=" * 80)


for metric_name, metric_value in mlflow_metrics.items():

    print(
        f"{metric_name}: "
        f"{metric_value:,.4f}"
    )


# ============================================================
# 04.02.06 VALIDACIÓN FINAL
# ============================================================

print()
print("=" * 80)
print("VALIDACIÓN FINAL")
print("=" * 80)


if not mlflow_run_id:

    raise RuntimeError(
        "No se ha generado un Run ID válido."
    )


if not mlflow_experiment_id:

    raise RuntimeError(
        "No se ha generado un Experiment ID válido."
    )


print(
    "OK - Run de MLflow creada"
)


print(
    "OK - Parámetros registrados"
)


print(
    "OK - Métricas de VALIDATION registradas"
)


print(
    "OK - Métricas de TEST registradas"
)


print(
    "OK - Métricas de generalización registradas"
)


print(
    "OK - Tags registrados"
)


print(
    "OK - OOT continúa aislado"
)


print()
print(
    "Registro de parámetros y métricas completado."
)

# COMMAND ----------

# DBTITLE 1,04.03 REGISTRO DEL MODELO SPARK ML EN MLFLOW
# ============================================================
# 04.03 REGISTRO DEL MODELO SPARK ML EN MLFLOW
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Registrar en MLflow + Unity Catalog exactamente el modelo
# Spark ML final que ya fue evaluado sobre TEST.
#
# Requisitos específicos de Databricks Serverless / UC:
#
# - utilizar un Unity Catalog Volume como almacenamiento
#   temporal para serializar Spark ML
#
# - registrar una Model Signature explícita
#
# IMPORTANTE
# ------------------------------------------------------------
#
# - NO se reconstruye el modelo.
# - NO se vuelve a entrenar.
# - Se registra exactamente final_rf_model.
# - Se reutiliza la Run creada en 04.02.
# - TEST no se utiliza para modificar el modelo.
# - OOT continúa completamente aislado.
#
# ============================================================


import os

import mlflow
import mlflow.spark

from mlflow.models import infer_signature


# ============================================================
# 04.03.01 VALIDACIONES PREVIAS
# ============================================================

if "final_rf_model" not in globals():

    raise RuntimeError(

        "No existe final_rf_model. "
        "Ejecuta primero el bloque 02."

    )


if "mlflow_run_id" not in globals():

    raise RuntimeError(

        "No existe mlflow_run_id. "
        "Ejecuta primero el bloque 04.02."

    )


if "train_final_df" not in globals():

    raise RuntimeError(

        "No existe train_final_df. "
        "Ejecuta primero el bloque 02."

    )


# ============================================================
# 04.03.02 CONFIGURACIÓN
# ============================================================

MODEL_ARTIFACT_PATH = (
    "model"
)


MODEL_REGISTERED_NAME = (
    "retail_analytics.5_ml.sales_forecasting_random_forest"
)


# ------------------------------------------------------------
# UC VOLUME TEMPORAL PARA SPARK ML
# ------------------------------------------------------------

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
# 04.03.03 DATASET DE EJEMPLO PARA LA SIGNATURE
# ============================================================
#
# Unity Catalog exige que el modelo registrado tenga una
# Model Signature.
#
# Para inferirla utilizamos una pequeña muestra del dataset
# de entrenamiento.
#
# IMPORTANTE:
#
# Esto NO entrena el modelo.
#
# Únicamente utilizamos el modelo ya entrenado para obtener
# el esquema real de entrada y salida.
#
# ============================================================


signature_input_df = (

    train_final_df

    .limit(
        10
    )

)


# ------------------------------------------------------------
# Generamos las predicciones correspondientes.
# ------------------------------------------------------------

signature_output_df = (

    final_rf_model

    .transform(
        signature_input_df
    )

    .select(
        "prediction"
    )

)


# ============================================================
# 04.03.04 CONVERSIÓN PARA INFER_SIGNATURE
# ============================================================
#
# infer_signature trabaja correctamente con estructuras
# tabulares tipo pandas.
#
# Como solo utilizamos 10 registros, la conversión es
# totalmente segura para memoria.
#
# ============================================================


signature_input_pdf = (

    signature_input_df

    .toPandas()

)


signature_output_pdf = (

    signature_output_df

    .toPandas()

)


# ============================================================
# 04.03.05 CREACIÓN DE MODEL SIGNATURE
# ============================================================

model_signature = infer_signature(

    signature_input_pdf,

    signature_output_pdf

)


print("=" * 80)
print("MODEL SIGNATURE")
print("=" * 80)


print(
    model_signature
)


# ============================================================
# 04.03.06 INFORMACIÓN DEL REGISTRO
# ============================================================

print()
print("=" * 80)
print("REGISTRO DEL MODELO SPARK ML EN MLFLOW")
print("=" * 80)


print(
    f"Run ID: "
    f"{mlflow_run_id}"
)


print(
    f"Artifact path: "
    f"{MODEL_ARTIFACT_PATH}"
)


print(
    f"Registered Model: "
    f"{MODEL_REGISTERED_NAME}"
)


print(
    f"MLflow DFS temp: "
    f"{MLFLOW_DFS_TMP}"
)


# ============================================================
# 04.03.07 REGISTRO DEL MODELO
# ============================================================
#
# Reabrimos exactamente la misma Run creada en 04.02.
#
# De esta forma quedan asociados:
#
# - parámetros
# - métricas
# - tags
# - modelo
#
# dentro de una única ejecución MLflow.
#
# ============================================================


with mlflow.start_run(
    run_id=mlflow_run_id
):


    model_info = mlflow.spark.log_model(

        spark_model=
            final_rf_model,

        artifact_path=
            MODEL_ARTIFACT_PATH,

        registered_model_name=
            MODEL_REGISTERED_NAME,

        dfs_tmpdir=
            MLFLOW_DFS_TMP,

        signature=
            model_signature

    )


# ============================================================
# 04.03.08 INFORMACIÓN DEL MODELO REGISTRADO
# ============================================================

logged_model_uri = (
    model_info.model_uri
)


print()
print("=" * 80)
print("MODELO REGISTRADO")
print("=" * 80)


print(
    f"Model URI: "
    f"{logged_model_uri}"
)


print(
    f"Registered Model: "
    f"{MODEL_REGISTERED_NAME}"
)


# ============================================================
# 04.03.09 VALIDACIONES
# ============================================================

if not logged_model_uri:

    raise RuntimeError(

        "MLflow no ha devuelto "
        "un Model URI válido."

    )


if model_signature is None:

    raise RuntimeError(

        "No se ha generado una "
        "Model Signature válida."

    )


# ============================================================
# 04.03.10 VARIABLES PARA BLOQUES POSTERIORES
# ============================================================

registered_model_name = (
    MODEL_REGISTERED_NAME
)


registered_model_uri = (
    logged_model_uri
)


# ============================================================
# 04.03.11 VALIDACIÓN FINAL
# ============================================================

print()
print("=" * 80)
print("VALIDACIÓN FINAL")
print("=" * 80)


print(
    "OK - UC Volume temporal configurado"
)


print(
    "OK - Model Signature generada"
)


print(
    "OK - Modelo Spark ML registrado en MLflow"
)


print(
    "OK - Modelo registrado en Unity Catalog"
)


print(
    "OK - Modelo asociado a la Run de 04.02"
)


print(
    f"OK - Registered Model: "
    f"{registered_model_name}"
)


print(
    f"OK - Model URI: "
    f"{registered_model_uri}"
)


print(
    "OK - No se ha realizado un nuevo entrenamiento"
)


print(
    "OK - Se ha registrado el modelo evaluado sobre TEST"
)


print(
    "OK - OOT continúa aislado"
)


print()
print(
    "Registro del modelo Spark ML completado."
)

# COMMAND ----------

# DBTITLE 1,04.04 VALIDACIÓN DEL MODELO REGISTRADO EN UNITY CATALOG
# ============================================================
# 04.04 VALIDACIÓN DEL MODELO REGISTRADO EN UNITY CATALOG
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Validar que el modelo registrado en Unity Catalog puede:
#
# 1. localizarse correctamente
# 2. cargarse desde Model Registry
# 3. ejecutar inferencia sobre TEST
# 4. reproducir las métricas obtenidas antes del registro
#
# De esta forma comprobamos que el artefacto registrado es
# realmente el mismo modelo que fue evaluado en 03.01.
#
# IMPORTANTE
# ------------------------------------------------------------
#
# - NO se entrena ningún modelo.
# - NO se modifican hiperparámetros.
# - TEST se utiliza únicamente para validación/reproducibilidad.
# - OOT continúa completamente aislado.
#
# ============================================================


import mlflow
import mlflow.spark

from mlflow import MlflowClient

from pyspark.sql import functions as F

from pyspark.ml.evaluation import (
    RegressionEvaluator
)


# ============================================================
# 04.04.01 VALIDACIONES PREVIAS
# ============================================================

if "test_df" not in globals():

    raise RuntimeError(

        "No existe test_df. "
        "Ejecuta primero el bloque 01."

    )


if "final_test_metrics" not in globals():

    raise RuntimeError(

        "No existe final_test_metrics. "
        "Ejecuta primero el bloque 03.01."

    )


if "registered_model_name" not in globals():

    raise RuntimeError(

        "No existe registered_model_name. "
        "Ejecuta primero el bloque 04.03."

    )


# ============================================================
# 04.04.02 CONFIGURACIÓN DE UNITY CATALOG
# ============================================================

mlflow.set_registry_uri(
    "databricks-uc"
)


client = (
    MlflowClient()
)


# ============================================================
# 04.04.03 LOCALIZAMOS LA ÚLTIMA VERSIÓN
# ============================================================
#
# No hardcodeamos "2".
#
# Consultamos Unity Catalog y obtenemos las versiones
# disponibles del modelo.
#
# Esto permite que el notebook siga funcionando si en el
# futuro se registra una versión 3, 4, etc.
#
# ============================================================

model_versions = list(

    client.search_model_versions(

        f"name='{registered_model_name}'"

    )

)


if not model_versions:

    raise RuntimeError(

        "No se han encontrado versiones para el modelo "
        f"{registered_model_name}."

    )


latest_model_version = max(

    model_versions,

    key=lambda model_version:
        int(
            model_version.version
        )

)


registered_model_version = (
    str(
        latest_model_version.version
    )
)


registered_model_run_id = (
    latest_model_version.run_id
)


registered_model_source = (
    latest_model_version.source
)


# ============================================================
# 04.04.04 URI DEL MODELO
# ============================================================

REGISTERED_MODEL_URI = (

    f"models:/{registered_model_name}/"
    f"{registered_model_version}"

)


print("=" * 80)
print("MODELO REGISTRADO EN UNITY CATALOG")
print("=" * 80)


print(
    f"Modelo:  "
    f"{registered_model_name}"
)


print(
    f"Versión: "
    f"{registered_model_version}"
)


print(
    f"Run ID:  "
    f"{registered_model_run_id}"
)


print(
    f"URI:     "
    f"{REGISTERED_MODEL_URI}"
)


# ============================================================
# 04.04.05 VALIDACIÓN DE TRAZABILIDAD
# ============================================================
#
# La versión registrada debe proceder de la misma Run creada
# en 04.02 y utilizada en 04.03.
#
# ============================================================

if "mlflow_run_id" in globals():

    if (
        registered_model_run_id
        !=
        mlflow_run_id
    ):

        raise RuntimeError(

            "La última versión registrada no pertenece "
            "a la Run MLflow creada en 04.02. "
            f"Esperada: {mlflow_run_id} | "
            f"Encontrada: {registered_model_run_id}"

        )


print()
print(
    "OK - La versión registrada pertenece "
    "a la Run esperada"
)


# ============================================================
# 04.04.06 CARGA DEL MODELO DESDE UNITY CATALOG
# ============================================================
#
# A partir de aquí NO utilizamos final_rf_model.
#
# Cargamos el modelo directamente desde Model Registry.
#
# ============================================================

print()
print("=" * 80)
print("CARGA DEL MODELO DESDE UNITY CATALOG")
print("=" * 80)


loaded_registered_model = (

    mlflow.spark.load_model(
        REGISTERED_MODEL_URI
    )

)


print(
    "OK - Modelo cargado correctamente "
    "desde Unity Catalog"
)


# ============================================================
# 04.04.07 INFERENCIA SOBRE TEST
# ============================================================

registered_test_predictions = (

    loaded_registered_model

    .transform(
        test_df
    )

    .select(

        "sale_date",
        "store_id",
        "net_sales",
        "prediction"

    )

)


registered_prediction_rows = (

    registered_test_predictions
    .count()

)


expected_test_rows = (
    test_df.count()
)


if (
    registered_prediction_rows
    !=
    expected_test_rows
):

    raise RuntimeError(

        "El modelo registrado no ha generado el mismo "
        "número de predicciones que registros existen "
        "en TEST."

    )


print()
print(
    f"Registros TEST:       "
    f"{expected_test_rows:,}"
)


print(
    f"Predicciones modelo:  "
    f"{registered_prediction_rows:,}"
)


# ============================================================
# 04.04.08 CALIDAD DE LAS PREDICCIONES
# ============================================================

registered_prediction_quality = (

    registered_test_predictions

    .agg(

        F.sum(

            F.col("prediction")
            .isNull()
            .cast("int")

        ).alias(
            "prediction_nulls"
        ),

        F.sum(

            F.isnan(
                F.col("prediction")
            )
            .cast("int")

        ).alias(
            "prediction_nans"
        )

    )

    .first()

)


registered_prediction_nulls = int(

    registered_prediction_quality[
        "prediction_nulls"
    ] or 0

)


registered_prediction_nans = int(

    registered_prediction_quality[
        "prediction_nans"
    ] or 0

)


if registered_prediction_nulls != 0:

    raise RuntimeError(

        "El modelo registrado genera "
        "predicciones NULL."

    )


if registered_prediction_nans != 0:

    raise RuntimeError(

        "El modelo registrado genera "
        "predicciones NaN."

    )


print(
    f"NULL predictions:     "
    f"{registered_prediction_nulls:,}"
)


print(
    f"NaN predictions:      "
    f"{registered_prediction_nans:,}"
)


# ============================================================
# 04.04.09 MAE
# ============================================================

registered_mae_evaluator = RegressionEvaluator(

    labelCol=
        "net_sales",

    predictionCol=
        "prediction",

    metricName=
        "mae"

)


registered_test_mae = (

    registered_mae_evaluator

    .evaluate(
        registered_test_predictions
    )

)


# ============================================================
# 04.04.10 RMSE
# ============================================================

registered_rmse_evaluator = RegressionEvaluator(

    labelCol=
        "net_sales",

    predictionCol=
        "prediction",

    metricName=
        "rmse"

)


registered_test_rmse = (

    registered_rmse_evaluator

    .evaluate(
        registered_test_predictions
    )

)


# ============================================================
# 04.04.11 WAPE
# ============================================================

registered_wape_stats = (

    registered_test_predictions

    .agg(

        F.sum(

            F.abs(

                F.col("net_sales")
                -
                F.col("prediction")

            )

        ).alias(
            "absolute_error"
        ),

        F.sum(

            F.abs(
                F.col("net_sales")
            )

        ).alias(
            "actual_total"
        )

    )

    .first()

)


registered_absolute_error = float(

    registered_wape_stats[
        "absolute_error"
    ]

)


registered_actual_total = float(

    registered_wape_stats[
        "actual_total"
    ]

)


if registered_actual_total == 0:

    raise RuntimeError(

        "No se puede calcular WAPE porque "
        "el total real de TEST es 0."

    )


registered_test_wape = (

    registered_absolute_error
    /
    registered_actual_total
    *
    100.0

)


# ============================================================
# 04.04.12 MÉTRICAS ORIGINALES
# ============================================================

original_test_mae = float(

    final_test_metrics[
        "mae"
    ]

)


original_test_rmse = float(

    final_test_metrics[
        "rmse"
    ]

)


original_test_wape = float(

    final_test_metrics[
        "wape"
    ]

)


# ============================================================
# 04.04.13 COMPARACIÓN
# ============================================================

mae_difference = abs(

    registered_test_mae
    -
    original_test_mae

)


rmse_difference = abs(

    registered_test_rmse
    -
    original_test_rmse

)


wape_difference = abs(

    registered_test_wape
    -
    original_test_wape

)


print()
print("=" * 80)
print("REPRODUCIBILIDAD DEL MODELO REGISTRADO")
print("=" * 80)


print()
print(
    f"{'Métrica':10} | "
    f"{'Original':>14} | "
    f"{'Registrado':>14} | "
    f"{'Diferencia':>14}"
)


print("-" * 62)


print(

    f"{'MAE':10} | "
    f"{original_test_mae:>14,.6f} | "
    f"{registered_test_mae:>14,.6f} | "
    f"{mae_difference:>14,.10f}"

)


print(

    f"{'RMSE':10} | "
    f"{original_test_rmse:>14,.6f} | "
    f"{registered_test_rmse:>14,.6f} | "
    f"{rmse_difference:>14,.10f}"

)


print(

    f"{'WAPE':10} | "
    f"{original_test_wape:>13.6f}% | "
    f"{registered_test_wape:>13.6f}% | "
    f"{wape_difference:>14.10f}"

)


# ============================================================
# 04.04.14 TOLERANCIA NUMÉRICA
# ============================================================
#
# Esperamos que las métricas sean esencialmente idénticas.
#
# Utilizamos una tolerancia muy pequeña para evitar falsos
# errores derivados únicamente de precisión numérica.
#
# ============================================================

METRIC_TOLERANCE = (
    1e-6
)


if mae_difference > METRIC_TOLERANCE:

    raise RuntimeError(

        "El MAE del modelo registrado no reproduce "
        "el MAE original."

    )


if rmse_difference > METRIC_TOLERANCE:

    raise RuntimeError(

        "El RMSE del modelo registrado no reproduce "
        "el RMSE original."

    )


if wape_difference > METRIC_TOLERANCE:

    raise RuntimeError(

        "El WAPE del modelo registrado no reproduce "
        "el WAPE original."

    )


# ============================================================
# 04.04.15 RESULTADO PARA BLOQUES POSTERIORES
# ============================================================

registered_model_validation = {

    "model_name":
        registered_model_name,

    "model_version":
        registered_model_version,

    "run_id":
        registered_model_run_id,

    "model_uri":
        REGISTERED_MODEL_URI,

    "test_rows":
        int(
            registered_prediction_rows
        ),

    "test_mae":
        float(
            registered_test_mae
        ),

    "test_rmse":
        float(
            registered_test_rmse
        ),

    "test_wape":
        float(
            registered_test_wape
        ),

    "reproducible":
        True

}


# ============================================================
# 04.04.16 VALIDACIÓN FINAL
# ============================================================

print()
print("=" * 80)
print("VALIDACIÓN FINAL")
print("=" * 80)


print(
    "OK - Modelo localizado en Unity Catalog"
)


print(
    f"OK - Versión validada: "
    f"{registered_model_version}"
)


print(
    "OK - Modelo cargado desde Model Registry"
)


print(
    f"OK - "
    f"{registered_prediction_rows:,} "
    f"predicciones generadas sobre TEST"
)


print(
    "OK - Sin predicciones NULL"
)


print(
    "OK - Sin predicciones NaN"
)


print(
    "OK - MAE reproducido"
)


print(
    "OK - RMSE reproducido"
)


print(
    "OK - WAPE reproducido"
)


print(
    "OK - Modelo registrado reproducible"
)


print(
    "OK - No se ha realizado ningún reentrenamiento"
)


print(
    "OK - OOT continúa aislado"
)


print()
print(
    "Validación del modelo registrado completada."
)

# COMMAND ----------

# DBTITLE 1,04.05 CIERRE DE MLFLOW Y MLOPS
# ============================================================
# 04.05 CIERRE DE MLFLOW Y MLOPS
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Realizar una validación final de toda la fase MLOps y dejar
# preparados los identificadores principales del modelo para
# los siguientes notebooks/procesos.
#
# En esta fase ya hemos completado:
#
# 04.01 - Configuración de MLflow
# 04.02 - Registro de parámetros y métricas
# 04.03 - Registro del modelo Spark ML
# 04.04 - Validación del modelo registrado
#
# IMPORTANTE
# ------------------------------------------------------------
#
# Este bloque:
#
# - NO entrena modelos
# - NO registra nuevas versiones
# - NO utiliza OOT
# - únicamente valida y resume el estado final
#
# ============================================================


import mlflow

from mlflow import MlflowClient


# ============================================================
# 04.05.01 VALIDACIONES PREVIAS
# ============================================================

required_objects = [

    "mlflow_run_id",
    "registered_model_name",
    "registered_model_version",
    "registered_model_validation",
    "generalization_metrics"

]


missing_objects = [

    object_name

    for object_name in required_objects

    if object_name not in globals()

]


if missing_objects:

    raise RuntimeError(

        "Faltan objetos necesarios para cerrar MLOps: "
        f"{missing_objects}"

    )


# ============================================================
# 04.05.02 CONFIGURACIÓN DE MLFLOW
# ============================================================

mlflow.set_registry_uri(
    "databricks-uc"
)


client = (
    MlflowClient()
)


# ============================================================
# 04.05.03 RECUPERAMOS LA RUN
# ============================================================

run_info = (

    client.get_run(
        mlflow_run_id
    )

)


run_status = (
    run_info.info.status
)


run_experiment_id = (
    run_info.info.experiment_id
)


# ============================================================
# 04.05.04 RECUPERAMOS LA VERSIÓN REGISTRADA
# ============================================================

model_version_info = (

    client.get_model_version(

        name=
            registered_model_name,

        version=
            registered_model_version

    )

)


model_version_run_id = (
    model_version_info.run_id
)


model_version_status = (
    model_version_info.status
)


# ============================================================
# 04.05.05 VALIDACIONES DE TRAZABILIDAD
# ============================================================

if (
    model_version_run_id
    !=
    mlflow_run_id
):

    raise RuntimeError(

        "La versión registrada no está asociada "
        "a la Run MLflow esperada."

    )


if not registered_model_validation.get(
    "reproducible",
    False
):

    raise RuntimeError(

        "El modelo registrado no consta "
        "como reproducible."

    )


# ============================================================
# 04.05.06 MÉTRICAS PRINCIPALES
# ============================================================

validation_wape = float(

    generalization_metrics[
        "validation_wape"
    ]

)


test_wape = float(

    generalization_metrics[
        "test_wape"
    ]

)


wape_difference_pp = float(

    generalization_metrics[
        "wape_difference_pp"
    ]

)


test_mae = float(

    generalization_metrics[
        "test_mae"
    ]

)


test_rmse = float(

    generalization_metrics[
        "test_rmse"
    ]

)


# ============================================================
# 04.05.07 RESUMEN
# ============================================================

print("=" * 80)
print("CIERRE DE MLFLOW Y MLOPS")
print("=" * 80)


print()
print("MLFLOW")
print("-" * 80)


print(
    f"Experiment ID: "
    f"{run_experiment_id}"
)


print(
    f"Run ID:        "
    f"{mlflow_run_id}"
)


print(
    f"Run status:    "
    f"{run_status}"
)


print()
print("MODEL REGISTRY")
print("-" * 80)


print(
    f"Modelo:        "
    f"{registered_model_name}"
)


print(
    f"Versión:       "
    f"{registered_model_version}"
)


print(
    f"Version status:"
    f" {model_version_status}"
)


print()
print("RENDIMIENTO")
print("-" * 80)


print(
    f"Validation WAPE: "
    f"{validation_wape:.4f}%"
)


print(
    f"Test MAE:        "
    f"{test_mae:,.2f}"
)


print(
    f"Test RMSE:       "
    f"{test_rmse:,.2f}"
)


print(
    f"Test WAPE:       "
    f"{test_wape:.4f}%"
)


print(
    f"Diferencia WAPE: "
    f"{wape_difference_pp:+.4f} pp"
)


print()
print("REPRODUCIBILIDAD")
print("-" * 80)


print(
    f"Modelo reproducible: "
    f"{registered_model_validation['reproducible']}"
)


print(
    f"TEST validado: "
    f"{registered_model_validation['test_rows']:,} registros"
)


# ============================================================
# 04.05.08 REFERENCIA FINAL DEL MODELO
# ============================================================
#
# Dejamos una estructura sencilla que podrá utilizarse en
# notebooks posteriores de inferencia, monitoring, etc.
#
# ============================================================

production_model_reference = {

    "model_name":
        registered_model_name,

    "model_version":
        str(
            registered_model_version
        ),

    "model_uri":

        (
            f"models:/"
            f"{registered_model_name}/"
            f"{registered_model_version}"
        ),

    "run_id":
        mlflow_run_id,

    "experiment_id":
        str(
            run_experiment_id
        ),

    "framework":
        "Spark ML",

    "algorithm":
        "RandomForestRegressor",

    "config_name":
        "RF_01_BASE",

    "target":
        "net_sales",

    "granularity":
        "store_day",

    "validation_wape":
        validation_wape,

    "test_wape":
        test_wape,

    "reproducible":
        True

}


# ============================================================
# 04.05.09 VALIDACIÓN FINAL
# ============================================================

print()
print("=" * 80)
print("VALIDACIÓN FINAL")
print("=" * 80)


print(
    "OK - Run MLflow localizada"
)


print(
    "OK - Parámetros y métricas registrados"
)


print(
    "OK - Modelo Spark ML registrado"
)


print(
    f"OK - Model Version "
    f"{registered_model_version} localizada"
)


print(
    "OK - Model Version asociada a la Run correcta"
)


print(
    "OK - Modelo cargado y validado desde Unity Catalog"
)


print(
    "OK - Predicciones reproducidas exactamente"
)


print(
    "OK - TEST utilizado únicamente para evaluación"
)


print(
    "OK - OOT continúa aislado"
)


print(
    "OK - production_model_reference preparado"
)


print()
print(
    "Fase MLflow / MLOps cerrada correctamente."
)

# COMMAND ----------

# MAGIC %md
# MAGIC # 05. DOCUMENTACIÓN Y GOBIERNO DEL MODELO
# MAGIC
# MAGIC ## Objetivo
# MAGIC
# MAGIC Documentar formalmente el modelo final registrado en **Unity Catalog** para facilitar su gobierno, trazabilidad, mantenimiento y reutilización.
# MAGIC
# MAGIC El modelo ya ha completado las fases de:
# MAGIC
# MAGIC - desarrollo y selección
# MAGIC - entrenamiento final
# MAGIC - evaluación sobre Test
# MAGIC - MLflow Tracking
# MAGIC - registro en Unity Catalog
# MAGIC - validación de reproducibilidad
# MAGIC
# MAGIC Modelo registrado:
# MAGIC
# MAGIC **`retail_analytics.5_ml.sales_forecasting_random_forest`**
# MAGIC
# MAGIC Versión actual:
# MAGIC
# MAGIC **Version 2**
# MAGIC
# MAGIC Framework:
# MAGIC
# MAGIC **Spark ML**
# MAGIC
# MAGIC Algoritmo:
# MAGIC
# MAGIC **RandomForestRegressor**
# MAGIC
# MAGIC En esta fase añadiremos metadata descriptiva al modelo registrado para que pueda comprenderse y gestionarse sin depender exclusivamente del notebook donde fue desarrollado.
# MAGIC
# MAGIC Se documentarán aspectos como:
# MAGIC
# MAGIC - propósito del modelo
# MAGIC - caso de uso
# MAGIC - variable objetivo
# MAGIC - granularidad del forecast
# MAGIC - framework y algoritmo
# MAGIC - configuración seleccionada
# MAGIC - estrategia de entrenamiento
# MAGIC - métricas de Validation y Test
# MAGIC - estado de generalización
# MAGIC - procedencia de la Run de MLflow
# MAGIC - reproducibilidad del modelo

# COMMAND ----------

# DBTITLE 1,05.01 DOCUMENTACIÓN Y METADATOS DEL MODELO
# ============================================================
# 05.01 DOCUMENTACIÓN Y METADATOS DEL MODELO
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Añadir documentación y metadata al modelo registrado en
# Unity Catalog.
#
# Documentaremos dos niveles:
#
# 1. Registered Model
#    -> descripción general del modelo y su propósito.
#
# 2. Model Version
#    -> información específica de la versión actualmente
#       validada.
#
# IMPORTANTE
# ------------------------------------------------------------
#
# - NO se registra una nueva versión.
# - NO se modifica el modelo.
# - NO se vuelve a entrenar.
# - La versión se obtiene dinámicamente.
# - OOT continúa aislado.
#
# ============================================================


import mlflow

from mlflow import MlflowClient


# ============================================================
# 05.01.01 VALIDACIONES PREVIAS
# ============================================================

required_objects = [

    "registered_model_name",
    "registered_model_version",
    "mlflow_run_id",
    "generalization_metrics",
    "registered_model_validation",
    "production_model_reference"

]


missing_objects = [

    object_name

    for object_name in required_objects

    if object_name not in globals()

]


if missing_objects:

    raise RuntimeError(

        "Faltan objetos necesarios para documentar el modelo: "
        f"{missing_objects}"

    )


# ============================================================
# 05.01.02 CONFIGURACIÓN DE UNITY CATALOG
# ============================================================

mlflow.set_registry_uri(
    "databricks-uc"
)


client = (
    MlflowClient()
)


# ============================================================
# 05.01.03 INFORMACIÓN PRINCIPAL
# ============================================================

MODEL_NAME = (
    registered_model_name
)


MODEL_VERSION = (
    str(
        registered_model_version
    )
)


MODEL_RUN_ID = (
    mlflow_run_id
)


print("=" * 80)
print("DOCUMENTACIÓN Y METADATOS DEL MODELO")
print("=" * 80)


print(
    f"Modelo:  "
    f"{MODEL_NAME}"
)


print(
    f"Versión: "
    f"{MODEL_VERSION}"
)


print(
    f"Run ID:  "
    f"{MODEL_RUN_ID}"
)


# ============================================================
# 05.01.04 VALIDAMOS LA VERSIÓN
# ============================================================

model_version_info = (

    client.get_model_version(

        name=
            MODEL_NAME,

        version=
            MODEL_VERSION

    )

)


if (
    model_version_info.run_id
    !=
    MODEL_RUN_ID
):

    raise RuntimeError(

        "La Model Version indicada no pertenece "
        "a la Run MLflow esperada."

    )


print()
print(
    "OK - Model Version localizada"
)


print(
    "OK - Model Version asociada a la Run correcta"
)


# ============================================================
# 05.01.05 DESCRIPCIÓN DEL REGISTERED MODEL
# ============================================================

registered_model_description = """
Modelo de forecasting de ventas del proyecto Retail Analytics.

El modelo predice net_sales a nivel diario por tienda utilizando
un RandomForestRegressor de Spark ML y features temporales,
lags, rolling statistics y la identificación de la tienda.

El modelo forma parte de un pipeline de Data Engineering y
Machine Learning desarrollado en Databricks sobre una
arquitectura Medallion.

El ciclo de vida del modelo incluye:

- Feature Engineering persistido en Delta
- separación temporal TRAIN / VALIDATION / TEST / OOT
- Hyperparameter Tuning
- selección del modelo
- entrenamiento final con TRAIN + VALIDATION
- evaluación independiente sobre TEST
- MLflow Tracking
- registro en Unity Catalog
- validación de reproducibilidad
- inferencia
- monitoring

La versión registrada puede cargarse directamente desde
Unity Catalog para procesos posteriores de inferencia y
monitorización.
""".strip()


client.update_registered_model(

    name=
        MODEL_NAME,

    description=
        registered_model_description

)


# ============================================================
# 05.01.06 DESCRIPCIÓN DE LA MODEL VERSION
# ============================================================

version_description = f"""
Modelo final de forecasting de ventas.

Framework:
Spark ML

Algoritmo:
RandomForestRegressor

Configuración:
RF_01_BASE

Hiperparámetros:
- numTrees: 60
- maxDepth: 8
- minInstancesPerNode: 3
- featureSubsetStrategy: auto
- seed: 42

Variable objetivo:
net_sales

Granularidad:
store_day

Estrategia de entrenamiento:
TRAIN + VALIDATION

Periodo de entrenamiento:
2024-01-29 -> 2026-04-30

Registros de entrenamiento:
25,513

Validation:
- MAE: {generalization_metrics["validation_mae"]:.4f}
- RMSE: {generalization_metrics["validation_rmse"]:.4f}
- WAPE: {generalization_metrics["validation_wape"]:.4f}%

Test:
- MAE: {generalization_metrics["test_mae"]:.4f}
- RMSE: {generalization_metrics["test_rmse"]:.4f}
- WAPE: {generalization_metrics["test_wape"]:.4f}%

Diferencia Validation -> Test:
- WAPE: {generalization_metrics["wape_difference_pp"]:+.4f} pp

Estado de generalización:
ESTABLE

Reproducibilidad:
VALIDADA

Run MLflow:
{MODEL_RUN_ID}

TEST se utilizó exclusivamente para evaluación final.

OOT permanece reservado para evaluación fuera de tiempo y
monitorización posterior.
""".strip()


client.update_model_version(

    name=
        MODEL_NAME,

    version=
        MODEL_VERSION,

    description=
        version_description

)


# ============================================================
# 05.01.07 TAGS DEL REGISTERED MODEL
# ============================================================
#
# Los tags permiten identificar rápidamente el propósito
# y características principales del modelo en Unity Catalog.
#
# ============================================================

registered_model_tags = {

    "project":
        "retail_analytics",

    "use_case":
        "sales_forecasting",

    "framework":
        "spark_ml",

    "algorithm":
        "random_forest",

    "target":
        "net_sales",

    "granularity":
        "store_day",

    "data_platform":
        "databricks"

}


for tag_key, tag_value in registered_model_tags.items():

    client.set_registered_model_tag(

        name=
            MODEL_NAME,

        key=
            tag_key,

        value=
            str(
                tag_value
            )

    )


# ============================================================
# 05.01.08 TAGS DE LA MODEL VERSION
# ============================================================

model_version_tags = {

    "config_name":
        "RF_01_BASE",

    "training_strategy":
        "train_plus_validation",

    "validation_status":
        "validated",

    "generalization_status":
        "stable",

    "reproducibility":
        "validated",

    "test_usage":
        "final_evaluation_only",

    "oot_status":
        "isolated",

    "mlflow_run_id":
        MODEL_RUN_ID

}


for tag_key, tag_value in model_version_tags.items():

    client.set_model_version_tag(

        name=
            MODEL_NAME,

        version=
            MODEL_VERSION,

        key=
            tag_key,

        value=
            str(
                tag_value
            )

    )


# ============================================================
# 05.01.09 RECUPERAMOS EL MODELO DOCUMENTADO
# ============================================================
#
# Volvemos a consultar Unity Catalog para verificar que las
# descripciones y tags se han persistido.
#
# ============================================================

documented_registered_model = (

    client.get_registered_model(
        MODEL_NAME
    )

)


documented_model_version = (

    client.get_model_version(

        name=
            MODEL_NAME,

        version=
            MODEL_VERSION

    )

)


# ============================================================
# 05.01.10 VALIDACIÓN DE DOCUMENTACIÓN
# ============================================================

if not documented_registered_model.description:

    raise RuntimeError(

        "El Registered Model no contiene descripción."

    )


if not documented_model_version.description:

    raise RuntimeError(

        "La Model Version no contiene descripción."

    )


# ============================================================
# 05.01.11 RESUMEN
# ============================================================

print()
print("=" * 80)
print("METADATA REGISTRADA")
print("=" * 80)


print(
    f"Registered Model: "
    f"{MODEL_NAME}"
)


print(
    f"Model Version:    "
    f"{MODEL_VERSION}"
)


print(
    f"Run ID:           "
    f"{MODEL_RUN_ID}"
)


print(
    f"Framework:        "
    f"{production_model_reference['framework']}"
)


print(
    f"Algoritmo:        "
    f"{production_model_reference['algorithm']}"
)


print(
    f"Target:           "
    f"{production_model_reference['target']}"
)


print(
    f"Granularidad:     "
    f"{production_model_reference['granularity']}"
)


print(
    f"Validation WAPE:  "
    f"{generalization_metrics['validation_wape']:.4f}%"
)


print(
    f"Test WAPE:        "
    f"{generalization_metrics['test_wape']:.4f}%"
)


print(
    f"Reproducible:     "
    f"{registered_model_validation['reproducible']}"
)


# ============================================================
# 05.01.12 VALIDACIÓN FINAL
# ============================================================

print()
print("=" * 80)
print("VALIDACIÓN FINAL")
print("=" * 80)


print(
    "OK - Registered Model documentado"
)


print(
    f"OK - Model Version "
    f"{MODEL_VERSION} documentada"
)


print(
    "OK - Descripción del modelo registrada"
)


print(
    "OK - Descripción de la versión registrada"
)


print(
    "OK - Tags del Registered Model registrados"
)


print(
    "OK - Tags de la Model Version registrados"
)


print(
    "OK - Trazabilidad con MLflow preservada"
)


print(
    "OK - Métricas de Validation documentadas"
)


print(
    "OK - Métricas de Test documentadas"
)


print(
    "OK - Estado de reproducibilidad documentado"
)


print(
    "OK - No se ha creado una nueva Model Version"
)


print(
    "OK - OOT continúa aislado"
)


print()
print(
    "Documentación y metadata del modelo completadas."
)

# COMMAND ----------

# MAGIC %md
# MAGIC # 05.02 CONCLUSIONES DEL MODELO
# MAGIC
# MAGIC ## Resultado final
# MAGIC
# MAGIC El modelo final de **Sales Forecasting** ha completado correctamente todo su ciclo de desarrollo, evaluación, registro y documentación.
# MAGIC
# MAGIC Modelo registrado:
# MAGIC
# MAGIC **`retail_analytics.5_ml.sales_forecasting_random_forest`**
# MAGIC
# MAGIC Versión validada:
# MAGIC
# MAGIC **Version 2**
# MAGIC
# MAGIC Framework:
# MAGIC
# MAGIC **Spark ML**
# MAGIC
# MAGIC Algoritmo:
# MAGIC
# MAGIC **RandomForestRegressor**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Configuración seleccionada
# MAGIC
# MAGIC La configuración final seleccionada fue:
# MAGIC
# MAGIC - `numTrees = 60`
# MAGIC - `maxDepth = 8`
# MAGIC - `minInstancesPerNode = 3`
# MAGIC - `featureSubsetStrategy = auto`
# MAGIC - `seed = 42`
# MAGIC
# MAGIC Configuración:
# MAGIC
# MAGIC **RF_01_BASE**
# MAGIC
# MAGIC Durante el Hyperparameter Tuning, una configuración con mayor profundidad obtuvo un WAPE ligeramente inferior, pero la mejora relativa fue únicamente del **0,0268 %**.
# MAGIC
# MAGIC Al no superar el umbral de materialidad definido del **0,5 %**, se mantuvo la configuración base por simplicidad y menor complejidad.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Estrategia temporal
# MAGIC
# MAGIC El desarrollo del modelo ha mantenido una separación temporal estricta:
# MAGIC
# MAGIC **TRAIN → VALIDATION → TEST → OOT**
# MAGIC
# MAGIC ### TRAIN
# MAGIC
# MAGIC Periodo:
# MAGIC
# MAGIC **29/01/2024 → 31/12/2025**
# MAGIC
# MAGIC Registros:
# MAGIC
# MAGIC **21.793**
# MAGIC
# MAGIC ### VALIDATION
# MAGIC
# MAGIC Periodo:
# MAGIC
# MAGIC **01/01/2026 → 30/04/2026**
# MAGIC
# MAGIC Registros:
# MAGIC
# MAGIC **3.720**
# MAGIC
# MAGIC ### TEST
# MAGIC
# MAGIC Periodo:
# MAGIC
# MAGIC **01/05/2026 → 31/07/2026**
# MAGIC
# MAGIC Registros:
# MAGIC
# MAGIC **2.852**
# MAGIC
# MAGIC ### OOT
# MAGIC
# MAGIC Periodo:
# MAGIC
# MAGIC **01/08/2026 → 31/08/2026**
# MAGIC
# MAGIC Registros:
# MAGIC
# MAGIC **961**
# MAGIC
# MAGIC OOT permanece reservado para procesos posteriores de inferencia y monitoring.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Entrenamiento final
# MAGIC
# MAGIC Una vez seleccionados los hiperparámetros, el modelo definitivo fue entrenado con:
# MAGIC
# MAGIC **TRAIN + VALIDATION**
# MAGIC
# MAGIC Registros utilizados:
# MAGIC
# MAGIC **25.513**
# MAGIC
# MAGIC Periodo:
# MAGIC
# MAGIC **29/01/2024 → 30/04/2026**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Rendimiento
# MAGIC
# MAGIC ### Validation
# MAGIC
# MAGIC - **MAE:** 2.142,99
# MAGIC - **RMSE:** 2.730,90
# MAGIC - **WAPE:** 35,4789 %
# MAGIC
# MAGIC ### Test
# MAGIC
# MAGIC - **MAE:** 2.126,96
# MAGIC - **RMSE:** 2.710,35
# MAGIC - **WAPE:** 35,7727 %
# MAGIC
# MAGIC Diferencia de WAPE:
# MAGIC
# MAGIC **+0,2938 puntos porcentuales**
# MAGIC
# MAGIC El comportamiento del modelo entre Validation y Test se considera:
# MAGIC
# MAGIC **ESTABLE**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## MLflow
# MAGIC
# MAGIC El modelo dispone de trazabilidad completa mediante MLflow.
# MAGIC
# MAGIC Run utilizada:
# MAGIC
# MAGIC **`74b2d392491b49419fb20b9f2d4eacf3`**
# MAGIC
# MAGIC Experiment ID:
# MAGIC
# MAGIC **`2440782200688128`**
# MAGIC
# MAGIC La Run contiene:
# MAGIC
# MAGIC - hiperparámetros
# MAGIC - métricas de Validation
# MAGIC - métricas de Test
# MAGIC - métricas de generalización
# MAGIC - tags
# MAGIC - modelo Spark ML
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Unity Catalog Model Registry
# MAGIC
# MAGIC El modelo fue registrado en Unity Catalog como:
# MAGIC
# MAGIC **`retail_analytics.5_ml.sales_forecasting_random_forest`**
# MAGIC
# MAGIC Versión:
# MAGIC
# MAGIC **2**
# MAGIC
# MAGIC Estado:
# MAGIC
# MAGIC **READY**
# MAGIC
# MAGIC La versión registrada fue cargada directamente desde Unity Catalog mediante:
# MAGIC
# MAGIC **`models:/retail_analytics.5_ml.sales_forecasting_random_forest/2`**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Reproducibilidad
# MAGIC
# MAGIC El modelo cargado desde Unity Catalog fue evaluado nuevamente sobre los **2.852 registros de TEST**.
# MAGIC
# MAGIC Las diferencias frente al modelo original fueron:
# MAGIC
# MAGIC - **MAE:** 0
# MAGIC - **RMSE:** 0
# MAGIC - **WAPE:** 0
# MAGIC
# MAGIC Por tanto, la reproducibilidad del modelo registrado queda:
# MAGIC
# MAGIC **VALIDADA**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Gobierno y metadata
# MAGIC
# MAGIC El Registered Model y su Version 2 disponen de:
# MAGIC
# MAGIC - descripción funcional
# MAGIC - framework
# MAGIC - algoritmo
# MAGIC - variable objetivo
# MAGIC - granularidad
# MAGIC - estrategia de entrenamiento
# MAGIC - métricas
# MAGIC - Run de MLflow
# MAGIC - estado de generalización
# MAGIC - estado de reproducibilidad
# MAGIC - tags de gobierno
# MAGIC
# MAGIC Esto permite identificar y reutilizar el modelo sin depender exclusivamente del notebook donde fue desarrollado.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Arquitectura final
# MAGIC
# MAGIC El flujo completo queda:
# MAGIC
# MAGIC **Gold Data**
# MAGIC
# MAGIC ↓
# MAGIC
# MAGIC **Feature Engineering con Spark**
# MAGIC
# MAGIC ↓
# MAGIC
# MAGIC **Delta Feature Table**
# MAGIC
# MAGIC ↓
# MAGIC
# MAGIC **TRAIN / VALIDATION / TEST / OOT**
# MAGIC
# MAGIC ↓
# MAGIC
# MAGIC **Hyperparameter Tuning**
# MAGIC
# MAGIC ↓
# MAGIC
# MAGIC **Modelo final Spark ML**
# MAGIC
# MAGIC ↓
# MAGIC
# MAGIC **TRAIN + VALIDATION**
# MAGIC
# MAGIC ↓
# MAGIC
# MAGIC **Evaluación sobre TEST**
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
# MAGIC **Version 2**
# MAGIC
# MAGIC ↓
# MAGIC
# MAGIC **Validación de reproducibilidad**
# MAGIC
# MAGIC ↓
# MAGIC
# MAGIC **Documentación y gobierno**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Estado del proyecto
# MAGIC
# MAGIC La fase de:
# MAGIC
# MAGIC - desarrollo
# MAGIC - tuning
# MAGIC - selección
# MAGIC - entrenamiento final
# MAGIC - evaluación
# MAGIC - MLflow
# MAGIC - Model Registry
# MAGIC - reproducibilidad
# MAGIC - documentación
# MAGIC
# MAGIC queda completada.
# MAGIC
# MAGIC El siguiente paso será utilizar el modelo registrado para procesos de:
# MAGIC
# MAGIC **Inference → Monitoring → consumo analítico**