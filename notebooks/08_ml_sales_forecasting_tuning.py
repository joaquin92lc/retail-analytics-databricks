# Databricks notebook source
# DBTITLE 1,01. CARGA Y VALIDACIÓN DEL DATASET DE FEATURES ML
# ============================================================
# 01. CARGA Y VALIDACIÓN DEL DATASET DE FEATURES ML
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Cargar el dataset de Machine Learning previamente construido
# y materializado por:
#
#       08_ml_sales_forecasting
#
# Tabla:
#
#       retail_analytics.5_ml.daily_store_features
#
# Este notebook NO reconstruye:
#
# - ventas diarias
# - calendario completo tienda × día
# - regularización de días sin ventas
# - lags
# - rolling averages
# - features temporales
#
# Todo ese Data Engineering pertenece al notebook principal.
#
# Este notebook se centrará exclusivamente en:
#
# - cargar el dataset ML
# - validar su integridad
# - realizar el split temporal
# - preparar las features
# - ejecutar Hyperparameter Tuning
# - seleccionar la mejor configuración
#
# ============================================================


from pyspark.sql import functions as F


# ============================================================
# 01.01 CONFIGURACIÓN
# ============================================================

ML_FEATURE_TABLE = (
    "retail_analytics.5_ml.daily_store_features"
)


# ============================================================
# 01.02 VALIDAMOS QUE LA TABLA EXISTA
# ============================================================

if not spark.catalog.tableExists(
    ML_FEATURE_TABLE
):

    raise RuntimeError(

        f"No existe la tabla ML:\n"
        f"{ML_FEATURE_TABLE}\n\n"
        "Ejecuta primero el notebook "
        "08_ml_sales_forecasting."

    )


print("=" * 80)
print("CARGA DEL DATASET DE FEATURES ML")
print("=" * 80)

print(
    f"OK - Tabla encontrada: {ML_FEATURE_TABLE}"
)


# ============================================================
# 01.03 CARGAMOS EL DATASET
# ============================================================

ml_model_df = spark.table(
    ML_FEATURE_TABLE
)


# ============================================================
# 01.04 COLUMNAS OBLIGATORIAS
# ============================================================
#
# Estas son las columnas necesarias para el modelo actual.
#
# Separamos:
#
# - identificadores
# - target
# - features temporales
# - lags
# - rolling features
# - features derivadas
#
# IMPORTANTE:
#
# Utilizamos "lag1_minus_lag7".
#
# No utilizamos el antiguo nombre "diff_7", ya que realmente
# esa feature representa:
#
#       lag_1 - lag_7
#
# ============================================================

identifier_columns = [

    "sale_date",
    "store_id"

]


target_column = (
    "net_sales"
)


temporal_features = [

    "year",
    "month",
    "day",
    "day_of_week",
    "week_of_year",
    "is_weekend"

]


lag_features = [

    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28"

]


rolling_features = [

    "rolling_mean_7",
    "rolling_mean_28",
    "rolling_std_7"

]


derived_features = [

    "lag1_minus_lag7",
    "lag1_vs_mean7"

]


numeric_features = (

    temporal_features
    +
    lag_features
    +
    rolling_features
    +
    derived_features

)


categorical_features = [

    "store_id"

]


required_columns = (

    identifier_columns
    +
    [target_column]
    +
    numeric_features

)


# Eliminamos posibles repeticiones conservando el orden.

required_columns = list(
    dict.fromkeys(
        required_columns
    )
)


# ============================================================
# 01.05 VALIDAMOS COLUMNAS
# ============================================================

available_columns = set(
    ml_model_df.columns
)


missing_columns = [

    column_name

    for column_name in required_columns

    if column_name not in available_columns

]


if missing_columns:

    raise RuntimeError(

        "Faltan columnas necesarias para el tuning:\n"
        +
        "\n".join(
            f" - {column_name}"
            for column_name in missing_columns
        )

    )


print(
    "OK - Todas las columnas necesarias están disponibles"
)


# ============================================================
# 01.06 ESTADÍSTICAS GENERALES
# ============================================================

dataset_stats = (

    ml_model_df

    .agg(

        F.count("*")
        .alias(
            "rows"
        ),

        F.countDistinct(
            "store_id"
        )
        .alias(
            "stores"
        ),

        F.min(
            "sale_date"
        )
        .alias(
            "min_date"
        ),

        F.max(
            "sale_date"
        )
        .alias(
            "max_date"
        )

    )

    .first()

)


total_rows = int(
    dataset_stats["rows"]
)

total_stores = int(
    dataset_stats["stores"]
)

min_date = (
    dataset_stats["min_date"]
)

max_date = (
    dataset_stats["max_date"]
)


# ============================================================
# 01.07 VALIDAMOS DUPLICADOS
# ============================================================
#
# La granularidad del dataset ML debe seguir siendo:
#
#       1 fila = 1 tienda + 1 día
#
# ============================================================

duplicate_store_dates = (

    ml_model_df

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
# 01.08 VALIDAMOS TARGET
# ============================================================

null_target_rows = (

    ml_model_df

    .filter(
        F.col(
            target_column
        ).isNull()
    )

    .count()

)


# ============================================================
# 01.09 VALIDAMOS FEATURES
# ============================================================
#
# El dataset ya pasó por la eliminación del periodo inicial
# necesario para construir lag_28 y rolling_mean_28.
#
# Por tanto, ninguna feature utilizada para entrenar debería
# contener NULL.
#
# ============================================================

feature_null_counts = {}


for feature in numeric_features:

    null_count = (

        ml_model_df

        .filter(
            F.col(
                feature
            ).isNull()
        )

        .count()

    )

    feature_null_counts[
        feature
    ] = null_count


# ============================================================
# 01.10 VALIDAMOS STORE_ID
# ============================================================

null_store_ids = (

    ml_model_df

    .filter(
        F.col(
            "store_id"
        ).isNull()
    )

    .count()

)


# ============================================================
# 01.11 VALIDAMOS FECHAS
# ============================================================

null_dates = (

    ml_model_df

    .filter(
        F.col(
            "sale_date"
        ).isNull()
    )

    .count()

)


# ============================================================
# 01.12 VALIDAMOS VALORES INFINITOS / NaN
# ============================================================
#
# Spark ML puede fallar si alguna feature contiene NaN.
#
# Las columnas actuales deberían ser numéricas y finitas.
#
# ============================================================

nan_counts = {}


for feature in numeric_features:

    data_type = dict(
        ml_model_df.dtypes
    ).get(
        feature
    )


    # isnan únicamente es aplicable a columnas numéricas.
    # Todas nuestras numeric_features deberían serlo.

    if data_type in [

        "double",
        "float"

    ]:

        nan_count = (

            ml_model_df

            .filter(
                F.isnan(
                    F.col(
                        feature
                    )
                )
            )

            .count()

        )

    else:

        nan_count = 0


    nan_counts[
        feature
    ] = nan_count


# ============================================================
# 01.13 RESULTADOS GENERALES
# ============================================================

print()
print("=" * 80)
print("RESUMEN DEL DATASET ML")
print("=" * 80)

print(
    f"Registros:               {total_rows:,}"
)

print(
    f"Tiendas:                 {total_stores:,}"
)

print(
    f"Primera fecha:           {min_date}"
)

print(
    f"Última fecha:            {max_date}"
)

print(
    f"Duplicados tienda-fecha: {duplicate_store_dates:,}"
)

print(
    f"NULLs en target:         {null_target_rows:,}"
)

print(
    f"NULLs en store_id:       {null_store_ids:,}"
)

print(
    f"NULLs en sale_date:      {null_dates:,}"
)


# ============================================================
# 01.14 RESULTADOS POR FEATURE
# ============================================================

print()
print("=" * 80)
print("VALIDACIÓN DE FEATURES")
print("=" * 80)


for feature in numeric_features:

    null_count = (
        feature_null_counts[
            feature
        ]
    )

    nan_count = (
        nan_counts[
            feature
        ]
    )


    if (
        null_count == 0
        and
        nan_count == 0
    ):

        status = "OK"

    else:

        status = "ERROR"


    print(

        f"{status:8} | "
        f"{feature:22} | "
        f"NULL: {null_count:,} | "
        f"NaN: {nan_count:,}"

    )


# ============================================================
# 01.15 VALIDACIÓN FINAL
# ============================================================

validation_errors = []


if total_rows == 0:

    validation_errors.append(
        "El dataset ML está vacío."
    )


if total_stores == 0:

    validation_errors.append(
        "No existen tiendas en el dataset ML."
    )


if duplicate_store_dates > 0:

    validation_errors.append(

        "Existen duplicados por "
        "store_id + sale_date."

    )


if null_target_rows > 0:

    validation_errors.append(
        "El target net_sales contiene NULL."
    )


if null_store_ids > 0:

    validation_errors.append(
        "store_id contiene NULL."
    )


if null_dates > 0:

    validation_errors.append(
        "sale_date contiene NULL."
    )


features_with_nulls = [

    feature

    for feature, null_count
    in feature_null_counts.items()

    if null_count > 0

]


if features_with_nulls:

    validation_errors.append(

        "Existen NULLs en features: "
        +
        ", ".join(
            features_with_nulls
        )

    )


features_with_nan = [

    feature

    for feature, nan_count
    in nan_counts.items()

    if nan_count > 0

]


if features_with_nan:

    validation_errors.append(

        "Existen NaN en features: "
        +
        ", ".join(
            features_with_nan
        )

    )


# ============================================================
# RESULTADO FINAL
# ============================================================

print()
print("=" * 80)
print("VALIDACIÓN FINAL")
print("=" * 80)


if validation_errors:

    for error in validation_errors:

        print(
            f"ERROR - {error}"
        )


    raise RuntimeError(

        "El dataset ML no está preparado "
        "para ejecutar Hyperparameter Tuning."

    )


print(
    "OK - Dataset ML cargado correctamente"
)

print(
    "OK - Granularidad única store_id + sale_date"
)

print(
    "OK - Target disponible y sin NULLs"
)

print(
    "OK - Features disponibles y sin NULLs"
)

print(
    "OK - Features sin valores NaN"
)

print(
    "OK - Dataset preparado para Hyperparameter Tuning"
)


# ============================================================
# 01.16 INSPECCIÓN DEL DATASET
# ============================================================

display(

    ml_model_df

    .select(

        "sale_date",
        "store_id",
        "net_sales",

        *numeric_features

    )

    .orderBy(
        "sale_date",
        "store_id"
    )

    .limit(
        50
    )

)

# COMMAND ----------

# DBTITLE 1,02. SPLIT TEMPORAL PARA TUNING
# ============================================================
# 02. SPLIT TEMPORAL PARA TUNING
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Dividir el dataset ML temporalmente en:
#
# TRAIN
#       hasta 2025-12-31
#
# VALIDATION
#       2026-01-01 -> 2026-04-30
#
# TEST
#       2026-05-01 -> 2026-07-31
#
# OOT
#       desde 2026-08-01 hasta la última fecha disponible
#
# METODOLOGÍA
# ------------------------------------------------------------
#
# TRAIN:
#       entrenamiento de cada configuración.
#
# VALIDATION:
#       comparación y selección de hiperparámetros.
#
# TEST:
#       NO se utiliza durante el tuning.
#       Se reserva para evaluar una única vez el modelo final.
#
# OOT:
#       periodo posterior completamente separado.
#
# IMPORTANTE
# ------------------------------------------------------------
#
# No realizamos split aleatorio porque estamos trabajando
# con forecasting.
#
# Mantener el orden temporal evita Data Leakage.
#
# ============================================================


from pyspark.sql import functions as F


# ============================================================
# 02.01 VALIDAMOS DATASET DE ENTRADA
# ============================================================

if "ml_model_df" not in globals():

    raise RuntimeError(
        "No existe ml_model_df. "
        "Ejecuta primero el bloque 01."
    )


# ============================================================
# 02.02 DEFINICIÓN DE PERIODOS
# ============================================================

TRAIN_END = "2025-12-31"

VALIDATION_START = "2026-01-01"
VALIDATION_END = "2026-04-30"

TEST_START = "2026-05-01"
TEST_END = "2026-07-31"

OOT_START = "2026-08-01"


# ============================================================
# 02.03 RANGO TEMPORAL DISPONIBLE
# ============================================================

dataset_date_stats = (

    ml_model_df

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


dataset_min_date = (
    dataset_date_stats["min_date"]
)

dataset_max_date = (
    dataset_date_stats["max_date"]
)


if (
    dataset_min_date is None
    or
    dataset_max_date is None
):

    raise RuntimeError(
        "No se ha podido determinar "
        "el rango temporal del dataset."
    )


# ============================================================
# 02.04 TRAIN
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
# 02.05 VALIDATION
# ============================================================

validation_df = (

    ml_model_df

    .filter(

        (F.col("sale_date") >= F.lit(VALIDATION_START))

        &

        (F.col("sale_date") <= F.lit(VALIDATION_END))

    )

)


# ============================================================
# 02.06 TEST
# ============================================================

test_df = (

    ml_model_df

    .filter(

        (F.col("sale_date") >= F.lit(TEST_START))

        &

        (F.col("sale_date") <= F.lit(TEST_END))

    )

)


# ============================================================
# 02.07 OOT
# ============================================================

oot_df = (

    ml_model_df

    .filter(
        F.col("sale_date")
        >=
        F.lit(OOT_START)
    )

)


# ============================================================
# 02.08 FUNCIÓN AUXILIAR PARA ESTADÍSTICAS
# ============================================================

def get_split_stats(df):

    stats = (

        df

        .agg(

            F.count("*")
            .alias(
                "rows"
            ),

            F.countDistinct(
                "store_id"
            )
            .alias(
                "stores"
            ),

            F.min(
                "sale_date"
            )
            .alias(
                "min_date"
            ),

            F.max(
                "sale_date"
            )
            .alias(
                "max_date"
            )

        )

        .first()

    )

    return {

        "rows": int(
            stats["rows"]
        ),

        "stores": int(
            stats["stores"]
        ),

        "min_date": stats["min_date"],

        "max_date": stats["max_date"]

    }


# ============================================================
# 02.09 ESTADÍSTICAS DE LOS SPLITS
# ============================================================

train_stats = get_split_stats(
    train_df
)

validation_stats = get_split_stats(
    validation_df
)

test_stats = get_split_stats(
    test_df
)

oot_stats = get_split_stats(
    oot_df
)


# ============================================================
# 02.10 VALIDAMOS CONSERVACIÓN DE REGISTROS
# ============================================================

total_dataset_rows = (
    ml_model_df
    .count()
)


total_split_rows = (

    train_stats["rows"]
    +
    validation_stats["rows"]
    +
    test_stats["rows"]
    +
    oot_stats["rows"]

)


row_difference = (

    total_dataset_rows
    -
    total_split_rows

)


# ============================================================
# 02.11 VALIDAMOS SOLAPAMIENTOS
# ============================================================
#
# Ninguna fila debe aparecer en más de un split.
#
# Utilizamos:
#
#       store_id + sale_date
#
# como clave temporal del dataset.
#
# ============================================================

split_key_columns = [
    "store_id",
    "sale_date"
]


train_validation_overlap = (

    train_df
    .select(
        *split_key_columns
    )

    .join(

        validation_df
        .select(
            *split_key_columns
        ),

        on=split_key_columns,

        how="inner"

    )

    .count()

)


train_test_overlap = (

    train_df
    .select(
        *split_key_columns
    )

    .join(

        test_df
        .select(
            *split_key_columns
        ),

        on=split_key_columns,

        how="inner"

    )

    .count()

)


train_oot_overlap = (

    train_df
    .select(
        *split_key_columns
    )

    .join(

        oot_df
        .select(
            *split_key_columns
        ),

        on=split_key_columns,

        how="inner"

    )

    .count()

)


validation_test_overlap = (

    validation_df
    .select(
        *split_key_columns
    )

    .join(

        test_df
        .select(
            *split_key_columns
        ),

        on=split_key_columns,

        how="inner"

    )

    .count()

)


validation_oot_overlap = (

    validation_df
    .select(
        *split_key_columns
    )

    .join(

        oot_df
        .select(
            *split_key_columns
        ),

        on=split_key_columns,

        how="inner"

    )

    .count()

)


test_oot_overlap = (

    test_df
    .select(
        *split_key_columns
    )

    .join(

        oot_df
        .select(
            *split_key_columns
        ),

        on=split_key_columns,

        how="inner"

    )

    .count()

)


# ============================================================
# 02.12 VALIDAMOS NÚMERO DE TIENDAS
# ============================================================

total_stores = (

    ml_model_df

    .select(
        "store_id"
    )

    .distinct()

    .count()

)


# ============================================================
# 02.13 RESULTADO
# ============================================================

print("=" * 80)
print("SPLIT TEMPORAL PARA HYPERPARAMETER TUNING")
print("=" * 80)

print(
    f"Dataset completo: {total_dataset_rows:,} registros"
)

print(
    f"Periodo completo: "
    f"{dataset_min_date} -> {dataset_max_date}"
)

print(
    f"Tiendas:          {total_stores:,}"
)


print()
print("-" * 80)

print(
    f"TRAIN       | "
    f"{train_stats['rows']:,} registros | "
    f"{train_stats['stores']} tiendas | "
    f"{train_stats['min_date']} -> "
    f"{train_stats['max_date']}"
)

print(
    f"VALIDATION  | "
    f"{validation_stats['rows']:,} registros | "
    f"{validation_stats['stores']} tiendas | "
    f"{validation_stats['min_date']} -> "
    f"{validation_stats['max_date']}"
)

print(
    f"TEST        | "
    f"{test_stats['rows']:,} registros | "
    f"{test_stats['stores']} tiendas | "
    f"{test_stats['min_date']} -> "
    f"{test_stats['max_date']}"
)

print(
    f"OOT         | "
    f"{oot_stats['rows']:,} registros | "
    f"{oot_stats['stores']} tiendas | "
    f"{oot_stats['min_date']} -> "
    f"{oot_stats['max_date']}"
)


print()
print("=" * 80)
print("CONSERVACIÓN DE REGISTROS")
print("=" * 80)

print(
    f"Dataset original:      {total_dataset_rows:,}"
)

print(
    f"Suma de splits:        {total_split_rows:,}"
)

print(
    f"Diferencia:            {row_difference:,}"
)


print()
print("=" * 80)
print("SOLAPAMIENTOS")
print("=" * 80)

print(
    f"TRAIN / VALIDATION:    {train_validation_overlap:,}"
)

print(
    f"TRAIN / TEST:          {train_test_overlap:,}"
)

print(
    f"TRAIN / OOT:           {train_oot_overlap:,}"
)

print(
    f"VALIDATION / TEST:     {validation_test_overlap:,}"
)

print(
    f"VALIDATION / OOT:      {validation_oot_overlap:,}"
)

print(
    f"TEST / OOT:            {test_oot_overlap:,}"
)


# ============================================================
# 02.14 VALIDACIÓN FINAL
# ============================================================

validation_errors = []


# ------------------------------------------------------------
# Splits vacíos
# ------------------------------------------------------------

if train_stats["rows"] == 0:

    validation_errors.append(
        "TRAIN está vacío."
    )


if validation_stats["rows"] == 0:

    validation_errors.append(
        "VALIDATION está vacío."
    )


if test_stats["rows"] == 0:

    validation_errors.append(
        "TEST está vacío."
    )


if oot_stats["rows"] == 0:

    validation_errors.append(
        "OOT está vacío."
    )


# ------------------------------------------------------------
# Conservación
# ------------------------------------------------------------

if row_difference != 0:

    validation_errors.append(

        "La suma de TRAIN + VALIDATION + TEST + OOT "
        "no coincide con el dataset original."

    )


# ------------------------------------------------------------
# Solapamientos
# ------------------------------------------------------------

overlaps = {

    "TRAIN / VALIDATION":
        train_validation_overlap,

    "TRAIN / TEST":
        train_test_overlap,

    "TRAIN / OOT":
        train_oot_overlap,

    "VALIDATION / TEST":
        validation_test_overlap,

    "VALIDATION / OOT":
        validation_oot_overlap,

    "TEST / OOT":
        test_oot_overlap

}


invalid_overlaps = [

    split_name

    for split_name, overlap_count
    in overlaps.items()

    if overlap_count > 0

]


if invalid_overlaps:

    validation_errors.append(

        "Existen solapamientos entre splits: "
        +
        ", ".join(
            invalid_overlaps
        )

    )


# ------------------------------------------------------------
# Todas las tiendas deberían estar representadas
# en todos los periodos.
# ------------------------------------------------------------

for split_name, split_stats in [

    ("TRAIN", train_stats),
    ("VALIDATION", validation_stats),
    ("TEST", test_stats),
    ("OOT", oot_stats)

]:

    if split_stats["stores"] != total_stores:

        validation_errors.append(

            f"{split_name} contiene "
            f"{split_stats['stores']} tiendas, "
            f"pero el dataset contiene {total_stores}."

        )


# ============================================================
# RESULTADO FINAL
# ============================================================

print()
print("=" * 80)
print("VALIDACIÓN FINAL")
print("=" * 80)


if validation_errors:

    for error in validation_errors:

        print(
            f"ERROR - {error}"
        )

    raise RuntimeError(

        "El split temporal no ha superado "
        "las validaciones."

    )


print(
    "OK - TRAIN disponible"
)

print(
    "OK - VALIDATION disponible"
)

print(
    "OK - TEST disponible"
)

print(
    "OK - OOT disponible"
)

print(
    "OK - Todas las tiendas están presentes en todos los splits"
)

print(
    "OK - No existen solapamientos temporales"
)

print(
    "OK - Se conserva el 100% de los registros"
)

print()
print(
    "IMPORTANTE - Hyperparameter Tuning utilizará "
    "únicamente TRAIN + VALIDATION"
)

print(
    "IMPORTANTE - TEST y OOT permanecen aislados"
)

# COMMAND ----------

# DBTITLE 1,03. PREPARACIÓN DE FEATURES PARA HYPERPARAMETER TUNING
# ============================================================
# 03. PREPARACIÓN DE FEATURES PARA HYPERPARAMETER TUNING
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Preparar las variables que utilizarán los modelos durante
# el Hyperparameter Tuning.
#
# Utilizaremos exactamente el mismo conjunto de features
# utilizado en el notebook principal para que la comparación
# sea consistente.
#
# FEATURES:
#
# - variables temporales
# - lags de ventas
# - rolling statistics
# - tendencias
# - store_id como variable categórica
#
# IMPORTANTE
# ------------------------------------------------------------
#
# En este bloque NO entrenamos ningún modelo.
#
# Únicamente definimos:
#
#       StringIndexer
#       OneHotEncoder
#       VectorAssembler
#
# Estas transformaciones se incorporarán posteriormente al
# Pipeline de cada Random Forest.
#
# ============================================================


from pyspark.sql import functions as F

from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler
)


# ============================================================
# 03.01 TARGET
# ============================================================

target_column = (
    "net_sales"
)


# ============================================================
# 03.02 FEATURES NUMÉRICAS
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
    # TENDENCIAS
    # --------------------------------------------------------

    "lag1_minus_lag7",
    "lag1_vs_mean7"
]


# ============================================================
# 03.03 FEATURES CATEGÓRICAS
# ============================================================

categorical_features = [

    "store_id"

]


# ============================================================
# 03.04 VALIDAMOS COLUMNAS
# ============================================================

required_model_columns = (

    [target_column]
    +
    numeric_features
    +
    categorical_features

)


required_model_columns = list(
    dict.fromkeys(
        required_model_columns
    )
)


missing_model_columns = [

    column_name

    for column_name in required_model_columns

    if column_name not in train_df.columns

]


if missing_model_columns:

    raise RuntimeError(

        "Faltan columnas necesarias para el modelo:\n"
        +
        "\n".join(
            f" - {column_name}"
            for column_name in missing_model_columns
        )

    )


# ============================================================
# 03.05 VALIDAMOS NULLS EN TRAIN Y VALIDATION
# ============================================================
#
# TEST y OOT no se utilizan para seleccionar hiperparámetros,
# por lo que no necesitamos acceder a ellos durante el tuning.
#
# ============================================================

columns_to_validate = (

    [target_column]
    +
    numeric_features
    +
    categorical_features

)


def count_nulls(df, columns):

    null_expression = [

        F.sum(

            F.when(
                F.col(column_name).isNull(),
                1
            )

            .otherwise(
                0
            )

        ).alias(
            column_name
        )

        for column_name in columns

    ]


    result = (

        df
        .agg(
            *null_expression
        )
        .first()
    )


    return {

        column_name:
            int(
                result[column_name]
            )

        for column_name in columns

    }


train_nulls = count_nulls(
    train_df,
    columns_to_validate
)


validation_nulls = count_nulls(
    validation_df,
    columns_to_validate
)


# ============================================================
# 03.06 VALIDAMOS NaN EN FEATURES NUMÉRICAS
# ============================================================

numeric_types = {
    "double",
    "float"
}


train_dtypes = dict(
    train_df.dtypes
)


validation_dtypes = dict(
    validation_df.dtypes
)


train_nan_counts = {}

validation_nan_counts = {}


for feature in numeric_features:

    if train_dtypes.get(feature) in numeric_types:

        train_nan_counts[feature] = (

            train_df

            .filter(
                F.isnan(
                    F.col(feature)
                )
            )

            .count()

        )

    else:

        train_nan_counts[feature] = 0


    if validation_dtypes.get(feature) in numeric_types:

        validation_nan_counts[feature] = (

            validation_df

            .filter(
                F.isnan(
                    F.col(feature)
                )
            )

            .count()

        )

    else:

        validation_nan_counts[feature] = 0


# ============================================================
# 03.07 STRING INDEXER
# ============================================================
#
# store_id es una variable categórica.
#
# StringIndexer transforma:
#
#       S001
#       S002
#       S003
#
# en índices internos.
#
# handleInvalid="keep" evita errores si posteriormente
# apareciera una categoría no observada durante TRAIN.
#
# ============================================================

store_indexer = StringIndexer(

    inputCol="store_id",

    outputCol="store_id_index",

    handleInvalid="keep"

)


# ============================================================
# 03.08 ONE HOT ENCODER
# ============================================================

store_encoder = OneHotEncoder(

    inputCol="store_id_index",

    outputCol="store_id_encoded",

    handleInvalid="keep"

)


# ============================================================
# 03.09 VECTOR ASSEMBLER
# ============================================================
#
# Spark ML necesita todas las variables predictoras dentro
# de una única columna vectorial:
#
#       features
#
# ============================================================

assembler = VectorAssembler(

    inputCols=(

        numeric_features

        +

        [
            "store_id_encoded"
        ]

    ),

    outputCol="features",

    handleInvalid="error"

)


# ============================================================
# 03.10 RESUMEN DE FEATURES
# ============================================================

print("=" * 80)
print("PREPARACIÓN DE FEATURES PARA TUNING")
print("=" * 80)

print(
    f"Target:                 {target_column}"
)

print(
    f"Features numéricas:     {len(numeric_features)}"
)

print(
    f"Features categóricas:   {len(categorical_features)}"
)

print(
    f"Total grupos features:  "
    f"{len(numeric_features) + len(categorical_features)}"
)


print()
print("=" * 80)
print("FEATURES NUMÉRICAS")
print("=" * 80)


for feature in numeric_features:

    print(
        f"OK - {feature}"
    )


print()
print("=" * 80)
print("FEATURES CATEGÓRICAS")
print("=" * 80)


for feature in categorical_features:

    print(
        f"OK - {feature}"
    )


# ============================================================
# 03.11 VALIDACIÓN DE NULLS
# ============================================================

print()
print("=" * 80)
print("VALIDACIÓN TRAIN / VALIDATION")
print("=" * 80)


validation_errors = []


for column_name in columns_to_validate:

    train_null_count = (
        train_nulls[column_name]
    )

    validation_null_count = (
        validation_nulls[column_name]
    )


    if (
        train_null_count == 0
        and
        validation_null_count == 0
    ):

        status = "OK"

    else:

        status = "ERROR"

        validation_errors.append(

            f"{column_name}: "
            f"TRAIN NULL={train_null_count}, "
            f"VALIDATION NULL={validation_null_count}"

        )


    print(

        f"{status:8} | "
        f"{column_name:22} | "
        f"TRAIN NULL: {train_null_count:,} | "
        f"VALIDATION NULL: {validation_null_count:,}"

    )


# ============================================================
# 03.12 VALIDACIÓN DE NaN
# ============================================================

for feature in numeric_features:

    train_nan = (
        train_nan_counts[feature]
    )

    validation_nan = (
        validation_nan_counts[feature]
    )


    if (
        train_nan > 0
        or
        validation_nan > 0
    ):

        validation_errors.append(

            f"{feature}: "
            f"TRAIN NaN={train_nan}, "
            f"VALIDATION NaN={validation_nan}"

        )


# ============================================================
# 03.13 VALIDACIÓN FINAL
# ============================================================

print()
print("=" * 80)
print("VALIDACIÓN FINAL")
print("=" * 80)


if validation_errors:

    for error in validation_errors:

        print(
            f"ERROR - {error}"
        )


    raise RuntimeError(

        "Las features no están preparadas "
        "para Hyperparameter Tuning."

    )


print(
    "OK - Todas las features necesarias están disponibles"
)

print(
    "OK - TRAIN no contiene NULLs ni NaN"
)

print(
    "OK - VALIDATION no contiene NULLs ni NaN"
)

print(
    "OK - store_id preparado mediante StringIndexer + OneHotEncoder"
)

print(
    "OK - VectorAssembler configurado"
)

print(
    "OK - TEST no se ha utilizado"
)

print(
    "OK - OOT no se ha utilizado"
)

print(
    "OK - Features preparadas para Hyperparameter Tuning"
)

# COMMAND ----------

# DBTITLE 1,04. DEFINICIÓN DEL ESPACIO DE HIPERPARÁMETROS
# ============================================================
# 04. DEFINICIÓN DEL ESPACIO DE HIPERPARÁMETROS
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Definir un conjunto controlado de configuraciones de
# Random Forest para evaluar durante el Hyperparameter Tuning.
#
# IMPORTANTE
# ------------------------------------------------------------
#
# Este notebook se creó precisamente porque el tuning anterior
# llegó a saturar la sesión al acumular demasiados modelos.
#
# Por ese motivo:
#
# - evitamos configuraciones demasiado pesadas
# - no utilizamos 150-200 árboles
# - no hacemos una búsqueda masiva
# - probamos únicamente variaciones razonables
#
# La estrategia será:
#
#   1. entrenar cada configuración con TRAIN
#   2. evaluar sobre VALIDATION
#   3. guardar únicamente métricas
#   4. seleccionar posteriormente la mejor configuración
#
# TEST y OOT permanecen completamente aislados.
#
# ============================================================


# ============================================================
# 04.01 CONFIGURACIONES RANDOM FOREST
# ============================================================
#
# RF_01_BASE
# ------------------------------------------------------------
# Configuración de referencia.
#
# RF_02_TREES_80
# ------------------------------------------------------------
# Analiza si aumentar ligeramente el número de árboles
# mejora la estabilidad.
#
# RF_03_DEPTH_6
# ------------------------------------------------------------
# Modelo más simple.
#
# RF_04_DEPTH_10
# ------------------------------------------------------------
# Modelo algo más complejo.
#
# RF_05_MIN_NODE_5
# ------------------------------------------------------------
# Aumenta la regularización al exigir más observaciones
# por nodo.
#
# RF_06_SQRT
# ------------------------------------------------------------
# Cambia la estrategia de selección de features en cada split.
#
# ============================================================


rf_configs = [

    # --------------------------------------------------------
    # CONFIGURACIÓN BASE
    # --------------------------------------------------------

    {
        "config_name": "RF_01_BASE",

        "numTrees": 60,
        "maxDepth": 8,
        "minInstancesPerNode": 3,

        "featureSubsetStrategy": "auto",

        "seed": 42
    },


    # --------------------------------------------------------
    # MÁS ÁRBOLES
    # --------------------------------------------------------

    {
        "config_name": "RF_02_TREES_80",

        "numTrees": 80,
        "maxDepth": 8,
        "minInstancesPerNode": 3,

        "featureSubsetStrategy": "auto",

        "seed": 42
    },


    # --------------------------------------------------------
    # MENOR PROFUNDIDAD
    # --------------------------------------------------------

    {
        "config_name": "RF_03_DEPTH_6",

        "numTrees": 60,
        "maxDepth": 6,
        "minInstancesPerNode": 3,

        "featureSubsetStrategy": "auto",

        "seed": 42
    },


    # --------------------------------------------------------
    # MAYOR PROFUNDIDAD
    # --------------------------------------------------------

    {
        "config_name": "RF_04_DEPTH_10",

        "numTrees": 60,
        "maxDepth": 10,
        "minInstancesPerNode": 3,

        "featureSubsetStrategy": "auto",

        "seed": 42
    },


    # --------------------------------------------------------
    # MAYOR REGULARIZACIÓN
    # --------------------------------------------------------

    {
        "config_name": "RF_05_MIN_NODE_5",

        "numTrees": 60,
        "maxDepth": 8,
        "minInstancesPerNode": 5,

        "featureSubsetStrategy": "auto",

        "seed": 42
    },


    # --------------------------------------------------------
    # FEATURE SUBSAMPLING
    # --------------------------------------------------------

    {
        "config_name": "RF_06_SQRT",

        "numTrees": 60,
        "maxDepth": 8,
        "minInstancesPerNode": 3,

        "featureSubsetStrategy": "sqrt",

        "seed": 42
    }

]


# ============================================================
# 04.02 VALIDACIÓN DE CONFIGURACIONES
# ============================================================

required_config_parameters = [

    "config_name",
    "numTrees",
    "maxDepth",
    "minInstancesPerNode",
    "featureSubsetStrategy",
    "seed"

]


validation_errors = []


# ------------------------------------------------------------
# NOMBRES DUPLICADOS
# ------------------------------------------------------------

config_names = [

    config["config_name"]
    for config in rf_configs

]


duplicate_names = {

    name

    for name in config_names

    if config_names.count(name) > 1

}


if duplicate_names:

    validation_errors.append(

        "Existen nombres de configuración duplicados: "
        +
        ", ".join(
            sorted(
                duplicate_names
            )
        )

    )


# ------------------------------------------------------------
# PARÁMETROS OBLIGATORIOS
# ------------------------------------------------------------

for config in rf_configs:

    missing_parameters = [

        parameter

        for parameter in required_config_parameters

        if parameter not in config

    ]


    if missing_parameters:

        validation_errors.append(

            f"{config.get('config_name', 'CONFIG_SIN_NOMBRE')}: "
            f"faltan parámetros "
            f"{', '.join(missing_parameters)}"

        )


# ============================================================
# 04.03 VALIDACIÓN DE VALORES
# ============================================================

for config in rf_configs:

    config_name = config[
        "config_name"
    ]


    # --------------------------------------------------------
    # NUM TREES
    # --------------------------------------------------------

    if config["numTrees"] <= 0:

        validation_errors.append(

            f"{config_name}: "
            "numTrees debe ser > 0."

        )


    # --------------------------------------------------------
    # MAX DEPTH
    # --------------------------------------------------------

    if config["maxDepth"] <= 0:

        validation_errors.append(

            f"{config_name}: "
            "maxDepth debe ser > 0."

        )


    # --------------------------------------------------------
    # MIN INSTANCES PER NODE
    # --------------------------------------------------------

    if config["minInstancesPerNode"] <= 0:

        validation_errors.append(

            f"{config_name}: "
            "minInstancesPerNode debe ser > 0."

        )


    # --------------------------------------------------------
    # FEATURE SUBSET STRATEGY
    # --------------------------------------------------------

    valid_feature_subset_strategies = {

        "auto",
        "all",
        "onethird",
        "sqrt",
        "log2"

    }


    if (
        config["featureSubsetStrategy"]
        not in
        valid_feature_subset_strategies
    ):

        validation_errors.append(

            f"{config_name}: "
            "featureSubsetStrategy no válido."

        )


# ============================================================
# 04.04 MOSTRAMOS CONFIGURACIONES
# ============================================================

print("=" * 80)
print("ESPACIO DE HIPERPARÁMETROS - RANDOM FOREST")
print("=" * 80)

print(
    f"Número de configuraciones: {len(rf_configs)}"
)


print()

for config in rf_configs:

    print("-" * 80)

    print(
        f"Configuración:          "
        f"{config['config_name']}"
    )

    print(
        f"numTrees:              "
        f"{config['numTrees']}"
    )

    print(
        f"maxDepth:              "
        f"{config['maxDepth']}"
    )

    print(
        f"minInstancesPerNode:    "
        f"{config['minInstancesPerNode']}"
    )

    print(
        f"featureSubsetStrategy:  "
        f"{config['featureSubsetStrategy']}"
    )

    print(
        f"seed:                  "
        f"{config['seed']}"
    )


# ============================================================
# 04.05 COMPARACIÓN RESUMIDA
# ============================================================

config_summary = [

    (
        config["config_name"],
        config["numTrees"],
        config["maxDepth"],
        config["minInstancesPerNode"],
        config["featureSubsetStrategy"]
    )

    for config in rf_configs

]


config_summary_df = spark.createDataFrame(

    config_summary,

    [

        "config_name",
        "numTrees",
        "maxDepth",
        "minInstancesPerNode",
        "featureSubsetStrategy"

    ]

)


print()
print("=" * 80)
print("RESUMEN DE CONFIGURACIONES")
print("=" * 80)


display(
    config_summary_df
)


# ============================================================
# 04.06 VALIDACIÓN FINAL
# ============================================================

print()
print("=" * 80)
print("VALIDACIÓN FINAL")
print("=" * 80)


if validation_errors:

    for error in validation_errors:

        print(
            f"ERROR - {error}"
        )


    raise RuntimeError(

        "El espacio de hiperparámetros "
        "no ha superado las validaciones."

    )


print(
    "OK - Nombres de configuración únicos"
)

print(
    "OK - Parámetros obligatorios disponibles"
)

print(
    "OK - Valores de hiperparámetros válidos"
)

print(
    "OK - Espacio de búsqueda limitado a 6 configuraciones"
)

print(
    "OK - Configuraciones preparadas para entrenamiento"
)

print()
print(
    "IMPORTANTE - En este bloque todavía "
    "no se ha entrenado ningún modelo"
)

print(
    "IMPORTANTE - TEST y OOT continúan sin utilizarse"
)

# COMMAND ----------

# DBTITLE 1,05. ENTRENAMIENTO Y EVALUACIÓN DEL HYPERPARAMETER TUNING
# ============================================================
# 05. ENTRENAMIENTO Y EVALUACIÓN DEL HYPERPARAMETER TUNING
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Entrenar secuencialmente las configuraciones de Random Forest
# definidas en el bloque 04.
#
# METODOLOGÍA
# ------------------------------------------------------------
#
# TRAIN:
#       utilizado para entrenar cada configuración.
#
# VALIDATION:
#       utilizado para calcular:
#
#       - MAE
#       - RMSE
#       - WAPE
#
# TEST:
#       NO utilizado.
#
# OOT:
#       NO utilizado.
#
# IMPORTANTE
# ------------------------------------------------------------
#
# No almacenaremos todos los modelos entrenados.
#
# Después de evaluar cada configuración:
#
#       1. guardamos únicamente sus métricas
#       2. eliminamos las referencias al modelo
#       3. eliminamos las predicciones
#       4. ejecutamos garbage collection
#
# Esto reduce el riesgo de saturar la sesión durante el tuning.
#
# ============================================================


import gc
import time

from pyspark.sql import functions as F

from pyspark.ml import Pipeline

from pyspark.ml.regression import (
    RandomForestRegressor
)

from pyspark.ml.evaluation import (
    RegressionEvaluator
)


# ============================================================
# 05.01 VALIDACIONES PREVIAS
# ============================================================

required_objects = [

    "train_df",
    "validation_df",

    "store_indexer",
    "store_encoder",
    "assembler",

    "rf_configs"

]


missing_objects = [

    object_name

    for object_name in required_objects

    if object_name not in globals()

]


if missing_objects:

    raise RuntimeError(

        "Faltan objetos necesarios para ejecutar el tuning: "
        +
        ", ".join(
            missing_objects
        )

    )


if len(rf_configs) == 0:

    raise RuntimeError(
        "rf_configs está vacío."
    )


# ============================================================
# 05.02 EVALUADORES
# ============================================================

mae_evaluator = RegressionEvaluator(

    labelCol="net_sales",
    predictionCol="prediction",
    metricName="mae"

)


rmse_evaluator = RegressionEvaluator(

    labelCol="net_sales",
    predictionCol="prediction",
    metricName="rmse"

)


# ============================================================
# 05.03 RESULTADOS DEL TUNING
# ============================================================
#
# Aquí guardaremos únicamente:
#
# - configuración
# - hiperparámetros
# - métricas
# - tiempo
#
# NO guardaremos PipelineModel.
#
# ============================================================

rf_tuning_results = []


# ============================================================
# 05.04 ENTRENAMIENTO SECUENCIAL
# ============================================================

print("=" * 80)
print("HYPERPARAMETER TUNING - RANDOM FOREST")
print("=" * 80)

print(
    f"Configuraciones a evaluar: {len(rf_configs)}"
)

print(
    "Dataset entrenamiento:     TRAIN"
)

print(
    "Dataset evaluación:        VALIDATION"
)

print()

print(
    "IMPORTANTE - TEST y OOT no se utilizarán"
)


for config_number, config in enumerate(
    rf_configs,
    start=1
):

    config_name = (
        config["config_name"]
    )


    print()
    print("=" * 80)

    print(
        f"[{config_number}/{len(rf_configs)}] "
        f"{config_name}"
    )

    print("=" * 80)


    print(
        f"numTrees:              "
        f"{config['numTrees']}"
    )

    print(
        f"maxDepth:              "
        f"{config['maxDepth']}"
    )

    print(
        f"minInstancesPerNode:    "
        f"{config['minInstancesPerNode']}"
    )

    print(
        f"featureSubsetStrategy:  "
        f"{config['featureSubsetStrategy']}"
    )


    # ========================================================
    # RANDOM FOREST
    # ========================================================

    rf_candidate = RandomForestRegressor(

        featuresCol="features",

        labelCol="net_sales",

        predictionCol="prediction",

        numTrees=config[
            "numTrees"
        ],

        maxDepth=config[
            "maxDepth"
        ],

        minInstancesPerNode=config[
            "minInstancesPerNode"
        ],

        featureSubsetStrategy=config[
            "featureSubsetStrategy"
        ],

        seed=config[
            "seed"
        ]

    )


    # ========================================================
    # PIPELINE
    # ========================================================

    candidate_pipeline = Pipeline(

        stages=[

            store_indexer,
            store_encoder,
            assembler,
            rf_candidate

        ]

    )


    # ========================================================
    # ENTRENAMIENTO
    # ========================================================

    start_time = (
        time.perf_counter()
    )


    candidate_model = (

        candidate_pipeline

        .fit(
            train_df
        )

    )


    training_time_seconds = (

        time.perf_counter()
        -
        start_time

    )


    # ========================================================
    # PREDICCIÓN SOBRE VALIDATION
    # ========================================================

    validation_predictions = (

        candidate_model

        .transform(
            validation_df
        )

        .select(

            "sale_date",
            "store_id",
            "net_sales",
            "prediction"

        )

    )


    # ========================================================
    # MAE
    # ========================================================

    candidate_mae = float(

        mae_evaluator.evaluate(
            validation_predictions
        )

    )


    # ========================================================
    # RMSE
    # ========================================================

    candidate_rmse = float(

        rmse_evaluator.evaluate(
            validation_predictions
        )

    )


    # ========================================================
    # WAPE
    # ============================================================
    #
    #            SUM(|actual - prediction|)
    # WAPE = ---------------------------------- × 100
    #                  SUM(|actual|)
    #
    # ========================================================

    wape_stats = (

        validation_predictions

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
                "absolute_actual"
            )

        )

        .first()

    )


    absolute_error = float(
        wape_stats["absolute_error"] or 0.0
    )


    absolute_actual = float(
        wape_stats["absolute_actual"] or 0.0
    )


    if absolute_actual == 0:

        raise RuntimeError(

            f"{config_name}: "
            "no se puede calcular WAPE porque "
            "SUM(abs(net_sales)) = 0."

        )


    candidate_wape = float(

        (
            absolute_error
            /
            absolute_actual
        )

        *
        100.0

    )


    # ========================================================
    # GUARDAMOS ÚNICAMENTE MÉTRICAS
    # ========================================================

    rf_tuning_results.append(

        {

            "config_name":
                config_name,

            "numTrees":
                int(
                    config["numTrees"]
                ),

            "maxDepth":
                int(
                    config["maxDepth"]
                ),

            "minInstancesPerNode":
                int(
                    config[
                        "minInstancesPerNode"
                    ]
                ),

            "featureSubsetStrategy":
                config[
                    "featureSubsetStrategy"
                ],

            "seed":
                int(
                    config["seed"]
                ),

            "mae":
                candidate_mae,

            "rmse":
                candidate_rmse,

            "wape":
                candidate_wape,

            "training_time_seconds":
                float(
                    training_time_seconds
                )

        }

    )


    # ========================================================
    # RESULTADO CONFIGURACIÓN
    # ========================================================

    print()

    print(
        f"MAE:                   "
        f"{candidate_mae:,.2f}"
    )

    print(
        f"RMSE:                  "
        f"{candidate_rmse:,.2f}"
    )

    print(
        f"WAPE:                  "
        f"{candidate_wape:.4f}%"
    )

    print(
        f"Tiempo entrenamiento:  "
        f"{training_time_seconds:.2f} s"
    )


    # ========================================================
    # LIBERACIÓN DE REFERENCIAS
    # ========================================================
    #
    # No necesitamos conservar este modelo.
    #
    # El modelo definitivo se volverá a entrenar posteriormente
    # utilizando la configuración seleccionada.
    #
    # ========================================================

    del validation_predictions
    del candidate_model
    del candidate_pipeline
    del rf_candidate

    gc.collect()


    print(
        "OK - Métricas guardadas y referencias liberadas"
    )


# ============================================================
# 05.05 VALIDAMOS NÚMERO DE RESULTADOS
# ============================================================

expected_results = (
    len(rf_configs)
)


actual_results = (
    len(rf_tuning_results)
)


# ============================================================
# 05.06 DATAFRAME DE RESULTADOS
# ============================================================

rf_tuning_results_df = (

    spark.createDataFrame(
        rf_tuning_results
    )

    .select(

        "config_name",

        "numTrees",
        "maxDepth",
        "minInstancesPerNode",
        "featureSubsetStrategy",

        F.round(
            "mae",
            2
        ).alias(
            "mae"
        ),

        F.round(
            "rmse",
            2
        ).alias(
            "rmse"
        ),

        F.round(
            "wape",
            4
        ).alias(
            "wape"
        ),

        F.round(
            "training_time_seconds",
            2
        ).alias(
            "training_time_seconds"
        )

    )

)


# ============================================================
# 05.07 RESULTADOS ORDENADOS POR WAPE
# ============================================================

rf_tuning_results_ranked_df = (

    rf_tuning_results_df

    .orderBy(

        F.asc(
            "wape"
        ),

        F.asc(
            "mae"
        )

    )

)


print()
print("=" * 80)
print("RESULTADOS DEL HYPERPARAMETER TUNING")
print("=" * 80)


display(
    rf_tuning_results_ranked_df
)


# ============================================================
# 05.08 VALIDACIÓN FINAL
# ============================================================

print()
print("=" * 80)
print("VALIDACIÓN FINAL")
print("=" * 80)


if actual_results != expected_results:

    raise RuntimeError(

        "No se han evaluado todas las configuraciones. "
        f"Esperadas={expected_results}, "
        f"obtenidas={actual_results}."

    )


print(
    f"OK - Configuraciones esperadas: "
    f"{expected_results}"
)

print(
    f"OK - Configuraciones evaluadas: "
    f"{actual_results}"
)

print(
    "OK - Todas las configuraciones se entrenaron con TRAIN"
)

print(
    "OK - Todas las configuraciones se evaluaron con VALIDATION"
)

print(
    "OK - Solo se conservaron métricas del tuning"
)

print(
    "OK - TEST permanece aislado"
)

print(
    "OK - OOT permanece aislado"
)

print()
print(
    "Hyperparameter Tuning completado."
)

print(
    "El siguiente bloque realizará el ranking "
    "y la selección de la configuración."
)

# COMMAND ----------

# DBTITLE 1,06. RANKING Y SELECCIÓN DE LA CONFIGURACIÓN
# ============================================================
# 06. RANKING Y SELECCIÓN DE LA CONFIGURACIÓN
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Comparar los resultados obtenidos durante el
# Hyperparameter Tuning y seleccionar la configuración final.
#
# CRITERIO PRINCIPAL
# ------------------------------------------------------------
#
# Utilizamos WAPE como métrica principal.
#
# También observamos:
#
# - MAE
# - RMSE
# - tiempo de entrenamiento
#
# REGLA DE SELECCIÓN
# ------------------------------------------------------------
#
# No seleccionaremos automáticamente una configuración más
# compleja si su mejora frente al modelo base es insignificante.
#
# Umbral mínimo:
#
#       0.5 % de mejora RELATIVA en WAPE
#
# Si el mejor modelo no mejora al modelo base al menos un 0.5%
# relativo, conservamos RF_01_BASE.
#
# IMPORTANTE
# ------------------------------------------------------------
#
# La selección utiliza únicamente métricas de VALIDATION.
#
# TEST y OOT continúan completamente aislados.
#
# ============================================================


from pyspark.sql import functions as F


# ============================================================
# 06.01 VALIDACIONES PREVIAS
# ============================================================

if "rf_tuning_results" not in globals():

    raise RuntimeError(
        "No existe rf_tuning_results. "
        "Ejecuta primero el bloque 05."
    )


if len(rf_tuning_results) == 0:

    raise RuntimeError(
        "rf_tuning_results está vacío."
    )


# ============================================================
# 06.02 CONFIGURACIÓN BASE
# ============================================================

BASE_CONFIG_NAME = (
    "RF_01_BASE"
)


base_candidates = [

    result

    for result in rf_tuning_results

    if result["config_name"] == BASE_CONFIG_NAME

]


if len(base_candidates) != 1:

    raise RuntimeError(

        "Debe existir exactamente una configuración "
        f"{BASE_CONFIG_NAME}."

    )


base_result = (
    base_candidates[0]
)


base_wape = float(
    base_result["wape"]
)


base_mae = float(
    base_result["mae"]
)


base_rmse = float(
    base_result["rmse"]
)


# ============================================================
# 06.03 RANKING TÉCNICO
# ============================================================
#
# Ordenamos primero por:
#
#   1. menor WAPE
#   2. menor MAE
#   3. menor RMSE
#
# ============================================================

ranked_results = sorted(

    rf_tuning_results,

    key=lambda result: (

        result["wape"],
        result["mae"],
        result["rmse"]

    )

)


technical_winner = (
    ranked_results[0]
)


technical_winner_name = (
    technical_winner["config_name"]
)


technical_winner_wape = float(
    technical_winner["wape"]
)


technical_winner_mae = float(
    technical_winner["mae"]
)


technical_winner_rmse = float(
    technical_winner["rmse"]
)


# ============================================================
# 06.04 MEJORA FRENTE AL MODELO BASE
# ============================================================
#
# Mejora absoluta:
#
#       WAPE_BASE - WAPE_WINNER
#
#
# Mejora relativa:
#
#       (WAPE_BASE - WAPE_WINNER)
#       -------------------------
#              WAPE_BASE
#
#                   × 100
#
# ============================================================

absolute_wape_improvement = (

    base_wape
    -
    technical_winner_wape

)


if base_wape == 0:

    relative_wape_improvement = (
        0.0
    )

else:

    relative_wape_improvement = (

        absolute_wape_improvement
        /
        base_wape

        *
        100.0

    )


# ============================================================
# 06.05 UMBRAL DE MATERIALIDAD
# ============================================================

MIN_RELATIVE_WAPE_IMPROVEMENT = (
    0.5
)


# ============================================================
# 06.06 SELECCIÓN FINAL
# ============================================================
#
# Caso 1
# ------------------------------------------------------------
#
# El modelo base ya es el ganador técnico.
#
#       -> seleccionamos BASE.
#
#
# Caso 2
# ------------------------------------------------------------
#
# Otro modelo gana técnicamente y mejora al BASE al menos
# un 0.5% relativo.
#
#       -> seleccionamos el ganador técnico.
#
#
# Caso 3
# ------------------------------------------------------------
#
# Otro modelo gana técnicamente pero la mejora es inferior
# al 0.5%.
#
#       -> mantenemos BASE.
#
# ============================================================

if technical_winner_name == BASE_CONFIG_NAME:

    selected_result = (
        base_result
    )

    selection_reason = (

        "La configuración BASE es también "
        "el ganador técnico del tuning."

    )


elif (
    relative_wape_improvement
    >=
    MIN_RELATIVE_WAPE_IMPROVEMENT
):

    selected_result = (
        technical_winner
    )

    selection_reason = (

        "El ganador técnico supera el umbral mínimo "
        "de mejora relativa frente al modelo BASE."

    )


else:

    selected_result = (
        base_result
    )

    selection_reason = (

        "El ganador técnico mejora ligeramente al modelo BASE, "
        "pero la mejora no alcanza el umbral mínimo del "
        f"{MIN_RELATIVE_WAPE_IMPROVEMENT:.2f}% relativo. "
        "Se mantiene la configuración BASE por simplicidad."

    )


# ============================================================
# 06.07 CONFIGURACIÓN SELECCIONADA
# ============================================================

selected_config_name = (
    selected_result["config_name"]
)


selected_numTrees = int(
    selected_result["numTrees"]
)


selected_maxDepth = int(
    selected_result["maxDepth"]
)


selected_minInstancesPerNode = int(
    selected_result[
        "minInstancesPerNode"
    ]
)


selected_featureSubsetStrategy = (
    selected_result[
        "featureSubsetStrategy"
    ]
)


selected_seed = int(
    selected_result["seed"]
)


selected_mae = float(
    selected_result["mae"]
)


selected_rmse = float(
    selected_result["rmse"]
)


selected_wape = float(
    selected_result["wape"]
)


# ============================================================
# 06.08 DATAFRAME DE RANKING
# ============================================================

ranking_rows = []


for rank_position, result in enumerate(
    ranked_results,
    start=1
):

    ranking_rows.append(

        (

            int(
                rank_position
            ),

            result[
                "config_name"
            ],

            int(
                result[
                    "numTrees"
                ]
            ),

            int(
                result[
                    "maxDepth"
                ]
            ),

            int(
                result[
                    "minInstancesPerNode"
                ]
            ),

            result[
                "featureSubsetStrategy"
            ],

            float(
                result[
                    "mae"
                ]
            ),

            float(
                result[
                    "rmse"
                ]
            ),

            float(
                result[
                    "wape"
                ]
            ),

            float(
                result[
                    "training_time_seconds"
                ]
            ),

            (
                result["config_name"]
                ==
                technical_winner_name
            ),

            (
                result["config_name"]
                ==
                selected_config_name
            )

        )

    )


ranking_df = spark.createDataFrame(

    ranking_rows,

    [

        "rank",
        "config_name",

        "numTrees",
        "maxDepth",
        "minInstancesPerNode",
        "featureSubsetStrategy",

        "mae",
        "rmse",
        "wape",
        "training_time_seconds",

        "technical_winner",
        "selected_model"

    ]

)


ranking_df = (

    ranking_df

    .select(

        "rank",
        "config_name",

        "numTrees",
        "maxDepth",
        "minInstancesPerNode",
        "featureSubsetStrategy",

        F.round(
            "mae",
            2
        ).alias(
            "mae"
        ),

        F.round(
            "rmse",
            2
        ).alias(
            "rmse"
        ),

        F.round(
            "wape",
            4
        ).alias(
            "wape"
        ),

        F.round(
            "training_time_seconds",
            2
        ).alias(
            "training_time_seconds"
        ),

        "technical_winner",
        "selected_model"

    )

    .orderBy(
        "rank"
    )

)


# ============================================================
# 06.09 MOSTRAMOS RANKING
# ============================================================

print("=" * 80)
print("RANKING DEL HYPERPARAMETER TUNING")
print("=" * 80)


display(
    ranking_df
)


# ============================================================
# 06.10 COMPARACIÓN GANADOR TÉCNICO VS BASE
# ============================================================

print()
print("=" * 80)
print("COMPARACIÓN CONTRA RF_01_BASE")
print("=" * 80)


print(
    f"BASE"
)

print(
    f"  WAPE:  {base_wape:.4f}%"
)

print(
    f"  MAE:   {base_mae:,.2f}"
)

print(
    f"  RMSE:  {base_rmse:,.2f}"
)


print()


print(
    f"GANADOR TÉCNICO: "
    f"{technical_winner_name}"
)

print(
    f"  WAPE:  {technical_winner_wape:.4f}%"
)

print(
    f"  MAE:   {technical_winner_mae:,.2f}"
)

print(
    f"  RMSE:  {technical_winner_rmse:,.2f}"
)


print()


print(
    f"Mejora absoluta WAPE: "
    f"{absolute_wape_improvement:.4f} "
    f"puntos porcentuales"
)

print(
    f"Mejora relativa WAPE: "
    f"{relative_wape_improvement:.4f}%"
)

print(
    f"Umbral mínimo exigido: "
    f"{MIN_RELATIVE_WAPE_IMPROVEMENT:.2f}%"
)


# ============================================================
# 06.11 MODELO SELECCIONADO
# ============================================================

print()
print("=" * 80)
print("CONFIGURACIÓN SELECCIONADA")
print("=" * 80)


print(
    f"Modelo:                 "
    f"{selected_config_name}"
)

print(
    f"numTrees:               "
    f"{selected_numTrees}"
)

print(
    f"maxDepth:               "
    f"{selected_maxDepth}"
)

print(
    f"minInstancesPerNode:     "
    f"{selected_minInstancesPerNode}"
)

print(
    f"featureSubsetStrategy:   "
    f"{selected_featureSubsetStrategy}"
)

print(
    f"seed:                   "
    f"{selected_seed}"
)


print()


print(
    f"Validation MAE:          "
    f"{selected_mae:,.2f}"
)

print(
    f"Validation RMSE:         "
    f"{selected_rmse:,.2f}"
)

print(
    f"Validation WAPE:         "
    f"{selected_wape:.4f}%"
)


print()
print(
    f"Motivo: {selection_reason}"
)


# ============================================================
# 06.12 EXPORTAMOS PARÁMETROS SELECCIONADOS
# ============================================================
#
# Dejamos un diccionario limpio que podrá reutilizarse en
# el siguiente notebook sin depender de todo el ranking.
#
# ============================================================

selected_rf_params = {

    "config_name":
        selected_config_name,

    "numTrees":
        selected_numTrees,

    "maxDepth":
        selected_maxDepth,

    "minInstancesPerNode":
        selected_minInstancesPerNode,

    "featureSubsetStrategy":
        selected_featureSubsetStrategy,

    "seed":
        selected_seed

}


# ============================================================
# 06.13 VALIDACIÓN FINAL
# ============================================================

print()
print("=" * 80)
print("VALIDACIÓN FINAL")
print("=" * 80)


if selected_config_name is None:

    raise RuntimeError(
        "No se ha seleccionado ninguna configuración."
    )


if technical_winner_wape > base_wape:

    raise RuntimeError(

        "El ranking contiene una inconsistencia: "
        "el ganador técnico tiene peor WAPE que el BASE."

    )


print(
    "OK - Ranking generado"
)

print(
    "OK - Ganador técnico identificado"
)

print(
    "OK - Comparación contra BASE realizada"
)

print(
    "OK - Regla de materialidad aplicada"
)

print(
    f"OK - Configuración final seleccionada: "
    f"{selected_config_name}"
)

print(
    "OK - selected_rf_params preparado"
)

print(
    "OK - TEST permanece aislado"
)

print(
    "OK - OOT permanece aislado"
)

print()
print(
    "Hyperparameter Tuning cerrado correctamente."
)