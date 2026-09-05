# Databricks notebook source
# DBTITLE 1,00. VALIDACION OPERATIVA
# ============================================================
# 00. VALIDACIÓN OPERATIVA
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Realizar una inspección operativa de la tabla Bronze SALES
# sin depender de un año o periodo concreto.
#
# Mostraremos:
#
# - número total de filas físicas en Bronze
# - número de claves de negocio únicas
# - número de filas físicas duplicadas
# - número de archivos procesados
# - rango temporal disponible
# - detalle de registros por archivo origen
#
# IMPORTANTE
# ------------------------------------------------------------
#
# Bronze representa la información recibida desde el origen.
#
# Por tanto, pueden existir varias filas físicas con la misma
# clave de negocio:
#
#       sale_id + line_id
#
# Esto NO implica automáticamente un error de ingestión.
#
# La unicidad de negocio se validará posteriormente en Silver.
#
# ============================================================


from pyspark.sql import functions as F


# ============================================================
# 1. CARGAMOS SALES BRONZE
# ============================================================

sales_bronze_df = spark.table(
    "retail_analytics.1_bronze.sales"
)


# ============================================================
# 2. TOTAL DE FILAS FÍSICAS
# ============================================================

bronze_total_rows = sales_bronze_df.count()


# ============================================================
# 3. CLAVES DE NEGOCIO ÚNICAS
# ============================================================
#
# La granularidad lógica de SALES es:
#
#       sale_id + line_id
#
# ============================================================

bronze_unique_keys = (
    sales_bronze_df
    .select(
        "sale_id",
        "line_id"
    )
    .distinct()
    .count()
)


# ============================================================
# 4. FILAS FÍSICAS DUPLICADAS
# ============================================================
#
# Diferencia entre:
#
# filas físicas
# -
# claves de negocio únicas
#
# Estas filas se muestran como información operativa.
#
# No provocan ERROR en Bronze.
#
# ============================================================

bronze_duplicate_rows = (
    bronze_total_rows
    -
    bronze_unique_keys
)


# ============================================================
# 5. NÚMERO DE ARCHIVOS PROCESADOS
# ============================================================

bronze_source_files = (
    sales_bronze_df
    .select("_source_file")
    .distinct()
    .count()
)


# ============================================================
# 6. RANGO TEMPORAL DISPONIBLE
# ============================================================

bronze_date_range = (
    sales_bronze_df
    .select(

        F.min("sale_timestamp").alias(
            "fecha_minima"
        ),

        F.max("sale_timestamp").alias(
            "fecha_maxima"
        )

    )
    .first()
)


# ============================================================
# 7. RESUMEN OPERATIVO
# ============================================================

print("============================================")
print("VALIDACIÓN OPERATIVA - BRONZE SALES")
print("============================================")

print(
    "Filas físicas Bronze:",
    bronze_total_rows
)

print(
    "Claves de negocio únicas:",
    bronze_unique_keys
)

print(
    "Filas físicas duplicadas:",
    bronze_duplicate_rows
)

print(
    "Archivos procesados:",
    bronze_source_files
)

print(
    "Fecha mínima:",
    bronze_date_range["fecha_minima"]
)

print(
    "Fecha máxima:",
    bronze_date_range["fecha_maxima"]
)


# ============================================================
# 8. DETALLE POR ARCHIVO ORIGEN
# ============================================================
#
# Ordenamos por _source_file para poder inspeccionar fácilmente
# qué archivos han sido incorporados por Auto Loader.
#
# Ya NO filtramos específicamente 2026.
#
# ============================================================

display(

    sales_bronze_df

    .groupBy(
        "_source_file"
    )

    .agg(

        F.count("*").alias(
            "physical_rows"
        ),

        F.countDistinct(
            "sale_id",
            "line_id"
        ).alias(
            "unique_business_keys"
        ),

        F.min(
            "sale_timestamp"
        ).alias(
            "min_sale_timestamp"
        ),

        F.max(
            "sale_timestamp"
        ).alias(
            "max_sale_timestamp"
        )

    )

    .orderBy(
        "_source_file"
    )

)

# COMMAND ----------

# DBTITLE 1,02. VALIDACIÓN DE INCREMENTALIDAD EN BRONZE
# ============================================================
# 02. VALIDACIÓN DE INCREMENTALIDAD EN BRONZE
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Validar automáticamente el periodo más reciente incorporado
# a Bronze sin depender de:
#
# - un año concreto
# - un mes concreto
# - un número esperado de registros
#
# El periodo incremental se obtendrá directamente de:
#
#       _source_file
#
# cuya estructura contiene:
#
#       year=YYYY/month=MM
#
#
# VALIDAREMOS
# ------------------------------------------------------------
#
# 1. Detectar automáticamente el último periodo disponible.
#
# 2. Comprobar que existen registros para ese periodo.
#
# 3. Comprobar que existen archivos origen.
#
# 4. Comprobar que sale_id y line_id no sean NULL.
#
# 5. Analizar duplicados físicos de la clave:
#
#       sale_id + line_id
#
#    En Bronze los duplicados se consideran INFORMACIÓN
#    OPERATIVA, no un ERROR automático.
#
# 6. Comprobar que las fechas de negocio pertenecen al
#    year/month indicado por la ruta del archivo.
#
# ============================================================


from pyspark.sql import functions as F


# ============================================================
# 1. CARGAMOS SALES BRONZE
# ============================================================

sales_bronze_df = spark.table(
    "retail_analytics.1_bronze.sales"
)


# ============================================================
# 2. EXTRAEMOS YEAR Y MONTH DESDE _SOURCE_FILE
# ============================================================
#
# Ejemplo:
#
# /Volumes/.../sales/year=2026/month=08/sales_2026_08_01.csv
#
# Se convertirá en:
#
# source_year  = 2026
# source_month = 8
#
# No dependemos así de ningún periodo hardcodeado.
#
# ============================================================

sales_bronze_period_df = (

    sales_bronze_df

    .withColumn(

        "_source_year",

        F.regexp_extract(
            F.col("_source_file"),
            r"year=(\d{4})",
            1
        ).cast("int")

    )

    .withColumn(

        "_source_month",

        F.regexp_extract(
            F.col("_source_file"),
            r"month=(\d{1,2})",
            1
        ).cast("int")

    )

)


# ============================================================
# 3. VALIDAMOS QUE EL PERIODO PUEDA EXTRAERSE
# ============================================================
#
# Si aparecen registros sin year/month en la ruta,
# la estructura del Landing habría cambiado y esta validación
# ya no sería fiable.
#
# ============================================================

invalid_source_period_rows = (

    sales_bronze_period_df

    .filter(

        F.col("_source_year").isNull()

        |

        F.col("_source_month").isNull()

    )

    .count()

)


# ============================================================
# 4. DETECTAMOS EL ÚLTIMO PERIODO DISPONIBLE
# ============================================================
#
# Ordenamos primero por año y después por mes.
#
# Ejemplos:
#
# 2026 / 08  >  2026 / 07
# 2027 / 01  >  2026 / 12
#
# ============================================================

latest_period = (

    sales_bronze_period_df

    .filter(

        F.col("_source_year").isNotNull()

        &

        F.col("_source_month").isNotNull()

    )

    .select(
        "_source_year",
        "_source_month"
    )

    .distinct()

    .orderBy(
        F.col("_source_year").desc(),
        F.col("_source_month").desc()
    )

    .first()

)


# ============================================================
# 5. CONTROLAMOS QUE EXISTA ALGÚN PERIODO
# ============================================================

if latest_period is None:

    raise RuntimeError(
        "ERROR - No se ha podido detectar ningún periodo "
        "year/month en _source_file."
    )


latest_year = latest_period["_source_year"]
latest_month = latest_period["_source_month"]


print(
    f"Último periodo detectado: "
    f"{latest_year}-{latest_month:02d}"
)


# ============================================================
# 6. AISLAMOS EL ÚLTIMO PERIODO
# ============================================================

latest_bronze_df = (

    sales_bronze_period_df

    .filter(

        (F.col("_source_year") == latest_year)

        &

        (F.col("_source_month") == latest_month)

    )

)


# ============================================================
# 7. VOLUMEN DEL ÚLTIMO PERIODO
# ============================================================

latest_period_rows = (
    latest_bronze_df
    .count()
)


# ============================================================
# 8. ARCHIVOS DEL ÚLTIMO PERIODO
# ============================================================

latest_period_files = (

    latest_bronze_df

    .select(
        "_source_file"
    )

    .distinct()

    .count()

)


# ============================================================
# 9. CLAVES DE NEGOCIO ÚNICAS
# ============================================================

latest_unique_keys = (

    latest_bronze_df

    .select(
        "sale_id",
        "line_id"
    )

    .distinct()

    .count()

)


# ============================================================
# 10. FILAS FÍSICAS DUPLICADAS
# ============================================================
#
# IMPORTANTE:
#
# Bronze conserva el dato recibido desde el origen.
#
# Por tanto:
#
#     physical_rows > unique_business_keys
#
# no implica automáticamente un problema del pipeline.
#
# La unicidad estricta se exigirá en Silver.
#
# ============================================================

latest_duplicate_rows = (
    latest_period_rows
    -
    latest_unique_keys
)


# ============================================================
# 11. CLAVES DE NEGOCIO NULL
# ============================================================
#
# sale_id + line_id identifican una línea de venta.
#
# Estos campos sí deberían estar informados.
#
# ============================================================

null_business_keys = (

    latest_bronze_df

    .filter(

        F.col("sale_id").isNull()

        |

        F.col("line_id").isNull()

    )

    .count()

)


# ============================================================
# 12. VALIDACIÓN TEMPORAL
# ============================================================
#
# Comprobamos que el timestamp de negocio corresponda al mismo
# year/month que indica la carpeta de origen.
#
# Si un fichero situado en:
#
#       year=2026/month=08
#
# contiene una venta de julio, la clasificaremos como error.
#
# ============================================================

invalid_period_dates = (

    latest_bronze_df

    .filter(

        (F.year("sale_timestamp") != latest_year)

        |

        (F.month("sale_timestamp") != latest_month)

        |

        F.col("sale_timestamp").isNull()

    )

    .count()

)


# ============================================================
# 13. RANGO TEMPORAL REAL
# ============================================================

latest_date_range = (

    latest_bronze_df

    .select(

        F.min(
            "sale_timestamp"
        ).alias(
            "fecha_minima"
        ),

        F.max(
            "sale_timestamp"
        ).alias(
            "fecha_maxima"
        )

    )

    .first()

)


# ============================================================
# 14. MOSTRAMOS EL RESUMEN
# ============================================================

print()
print("============================================")
print("VALIDACIÓN INCREMENTAL - BRONZE")
print("============================================")

print(
    "Periodo:",
    f"{latest_year}-{latest_month:02d}"
)

print(
    "Filas físicas:",
    latest_period_rows
)

print(
    "Claves de negocio únicas:",
    latest_unique_keys
)

print(
    "Filas físicas duplicadas:",
    latest_duplicate_rows
)

print(
    "Archivos procesados:",
    latest_period_files
)

print(
    "Claves NULL:",
    null_business_keys
)

print(
    "Registros fuera del periodo:",
    invalid_period_dates
)

print(
    "Fecha mínima:",
    latest_date_range["fecha_minima"]
)

print(
    "Fecha máxima:",
    latest_date_range["fecha_maxima"]
)

print(
    "Registros con year/month no extraíble:",
    invalid_source_period_rows
)


# ============================================================
# 15. DETALLE POR ARCHIVO DEL ÚLTIMO PERIODO
# ============================================================

display(

    latest_bronze_df

    .groupBy(
        "_source_file"
    )

    .agg(

        F.count("*").alias(
            "physical_rows"
        ),

        F.countDistinct(
            "sale_id",
            "line_id"
        ).alias(
            "unique_business_keys"
        ),

        F.min(
            "sale_timestamp"
        ).alias(
            "min_sale_timestamp"
        ),

        F.max(
            "sale_timestamp"
        ).alias(
            "max_sale_timestamp"
        )

    )

    .orderBy(
        "_source_file"
    )

)


# ============================================================
# 16. RESULTADO FINAL
# ============================================================
#
# ERRORES CRÍTICOS:
#
# - no existen registros en el último periodo
# - no existen archivos
# - claves de negocio NULL
# - fechas incompatibles con year/month
# - no puede extraerse year/month de alguna ruta
#
# Los duplicados físicos de Bronze NO provocan el fallo.
#
# ============================================================

validation_errors = []


if latest_period_rows == 0:

    validation_errors.append(
        "El último periodo no contiene registros"
    )


if latest_period_files == 0:

    validation_errors.append(
        "No existen archivos origen para el último periodo"
    )


if null_business_keys > 0:

    validation_errors.append(
        f"{null_business_keys} registros tienen "
        "sale_id o line_id NULL"
    )


if invalid_period_dates > 0:

    validation_errors.append(
        f"{invalid_period_dates} registros no pertenecen "
        "al periodo indicado por _source_file"
    )


if invalid_source_period_rows > 0:

    validation_errors.append(
        f"{invalid_source_period_rows} registros tienen una "
        "ruta _source_file sin year/month válido"
    )


# ============================================================
# 17. LANZAMOS RESULTADO
# ============================================================

if validation_errors:

    print()
    print("ERROR - Validación incremental Bronze fallida")

    for error in validation_errors:
        print(" -", error)

    raise RuntimeError(
        " | ".join(validation_errors)
    )


else:

    print()
    print(
        "OK - Incrementalidad Bronze validada correctamente"
    )

    if latest_duplicate_rows > 0:

        print(
            "INFO - Bronze contiene "
            f"{latest_duplicate_rows} filas físicas duplicadas. "
            "La unicidad de negocio se validará en Silver."
        )

# COMMAND ----------

# DBTITLE 1,03.00 DIAGNÓSTICO DE CLAVES BRONZE AUSENTES EN SILVER
# ============================================================
# 03.00 DIAGNÓSTICO DE CLAVES BRONZE AUSENTES EN SILVER
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Identificar exactamente qué registros existen en Bronze
# pero no aparecen en Silver para el último periodo.
#
# No modificamos todavía ninguna validación.
#
# Primero necesitamos determinar si estas 23 claves:
#
# - son registros inválidos correctamente descartados,
# - tienen problemas de calidad,
# - han sido filtrados por alguna regla de Silver,
# - o realmente se han perdido durante el procesamiento.
#
# ============================================================


# ============================================================
# 1. OBTENEMOS LAS CLAVES AUSENTES
# ============================================================

missing_keys_df = (

    bronze_latest_df

    .select(
        "sale_id",
        "line_id"
    )

    .distinct()

    .join(

        silver_latest_df
        .select(
            "sale_id",
            "line_id"
        )
        .distinct(),

        on=[
            "sale_id",
            "line_id"
        ],

        how="left_anti"

    )

)


# ============================================================
# 2. RECUPERAMOS LOS REGISTROS COMPLETOS DE BRONZE
# ============================================================

missing_bronze_rows_df = (

    bronze_latest_df

    .join(

        missing_keys_df,

        on=[
            "sale_id",
            "line_id"
        ],

        how="inner"

    )

)


# ============================================================
# 3. MOSTRAMOS LAS 23 FILAS
# ============================================================

print(
    "Claves Bronze ausentes en Silver:",
    missing_keys_df.count()
)


display(

    missing_bronze_rows_df

    .orderBy(
        "sale_id",
        "line_id"
    )

)

# COMMAND ----------

# DBTITLE 1,03.01 COMPROBAR SI LAS 23 CLAVES YA EXISTEN EN SILVER
#============================================================
# 03.02 COMPROBAR SI LAS 23 CLAVES YA EXISTEN EN SILVER
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Comprobar si las claves que parecen "ausentes" en agosto
# realmente existen en Silver, pero asociadas a otro periodo.
#
# Esto nos permitirá distinguir entre:
#
# 1. Pérdida real de registros
#
# 2. Claves de negocio reutilizadas en Bronze que Silver
#    correctamente no vuelve a insertar
#
# ============================================================


# ============================================================
# 1. BUSCAMOS LAS CLAVES EN TODA LA TABLA SILVER
# ============================================================

missing_keys_existing_in_silver_df = (

    missing_keys_df

    .join(

        sales_silver_df,

        on=[
            "sale_id",
            "line_id"
        ],

        how="inner"

    )

)


# ============================================================
# 2. CONTAMOS CUÁNTAS DE LAS 23 EXISTEN YA EN SILVER
# ============================================================

existing_in_silver = (
    missing_keys_existing_in_silver_df
    .select(
        "sale_id",
        "line_id"
    )
    .distinct()
    .count()
)


print(
    "Claves aparentemente ausentes:",
    missing_keys_df.count()
)

print(
    "Claves que ya existen en Silver:",
    existing_in_silver
)


# ============================================================
# 3. MOSTRAMOS DÓNDE ESTÁN
# ============================================================

display(

    missing_keys_existing_in_silver_df

    .select(
        "sale_id",
        "line_id",
        "sale_timestamp"
    )

    .orderBy(
        "sale_id",
        "line_id"
    )

)

# COMMAND ----------

# DBTITLE 1,03. VALIDACIÓN DE SILVER
# ============================================================
# 03. VALIDACIÓN DE SILVER
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Validar que el último periodo disponible en Bronze ha sido
# correctamente procesado en Silver.
#
# VALIDAREMOS
# ------------------------------------------------------------
#
# 1. Detectar automáticamente el último periodo de Bronze.
#
# 2. Comprobar que Silver contiene registros de ese periodo.
#
# 3. Exigir unicidad estricta de:
#
#       sale_id + line_id
#
# 4. Comprobar claves obligatorias NULL.
#
# 5. Validar que TODAS las claves del último periodo de Bronze
#    existen finalmente en Silver.
#
# IMPORTANTE
# ------------------------------------------------------------
#
# Una clave recibida en el último periodo de Bronze puede haber
# sido procesada anteriormente en Silver.
#
# Por tanto, NO exigimos:
#
#     claves Bronze agosto == filas Silver agosto
#
# sino:
#
#     todas las claves Bronze del último periodo
#              ↓
#     deben existir en Silver GLOBAL
#
# 6. Validar el rango temporal de las filas Silver que sí
#    pertenecen al último periodo.
#
# 7. Lanzar ERROR si alguna condición crítica falla.
#
# ============================================================


from pyspark.sql import functions as F


# ============================================================
# 1. CARGAMOS BRONZE Y SILVER
# ============================================================

sales_bronze_df = spark.table(
    "retail_analytics.1_bronze.sales"
)

sales_silver_df = spark.table(
    "retail_analytics.2_silver.sales"
)


# ============================================================
# 2. EXTRAEMOS YEAR / MONTH DESDE _SOURCE_FILE EN BRONZE
# ============================================================

bronze_period_df = (

    sales_bronze_df

    .withColumn(

        "_source_year",

        F.regexp_extract(
            F.col("_source_file"),
            r"year=(\d{4})",
            1
        ).cast("int")

    )

    .withColumn(

        "_source_month",

        F.regexp_extract(
            F.col("_source_file"),
            r"month=(\d{1,2})",
            1
        ).cast("int")

    )

)


# ============================================================
# 3. DETECTAMOS EL ÚLTIMO PERIODO DISPONIBLE EN BRONZE
# ============================================================

latest_period = (

    bronze_period_df

    .filter(

        F.col("_source_year").isNotNull()

        &

        F.col("_source_month").isNotNull()

    )

    .select(
        "_source_year",
        "_source_month"
    )

    .distinct()

    .orderBy(
        F.col("_source_year").desc(),
        F.col("_source_month").desc()
    )

    .first()

)


if latest_period is None:

    raise RuntimeError(
        "ERROR - No se ha podido detectar el último periodo "
        "disponible en Bronze."
    )


latest_year = latest_period["_source_year"]
latest_month = latest_period["_source_month"]


print(
    f"Último periodo detectado en Bronze: "
    f"{latest_year}-{latest_month:02d}"
)


# ============================================================
# 4. FILTRAMOS BRONZE AL ÚLTIMO PERIODO
# ============================================================

bronze_latest_df = (

    bronze_period_df

    .filter(

        (F.col("_source_year") == latest_year)

        &

        (F.col("_source_month") == latest_month)

    )

)


# ============================================================
# 5. FILTRAMOS SILVER AL MISMO PERIODO
# ============================================================

silver_latest_df = (

    sales_silver_df

    .filter(

        (F.year("sale_timestamp") == latest_year)

        &

        (F.month("sale_timestamp") == latest_month)

    )

)


# ============================================================
# 6. CLAVES ÚNICAS DE BRONZE DEL ÚLTIMO PERIODO
# ============================================================
#
# Bronze puede contener duplicados físicos.
#
# Por tanto, trabajamos con claves de negocio únicas:
#
#       sale_id + line_id
#
# ============================================================

bronze_latest_unique_keys_df = (

    bronze_latest_df

    .select(
        "sale_id",
        "line_id"
    )

    .distinct()

)


bronze_latest_unique_keys = (
    bronze_latest_unique_keys_df
    .count()
)


# ============================================================
# 7. FILAS SILVER DEL ÚLTIMO PERIODO
# ============================================================

silver_latest_rows = (
    silver_latest_df
    .count()
)


# ============================================================
# 8. DUPLICADOS EN SILVER DEL ÚLTIMO PERIODO
# ============================================================
#
# Silver sí debe mantener unicidad estricta de:
#
#       sale_id + line_id
#
# ============================================================

silver_duplicate_keys = (

    silver_latest_df

    .groupBy(
        "sale_id",
        "line_id"
    )

    .count()

    .filter(
        F.col("count") > 1
    )

    .count()

)


# ============================================================
# 9. CLAVES OBLIGATORIAS NULL EN SILVER
# ============================================================

silver_null_business_keys = (

    sales_silver_df

    .filter(

        F.col("sale_id").isNull()

        |

        F.col("line_id").isNull()

    )

    .count()

)


# ============================================================
# 10. COMPROBAMOS QUE TODAS LAS CLAVES BRONZE EXISTEN
#     FINALMENTE EN SILVER GLOBAL
# ============================================================
#
# Este es el punto clave.
#
# NO comparamos únicamente contra silver_latest_df.
#
# Puede ocurrir que una clave recibida de nuevo en agosto ya
# estuviera cargada en enero, febrero, mayo, etc.
#
# En ese caso Silver no debe duplicarla.
#
# Por tanto buscamos contra TODA Silver.
#
# ============================================================

silver_global_keys_df = (

    sales_silver_df

    .select(
        "sale_id",
        "line_id"
    )

    .distinct()

)


missing_in_silver_df = (

    bronze_latest_unique_keys_df

    .join(

        silver_global_keys_df,

        on=[
            "sale_id",
            "line_id"
        ],

        how="left_anti"

    )

)


missing_in_silver = (
    missing_in_silver_df
    .count()
)


# ============================================================
# 11. IDENTIFICAMOS CLAVES YA EXISTENTES DE PERIODOS ANTERIORES
# ============================================================
#
# Queremos saber cuántas claves del último periodo Bronze:
#
# - existen en Silver
# - pero su registro Silver pertenece a un periodo anterior
#
# Esto es INFO, no ERROR.
#
# ============================================================

bronze_keys_with_silver_date_df = (

    bronze_latest_unique_keys_df

    .join(

        sales_silver_df.select(
            "sale_id",
            "line_id",
            "sale_timestamp"
        ),

        on=[
            "sale_id",
            "line_id"
        ],

        how="inner"

    )

)


previous_period_existing_keys = (

    bronze_keys_with_silver_date_df

    .filter(

        (F.year("sale_timestamp") != latest_year)

        |

        (F.month("sale_timestamp") != latest_month)

    )

    .select(
        "sale_id",
        "line_id"
    )

    .distinct()

    .count()

)


# ============================================================
# 12. VALIDAMOS FECHAS DE LAS FILAS SILVER DEL ÚLTIMO PERIODO
# ============================================================

silver_invalid_period_dates = (

    silver_latest_df

    .filter(

        (F.year("sale_timestamp") != latest_year)

        |

        (F.month("sale_timestamp") != latest_month)

        |

        F.col("sale_timestamp").isNull()

    )

    .count()

)


# ============================================================
# 13. RANGO TEMPORAL SILVER DEL ÚLTIMO PERIODO
# ============================================================

silver_date_range = (

    silver_latest_df

    .select(

        F.min(
            "sale_timestamp"
        ).alias(
            "fecha_minima"
        ),

        F.max(
            "sale_timestamp"
        ).alias(
            "fecha_maxima"
        )

    )

    .first()

)


# ============================================================
# 14. VOLUMEN TOTAL SILVER
# ============================================================

total_silver = (
    sales_silver_df
    .count()
)


# ============================================================
# 15. RESUMEN
# ============================================================

print()
print("============================================")
print("VALIDACIÓN INCREMENTAL - SILVER")
print("============================================")

print(
    "Periodo:",
    f"{latest_year}-{latest_month:02d}"
)

print(
    "Filas totales Silver:",
    total_silver
)

print(
    "Claves Bronze del periodo:",
    bronze_latest_unique_keys
)

print(
    "Filas Silver del periodo:",
    silver_latest_rows
)

print(
    "Claves Bronze ausentes en Silver global:",
    missing_in_silver
)

print(
    "Claves Bronze ya existentes de periodos anteriores:",
    previous_period_existing_keys
)

print(
    "Claves duplicadas en Silver:",
    silver_duplicate_keys
)

print(
    "Claves NULL en Silver:",
    silver_null_business_keys
)

print(
    "Registros Silver fuera del periodo:",
    silver_invalid_period_dates
)

print(
    "Fecha mínima Silver:",
    silver_date_range["fecha_minima"]
)

print(
    "Fecha máxima Silver:",
    silver_date_range["fecha_maxima"]
)


# ============================================================
# 16. VALIDACIONES CRÍTICAS
# ============================================================

validation_errors = []


# ------------------------------------------------------------
# SILVER DEBE CONTENER DATOS DEL ÚLTIMO PERIODO
# ------------------------------------------------------------

if silver_latest_rows == 0:

    validation_errors.append(
        "Silver no contiene registros para el último periodo"
    )


# ------------------------------------------------------------
# SILVER NO DEBE TENER CLAVES DUPLICADAS
# ------------------------------------------------------------

if silver_duplicate_keys > 0:

    validation_errors.append(
        f"Silver contiene {silver_duplicate_keys} "
        "claves duplicadas en el último periodo"
    )


# ------------------------------------------------------------
# SALE_ID Y LINE_ID SON OBLIGATORIOS
# ------------------------------------------------------------

if silver_null_business_keys > 0:

    validation_errors.append(
        f"Silver contiene {silver_null_business_keys} "
        "registros con sale_id o line_id NULL"
    )


# ------------------------------------------------------------
# TODAS LAS CLAVES BRONZE DEBEN EXISTIR EN SILVER GLOBAL
# ------------------------------------------------------------

if missing_in_silver > 0:

    validation_errors.append(
        f"Faltan {missing_in_silver} claves del último "
        "periodo Bronze en Silver"
    )


# ------------------------------------------------------------
# VALIDACIÓN TEMPORAL
# ------------------------------------------------------------

if silver_invalid_period_dates > 0:

    validation_errors.append(
        f"Silver contiene {silver_invalid_period_dates} "
        "registros fuera del periodo esperado"
    )


# ============================================================
# 17. RESULTADO FINAL
# ============================================================

if validation_errors:

    print()
    print(
        "ERROR - Validación incremental Silver fallida"
    )

    for error in validation_errors:

        print(
            " -",
            error
        )

    raise RuntimeError(
        " | ".join(validation_errors)
    )


else:

    print()
    print(
        "OK - Último periodo procesado correctamente en Silver"
    )


    # ========================================================
    # INFORMACIÓN OPERATIVA
    # ========================================================

    if previous_period_existing_keys > 0:

        print(
            "INFO - "
            f"{previous_period_existing_keys} claves recibidas "
            "en el último periodo Bronze ya existían en Silver "
            "de periodos anteriores."
        )

        print(
            "INFO - No se consideran registros perdidos ni "
            "duplicados Silver."
        )