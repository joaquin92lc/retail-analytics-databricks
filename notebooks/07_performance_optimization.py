# Databricks notebook source
# DBTITLE 1,01. ANÁLISIS INICIAL DE TABLAS DELTA
# ============================================================
# 01. ANÁLISIS INICIAL DE TABLAS DELTA
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Analizar el estado físico de las principales tablas Delta
# antes de aplicar cualquier técnica de optimización.
#
# Queremos conocer:
#
# - número de registros
# - número de archivos Delta
# - tamaño total
# - tamaño medio por archivo
# - filas medias por archivo
# - columnas de partición
# - posible presencia de small files
#
# PRINCIPIO
# ------------------------------------------------------------
#
# No aplicamos OPTIMIZE, particionado o Liquid Clustering
# automáticamente.
#
# Primero medimos.
#
# Una tabla pequeña no necesita necesariamente optimización,
# aunque tenga varios archivos.
#
# ============================================================


from pyspark.sql import functions as F


# ============================================================
# 1. TABLAS PRINCIPALES A ANALIZAR
# ============================================================

tables_to_analyze = [

    "retail_analytics.1_bronze.sales",

    "retail_analytics.2_silver.sales",

    "retail_analytics.3_gold.fact_sales"

]


# ============================================================
# 2. UMBRALES ORIENTATIVOS
# ============================================================
#
# Estos valores NO son reglas absolutas de Databricks.
#
# Los utilizamos únicamente para detectar casos que merece
# la pena revisar.
#
# Un tamaño medio de archivo muy pequeño puede indicar
# fragmentación / small files.
#
# ============================================================

SMALL_FILE_THRESHOLD_MB = 16

MIN_FILES_FOR_REVIEW = 10


# ============================================================
# 3. LISTA DONDE GUARDAREMOS EL RESUMEN
# ============================================================

analysis_results = []


# ============================================================
# 4. ANALIZAMOS CADA TABLA
# ============================================================

for table_name in tables_to_analyze:

    print("=" * 80)
    print(f"TABLA: {table_name}")
    print("=" * 80)


    # ========================================================
    # 4.1 NÚMERO TOTAL DE REGISTROS
    # ========================================================

    table_df = spark.table(
        table_name
    )

    row_count = (
        table_df
        .count()
    )


    # ========================================================
    # 4.2 INFORMACIÓN FÍSICA DELTA
    # ========================================================
    #
    # DESCRIBE DETAIL nos devuelve información como:
    #
    # - numFiles
    # - sizeInBytes
    # - partitionColumns
    #
    # ========================================================

    detail = (

        spark.sql(
            f"DESCRIBE DETAIL {table_name}"
        )

        .collect()[0]

    )


    num_files = (
        detail["numFiles"]
    )


    size_bytes = (
        detail["sizeInBytes"]
    )


    partition_columns = (
        detail["partitionColumns"]
    )


    # ========================================================
    # 4.3 CONVERSIÓN A MB
    # ========================================================

    size_mb = (
        size_bytes
        /
        1024
        /
        1024
    )


    # ========================================================
    # 4.4 TAMAÑO MEDIO POR ARCHIVO
    # ========================================================

    if num_files > 0:

        avg_file_size_mb = (
            size_mb
            /
            num_files
        )

    else:

        avg_file_size_mb = 0


    # ========================================================
    # 4.5 FILAS MEDIAS POR ARCHIVO
    # ========================================================

    if num_files > 0:

        avg_rows_per_file = (
            row_count
            /
            num_files
        )

    else:

        avg_rows_per_file = 0


    # ========================================================
    # 4.6 CLASIFICACIÓN DE PERFORMANCE
    # ========================================================
    #
    # REVIEW:
    #
    # - existen bastantes archivos
    # - y el tamaño medio por archivo es pequeño
    #
    # Esto puede indicar fragmentación.
    #
    # IMPORTANTE:
    #
    # REVIEW no significa automáticamente que haya que ejecutar
    # OPTIMIZE.
    #
    # Solo indica que merece la pena analizarlo.
    #
    # ========================================================

    if (
        num_files >= MIN_FILES_FOR_REVIEW
        and
        avg_file_size_mb < SMALL_FILE_THRESHOLD_MB
    ):

        performance_status = "REVIEW - POSSIBLE SMALL FILES"

    else:

        performance_status = "OK"


    # ========================================================
    # 4.7 MOSTRAMOS RESULTADOS
    # ========================================================

    print(
        f"Registros: {row_count:,}"
    )

    print(
        f"Archivos Delta: {num_files:,}"
    )

    print(
        f"Tamaño total: {size_mb:,.2f} MB"
    )

    print(
        f"Tamaño medio por archivo: "
        f"{avg_file_size_mb:,.2f} MB"
    )

    print(
        f"Filas medias por archivo: "
        f"{avg_rows_per_file:,.0f}"
    )

    print(
        "Columnas de partición:",
        partition_columns
    )

    print(
        "Estado:",
        performance_status
    )

    print()


    # ========================================================
    # 4.8 GUARDAMOS RESULTADOS
    # ========================================================

    analysis_results.append({

        "table_name":
            table_name,

        "row_count":
            row_count,

        "num_files":
            num_files,

        "size_mb":
            round(
                size_mb,
                2
            ),

        "avg_file_size_mb":
            round(
                avg_file_size_mb,
                2
            ),

        "avg_rows_per_file":
            round(
                avg_rows_per_file,
                0
            ),

        "partition_columns":
            str(
                partition_columns
            ),

        "performance_status":
            performance_status

    })


# ============================================================
# 5. RESUMEN GLOBAL
# ============================================================
#
# Convertimos los resultados en DataFrame para facilitar la
# comparación entre Bronze, Silver y Gold.
#
# ============================================================

performance_summary_df = (
    spark.createDataFrame(
        analysis_results
    )
)


display(

    performance_summary_df

    .select(

        "table_name",

        "row_count",

        "num_files",

        "size_mb",

        "avg_file_size_mb",

        "avg_rows_per_file",

        "partition_columns",

        "performance_status"

    )

    .orderBy(
        "table_name"
    )

)

# COMMAND ----------

# DBTITLE 1,02. ANÁLISIS DEL HISTORIAL DELTA
# ============================================================
# 02. ANÁLISIS DEL HISTORIAL DELTA
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Revisar las operaciones recientes de las principales tablas
# Delta para validar cómo están siendo escritas.
#
# Queremos identificar:
#
# - WRITE
# - MERGE
# - STREAMING UPDATE
# - CREATE OR REPLACE TABLE
# - otras operaciones relevantes
#
# Esto nos permite comprobar si cada capa está funcionando de
# acuerdo con la estrategia esperada de carga.
#
# ============================================================


from pyspark.sql import functions as F


# ============================================================
# 1. TABLAS A ANALIZAR
# ============================================================

tables_to_analyze = [

    "retail_analytics.1_bronze.sales",

    "retail_analytics.2_silver.sales",

    "retail_analytics.3_gold.fact_sales"

]


# ============================================================
# 2. NÚMERO DE VERSIONES RECIENTES A MOSTRAR
# ============================================================

HISTORY_LIMIT = 10


# ============================================================
# 3. ANALIZAMOS CADA TABLA
# ============================================================

for table_name in tables_to_analyze:

    print("=" * 80)
    print(f"HISTORIAL DELTA: {table_name}")
    print("=" * 80)


    # ========================================================
    # 3.1 OBTENEMOS EL HISTORIAL DELTA
    # ========================================================

    history_df = spark.sql(
        f"DESCRIBE HISTORY {table_name}"
    )


    # ========================================================
    # 3.2 MOSTRAMOS OPERACIONES RECIENTES
    # ========================================================

    display(

        history_df

        .select(

            "version",

            "timestamp",

            "operation",

            "operationParameters",

            "operationMetrics"

        )

        .orderBy(
            F.col("version").desc()
        )

        .limit(
            HISTORY_LIMIT
        )

    )


    # ========================================================
    # 3.3 RESUMEN DE OPERACIONES
    # ========================================================
    #
    # Nos permite ver rápidamente qué tipos de escritura
    # aparecen en el historial de la tabla.
    #
    # ========================================================

    operation_summary_df = (

        history_df

        .groupBy(
            "operation"
        )

        .count()

        .orderBy(
            F.col("count").desc()
        )

    )


    print("Resumen de operaciones:")

    display(
        operation_summary_df
    )


    # ========================================================
    # 3.4 ÚLTIMA OPERACIÓN
    # ========================================================

    latest_operation = (

        history_df

        .orderBy(
            F.col("version").desc()
        )

        .select(
            "version",
            "timestamp",
            "operation",
            "operationMetrics"
        )

        .first()

    )


    print(
        "Última versión:",
        latest_operation["version"]
    )

    print(
        "Última operación:",
        latest_operation["operation"]
    )

    print(
        "Timestamp:",
        latest_operation["timestamp"]
    )

    print()

# COMMAND ----------

# DBTITLE 1,03. VALIDACIÓN DEL INCREMENTAL EN GOLD
# ============================================================
# 03. VALIDACIÓN DEL INCREMENTAL EN GOLD
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Validar que FACT_SALES está funcionando actualmente mediante
# procesamiento incremental.
#
# Comprobaremos:
#
# 1. Última operación Delta realizada.
# 2. Que la última escritura sea MERGE.
# 3. Métricas generadas por el último MERGE.
# 4. Unicidad de la clave de negocio:
#
#       sale_id + line_id
#
# 5. Coherencia entre Silver y Gold.
#
# Si alguna validación crítica falla, el notebook terminará
# con ERROR.
#
# ============================================================


from pyspark.sql import functions as F


# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

SILVER_TABLE = (
    "retail_analytics.2_silver.sales"
)

GOLD_TABLE = (
    "retail_analytics.3_gold.fact_sales"
)


# ============================================================
# 2. CARGAMOS SILVER Y GOLD
# ============================================================

silver_df = spark.table(
    SILVER_TABLE
)

gold_df = spark.table(
    GOLD_TABLE
)


# ============================================================
# 3. HISTORIAL DELTA DE FACT_SALES
# ============================================================

gold_history_df = spark.sql(
    f"DESCRIBE HISTORY {GOLD_TABLE}"
)


# ============================================================
# 4. ÚLTIMA OPERACIÓN
# ============================================================

latest_operation = (

    gold_history_df

    .orderBy(
        F.col("version").desc()
    )

    .select(
        "version",
        "timestamp",
        "operation",
        "operationMetrics"
    )

    .first()

)


latest_version = (
    latest_operation["version"]
)

latest_timestamp = (
    latest_operation["timestamp"]
)

latest_operation_name = (
    latest_operation["operation"]
)

latest_metrics = (
    latest_operation["operationMetrics"]
)


# ============================================================
# 5. RECUENTOS SILVER / GOLD
# ============================================================

silver_rows = (
    silver_df
    .count()
)

gold_rows = (
    gold_df
    .count()
)


# ============================================================
# 6. UNICIDAD DE FACT_SALES
# ============================================================
#
# La granularidad de FACT_SALES es:
#
#       sale_id + line_id
#
# Gold no debe contener duplicados para esta clave.
#
# ============================================================

gold_duplicate_keys = (

    gold_df

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
# 7. CLAVES SILVER AUSENTES EN GOLD
# ============================================================
#
# Todas las claves existentes en Silver deben estar
# materializadas en FACT_SALES.
#
# ============================================================

silver_keys_df = (

    silver_df

    .select(
        "sale_id",
        "line_id"
    )

    .distinct()

)


gold_keys_df = (

    gold_df

    .select(
        "sale_id",
        "line_id"
    )

    .distinct()

)


missing_in_gold = (

    silver_keys_df

    .join(

        gold_keys_df,

        on=[
            "sale_id",
            "line_id"
        ],

        how="left_anti"

    )

    .count()

)


# ============================================================
# 8. CLAVES GOLD QUE NO EXISTEN EN SILVER
# ============================================================
#
# También hacemos la comprobación inversa.
#
# Esto permite detectar registros residuales en Gold.
#
# ============================================================

extra_in_gold = (

    gold_keys_df

    .join(

        silver_keys_df,

        on=[
            "sale_id",
            "line_id"
        ],

        how="left_anti"

    )

    .count()

)


# ============================================================
# 9. RESUMEN
# ============================================================

print()
print("============================================")
print("VALIDACIÓN INCREMENTAL - GOLD FACT_SALES")
print("============================================")

print(
    "Última versión Delta:",
    latest_version
)

print(
    "Última operación:",
    latest_operation_name
)

print(
    "Timestamp:",
    latest_timestamp
)

print()

print(
    "Filas Silver:",
    silver_rows
)

print(
    "Filas Gold:",
    gold_rows
)

print(
    "Diferencia Gold - Silver:",
    gold_rows - silver_rows
)

print()

print(
    "Claves duplicadas Gold:",
    gold_duplicate_keys
)

print(
    "Claves Silver ausentes en Gold:",
    missing_in_gold
)

print(
    "Claves extra en Gold:",
    extra_in_gold
)


# ============================================================
# 10. MÉTRICAS DEL ÚLTIMO MERGE
# ============================================================

print()
print("Métricas última operación:")

if latest_metrics:

    for metric_name, metric_value in latest_metrics.items():

        print(
            f" - {metric_name}: {metric_value}"
        )

else:

    print(
        " - No existen métricas disponibles"
    )


# ============================================================
# 11. VALIDACIONES CRÍTICAS
# ============================================================

validation_errors = []


# ------------------------------------------------------------
# LA ÚLTIMA ESCRITURA DEBE SER MERGE
# ------------------------------------------------------------

if latest_operation_name != "MERGE":

    validation_errors.append(
        "La última operación de FACT_SALES no es MERGE "
        f"(operación encontrada: {latest_operation_name})"
    )


# ------------------------------------------------------------
# SILVER Y GOLD DEBEN TENER EL MISMO VOLUMEN
# ------------------------------------------------------------

if gold_rows != silver_rows:

    validation_errors.append(
        f"Silver contiene {silver_rows} filas y Gold "
        f"{gold_rows}"
    )


# ------------------------------------------------------------
# GOLD NO DEBE CONTENER DUPLICADOS
# ------------------------------------------------------------

if gold_duplicate_keys > 0:

    validation_errors.append(
        f"FACT_SALES contiene {gold_duplicate_keys} "
        "claves duplicadas"
    )


# ------------------------------------------------------------
# NO PUEDEN FALTAR CLAVES SILVER EN GOLD
# ------------------------------------------------------------

if missing_in_gold > 0:

    validation_errors.append(
        f"Faltan {missing_in_gold} claves de Silver en Gold"
    )


# ------------------------------------------------------------
# GOLD NO DEBE CONTENER CLAVES RESIDUALES
# ------------------------------------------------------------

if extra_in_gold > 0:

    validation_errors.append(
        f"Gold contiene {extra_in_gold} claves que no existen "
        "en Silver"
    )


# ============================================================
# 12. RESULTADO FINAL
# ============================================================

if validation_errors:

    print()
    print(
        "ERROR - Validación incremental Gold fallida"
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
        "OK - FACT_SALES validada correctamente"
    )

    print(
        "OK - Última escritura realizada mediante MERGE"
    )

    print(
        "OK - Silver y Gold están sincronizados"
    )

    print(
        "OK - No existen duplicados ni claves residuales"
    )