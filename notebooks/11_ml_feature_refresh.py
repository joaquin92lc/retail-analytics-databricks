# Databricks notebook source
# MAGIC %md
# MAGIC # 00. CONFIGURACIÓN Y LECTURA DE GOLD
# MAGIC
# MAGIC ## Objetivo
# MAGIC
# MAGIC Construir un proceso independiente de **Feature Engineering** para Machine Learning a partir de la capa Gold.
# MAGIC
# MAGIC Este notebook tendrá una única responsabilidad:
# MAGIC
# MAGIC **Gold → Feature Table**
# MAGIC
# MAGIC La tabla resultante será:
# MAGIC
# MAGIC `retail_analytics.5_ml.daily_store_features`
# MAGIC
# MAGIC y será utilizada posteriormente por el proceso de inference.
# MAGIC
# MAGIC El notebook debe poder ejecutarse desde una sesión limpia y desde Lakeflow Jobs sin depender de variables creadas manualmente en otros notebooks.

# COMMAND ----------
# Catálogo recibido desde Databricks Asset Bundles.
# DEV  -> retail_analytics_dev
# PROD -> retail_analytics

dbutils.widgets.text("catalog", "retail_analytics")
CATALOG = dbutils.widgets.get("catalog")

print(f"Environment catalog: {CATALOG}")

# DBTITLE 1,00.01 CONFIGURACIÓN Y LECTURA DE GOLD
# ============================================================
# 00.01 CONFIGURACIÓN Y LECTURA DE GOLD
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Cargar las tablas necesarias desde Gold y centralizar
# la configuración del proceso de Feature Engineering.
#
# El número esperado de tiendas se obtendrá dinámicamente
# de Gold para evitar hardcodes innecesarios.
#
# ============================================================


from pyspark.sql import functions as F


# ============================================================
# 00.01.01 TABLAS ORIGEN
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

FEATURE_TABLE = (
    f"{CATALOG}.5_ml.daily_store_features"
)


# ============================================================
# 00.01.03 PARÁMETROS FUNCIONALES
# ============================================================

MAX_LAG_DAYS = 28


# ============================================================
# 00.01.04 VALIDACIÓN DE TABLAS ORIGEN
# ============================================================

required_tables = [

    FACT_SALES_TABLE,
    DIM_DATE_TABLE,
    DIM_STORE_TABLE

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

        "Faltan tablas Gold necesarias: "
        f"{missing_tables}"

    )


print(
    "OK - Todas las tablas Gold necesarias están disponibles"
)


# ============================================================
# 00.01.05 LECTURA DE GOLD
# ============================================================

fact_sales_df = spark.table(
    FACT_SALES_TABLE
)


dim_date_df = spark.table(
    DIM_DATE_TABLE
)


dim_store_df = spark.table(
    DIM_STORE_TABLE
)


# ============================================================
# 00.01.06 NÚMERO ESPERADO DE TIENDAS
# ============================================================
#
# Utilizamos dim_store como referencia de las tiendas
# existentes en el modelo Gold.
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


if EXPECTED_STORES <= 0:

    raise RuntimeError(
        "No se han encontrado tiendas en dim_store."
    )


# ============================================================
# 00.01.07 VALIDACIÓN DE FACT SALES
# ============================================================

fact_rows = (
    fact_sales_df.count()
)


fact_stores = (

    fact_sales_df

    .select(
        "store_id"
    )

    .distinct()

    .count()

)


if fact_rows == 0:

    raise RuntimeError(
        "fact_sales está vacía."
    )


if fact_stores != EXPECTED_STORES:

    raise RuntimeError(

        "La cobertura de tiendas de fact_sales "
        "no coincide con dim_store. "
        f"dim_store: {EXPECTED_STORES} | "
        f"fact_sales: {fact_stores}"

    )


# ============================================================
# 00.01.08 RANGO TEMPORAL
# ============================================================

date_range = (

    fact_sales_df
    .alias("fact")

    .join(

        dim_date_df

        .select(

            "date_key",

            F.col(
                "date"
            ).alias(
                "sale_date"
            )

        )

        .alias("date"),

        on=
            "date_key",

        how=
            "inner"

    )

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


if (
    date_range["min_date"] is None
    or
    date_range["max_date"] is None
):

    raise RuntimeError(

        "No se ha podido determinar "
        "el periodo disponible en Gold."

    )


# ============================================================
# 00.01.09 RESULTADOS
# ============================================================

print()
print("=" * 80)
print("CONFIGURACIÓN FEATURE REFRESH")
print("=" * 80)


print(
    f"Fact Sales:       "
    f"{FACT_SALES_TABLE}"
)


print(
    f"Feature Table:    "
    f"{FEATURE_TABLE}"
)


print(
    f"Registros Gold:   "
    f"{fact_rows:,}"
)


print(
    f"Tiendas:          "
    f"{EXPECTED_STORES}"
)


print(
    f"Periodo:          "
    f"{date_range['min_date']} "
    f"-> "
    f"{date_range['max_date']}"
)


# ============================================================
# 00.01.10 VALIDACIÓN FINAL
# ============================================================

print()
print("=" * 80)
print("VALIDACIÓN FINAL")
print("=" * 80)


print(
    "OK - Tablas Gold disponibles"
)


print(
    "OK - fact_sales contiene registros"
)


print(
    f"OK - {EXPECTED_STORES} tiendas detectadas dinámicamente"
)


print(
    "OK - Cobertura de tiendas consistente"
)


print(
    f"OK - Periodo Gold: "
    f"{date_range['min_date']} "
    f"-> "
    f"{date_range['max_date']}"
)


print()
print(
    "Configuración y lectura de Gold correctas."
)

# COMMAND ----------

# DBTITLE 1,01. AGREGACIÓN DIARIA POR TIENDA
# ============================================================
# 01. AGREGACIÓN DIARIA POR TIENDA
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Construir desde Gold el dataset diario base para
# Feature Engineering.
#
# Granularidad:
#
#       1 fila = 1 tienda + 1 día
#
# Target principal:
#
#       net_sales
#
# También conservaremos métricas complementarias:
#
# - gross_sales
# - total_discount
# - total_units
# - total_tickets
#
# ============================================================


from pyspark.sql import functions as F


# ============================================================
# 1. INCORPORAMOS LA FECHA REAL
# ============================================================
#
# fact_sales contiene date_key.
#
# Recuperamos sale_date desde dim_date.
#
# ============================================================

sales_with_date_df = (

    fact_sales_df.alias("fact")

    .join(

        dim_date_df
        .select(
            "date_key",
            F.col("date").alias("sale_date")
        )
        .alias("date"),

        on="date_key",

        how="inner"
    )
)


# ============================================================
# 2. AGREGACIÓN DIARIA POR TIENDA
# ============================================================

daily_store_sales_df = (

    sales_with_date_df

    .groupBy(
        "sale_date",
        "store_id"
    )

    .agg(

        # ----------------------------------------------------
        # TARGET PRINCIPAL
        # ----------------------------------------------------

        F.round(
            F.sum("net_amount"),
            2
        ).alias(
            "net_sales"
        ),


        # ----------------------------------------------------
        # MÉTRICAS COMPLEMENTARIAS
        # ----------------------------------------------------

        F.round(
            F.sum("gross_amount"),
            2
        ).alias(
            "gross_sales"
        ),

        F.round(
            F.sum("discount_amount"),
            2
        ).alias(
            "total_discount"
        ),

        F.sum(
            "quantity"
        ).alias(
            "total_units"
        ),

        F.countDistinct(
            "sale_id"
        ).alias(
            "total_tickets"
        )
    )
)


# ============================================================
# 3. VALIDACIÓN DEL RESULTADO
# ============================================================

daily_rows = (
    daily_store_sales_df
    .count()
)

daily_stores = (
    daily_store_sales_df
    .select("store_id")
    .distinct()
    .count()
)

daily_dates = (
    daily_store_sales_df
    .select("sale_date")
    .distinct()
    .count()
)


date_range = (

    daily_store_sales_df

    .agg(

        F.min("sale_date")
        .alias("min_date"),

        F.max("sale_date")
        .alias("max_date")
    )

    .first()
)


print("=" * 75)
print("AGREGACIÓN DIARIA POR TIENDA")
print("=" * 75)

print(
    f"Registros:      {daily_rows:,}"
)

print(
    f"Tiendas:        {daily_stores}"
)

print(
    f"Días con datos: {daily_dates}"
)

print(
    f"Periodo:        {date_range['min_date']} "
    f"-> {date_range['max_date']}"
)


# ============================================================
# 4. VALIDACIÓN DE DUPLICADOS
# ============================================================

duplicate_store_dates = (

    daily_store_sales_df

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
    f"Duplicados tienda-fecha: {duplicate_store_dates}"
)


if (
    daily_stores == EXPECTED_STORES
    and
    duplicate_store_dates == 0
):

    print()
    print(
        "OK - Agregación diaria construida correctamente"
    )

else:

    raise RuntimeError(
        "La agregación diaria presenta inconsistencias"
    )


# ============================================================
# 5. INSPECCIÓN
# ============================================================

display(

    daily_store_sales_df

    .orderBy(
        "sale_date",
        "store_id"
    )

    .limit(50)
)

# COMMAND ----------

# DBTITLE 1,02. REGULARIZACIÓN TEMPORAL
# ============================================================
# 02. REGULARIZACIÓN TEMPORAL
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Construir un panel temporal completo:
#
#       1 fila = 1 tienda + 1 día
#
# Para ello:
#
# - generamos todas las fechas del periodo
# - generamos todas las tiendas
# - creamos todas las combinaciones posibles
# - hacemos LEFT JOIN con las ventas reales
# - rellenamos con 0 los días sin ventas
#
# IMPORTANTE
# ------------------------------------------------------------
#
# El número esperado de registros NO se hardcodea.
#
# Se calcula dinámicamente como:
#
#       nº tiendas x nº días del periodo
#
# Esto permite que el notebook siga funcionando cuando
# lleguen nuevos meses de datos.
#
# ============================================================


from pyspark.sql import functions as F


# ============================================================
# 1. RANGO DE FECHAS
# ============================================================

date_bounds = (

    daily_store_sales_df

    .agg(

        F.min("sale_date")
        .alias("min_date"),

        F.max("sale_date")
        .alias("max_date")
    )

    .first()
)


min_date = date_bounds["min_date"]
max_date = date_bounds["max_date"]


print("=" * 75)
print("REGULARIZACIÓN TEMPORAL")
print("=" * 75)

print(
    f"Periodo: {min_date} -> {max_date}"
)


# ============================================================
# 2. CALENDARIO COMPLETO
# ============================================================

calendar_df = (

    spark

    .range(1)

    .select(

        F.explode(

            F.sequence(

                F.lit(min_date),

                F.lit(max_date),

                F.expr("INTERVAL 1 DAY")
            )
        )

        .alias("sale_date")
    )
)


# ============================================================
# 3. LISTADO DE TIENDAS
# ============================================================

stores_df = (

    daily_store_sales_df

    .select(
        "store_id"
    )

    .distinct()
)


# ============================================================
# 4. TODAS LAS COMBINACIONES TIENDA + FECHA
# ============================================================

complete_store_calendar_df = (

    stores_df

    .crossJoin(
        calendar_df
    )
)


# ============================================================
# 5. LEFT JOIN CON VENTAS REALES
# ============================================================

regularized_daily_df = (

    complete_store_calendar_df
    .alias("calendar")

    .join(

        daily_store_sales_df
        .alias("sales"),

        on=[
            "sale_date",
            "store_id"
        ],

        how="left"
    )
)


# ============================================================
# 6. RELLENAMOS DÍAS SIN VENTAS
# ============================================================

regularized_daily_df = (

    regularized_daily_df

    .fillna({

        "net_sales":
            0.0,

        "gross_sales":
            0.0,

        "total_discount":
            0.0,

        "total_units":
            0,

        "total_tickets":
            0
    })
)


# ============================================================
# 7. VALIDACIÓN DE REGISTROS
# ============================================================

# Número real de registros generados
regularized_rows = (

    regularized_daily_df
    .count()
)


# ------------------------------------------------------------
# Número dinámico de tiendas
# ------------------------------------------------------------

num_stores = (

    stores_df
    .count()
)


# ------------------------------------------------------------
# Número dinámico de días
#
# datediff devuelve la diferencia entre fechas.
# Sumamos 1 porque incluimos tanto la fecha mínima
# como la fecha máxima.
# ------------------------------------------------------------

num_days = (

    spark

    .range(1)

    .select(

        (
            F.datediff(
                F.lit(max_date),
                F.lit(min_date)
            )
            +
            F.lit(1)
        )

        .alias("num_days")
    )

    .first()["num_days"]
)


# ------------------------------------------------------------
# Registros esperados
# ------------------------------------------------------------

expected_rows = (
    num_stores
    *
    num_days
)


# ============================================================
# 8. DÍAS SIN VENTAS INSERTADOS
# ============================================================

zero_sales_rows = (

    regularized_daily_df

    .filter(
        F.col("net_sales") == 0
    )

    .count()
)


# ============================================================
# 9. DUPLICADOS
# ============================================================

duplicate_store_dates = (

    regularized_daily_df

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
# 10. RESULTADOS
# ============================================================

print()

print(
    f"Número de tiendas:    {num_stores:,}"
)

print(
    f"Número de días:       {num_days:,}"
)

print(
    f"Registros finales:    {regularized_rows:,}"
)

print(
    f"Registros esperados:  {expected_rows:,}"
)

print(
    f"Días con ventas = 0:  {zero_sales_rows}"
)

print(
    f"Duplicados:            {duplicate_store_dates}"
)


# ============================================================
# 11. VALIDACIÓN FINAL
# ============================================================

if (
    regularized_rows == expected_rows
    and
    duplicate_store_dates == 0
):

    print()

    print(
        "OK - Panel temporal regularizado correctamente"
    )

else:

    raise RuntimeError(

        "La regularización temporal presenta inconsistencias. "
        f"Registros generados: {regularized_rows:,}. "
        f"Registros esperados: {expected_rows:,}. "
        f"Duplicados: {duplicate_store_dates:,}."
    )


# ============================================================
# 12. INSPECCIÓN DE LOS DÍAS SIN VENTAS
# ============================================================

print()

print("-" * 75)
print("COMBINACIONES TIENDA-DÍA SIN VENTAS")
print("-" * 75)

display(

    regularized_daily_df

    .filter(
        F.col("net_sales") == 0
    )

    .orderBy(
        "sale_date",
        "store_id"
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC # 03. FEATURE ENGINEERING
# MAGIC
# MAGIC ## Objetivo
# MAGIC
# MAGIC Construir las variables utilizadas por el modelo de Sales Forecasting a partir del panel temporal regularizado.
# MAGIC
# MAGIC Se generan tres grupos de features:
# MAGIC
# MAGIC ### Calendar features
# MAGIC
# MAGIC - `year`
# MAGIC - `month`
# MAGIC - `day`
# MAGIC - `day_of_week`
# MAGIC - `week_of_year`
# MAGIC - `is_weekend`
# MAGIC
# MAGIC ### Lag features
# MAGIC
# MAGIC - `lag_1`
# MAGIC - `lag_7`
# MAGIC - `lag_14`
# MAGIC - `lag_28`
# MAGIC
# MAGIC ### Rolling y trend features
# MAGIC
# MAGIC - `rolling_mean_7`
# MAGIC - `rolling_mean_28`
# MAGIC - `rolling_std_7`
# MAGIC - `lag1_minus_lag7`
# MAGIC - `lag1_vs_mean7`
# MAGIC
# MAGIC Todas las variables se calculan de forma independiente para cada tienda y respetando el orden temporal.
# MAGIC
# MAGIC Las ventanas rolling utilizan exclusivamente información anterior al día actual para evitar **Data Leakage**.
# MAGIC
# MAGIC La feature:
# MAGIC
# MAGIC `lag1_minus_lag7`
# MAGIC
# MAGIC representa:
# MAGIC
# MAGIC `lag_1 - lag_7`
# MAGIC
# MAGIC y sustituye definitivamente a la antigua denominación `diff_7`.

# COMMAND ----------

# DBTITLE 1,03.01 CALENDAR FEATURES Y LAGS
# ============================================================
# 03.01 CALENDAR FEATURES Y LAGS
# ============================================================

from pyspark.sql import functions as F
from pyspark.sql.window import Window


# ============================================================
# 1. ORDEN TEMPORAL POR TIENDA
# ============================================================

store_window = (

    Window

    .partitionBy(
        "store_id"
    )

    .orderBy(
        "sale_date"
    )
)


# ============================================================
# 2. CALENDAR FEATURES
# ============================================================

features_df = (

    regularized_daily_df

    .withColumn(
        "year",
        F.year("sale_date")
    )

    .withColumn(
        "month",
        F.month("sale_date")
    )

    .withColumn(
        "day",
        F.dayofmonth("sale_date")
    )

    .withColumn(
        "day_of_week",
        F.dayofweek("sale_date")
    )

    .withColumn(
        "week_of_year",
        F.weekofyear("sale_date")
    )

    .withColumn(
        "is_weekend",
        F.when(
            F.dayofweek("sale_date").isin(1, 7),
            1
        ).otherwise(0)
    )
)


# ============================================================
# 3. LAG FEATURES
# ============================================================

features_df = (

    features_df

    .withColumn(
        "lag_1",
        F.lag(
            "net_sales",
            1
        ).over(store_window)
    )

    .withColumn(
        "lag_7",
        F.lag(
            "net_sales",
            7
        ).over(store_window)
    )

    .withColumn(
        "lag_14",
        F.lag(
            "net_sales",
            14
        ).over(store_window)
    )

    .withColumn(
        "lag_28",
        F.lag(
            "net_sales",
            28
        ).over(store_window)
    )
)


# ============================================================
# 4. VALIDACIÓN
# ============================================================

print("=" * 75)
print("CALENDAR FEATURES Y LAGS")
print("=" * 75)

print(
    f"Registros: {features_df.count():,}"
)

print(
    f"Columnas:  {len(features_df.columns)}"
)

print()

print(
    "OK - Calendar features y lags generados"
)


display(

    features_df

    .select(
        "sale_date",
        "store_id",
        "net_sales",
        "year",
        "month",
        "day",
        "day_of_week",
        "week_of_year",
        "is_weekend",
        "lag_1",
        "lag_7",
        "lag_14",
        "lag_28"
    )

    .orderBy(
        "store_id",
        "sale_date"
    )

    .limit(50)
)

# COMMAND ----------

# DBTITLE 1,03.02 ROLLING FEATURES Y TENDENCIAS
# ============================================================
# 03.02 ROLLING FEATURES Y TENDENCIAS
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Construir estadísticas históricas por tienda utilizando
# únicamente información anterior al día actual.
#
# Esto evita Data Leakage.
#
# Features:
#
# - rolling_mean_7
# - rolling_mean_28
# - rolling_std_7
# - lag1_minus_lag7
# - lag1_vs_mean7
#
# ============================================================


from pyspark.sql import functions as F
from pyspark.sql.window import Window


# ============================================================
# 03.02.01 VENTANAS HISTÓRICAS
# ============================================================
#
# rowsBetween(-7, -1)
#
# significa:
#
# tomar las 7 filas anteriores
# sin incluir el día actual.
#
# rowsBetween(-28, -1)
#
# toma las 28 filas anteriores
# sin incluir el día actual.
#
# ============================================================

rolling_7_window = (

    Window

    .partitionBy(
        "store_id"
    )

    .orderBy(
        "sale_date"
    )

    .rowsBetween(
        -7,
        -1
    )
)


rolling_28_window = (

    Window

    .partitionBy(
        "store_id"
    )

    .orderBy(
        "sale_date"
    )

    .rowsBetween(
        -28,
        -1
    )
)


# ============================================================
# 03.02.02 ROLLING MEAN 7
# ============================================================

features_df = (

    features_df

    .withColumn(

        "rolling_mean_7",

        F.avg(
            "net_sales"
        ).over(
            rolling_7_window
        )

    )

)


# ============================================================
# 03.02.03 ROLLING MEAN 28
# ============================================================

features_df = (

    features_df

    .withColumn(

        "rolling_mean_28",

        F.avg(
            "net_sales"
        ).over(
            rolling_28_window
        )

    )

)


# ============================================================
# 03.02.04 ROLLING STANDARD DEVIATION 7
# ============================================================

features_df = (

    features_df

    .withColumn(

        "rolling_std_7",

        F.stddev_samp(
            "net_sales"
        ).over(
            rolling_7_window
        )

    )

)


# ============================================================
# 03.02.05 LAG 1 MENOS LAG 7
# ============================================================
#
# La antigua denominación "diff_7" era ambigua.
#
# La fórmula real siempre ha sido:
#
#       lag_1 - lag_7
#
# Por eso la feature se denomina:
#
#       lag1_minus_lag7
#
# ============================================================

features_df = (

    features_df

    .withColumn(

        "lag1_minus_lag7",

        F.col(
            "lag_1"
        )

        -

        F.col(
            "lag_7"
        )

    )

)


# ============================================================
# 03.02.06 RELACIÓN LAG 1 VS MEDIA 7 DÍAS
# ============================================================

features_df = (

    features_df

    .withColumn(

        "lag1_vs_mean7",

        F.when(

            F.col(
                "rolling_mean_7"
            ) != 0,

            F.col(
                "lag_1"
            )

            /

            F.col(
                "rolling_mean_7"
            )

        )

        .otherwise(
            F.lit(0.0)
        )

    )

)


# ============================================================
# 03.02.07 VALIDACIÓN
# ============================================================

required_rolling_features = [

    "rolling_mean_7",
    "rolling_mean_28",
    "rolling_std_7",
    "lag1_minus_lag7",
    "lag1_vs_mean7"

]


missing_rolling_features = [

    column

    for column in required_rolling_features

    if column not in features_df.columns

]


if missing_rolling_features:

    raise RuntimeError(

        "Faltan features rolling/trend esperadas: "
        f"{missing_rolling_features}"

    )


# ============================================================
# 03.02.08 RESULTADOS
# ============================================================

print("=" * 80)
print("ROLLING FEATURES Y TENDENCIAS")
print("=" * 80)


print(
    f"Registros: "
    f"{features_df.count():,}"
)


print(
    f"Columnas:  "
    f"{len(features_df.columns)}"
)


print()
print(
    "OK - rolling_mean_7 generada"
)


print(
    "OK - rolling_mean_28 generada"
)


print(
    "OK - rolling_std_7 generada"
)


print(
    "OK - lag1_minus_lag7 generada"
)


print(
    "OK - lag1_vs_mean7 generada"
)


print()
print(
    "Rolling features y tendencias generadas correctamente."
)


# ============================================================
# 03.02.09 INSPECCIÓN
# ============================================================

display(

    features_df

    .select(

        "sale_date",
        "store_id",
        "net_sales",

        "lag_1",
        "lag_7",
        "lag_14",
        "lag_28",

        "rolling_mean_7",
        "rolling_mean_28",
        "rolling_std_7",

        "lag1_minus_lag7",
        "lag1_vs_mean7"

    )

    .orderBy(
        "store_id",
        "sale_date"
    )

    .limit(50)

)

# COMMAND ----------

# DBTITLE 1,03.03 LIMPIEZA Y DATASET FINAL DE FEATURES
# ============================================================
# 03.03 LIMPIEZA Y DATASET FINAL DE FEATURES
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Eliminar las observaciones iniciales de cada tienda que no
# disponen todavía del histórico necesario para construir
# todas las features utilizadas por el modelo.
#
# El lag máximo utilizado es lag_28.
#
# Por tanto, las primeras 28 observaciones de cada tienda
# no pueden formar parte del dataset final de ML.
#
# ============================================================


from pyspark.sql import functions as F


# ============================================================
# 03.03.01 FEATURES NECESARIAS
# ============================================================

required_features = [

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
# 03.03.02 VALIDAMOS QUE TODAS LAS FEATURES EXISTEN
# ============================================================

missing_features = [

    feature_name

    for feature_name in required_features

    if feature_name not in features_df.columns

]


if missing_features:

    raise RuntimeError(

        "Faltan features necesarias para construir "
        f"el dataset final: {missing_features}"

    )


# ============================================================
# 03.03.03 ELIMINAMOS FILAS SIN HISTÓRICO SUFICIENTE
# ============================================================

ml_features_df = (

    features_df

    .dropna(
        subset=required_features
    )

)


# ============================================================
# 03.03.04 ORDEN Y SELECCIÓN FINAL DE COLUMNAS
# ============================================================

ml_features_df = (

    ml_features_df

    .select(

        # ----------------------------------------------------
        # IDENTIFICACIÓN
        # ----------------------------------------------------

        "sale_date",
        "store_id",


        # ----------------------------------------------------
        # MÉTRICAS DIARIAS
        # ----------------------------------------------------

        "net_sales",
        "gross_sales",
        "total_discount",
        "total_units",
        "total_tickets",


        # ----------------------------------------------------
        # CALENDAR FEATURES
        # ----------------------------------------------------

        "year",
        "month",
        "day",
        "day_of_week",
        "week_of_year",
        "is_weekend",


        # ----------------------------------------------------
        # LAG FEATURES
        # ----------------------------------------------------

        "lag_1",
        "lag_7",
        "lag_14",
        "lag_28",


        # ----------------------------------------------------
        # ROLLING FEATURES
        # ----------------------------------------------------

        "rolling_mean_7",
        "rolling_mean_28",
        "rolling_std_7",


        # ----------------------------------------------------
        # TREND FEATURES
        # ----------------------------------------------------

        "lag1_minus_lag7",
        "lag1_vs_mean7"

    )

    .orderBy(
        "sale_date",
        "store_id"
    )

)


# ============================================================
# 03.03.05 MÉTRICAS GENERALES
# ============================================================

rows_before_cleanup = (
    features_df.count()
)


final_rows = (
    ml_features_df.count()
)


removed_rows = (
    rows_before_cleanup
    -
    final_rows
)


final_stores = (

    ml_features_df

    .select(
        "store_id"
    )

    .distinct()

    .count()

)


# ============================================================
# 03.03.06 RANGO TEMPORAL FINAL
# ============================================================

final_date_range = (

    ml_features_df

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


# ============================================================
# 03.03.07 VALIDACIÓN DE DUPLICADOS
# ============================================================

duplicate_rows = (

    ml_features_df

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
# 03.03.08 VALIDACIÓN DE NULLS
# ============================================================

null_expressions = [

    F.sum(

        F.col(
            feature_name
        )
        .isNull()
        .cast("int")

    ).alias(
        feature_name
    )

    for feature_name in required_features

]


null_stats = (

    ml_features_df

    .agg(
        *null_expressions
    )

    .first()

)


total_feature_nulls = sum(

    int(
        null_stats[
            feature_name
        ] or 0
    )

    for feature_name in required_features

)


# ============================================================
# 03.03.09 VALIDACIÓN DE NAN
# ============================================================

numeric_feature_types = {

    field.name:
        field.dataType.simpleString()

    for field in ml_features_df.schema.fields

}


nan_expressions = [

    F.sum(

        F.isnan(
            F.col(
                feature_name
            )
        )
        .cast("int")

    ).alias(
        feature_name
    )

    for feature_name in required_features

    if numeric_feature_types[
        feature_name
    ] in (
        "double",
        "float"
    )

]


if nan_expressions:

    nan_stats = (

        ml_features_df

        .agg(
            *nan_expressions
        )

        .first()

    )


    total_feature_nans = sum(

        int(
            nan_stats[
                feature_name
            ] or 0
        )

        for feature_name in required_features

        if numeric_feature_types[
            feature_name
        ] in (
            "double",
            "float"
        )

    )

else:

    total_feature_nans = 0


# ============================================================
# 03.03.10 REGISTROS ESPERADOS
# ============================================================
#
# Cada tienda pierde exactamente las primeras 28 filas.
#
# ============================================================

expected_removed_rows = (
    EXPECTED_STORES
    *
    MAX_LAG_DAYS
)


expected_final_rows = (
    regularized_rows
    -
    expected_removed_rows
)


# ============================================================
# 03.03.11 RESULTADOS
# ============================================================

print("=" * 80)
print("DATASET FINAL DE FEATURES")
print("=" * 80)


print(
    f"Registros antes limpieza: "
    f"{rows_before_cleanup:,}"
)


print(
    f"Registros finales:        "
    f"{final_rows:,}"
)


print(
    f"Registros eliminados:     "
    f"{removed_rows:,}"
)


print(
    f"Eliminados esperados:     "
    f"{expected_removed_rows:,}"
)


print(
    f"Tiendas:                  "
    f"{final_stores}"
)


print(
    f"Periodo final:            "
    f"{final_date_range['min_date']} "
    f"-> "
    f"{final_date_range['max_date']}"
)


print(
    f"Duplicados:               "
    f"{duplicate_rows}"
)


print(
    f"NULLs en features:        "
    f"{total_feature_nulls}"
)


print(
    f"NaN en features:          "
    f"{total_feature_nans}"
)


# ============================================================
# 03.03.12 VALIDACIÓN FINAL
# ============================================================

if final_rows != expected_final_rows:

    raise RuntimeError(

        "El número final de registros "
        "no coincide con el esperado. "
        f"Esperado: {expected_final_rows:,} | "
        f"Actual: {final_rows:,}"

    )


if removed_rows != expected_removed_rows:

    raise RuntimeError(

        "El número de registros eliminados "
        "no coincide con las primeras 28 filas "
        "esperadas por tienda."

    )


if final_stores != EXPECTED_STORES:

    raise RuntimeError(

        "El número final de tiendas "
        "no coincide con el esperado."

    )


if duplicate_rows != 0:

    raise RuntimeError(

        "Existen duplicados sale_date + store_id "
        "en el dataset final de features."

    )


if total_feature_nulls != 0:

    raise RuntimeError(

        "Existen NULLs en las features "
        "del dataset final."

    )


if total_feature_nans != 0:

    raise RuntimeError(

        "Existen valores NaN en las features "
        "del dataset final."

    )


print()
print("=" * 80)
print("VALIDACIÓN FINAL")
print("=" * 80)


print(
    "OK - Primeras 28 observaciones eliminadas por tienda"
)


print(
    f"OK - Registros finales esperados: "
    f"{expected_final_rows:,}"
)


print(
    f"OK - {final_stores} tiendas"
)


print(
    "OK - Sin duplicados"
)


print(
    "OK - Sin NULLs en features"
)


print(
    "OK - Sin NaN en features"
)


print(
    "OK - lag1_minus_lag7 incluida"
)


print(
    "OK - Dataset compatible con Model Version 2"
)


print()
print(
    "Dataset final de features construido correctamente."
)


# ============================================================
# 03.03.13 INSPECCIÓN
# ============================================================

display(

    ml_features_df

    .orderBy(
        "sale_date",
        "store_id"
    )

    .limit(
        50
    )

)

# COMMAND ----------

# MAGIC %md
# MAGIC # 04. PERSISTENCIA DE LA FEATURE TABLE
# MAGIC
# MAGIC ## Objetivo
# MAGIC
# MAGIC Persistir el dataset generado por el proceso automatizable de Feature Engineering en:
# MAGIC
# MAGIC `retail_analytics.5_ml.daily_store_features`
# MAGIC
# MAGIC La tabla constituye la interfaz entre:
# MAGIC
# MAGIC **Data Engineering → Machine Learning**
# MAGIC
# MAGIC Cada ejecución reconstruirá las features a partir de la información disponible en Gold, garantizando que el posterior proceso de inference utiliza datos actualizados y reproducibles.

# COMMAND ----------

# DBTITLE 1,04.01 PERSISTENCIA DE LA FEATURE TABLE
# ============================================================
# 04.01 PERSISTENCIA DE LA FEATURE TABLE
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Persistir el dataset de features reconstruido desde Gold.
#
# En esta versión del pipeline utilizamos overwrite porque
# reconstruimos completamente la Feature Table.
#
# Después de persistirla, validaremos que la tabla Delta
# coincide estructuralmente con ml_features_df.
#
# ============================================================


from pyspark.sql import functions as F


# ============================================================
# 04.01.01 VALIDACIONES PREVIAS
# ============================================================

required_objects = [

    "ml_features_df",
    "FEATURE_TABLE",
    "final_rows",
    "EXPECTED_STORES"

]


missing_objects = [

    object_name

    for object_name in required_objects

    if object_name not in globals()

]


if missing_objects:

    raise RuntimeError(

        "Faltan objetos necesarios para persistir "
        f"la Feature Table: {missing_objects}"

    )


# ============================================================
# 04.01.02 INFORMACIÓN DEL DATASET ORIGEN
# ============================================================

source_rows = (
    ml_features_df.count()
)


source_stores = (

    ml_features_df

    .select(
        "store_id"
    )

    .distinct()

    .count()

)


source_dates = (

    ml_features_df

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


source_columns = (
    ml_features_df.columns
)


# ============================================================
# 04.01.03 ESCRITURA EN DELTA
# ============================================================

(

    ml_features_df

    .write

    .format(
        "delta"
    )

    .mode(
        "overwrite"
    )

    .option(
        "overwriteSchema",
        "true"
    )

    .saveAsTable(
        FEATURE_TABLE
    )

)


print(
    f"OK - Feature Table actualizada: "
    f"{FEATURE_TABLE}"
)


# ============================================================
# 04.01.04 RECUPERAMOS LA TABLA DESDE DELTA
# ============================================================

persisted_features_df = (

    spark.table(
        FEATURE_TABLE
    )

)


# ============================================================
# 04.01.05 VALIDACIONES GENERALES
# ============================================================

persisted_rows = (
    persisted_features_df.count()
)


persisted_stores = (

    persisted_features_df

    .select(
        "store_id"
    )

    .distinct()

    .count()

)


persisted_dates = (

    persisted_features_df

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


persisted_columns = (
    persisted_features_df.columns
)


# ============================================================
# 04.01.06 DUPLICADOS
# ============================================================

duplicate_rows = (

    persisted_features_df

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
# 04.01.07 COMPARACIÓN DE SCHEMA / COLUMNAS
# ============================================================

missing_persisted_columns = [

    column

    for column in source_columns

    if column not in persisted_columns

]


extra_persisted_columns = [

    column

    for column in persisted_columns

    if column not in source_columns

]


same_column_order = (

    source_columns
    ==
    persisted_columns

)


# ============================================================
# 04.01.08 VALIDACIÓN DE FEATURES CLAVE
# ============================================================

required_model_features = [

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


missing_model_features = [

    feature_name

    for feature_name in required_model_features

    if feature_name not in persisted_columns

]


# ============================================================
# 04.01.09 RESULTADOS
# ============================================================

print()
print("=" * 80)
print("VALIDACIÓN FEATURE TABLE")
print("=" * 80)


print(
    f"Tabla:            "
    f"{FEATURE_TABLE}"
)


print(
    f"Registros origen: "
    f"{source_rows:,}"
)


print(
    f"Registros Delta:  "
    f"{persisted_rows:,}"
)


print(
    f"Tiendas origen:   "
    f"{source_stores}"
)


print(
    f"Tiendas Delta:    "
    f"{persisted_stores}"
)


print(
    f"Periodo origen:   "
    f"{source_dates['min_date']} "
    f"-> "
    f"{source_dates['max_date']}"
)


print(
    f"Periodo Delta:    "
    f"{persisted_dates['min_date']} "
    f"-> "
    f"{persisted_dates['max_date']}"
)


print(
    f"Columnas origen:  "
    f"{len(source_columns)}"
)


print(
    f"Columnas Delta:   "
    f"{len(persisted_columns)}"
)


print(
    f"Duplicados:       "
    f"{duplicate_rows}"
)


# ============================================================
# 04.01.10 VALIDACIONES ESTRICTAS
# ============================================================

if source_rows != final_rows:

    raise RuntimeError(

        "El número de registros de ml_features_df "
        "no coincide con final_rows."

    )


if persisted_rows != source_rows:

    raise RuntimeError(

        "El número de registros persistidos "
        "no coincide con el dataset origen."

    )


if persisted_stores != EXPECTED_STORES:

    raise RuntimeError(

        "El número de tiendas persistido "
        "no coincide con el esperado."

    )


if source_stores != persisted_stores:

    raise RuntimeError(

        "La cobertura de tiendas cambió "
        "durante la persistencia."

    )


if (
    source_dates["min_date"]
    !=
    persisted_dates["min_date"]
):

    raise RuntimeError(

        "La fecha mínima persistida "
        "no coincide con el dataset origen."

    )


if (
    source_dates["max_date"]
    !=
    persisted_dates["max_date"]
):

    raise RuntimeError(

        "La fecha máxima persistida "
        "no coincide con el dataset origen."

    )


if duplicate_rows != 0:

    raise RuntimeError(

        "Se han detectado duplicados "
        "sale_date + store_id en la Feature Table."

    )


if missing_persisted_columns:

    raise RuntimeError(

        "Faltan columnas tras la persistencia: "
        f"{missing_persisted_columns}"

    )


if extra_persisted_columns:

    raise RuntimeError(

        "Han aparecido columnas inesperadas "
        f"tras la persistencia: {extra_persisted_columns}"

    )


if not same_column_order:

    raise RuntimeError(

        "El orden de columnas de la tabla persistida "
        "no coincide con ml_features_df."

    )


if missing_model_features:

    raise RuntimeError(

        "Faltan features necesarias para Model Version 2: "
        f"{missing_model_features}"

    )


# ============================================================
# 04.01.11 VALIDACIÓN FINAL
# ============================================================

print()
print("=" * 80)
print("VALIDACIÓN FINAL")
print("=" * 80)


print(
    "OK - Feature Table persistida"
)


print(
    f"OK - {persisted_rows:,} registros"
)


print(
    f"OK - {persisted_stores} tiendas"
)


print(
    "OK - Periodo temporal preservado"
)


print(
    "OK - Sin duplicados"
)


print(
    "OK - Schema preservado"
)


print(
    "OK - Orden de columnas preservado"
)


print(
    "OK - lag1_minus_lag7 persistida"
)


print(
    "OK - Features compatibles con Model Version 2"
)


print()
print(
    "Feature Table persistida y validada correctamente."
)

# COMMAND ----------

# MAGIC %md
# MAGIC # 05. VALIDACIONES DE CALIDAD DE LA FEATURE TABLE
# MAGIC
# MAGIC ## Objetivo
# MAGIC
# MAGIC Validar que la Feature Table generada está preparada para ser consumida por el proceso de inference.
# MAGIC
# MAGIC Se comprobará:
# MAGIC
# MAGIC - cobertura de las 31 tiendas
# MAGIC - ausencia de duplicados
# MAGIC - ausencia de NULLs en las features del modelo
# MAGIC - ausencia de valores infinitos
# MAGIC - coherencia temporal con Gold
# MAGIC - existencia de histórico suficiente por tienda
# MAGIC
# MAGIC La comprobación temporal es especialmente importante para la automatización:
# MAGIC
# MAGIC **MAX(fecha Gold) = MAX(fecha Feature Table)**
# MAGIC
# MAGIC Solo cuando ambas fechas coincidan se considerará que las features están actualizadas y el proceso de inference podrá ejecutarse.

# COMMAND ----------

# DBTITLE 1,05.01 QUALITY GATES DE LA FEATURE TABLE
# ============================================================
# 05.01 QUALITY GATES DE LA FEATURE TABLE
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Validar que la Feature Table está preparada para ser
# consumida por el proceso de inference.
#
# Quality Gates:
#
# - alineación temporal con Gold
# - cobertura completa de tiendas
# - ausencia de duplicados
# - ausencia de NULLs
# - ausencia de NaN
# - histórico suficiente por tienda
# - disponibilidad de todas las features de Model Version 2
#
# ============================================================


from pyspark.sql import functions as F


# ============================================================
# 05.01.01 VALIDACIONES PREVIAS
# ============================================================

required_objects = [

    "sales_with_date_df",
    "persisted_features_df",
    "EXPECTED_STORES",
    "MAX_LAG_DAYS"

]


missing_objects = [

    object_name

    for object_name in required_objects

    if object_name not in globals()

]


if missing_objects:

    raise RuntimeError(

        "Faltan objetos necesarios para ejecutar "
        f"los Quality Gates: {missing_objects}"

    )


# ============================================================
# 05.01.02 FEATURES DEL MODELO
# ============================================================

model_features = [

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


missing_model_features = [

    feature_name

    for feature_name in model_features

    if feature_name not in persisted_features_df.columns

]


if missing_model_features:

    raise RuntimeError(

        "Faltan features necesarias para Model Version 2: "
        f"{missing_model_features}"

    )


# ============================================================
# 05.01.03 FECHA MÁXIMA DE GOLD
# ============================================================

gold_max_date = (

    sales_with_date_df

    .agg(

        F.max(
            "sale_date"
        ).alias(
            "max_date"
        )

    )

    .first()[
        "max_date"
    ]

)


# ============================================================
# 05.01.04 FECHA MÁXIMA DE FEATURE TABLE
# ============================================================

feature_max_date = (

    persisted_features_df

    .agg(

        F.max(
            "sale_date"
        ).alias(
            "max_date"
        )

    )

    .first()[
        "max_date"
    ]

)


# ============================================================
# 05.01.05 COBERTURA DE TIENDAS
# ============================================================

feature_stores = (

    persisted_features_df

    .select(
        "store_id"
    )

    .distinct()

    .count()

)


# ============================================================
# 05.01.06 DUPLICADOS
# ============================================================

feature_duplicates = (

    persisted_features_df

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
# 05.01.07 NULLS EN FEATURES
# ============================================================

null_expressions = [

    F.sum(

        F.col(
            feature_name
        )
        .isNull()
        .cast("int")

    ).alias(
        feature_name
    )

    for feature_name in model_features

]


null_stats = (

    persisted_features_df

    .agg(
        *null_expressions
    )

    .first()

)


feature_nulls = sum(

    int(
        null_stats[
            feature_name
        ] or 0
    )

    for feature_name in model_features

)


# ============================================================
# 05.01.08 NAN EN FEATURES
# ============================================================

feature_types = {

    field.name:
        field.dataType.simpleString()

    for field in persisted_features_df.schema.fields

}


nan_feature_names = [

    feature_name

    for feature_name in model_features

    if feature_types[
        feature_name
    ] in (
        "double",
        "float"
    )

]


if nan_feature_names:

    nan_expressions = [

        F.sum(

            F.isnan(
                F.col(
                    feature_name
                )
            )
            .cast("int")

        ).alias(
            feature_name
        )

        for feature_name in nan_feature_names

    ]


    nan_stats = (

        persisted_features_df

        .agg(
            *nan_expressions
        )

        .first()

    )


    feature_nans = sum(

        int(
            nan_stats[
                feature_name
            ] or 0
        )

        for feature_name in nan_feature_names

    )

else:

    feature_nans = 0


# ============================================================
# 05.01.09 HISTÓRICO POR TIENDA
# ============================================================

history_by_store_df = (

    persisted_features_df

    .groupBy(
        "store_id"
    )

    .count()

)


history_stats = (

    history_by_store_df

    .agg(

        F.min(
            "count"
        ).alias(
            "min_history"
        ),

        F.max(
            "count"
        ).alias(
            "max_history"
        )

    )

    .first()

)


minimum_history = int(
    history_stats[
        "min_history"
    ]
)


maximum_history = int(
    history_stats[
        "max_history"
    ]
)


# ============================================================
# 05.01.10 REGISTROS Y PERIODO
# ============================================================

feature_rows = (
    persisted_features_df.count()
)


feature_min_date = (

    persisted_features_df

    .agg(

        F.min(
            "sale_date"
        ).alias(
            "min_date"
        )

    )

    .first()[
        "min_date"
    ]

)


# ============================================================
# 05.01.11 RESULTADOS
# ============================================================

print("=" * 80)
print("QUALITY GATES - FEATURE TABLE")
print("=" * 80)


print(
    f"Registros:                  "
    f"{feature_rows:,}"
)


print(
    f"Periodo Feature Table:      "
    f"{feature_min_date} "
    f"-> "
    f"{feature_max_date}"
)


print(
    f"Última fecha Gold:          "
    f"{gold_max_date}"
)


print(
    f"Última fecha Feature Table: "
    f"{feature_max_date}"
)


print(
    f"Tiendas:                    "
    f"{feature_stores}"
)


print(
    f"Duplicados:                 "
    f"{feature_duplicates}"
)


print(
    f"NULLs en features:          "
    f"{feature_nulls}"
)


print(
    f"NaN en features:            "
    f"{feature_nans}"
)


print(
    f"Mínimo histórico/tienda:    "
    f"{minimum_history:,}"
)


print(
    f"Máximo histórico/tienda:    "
    f"{maximum_history:,}"
)


# ============================================================
# 05.01.12 QUALITY GATES ESTRICTOS
# ============================================================

if gold_max_date != feature_max_date:

    raise RuntimeError(

        "QUALITY GATE FAILED - "
        "Feature Table desactualizada respecto a Gold. "
        f"Gold: {gold_max_date} | "
        f"Feature Table: {feature_max_date}"

    )


if feature_stores != EXPECTED_STORES:

    raise RuntimeError(

        "QUALITY GATE FAILED - "
        "Cobertura de tiendas incorrecta. "
        f"Esperadas: {EXPECTED_STORES} | "
        f"Encontradas: {feature_stores}"

    )


if feature_duplicates != 0:

    raise RuntimeError(

        "QUALITY GATE FAILED - "
        "Existen duplicados sale_date + store_id."

    )


if feature_nulls != 0:

    raise RuntimeError(

        "QUALITY GATE FAILED - "
        "Existen NULLs en las features del modelo."

    )


if feature_nans != 0:

    raise RuntimeError(

        "QUALITY GATE FAILED - "
        "Existen NaN en las features del modelo."

    )


if minimum_history < MAX_LAG_DAYS:

    raise RuntimeError(

        "QUALITY GATE FAILED - "
        "Existe alguna tienda sin histórico suficiente. "
        f"Mínimo: {minimum_history} | "
        f"Necesario: {MAX_LAG_DAYS}"

    )


# ============================================================
# 05.01.13 VALIDACIÓN FINAL
# ============================================================

print()
print("=" * 80)
print("VALIDACIÓN FINAL")
print("=" * 80)


print(
    "OK - Gold y Feature Table alineados temporalmente"
)


print(
    f"OK - {feature_stores} tiendas"
)


print(
    "OK - Sin duplicados"
)


print(
    "OK - Sin NULLs en features"
)


print(
    "OK - Sin NaN en features"
)


print(
    "OK - Histórico suficiente por tienda"
)


print(
    "OK - lag1_minus_lag7 disponible"
)


print(
    "OK - Features compatibles con Model Version 2"
)


print()
print(
    "OK - FEATURE TABLE READY FOR INFERENCE"
)

# COMMAND ----------

# MAGIC %md
# MAGIC # 06. PREPARACIÓN PARA LA ORQUESTACIÓN CON LAKEFLOW JOBS
# MAGIC
# MAGIC ## Objetivo
# MAGIC
# MAGIC Integrar el proceso de Feature Engineering dentro del flujo automatizado del proyecto.
# MAGIC
# MAGIC Este notebook funciona como un proceso independiente cuya responsabilidad es reconstruir, validar y persistir las features utilizadas posteriormente por el proceso de Machine Learning.
# MAGIC
# MAGIC ## Responsabilidad del proceso
# MAGIC
# MAGIC El flujo ejecutado es:
# MAGIC
# MAGIC **Gold**
# MAGIC
# MAGIC ↓
# MAGIC
# MAGIC **Agregación diaria por tienda**
# MAGIC
# MAGIC ↓
# MAGIC
# MAGIC **Regularización temporal**
# MAGIC
# MAGIC ↓
# MAGIC
# MAGIC **Calendar Features**
# MAGIC
# MAGIC ↓
# MAGIC
# MAGIC **Lag Features**
# MAGIC
# MAGIC ↓
# MAGIC
# MAGIC **Rolling Features**
# MAGIC
# MAGIC ↓
# MAGIC
# MAGIC **Trend Features**
# MAGIC
# MAGIC ↓
# MAGIC
# MAGIC **Delta Feature Table**
# MAGIC
# MAGIC ↓
# MAGIC
# MAGIC **Quality Gates**
# MAGIC
# MAGIC La tabla resultante es:
# MAGIC
# MAGIC `retail_analytics.5_ml.daily_store_features`
# MAGIC
# MAGIC El proceso reconstruye completamente la Feature Table utilizando la información disponible en Gold.
# MAGIC
# MAGIC ## Features del modelo
# MAGIC
# MAGIC La Feature Table contiene las features utilizadas por la Model Version 2:
# MAGIC
# MAGIC **Calendar Features**
# MAGIC
# MAGIC - `year`
# MAGIC - `month`
# MAGIC - `day`
# MAGIC - `day_of_week`
# MAGIC - `week_of_year`
# MAGIC - `is_weekend`
# MAGIC
# MAGIC **Lag Features**
# MAGIC
# MAGIC - `lag_1`
# MAGIC - `lag_7`
# MAGIC - `lag_14`
# MAGIC - `lag_28`
# MAGIC
# MAGIC **Rolling Features**
# MAGIC
# MAGIC - `rolling_mean_7`
# MAGIC - `rolling_mean_28`
# MAGIC - `rolling_std_7`
# MAGIC
# MAGIC **Trend Features**
# MAGIC
# MAGIC - `lag1_minus_lag7`
# MAGIC - `lag1_vs_mean7`
# MAGIC
# MAGIC Todas las features temporales se construyen utilizando exclusivamente información anterior al día actual para evitar Data Leakage.
# MAGIC
# MAGIC ## Quality Gates
# MAGIC
# MAGIC Antes de considerar válida una ejecución se comprueba:
# MAGIC
# MAGIC - cobertura completa de las tiendas disponibles en `dim_store`
# MAGIC - ausencia de duplicados por `sale_date + store_id`
# MAGIC - ausencia de NULLs en las features utilizadas por el modelo
# MAGIC - ausencia de NaN en las features utilizadas por el modelo
# MAGIC - histórico suficiente por tienda
# MAGIC - disponibilidad de todas las features requeridas por Model Version 2
# MAGIC - alineación temporal entre Gold y Feature Table
# MAGIC
# MAGIC La condición temporal principal es:
# MAGIC
# MAGIC `MAX(fecha Gold) = MAX(fecha Feature Table)`
# MAGIC
# MAGIC Si cualquiera de los Quality Gates falla, el notebook genera una excepción y la ejecución de Lakeflow Jobs debe finalizar como **FAILED**.
# MAGIC
# MAGIC Esto evita que el proceso posterior de inference utilice features incompletas, inconsistentes o desactualizadas.
# MAGIC
# MAGIC ## Integración en Lakeflow Jobs
# MAGIC
# MAGIC El notebook se utilizará como una task:
# MAGIC
# MAGIC `ml_feature_refresh`
# MAGIC
# MAGIC La dependencia será:
# MAGIC
# MAGIC **gold_modeling**
# MAGIC
# MAGIC ↓
# MAGIC
# MAGIC **ml_feature_refresh**
# MAGIC
# MAGIC ↓
# MAGIC
# MAGIC **sales_forecast_inference**
# MAGIC
# MAGIC Por tanto, el proceso de inference únicamente podrá ejecutarse cuando:
# MAGIC
# MAGIC 1. Gold se haya actualizado correctamente.
# MAGIC 2. La Feature Table se haya reconstruido.
# MAGIC 3. Todos los Quality Gates hayan finalizado correctamente.
# MAGIC
# MAGIC ## Estado
# MAGIC
# MAGIC El proceso de Feature Engineering queda preparado para su ejecución independiente y automatizada mediante Lakeflow Jobs.
# MAGIC
# MAGIC La Feature Table queda validada como interfaz entre:
# MAGIC
# MAGIC **Data Engineering → Machine Learning**
# MAGIC
# MAGIC y preparada para ser consumida por el proceso de inference utilizando la **Model Version 2**.
