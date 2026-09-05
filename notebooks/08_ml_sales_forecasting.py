# Databricks notebook source
# MAGIC %md
# MAGIC # 00. CONFIGURACIÓN Y DEFINICIÓN DEL PROBLEMA DE MACHINE LEARNING
# MAGIC
# MAGIC ## Objetivo
# MAGIC
# MAGIC En esta fase ampliamos el proyecto **Retail Analytics** incorporando una capa de **Machine Learning** sobre el Lakehouse construido previamente.
# MAGIC
# MAGIC Hasta este punto, el proyecto dispone de un pipeline de Data Engineering completo:
# MAGIC
# MAGIC **Landing → Bronze → Silver → Gold → SQL Analytics**
# MAGIC
# MAGIC Los datos se encuentran limpios, validados, modelados y disponibles en la capa Gold.
# MAGIC
# MAGIC El siguiente objetivo será utilizar este histórico para construir un modelo capaz de realizar **forecasting de ventas**.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Problema de negocio
# MAGIC
# MAGIC En un escenario Retail, disponer de una estimación de las ventas futuras puede ayudar a mejorar decisiones relacionadas con:
# MAGIC
# MAGIC - planificación de inventario
# MAGIC - reposición de productos
# MAGIC - planificación de demanda
# MAGIC - asignación de recursos
# MAGIC - análisis de tendencias
# MAGIC - detección anticipada de cambios en la demanda
# MAGIC
# MAGIC El objetivo será desarrollar un modelo que utilice el histórico disponible para **predecir ventas futuras**.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Fuente de datos
# MAGIC
# MAGIC El punto de partida será el modelo dimensional de la capa Gold:
# MAGIC
# MAGIC - `fact_sales`
# MAGIC - `dim_date`
# MAGIC - `dim_product`
# MAGIC - `dim_store`
# MAGIC - `dim_customer`
# MAGIC
# MAGIC La tabla principal será:
# MAGIC
# MAGIC `retail_analytics.3_gold.fact_sales`
# MAGIC
# MAGIC A partir de estas tablas construiremos un **dataset específico para Machine Learning**, evitando entrenar directamente sobre las tablas operacionales del modelo Gold.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Granularidad del forecasting
# MAGIC
# MAGIC Antes de entrenar ningún modelo debemos determinar cuál es la granularidad adecuada de predicción.
# MAGIC
# MAGIC Analizaremos diferentes posibilidades:
# MAGIC
# MAGIC **Ventas globales**
# MAGIC
# MAGIC `fecha → ventas`
# MAGIC
# MAGIC **Ventas por tienda**
# MAGIC
# MAGIC `fecha + store_id → ventas`
# MAGIC
# MAGIC **Ventas por producto**
# MAGIC
# MAGIC `fecha + product_id → ventas`
# MAGIC
# MAGIC **Ventas por tienda y producto**
# MAGIC
# MAGIC `fecha + store_id + product_id → ventas`
# MAGIC
# MAGIC La granularidad final se decidirá después de analizar la densidad y continuidad temporal de los datos.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Variable objetivo
# MAGIC
# MAGIC La principal candidata como variable objetivo será:
# MAGIC
# MAGIC `net_sales`
# MAGIC
# MAGIC calculada mediante la agregación de:
# MAGIC
# MAGIC `SUM(net_amount)`
# MAGIC
# MAGIC para cada unidad temporal definida en el dataset.
# MAGIC
# MAGIC También podremos analizar como variables complementarias:
# MAGIC
# MAGIC - unidades vendidas
# MAGIC - número de tickets
# MAGIC - gross sales
# MAGIC - descuentos
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Estrategia de Machine Learning
# MAGIC
# MAGIC El desarrollo seguirá las siguientes etapas:
# MAGIC
# MAGIC **Gold Data**
# MAGIC
# MAGIC ↓
# MAGIC
# MAGIC **Exploratory Data Analysis**
# MAGIC
# MAGIC ↓
# MAGIC
# MAGIC **Feature Engineering**
# MAGIC
# MAGIC ↓
# MAGIC
# MAGIC **ML Dataset**
# MAGIC
# MAGIC ↓
# MAGIC
# MAGIC **Temporal Train / Validation / Test**
# MAGIC
# MAGIC ↓
# MAGIC
# MAGIC **Baseline Model**
# MAGIC
# MAGIC ↓
# MAGIC
# MAGIC **Machine Learning Models**
# MAGIC
# MAGIC ↓
# MAGIC
# MAGIC **MLflow Experiment Tracking**
# MAGIC
# MAGIC ↓
# MAGIC
# MAGIC **Model Evaluation**
# MAGIC
# MAGIC ↓
# MAGIC
# MAGIC **Model Registration**
# MAGIC
# MAGIC ↓
# MAGIC
# MAGIC **Sales Forecast**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Principio importante
# MAGIC
# MAGIC Al tratarse de un problema temporal, los datos **NO se dividirán aleatoriamente**.
# MAGIC
# MAGIC El entrenamiento y evaluación respetarán siempre el orden cronológico:
# MAGIC
# MAGIC **PASADO → PRESENTE → FUTURO**
# MAGIC
# MAGIC Esto permitirá simular correctamente cómo funcionaría el modelo en un escenario real y evitará **Data Leakage**.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Resultado esperado
# MAGIC
# MAGIC Al finalizar esta fase tendremos:
# MAGIC
# MAGIC - dataset preparado para Machine Learning
# MAGIC - features temporales y de negocio
# MAGIC - baseline de forecasting
# MAGIC - varios modelos comparados
# MAGIC - experimentos registrados con MLflow
# MAGIC - modelo seleccionado y registrado
# MAGIC - predicciones de ventas futuras
# MAGIC - tabla Delta con los resultados del forecast
# MAGIC - integración de las predicciones con la capa analítica

# COMMAND ----------

# DBTITLE 1,01. ANÁLISIS DEL DATASET PARA FORECASTING
# ============================================================
# 01. ANÁLISIS DEL DATASET PARA FORECASTING
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Analizar la tabla de hechos Gold para determinar:
#
# - volumen disponible
# - periodo temporal cubierto
# - número de días con información
# - número de tiendas
# - número de productos
# - densidad de las combinaciones tienda-producto
#
# Con este análisis decidiremos posteriormente cuál será
# la granularidad más adecuada para nuestro modelo:
#
#   1. ventas globales por día
#   2. ventas por tienda y día
#   3. ventas por producto y día
#   4. ventas por tienda + producto + día
#
# IMPORTANTE:
#
# Todavía NO construiremos el dataset de Machine Learning.
# Primero analizamos los datos disponibles y después
# tomamos una decisión basada en evidencia.
# ============================================================


from pyspark.sql import functions as F


# ------------------------------------------------------------
# 01.01 LECTURA DE LAS TABLAS GOLD
# ------------------------------------------------------------
#
# fact_sales contiene nuestras transacciones.
#
# dim_date nos permitirá trabajar con información temporal.
#
# dim_product y dim_store se utilizarán posteriormente para
# incorporar atributos adicionales al modelo.
# ------------------------------------------------------------

fact_sales_df = spark.table(
    "retail_analytics.3_gold.fact_sales"
)

dim_date_df = spark.table(
    "retail_analytics.3_gold.dim_date"
)

dim_product_df = spark.table(
    "retail_analytics.3_gold.dim_product"
)

dim_store_df = spark.table(
    "retail_analytics.3_gold.dim_store"
)


# ------------------------------------------------------------
# 01.02 INFORMACIÓN GENERAL DEL DATASET
# ------------------------------------------------------------

total_rows = fact_sales_df.count()

total_tickets = (
    fact_sales_df
    .select("sale_id")
    .distinct()
    .count()
)

total_stores = (
    fact_sales_df
    .select("store_id")
    .distinct()
    .count()
)

total_products = (
    fact_sales_df
    .select("product_id")
    .distinct()
    .count()
)


print("=" * 80)
print("RESUMEN GENERAL DEL DATASET")
print("=" * 80)

print(
    f"Líneas de venta:      {total_rows:,}"
)

print(
    f"Tickets:              {total_tickets:,}"
)

print(
    f"Tiendas con ventas:    {total_stores:,}"
)

print(
    f"Productos con ventas:  {total_products:,}"
)


# ------------------------------------------------------------
# 01.03 PERIODO TEMPORAL DISPONIBLE
# ------------------------------------------------------------
#
# fact_sales contiene date_key.
#
# Recuperamos la fecha real desde dim_date para poder analizar
# correctamente la cobertura temporal.
# ------------------------------------------------------------

sales_with_date_df = (

    fact_sales_df.alias("f")

    .join(

        dim_date_df
        .select(
            "date_key",
            "date"
        )
        .alias("d"),

        F.col("f.date_key")
        ==
        F.col("d.date_key"),

        "inner"
    )

    .select(
        F.col("f.*"),
        F.col("d.date").alias("sale_date")
    )
)


date_stats = (

    sales_with_date_df

    .agg(

        F.min("sale_date")
        .alias("min_date"),

        F.max("sale_date")
        .alias("max_date"),

        F.countDistinct("sale_date")
        .alias("days_with_sales")
    )

    .first()
)


print("\n" + "=" * 80)
print("COBERTURA TEMPORAL")
print("=" * 80)

print(
    "Primera fecha:",
    date_stats["min_date"]
)

print(
    "Última fecha:",
    date_stats["max_date"]
)

print(
    "Días con ventas:",
    date_stats["days_with_sales"]
)


# ------------------------------------------------------------
# 01.04 VOLUMEN DE VENTAS POR AÑO
# ------------------------------------------------------------
#
# Nos interesa comprobar cuánto histórico tenemos disponible
# y cómo se distribuyen las observaciones entre años.
# ------------------------------------------------------------

sales_by_year_df = (

    sales_with_date_df

    .groupBy(
        F.year("sale_date")
        .alias("year")
    )

    .agg(

        F.count("*")
        .alias("sales_lines"),

        F.countDistinct("sale_id")
        .alias("tickets"),

        F.round(
            F.sum("net_amount"),
            2
        ).alias("net_sales")
    )

    .orderBy("year")
)


display(
    sales_by_year_df
)


# ------------------------------------------------------------
# 01.05 DENSIDAD TEMPORAL POR TIENDA
# ------------------------------------------------------------
#
# Queremos saber cuántos días diferentes tiene información
# cada tienda.
#
# Si prácticamente todas las tiendas tienen ventas todos los
# días, forecasting por tienda será una opción interesante.
# ------------------------------------------------------------

store_density_df = (

    sales_with_date_df

    .groupBy(
        "store_id"
    )

    .agg(

        F.countDistinct("sale_date")
        .alias("days_with_sales"),

        F.count("*")
        .alias("sales_lines"),

        F.round(
            F.sum("net_amount"),
            2
        ).alias("net_sales")
    )
)


display(

    store_density_df

    .orderBy(
        F.desc("days_with_sales")
    )
)


# ------------------------------------------------------------
# 01.06 DENSIDAD TEMPORAL POR PRODUCTO
# ------------------------------------------------------------
#
# Realizamos el mismo análisis para los productos.
#
# Un producto con pocas fechas de venta genera una serie
# temporal muy dispersa y será más difícil de modelar.
# ------------------------------------------------------------

product_density_df = (

    sales_with_date_df

    .groupBy(
        "product_id"
    )

    .agg(

        F.countDistinct("sale_date")
        .alias("days_with_sales"),

        F.count("*")
        .alias("sales_lines"),

        F.round(
            F.sum("net_amount"),
            2
        ).alias("net_sales")
    )
)


display(

    product_density_df

    .orderBy(
        F.desc("days_with_sales")
    )
)


# ------------------------------------------------------------
# 01.07 DENSIDAD TIENDA + PRODUCTO
# ------------------------------------------------------------
#
# Esta sería nuestra granularidad más detallada:
#
#     sale_date + store_id + product_id
#
# Pero antes debemos comprobar si existe suficiente histórico
# para cada combinación.
#
# Un gran número de combinaciones con muy pocos días de venta
# haría esta granularidad poco adecuada para nuestro primer
# modelo de forecasting.
# ------------------------------------------------------------

store_product_density_df = (

    sales_with_date_df

    .groupBy(
        "store_id",
        "product_id"
    )

    .agg(

        F.countDistinct("sale_date")
        .alias("days_with_sales"),

        F.count("*")
        .alias("sales_lines"),

        F.round(
            F.sum("net_amount"),
            2
        ).alias("net_sales")
    )
)


# ------------------------------------------------------------
# RESUMEN ESTADÍSTICO
# ------------------------------------------------------------
#
# Calculamos percentiles de días con ventas.
#
# Esto es mucho más informativo que mirar únicamente
# la media porque nos permite conocer la distribución real
# de la densidad de las series.
# ------------------------------------------------------------

store_product_density_summary_df = (

    store_product_density_df

    .select(

        F.count("*")
        .alias("store_product_combinations"),

        F.round(
            F.avg("days_with_sales"),
            2
        ).alias("avg_days_with_sales"),

        F.min("days_with_sales")
        .alias("min_days_with_sales"),

        F.expr(
            "percentile_approx(days_with_sales, 0.25)"
        ).alias("p25_days"),

        F.expr(
            "percentile_approx(days_with_sales, 0.50)"
        ).alias("median_days"),

        F.expr(
            "percentile_approx(days_with_sales, 0.75)"
        ).alias("p75_days"),

        F.max("days_with_sales")
        .alias("max_days_with_sales")
    )
)


display(
    store_product_density_summary_df
)


# ------------------------------------------------------------
# 01.08 DISTRIBUCIÓN DE LA DENSIDAD
# ------------------------------------------------------------
#
# Clasificamos las combinaciones tienda-producto según
# el número de días en los que han registrado ventas.
#
# Esto nos permitirá detectar rápidamente si las series son
# suficientemente densas para forecasting.
# ------------------------------------------------------------

density_distribution_df = (

    store_product_density_df

    .withColumn(

        "density_group",

        F.when(
            F.col("days_with_sales") < 30,
            "< 30 días"
        )

        .when(
            F.col("days_with_sales") < 90,
            "30-89 días"
        )

        .when(
            F.col("days_with_sales") < 180,
            "90-179 días"
        )

        .when(
            F.col("days_with_sales") < 365,
            "180-364 días"
        )

        .otherwise(
            ">= 365 días"
        )
    )

    .groupBy(
        "density_group"
    )

    .count()

    .orderBy(
        "density_group"
    )
)


display(
    density_distribution_df
)

# COMMAND ----------

# DBTITLE 1,02. CONSTRUCCIÓN DEL DATASET DIARIO POR TIENDA
# ============================================================
# 02. CONSTRUCCIÓN DEL DATASET DIARIO POR TIENDA
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Construir el dataset base para Machine Learning con la
# granularidad:
#
#       1 fila = 1 tienda + 1 día
#
# Target principal:
#
#       net_sales
#
# También conservaremos:
#
# - gross_sales
# - total_discount
# - total_units
# - total_tickets
#
# IMPORTANTE
# ------------------------------------------------------------
#
# Todo el cálculo será dinámico:
#
# - número de tiendas
# - primera fecha
# - última fecha
# - número de días
# - volumen esperado
# - combinaciones tienda-día ausentes
#
# No utilizaremos números hardcodeados.
#
# ============================================================


from pyspark.sql import functions as F

# ============================================================
# 0. CARGA DE DEPENDENCIAS DEL DATASET
# ============================================================
#
# Hacemos este bloque autosuficiente.
#
# No dependemos de que el bloque 01 haya sido ejecutado
# previamente en la sesión.
#
# ============================================================


# ------------------------------------------------------------
# FACT SALES
# ------------------------------------------------------------

fact_sales_df = spark.table(
    "retail_analytics.3_gold.fact_sales"
)


# ------------------------------------------------------------
# DIM DATE
# ------------------------------------------------------------

dim_date_df = spark.table(
    "retail_analytics.3_gold.dim_date"
)


# ------------------------------------------------------------
# DIM STORE
# ------------------------------------------------------------

dim_store_df = spark.table(
    "retail_analytics.3_gold.dim_store"
)


# ============================================================
# CONSTRUIMOS SALES_WITH_DATE_DF
# ============================================================
#
# FACT_SALES contiene date_key.
#
# Recuperamos la fecha real desde DIM_DATE para disponer de:
#
#       sale_date
#
# ============================================================

sales_with_date_df = (

    fact_sales_df.alias("f")

    .join(

        dim_date_df
        .select(
            "date_key",
            "date"
        )
        .alias("d"),

        F.col("f.date_key")
        ==
        F.col("d.date_key"),

        how="inner"

    )

    .select(

        F.col("f.*"),

        F.col(
            "d.date"
        ).alias(
            "sale_date"
        )

    )

)
# ============================================================
# 02.01 AGREGACIÓN DIARIA POR TIENDA
# ============================================================
#
# Partimos de sales_with_date_df, creado en el bloque 01.
#
# Granularidad origen:
#
#       1 fila = 1 línea de ticket
#
# Granularidad resultado:
#
#       1 fila = 1 tienda + 1 día
#
# ============================================================

daily_store_sales_df = (

    sales_with_date_df

    .groupBy(
        "sale_date",
        "store_id"
    )

    .agg(

        # ----------------------------------------------------
        # TARGET
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
# 02.02 ATRIBUTOS TEMPORALES
# ============================================================
#
# Son variables conocidas en el momento de la predicción,
# por tanto no generan Data Leakage.
#
# ============================================================

daily_store_sales_df = (

    daily_store_sales_df

    .withColumn(
        "year",
        F.year(
            F.col("sale_date")
        )
    )

    .withColumn(
        "month",
        F.month(
            F.col("sale_date")
        )
    )

    .withColumn(
        "day",
        F.dayofmonth(
            F.col("sale_date")
        )
    )

    .withColumn(
        "day_of_week",
        F.dayofweek(
            F.col("sale_date")
        )
    )

    .withColumn(
        "week_of_year",
        F.weekofyear(
            F.col("sale_date")
        )
    )

    .withColumn(

        "is_weekend",

        F.when(

            F.dayofweek(
                F.col("sale_date")
            ).isin(
                1,
                7
            ),

            1

        )

        .otherwise(
            0
        )

    )

)


# ============================================================
# 02.03 ATRIBUTOS DE TIENDA
# ============================================================

daily_store_sales_df = (

    daily_store_sales_df.alias(
        "sales"
    )

    .join(

        dim_store_df.alias(
            "store"
        ),

        on="store_id",

        how="left"

    )

    .select(

        F.col(
            "sales.sale_date"
        ),

        F.col(
            "sales.store_id"
        ),

        F.col(
            "store.store_name"
        ),

        F.col(
            "store.city"
        ),

        F.col(
            "store.region"
        ),

        F.col(
            "sales.net_sales"
        ),

        F.col(
            "sales.gross_sales"
        ),

        F.col(
            "sales.total_discount"
        ),

        F.col(
            "sales.total_units"
        ),

        F.col(
            "sales.total_tickets"
        ),

        F.col(
            "sales.year"
        ),

        F.col(
            "sales.month"
        ),

        F.col(
            "sales.day"
        ),

        F.col(
            "sales.day_of_week"
        ),

        F.col(
            "sales.week_of_year"
        ),

        F.col(
            "sales.is_weekend"
        )

    )

)


# ============================================================
# 02.04 ORDENAMOS EL DATASET
# ============================================================

daily_store_sales_df = (

    daily_store_sales_df

    .orderBy(
        "store_id",
        "sale_date"
    )

)


# ============================================================
# 02.05 VALIDACIÓN DEL VOLUMEN ESPERADO
# ============================================================
#
# Volumen esperado:
#
#       número de tiendas
#              ×
#       número de días naturales
#
# ============================================================


# ------------------------------------------------------------
# NÚMERO DE TIENDAS
# ------------------------------------------------------------

total_stores_ml = (

    daily_store_sales_df

    .select(
        "store_id"
    )

    .distinct()

    .count()

)


# ------------------------------------------------------------
# RANGO TEMPORAL
# ------------------------------------------------------------

ml_date_range = (

    daily_store_sales_df

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


min_ml_date = (
    ml_date_range["min_date"]
)

max_ml_date = (
    ml_date_range["max_date"]
)


# ------------------------------------------------------------
# NÚMERO DE DÍAS NATURALES
# ------------------------------------------------------------

total_calendar_days = (

    spark.range(1)

    .select(

        (
            F.datediff(
                F.lit(max_ml_date),
                F.lit(min_ml_date)
            )
            +
            1
        ).alias(
            "days"
        )

    )

    .first()["days"]

)


# ------------------------------------------------------------
# VOLUMEN ESPERADO
# ------------------------------------------------------------

expected_ml_rows = (
    total_stores_ml
    *
    total_calendar_days
)


# ------------------------------------------------------------
# VOLUMEN REAL
# ------------------------------------------------------------

total_ml_rows = (
    daily_store_sales_df
    .count()
)


# ------------------------------------------------------------
# COMBINACIONES AUSENTES
# ------------------------------------------------------------

missing_store_days = (
    expected_ml_rows
    -
    total_ml_rows
)


# ============================================================
# 02.06 COMPROBAMOS DUPLICADOS
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


# ============================================================
# 02.07 RESUMEN
# ============================================================

print("=" * 80)
print("VALIDACIÓN DEL DATASET DIARIO POR TIENDA")
print("=" * 80)

print(
    f"Tiendas:                 {total_stores_ml:,}"
)

print(
    f"Primera fecha:           {min_ml_date}"
)

print(
    f"Última fecha:            {max_ml_date}"
)

print(
    f"Días naturales:          {total_calendar_days:,}"
)

print(
    f"Registros obtenidos:     {total_ml_rows:,}"
)

print(
    f"Registros esperados:     {expected_ml_rows:,}"
)

print(
    f"Combinaciones ausentes:  {missing_store_days:,}"
)

print(
    f"Duplicados tienda-día:   {duplicate_store_dates:,}"
)


# ============================================================
# 02.08 RESULTADO DE LA VALIDACIÓN INICIAL
# ============================================================

if duplicate_store_dates > 0:

    raise RuntimeError(
        f"ERROR - Se han detectado "
        f"{duplicate_store_dates} claves "
        "sale_date + store_id duplicadas."
    )


if missing_store_days < 0:

    raise RuntimeError(
        "ERROR - El dataset contiene más combinaciones "
        "tienda-día de las esperadas."
    )


if missing_store_days == 0:

    print(
        "OK - El dataset contiene todas las combinaciones "
        "tienda-día esperadas."
    )

else:

    print(
        "INFO - Existen "
        f"{missing_store_days} combinaciones tienda-día "
        "sin transacciones."
    )

    print(
        "INFO - Estas combinaciones se regularizarán "
        "posteriormente con ventas = 0."
    )


# ============================================================
# 02.09 GENERAMOS EL CALENDARIO COMPLETO
# ============================================================
#
# Creamos:
#
#       todas las tiendas
#              ×
#       todas las fechas del histórico
#
# ============================================================


calendar_df = (

    dim_date_df

    .select(
        F.col(
            "date"
        ).alias(
            "sale_date"
        )
    )

    .filter(

        (F.col("sale_date") >= F.lit(min_ml_date))

        &

        (F.col("sale_date") <= F.lit(max_ml_date))

    )

    .distinct()

)


stores_df = (

    daily_store_sales_df

    .select(
        "store_id"
    )

    .distinct()

)


expected_store_dates_df = (

    stores_df

    .crossJoin(
        calendar_df
    )

)


# ============================================================
# 02.10 DETECTAMOS COMBINACIONES AUSENTES
# ============================================================

missing_store_dates_df = (

    expected_store_dates_df.alias(
        "expected"
    )

    .join(

        daily_store_sales_df
        .select(
            "store_id",
            "sale_date"
        )
        .alias(
            "actual"
        ),

        on=[
            "store_id",
            "sale_date"
        ],

        how="left_anti"

    )

    .orderBy(
        "store_id",
        "sale_date"
    )

)


missing_store_dates_count = (
    missing_store_dates_df
    .count()
)


# ============================================================
# 02.11 VALIDAMOS COHERENCIA DEL NÚMERO DE HUECOS
# ============================================================

print()
print("=" * 80)
print("ANÁLISIS DE COMBINACIONES TIENDA-DÍA AUSENTES")
print("=" * 80)

print(
    f"Huecos calculados por volumen: "
    f"{missing_store_days:,}"
)

print(
    f"Huecos detectados con LEFT ANTI: "
    f"{missing_store_dates_count:,}"
)


if (
    missing_store_days
    !=
    missing_store_dates_count
):

    raise RuntimeError(
        "ERROR - El número de combinaciones ausentes "
        "no coincide entre ambas validaciones."
    )


# ============================================================
# 02.12 REGULARIZACIÓN DEL DATASET TEMPORAL
# ============================================================
#
# Las combinaciones tienda-día sin transacciones se interpretan
# como días con ventas = 0.
#
# Esto garantiza una serie temporal continua.
#
# ============================================================


complete_store_calendar_df = (

    expected_store_dates_df

    .select(
        "sale_date",
        "store_id"
    )

)


daily_metrics_df = (

    daily_store_sales_df

    .select(

        "sale_date",
        "store_id",

        "net_sales",
        "gross_sales",
        "total_discount",
        "total_units",
        "total_tickets"

    )

)


ml_daily_store_df = (

    complete_store_calendar_df

    .join(

        daily_metrics_df,

        on=[
            "sale_date",
            "store_id"
        ],

        how="left"

    )

)


# ============================================================
# 02.13 COMPLETAMOS LAS MÉTRICAS CON CERO
# ============================================================

ml_daily_store_df = (

    ml_daily_store_df

    .fillna(

        {

            "net_sales": 0.0,

            "gross_sales": 0.0,

            "total_discount": 0.0,

            "total_units": 0,

            "total_tickets": 0

        }

    )

)


# ============================================================
# 02.14 INCORPORAMOS ATRIBUTOS DE TIENDA
# ============================================================

ml_daily_store_df = (

    ml_daily_store_df.alias(
        "sales"
    )

    .join(

        dim_store_df.alias(
            "store"
        ),

        on="store_id",

        how="left"

    )

    .select(

        F.col(
            "sales.sale_date"
        ),

        F.col(
            "sales.store_id"
        ),

        F.col(
            "store.store_name"
        ),

        F.col(
            "store.city"
        ),

        F.col(
            "store.region"
        ),

        F.col(
            "sales.net_sales"
        ),

        F.col(
            "sales.gross_sales"
        ),

        F.col(
            "sales.total_discount"
        ),

        F.col(
            "sales.total_units"
        ),

        F.col(
            "sales.total_tickets"
        )

    )

)


# ============================================================
# 02.15 RECONSTRUIMOS FEATURES TEMPORALES
# ============================================================

ml_daily_store_df = (

    ml_daily_store_df

    .withColumn(
        "year",
        F.year(
            "sale_date"
        )
    )

    .withColumn(
        "month",
        F.month(
            "sale_date"
        )
    )

    .withColumn(
        "day",
        F.dayofmonth(
            "sale_date"
        )
    )

    .withColumn(
        "day_of_week",
        F.dayofweek(
            "sale_date"
        )
    )

    .withColumn(
        "week_of_year",
        F.weekofyear(
            "sale_date"
        )
    )

    .withColumn(

        "is_weekend",

        F.when(

            F.dayofweek(
                "sale_date"
            ).isin(
                1,
                7
            ),

            1

        )

        .otherwise(
            0
        )

    )

    .orderBy(
        "store_id",
        "sale_date"
    )

)


# ============================================================
# 02.16 VALIDACIÓN FINAL DEL PANEL REGULARIZADO
# ============================================================

final_ml_rows = (
    ml_daily_store_df
    .count()
)


zero_sales_days = (

    ml_daily_store_df

    .filter(
        F.col("net_sales") == 0
    )

    .count()

)


duplicate_dates_final = (

    ml_daily_store_df

    .groupBy(
        "store_id",
        "sale_date"
    )

    .count()

    .filter(
        F.col("count") > 1
    )

    .count()

)


print()
print("=" * 80)
print("VALIDACIÓN DEL PANEL TEMPORAL REGULARIZADO")
print("=" * 80)

print(
    f"Registros finales:       {final_ml_rows:,}"
)

print(
    f"Registros esperados:     {expected_ml_rows:,}"
)

print(
    f"Días regularizados a 0:  {zero_sales_days:,}"
)

print(
    f"Duplicados:               {duplicate_dates_final:,}"
)


# ============================================================
# 02.17 RESULTADO FINAL
# ============================================================

validation_errors = []


if final_ml_rows != expected_ml_rows:

    validation_errors.append(
        "El volumen final del panel temporal "
        "no coincide con el esperado."
    )


if duplicate_dates_final > 0:

    validation_errors.append(
        f"Existen {duplicate_dates_final} "
        "duplicados tienda-día."
    )


if zero_sales_days != missing_store_dates_count:

    validation_errors.append(
        "El número de días regularizados a cero "
        "no coincide con los huecos detectados."
    )


if validation_errors:

    print()
    print(
        "ERROR - Validación del dataset ML fallida"
    )

    for error in validation_errors:

        print(
            " -",
            error
        )

    raise RuntimeError(
        " | ".join(
            validation_errors
        )
    )


else:

    print()
    print(
        "OK - Dataset diario por tienda "
        "regularizado correctamente"
    )

    print(
        "OK - Volumen dinámico validado"
    )

    print(
        "OK - No existen duplicados tienda-día"
    )

    print(
        "OK - Serie temporal continua preparada para "
        "Feature Engineering"
    )


# ============================================================
# 02.18 INSPECCIÓN DE LOS DÍAS REGULARIZADOS
# ============================================================

display(

    ml_daily_store_df

    .filter(
        F.col("net_sales") == 0
    )

    .orderBy(
        "sale_date",
        "store_id"
    )

)

# COMMAND ----------

# DBTITLE 1,02.08 ANÁLISIS DE COMBINACIONES TIENDA-DÍA AUSENTES
# ============================================================
# 02.08 ANÁLISIS DE COMBINACIONES TIENDA-DÍA AUSENTES
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Hemos detectado:
#
#       Registros esperados: 29.233
#       Registros obtenidos: 29.226
#
# Por tanto existen 7 combinaciones tienda-día sin ventas.
#
# Antes de construir features temporales debemos identificar:
#
# - qué tiendas están afectadas
# - qué fechas faltan
# - si se trata de días reales con 0 ventas
# - o de posibles huecos en los datos
#
# ============================================================


from pyspark.sql import functions as F


# ------------------------------------------------------------
# 1. NÚMERO DE DÍAS DISPONIBLES POR TIENDA
# ------------------------------------------------------------
#
# Esto nos permitirá localizar rápidamente qué tiendas
# no tienen los 943 días completos.
#
# ------------------------------------------------------------

store_day_coverage_df = (

    daily_store_sales_df

    .groupBy(
        "store_id"
    )

    .agg(

        F.countDistinct(
            "sale_date"
        ).alias(
            "days_available"
        )
    )

    .withColumn(

        "missing_days",

        F.lit(943)
        -
        F.col("days_available")
    )

    .orderBy(
        F.desc("missing_days")
    )
)


display(
    store_day_coverage_df
)


# ------------------------------------------------------------
# 2. GENERAMOS TODAS LAS COMBINACIONES ESPERADAS
# ------------------------------------------------------------
#
# Creamos el calendario completo:
#
#       943 días × 31 tiendas
#
# Esto representa todas las combinaciones que deberían existir
# en nuestro dataset diario.
#
# ------------------------------------------------------------

calendar_df = (

    dim_date_df

    .select(
        F.col("date").alias("sale_date")
    )

    .filter(

        (F.col("sale_date") >= F.lit("2024-01-01"))

        &

        (F.col("sale_date") <= F.lit("2026-07-31"))
    )
)


stores_df = (

    fact_sales_df

    .select(
        "store_id"
    )

    .distinct()
)


expected_store_dates_df = (

    stores_df

    .crossJoin(
        calendar_df
    )
)


# ------------------------------------------------------------
# 3. DETECTAMOS LAS COMBINACIONES AUSENTES
# ------------------------------------------------------------
#
# LEFT ANTI devuelve únicamente las combinaciones esperadas
# que NO aparecen en daily_store_sales_df.
#
# ------------------------------------------------------------

missing_store_dates_df = (

    expected_store_dates_df.alias("expected")

    .join(

        daily_store_sales_df
        .select(
            "store_id",
            "sale_date"
        )
        .alias("actual"),

        on=[
            "store_id",
            "sale_date"
        ],

        how="left_anti"
    )

    .orderBy(
        "store_id",
        "sale_date"
    )
)


# ------------------------------------------------------------
# 4. VALIDAMOS EL NÚMERO DE HUECOS
# ------------------------------------------------------------

missing_store_dates_count = (
    missing_store_dates_df
    .count()
)


print("=" * 80)
print("ANÁLISIS DE COMBINACIONES TIENDA-DÍA AUSENTES")
print("=" * 80)

print(
    f"Combinaciones ausentes: {missing_store_dates_count:,}"
)


# ------------------------------------------------------------
# 5. MOSTRAMOS EXACTAMENTE QUÉ DÍAS FALTAN
# ------------------------------------------------------------

display(
    missing_store_dates_df
)

# COMMAND ----------

# DBTITLE 1,02.09 REGULARIZACIÓN DEL DATASET TEMPORAL
# ============================================================
# 02.09 REGULARIZACIÓN DEL DATASET TEMPORAL
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Construir un dataset temporal completamente continuo:
#
#       31 tiendas × 943 días = 29.233 observaciones
#
# Hemos detectado únicamente 7 combinaciones tienda-día
# sin transacciones.
#
# Como nuestro dataset representa el conjunto completo de
# ventas disponibles, interpretaremos esas ausencias como:
#
#       net_sales = 0
#
# y no como datos desconocidos.
#
# Esto es especialmente importante para Machine Learning:
#
# - los lags deben representar días reales consecutivos
# - las rolling averages necesitan ventanas temporales regulares
# - no queremos que lag_1 signifique accidentalmente
#   "último día con ventas"
#
# ============================================================


from pyspark.sql import functions as F


# ------------------------------------------------------------
# 1. CREAMOS EL PANEL COMPLETO TIENDA × FECHA
# ------------------------------------------------------------
#
# expected_store_dates_df ya contiene:
#
#       todas las tiendas
#           ×
#       todas las fechas
#
# Por tanto contiene exactamente las 29.233 combinaciones
# que deberían existir.
#
# ------------------------------------------------------------

complete_store_calendar_df = (

    expected_store_dates_df

    .select(
        "sale_date",
        "store_id"
    )
)


# ------------------------------------------------------------
# 2. RECUPERAMOS ÚNICAMENTE LAS MEDIDAS DIARIAS
# ------------------------------------------------------------
#
# Evitamos reutilizar aquí los atributos temporales y de tienda
# porque los reconstruiremos posteriormente de forma limpia.
#
# ------------------------------------------------------------

daily_metrics_df = (

    daily_store_sales_df

    .select(

        "sale_date",
        "store_id",

        "net_sales",
        "gross_sales",
        "total_discount",
        "total_units",
        "total_tickets"
    )
)


# ------------------------------------------------------------
# 3. LEFT JOIN CONTRA EL CALENDARIO COMPLETO
# ------------------------------------------------------------
#
# Las 29.226 combinaciones existentes conservarán sus valores.
#
# Las 7 combinaciones ausentes tendrán inicialmente NULL.
#
# ------------------------------------------------------------

ml_daily_store_df = (

    complete_store_calendar_df.alias("calendar")

    .join(

        daily_metrics_df.alias("sales"),

        on=[
            "sale_date",
            "store_id"
        ],

        how="left"
    )
)


# ------------------------------------------------------------
# 4. COMPLETAMOS LOS DÍAS SIN TRANSACCIONES CON CERO
# ------------------------------------------------------------
#
# Para esos 7 días:
#
#       ventas          = 0
#       unidades        = 0
#       tickets         = 0
#       descuentos      = 0
#
# ------------------------------------------------------------

ml_daily_store_df = (

    ml_daily_store_df

    .fillna(

        {

            "net_sales": 0.0,

            "gross_sales": 0.0,

            "total_discount": 0.0,

            "total_units": 0,

            "total_tickets": 0
        }
    )
)


# ------------------------------------------------------------
# 5. INCORPORAMOS ATRIBUTOS DE LA TIENDA
# ------------------------------------------------------------

ml_daily_store_df = (

    ml_daily_store_df.alias("sales")

    .join(

        dim_store_df.alias("store"),

        on="store_id",

        how="left"
    )

    .select(

        F.col(
            "sales.sale_date"
        ),

        F.col(
            "sales.store_id"
        ),

        F.col(
            "store.store_name"
        ),

        F.col(
            "store.city"
        ),

        F.col(
            "store.region"
        ),

        F.col(
            "sales.net_sales"
        ),

        F.col(
            "sales.gross_sales"
        ),

        F.col(
            "sales.total_discount"
        ),

        F.col(
            "sales.total_units"
        ),

        F.col(
            "sales.total_tickets"
        )
    )
)


# ------------------------------------------------------------
# 6. RECONSTRUIMOS LAS FEATURES TEMPORALES
# ------------------------------------------------------------
#
# Todas estas variables son conocidas previamente y, por tanto,
# pueden utilizarse posteriormente como features sin provocar
# Data Leakage.
#
# ------------------------------------------------------------

ml_daily_store_df = (

    ml_daily_store_df

    .withColumn(
        "year",
        F.year(
            "sale_date"
        )
    )

    .withColumn(
        "month",
        F.month(
            "sale_date"
        )
    )

    .withColumn(
        "day",
        F.dayofmonth(
            "sale_date"
        )
    )

    .withColumn(
        "day_of_week",
        F.dayofweek(
            "sale_date"
        )
    )

    .withColumn(
        "week_of_year",
        F.weekofyear(
            "sale_date"
        )
    )

    .withColumn(

        "is_weekend",

        F.when(
            F.dayofweek(
                "sale_date"
            ).isin(
                1,
                7
            ),
            1
        )

        .otherwise(
            0
        )
    )

    .orderBy(
        "store_id",
        "sale_date"
    )
)


# ------------------------------------------------------------
# 7. VALIDACIÓN FINAL DEL PANEL TEMPORAL
# ------------------------------------------------------------

final_ml_rows = (
    ml_daily_store_df
    .count()
)


zero_sales_days = (

    ml_daily_store_df

    .filter(
        F.col("net_sales") == 0
    )

    .count()
)


duplicate_dates = (

    ml_daily_store_df

    .groupBy(
        "store_id",
        "sale_date"
    )

    .count()

    .filter(
        F.col("count") > 1
    )

    .count()
)


print("=" * 80)
print("VALIDACIÓN DEL PANEL TEMPORAL REGULARIZADO")
print("=" * 80)


print(
    f"Registros finales:         {final_ml_rows:,}"
)

print(
    f"Registros esperados:       {31 * 943:,}"
)

print(
    f"Días con ventas = 0:       {zero_sales_days:,}"
)

print(
    f"Claves duplicadas:         {duplicate_dates:,}"
)


if (
    final_ml_rows == 31 * 943
    and
    zero_sales_days == 7
    and
    duplicate_dates == 0
):

    print(
        "OK - Dataset temporal completamente regularizado"
    )

else:

    print(
        "REVISAR - El panel temporal no cumple las condiciones esperadas"
    )


# ------------------------------------------------------------
# 8. MOSTRAMOS LOS DÍAS COMPLETADOS
# ------------------------------------------------------------

display(

    ml_daily_store_df

    .filter(
        F.col("net_sales") == 0
    )

    .orderBy(
        "sale_date"
    )
)

# COMMAND ----------

# DBTITLE 1,03. FEATURE ENGINEERING TEMPORAL
# ============================================================
# 03. FEATURE ENGINEERING TEMPORAL
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Crear variables predictoras a partir del histórico de ventas
# de cada tienda.
#
# La variable objetivo seguirá siendo:
#
#       net_sales
#
# Crearemos:
#
# - lags temporales
# - medias móviles
# - desviación estándar reciente
# - tendencias recientes
#
# IMPORTANTE
# ------------------------------------------------------------
#
# Todas las ventanas deben utilizar únicamente información
# anterior al día que queremos predecir.
#
# Nunca incluimos el valor actual de net_sales dentro de una
# rolling average porque produciría Data Leakage.
#
# ============================================================


from pyspark.sql import functions as F
from pyspark.sql.window import Window


# ------------------------------------------------------------
# 03.01 DEFINIMOS LA VENTANA TEMPORAL POR TIENDA
# ------------------------------------------------------------
#
# Cada tienda representa una serie temporal independiente.
#
# Ordenamos cronológicamente por sale_date.
#
# ------------------------------------------------------------

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
# 03.02 LAGS DE VENTAS
# ============================================================
#
# Los lags permiten que el modelo conozca el comportamiento
# histórico reciente de la tienda.
#
# lag_1:
#       ventas de ayer
#
# lag_7:
#       ventas del mismo día de la semana anterior
#
# lag_14:
#       ventas de hace dos semanas
#
# lag_28:
#       ventas de hace cuatro semanas
#
# ============================================================

ml_features_df = (

    ml_daily_store_df


    .withColumn(
        "lag_1",

        F.lag(
            "net_sales",
            1
        ).over(
            store_window
        )
    )


    .withColumn(
        "lag_7",

        F.lag(
            "net_sales",
            7
        ).over(
            store_window
        )
    )


    .withColumn(
        "lag_14",

        F.lag(
            "net_sales",
            14
        ).over(
            store_window
        )
    )


    .withColumn(
        "lag_28",

        F.lag(
            "net_sales",
            28
        ).over(
            store_window
        )
    )
)


# ============================================================
# 03.03 VENTANAS MÓVILES
# ============================================================
#
# Creamos ventanas que terminan SIEMPRE en el día anterior.
#
# Ejemplo:
#
# rowsBetween(-7, -1)
#
# significa:
#
#       desde hace 7 días
#       hasta ayer
#
# De esta forma evitamos utilizar net_sales del día actual.
#
# ============================================================


rolling_7_window = (

    store_window

    .rowsBetween(
        -7,
        -1
    )
)


rolling_28_window = (

    store_window

    .rowsBetween(
        -28,
        -1
    )
)


# ------------------------------------------------------------
# MEDIA MÓVIL 7 DÍAS
# ------------------------------------------------------------

ml_features_df = (

    ml_features_df

    .withColumn(
        "rolling_mean_7",

        F.avg(
            "net_sales"
        ).over(
            rolling_7_window
        )
    )
)


# ------------------------------------------------------------
# MEDIA MÓVIL 28 DÍAS
# ------------------------------------------------------------

ml_features_df = (

    ml_features_df

    .withColumn(
        "rolling_mean_28",

        F.avg(
            "net_sales"
        ).over(
            rolling_28_window
        )
    )
)


# ------------------------------------------------------------
# DESVIACIÓN ESTÁNDAR ÚLTIMOS 7 DÍAS
# ------------------------------------------------------------
#
# Nos indica cuánto han variado recientemente las ventas.
#
# Una desviación alta implica mayor volatilidad.
#
# ------------------------------------------------------------

ml_features_df = (

    ml_features_df

    .withColumn(
        "rolling_std_7",

        F.stddev(
            "net_sales"
        ).over(
            rolling_7_window
        )
    )
)


# ============================================================
# 03.04 TENDENCIAS
# ============================================================
#
# Creamos variables sencillas que representen cambios recientes
# en el comportamiento de ventas.
#
# ============================================================


# ------------------------------------------------------------
# DIFERENCIA ENTRE AYER Y HACE 7 DÍAS
# ------------------------------------------------------------
#
# Comparamos:
#
#       ventas t-1
#          -
#       ventas t-7
#
# ============================================================

ml_features_df = (

    ml_features_df

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


# ------------------------------------------------------------
# RATIO AYER VS MEDIA DE LOS 7 DÍAS ANTERIORES
# ------------------------------------------------------------

ml_features_df = (

    ml_features_df

    .withColumn(

        "lag1_vs_mean7",

        F.when(

            F.col(
                "rolling_mean_7"
            ) > 0,

            F.col(
                "lag_1"
            )

            /

            F.col(
                "rolling_mean_7"
            )

        )

        .otherwise(
            0.0
        )

    )

)


# ============================================================
# 03.05 IDENTIFICAMOS FILAS UTILIZABLES PARA ML
# ============================================================
#
# Para disponer de lag_28 necesitamos 28 observaciones
# anteriores por tienda.
#
# Como el panel ya está regularizado:
#
#       1 fila = 1 tienda + 1 día
#
# los primeros 28 días de cada tienda deben quedar fuera
# del dataset final de entrenamiento.
#
# ============================================================

ml_model_df = (

    ml_features_df

    .filter(

        F.col(
            "lag_28"
        ).isNotNull()

        &

        F.col(
            "rolling_mean_28"
        ).isNotNull()

    )

)


# ============================================================
# 03.06 VALIDACIÓN DEL DATASET CON FEATURES
# ============================================================


# ------------------------------------------------------------
# VOLUMEN ORIGINAL
# ------------------------------------------------------------

original_rows = (
    ml_daily_store_df
    .count()
)


# ------------------------------------------------------------
# VOLUMEN DISPONIBLE PARA ML
# ------------------------------------------------------------

model_rows = (
    ml_model_df
    .count()
)


# ------------------------------------------------------------
# FILAS ELIMINADAS
# ------------------------------------------------------------

removed_rows = (
    original_rows
    -
    model_rows
)


# ------------------------------------------------------------
# NÚMERO DE TIENDAS
# ------------------------------------------------------------

model_store_count = (

    ml_daily_store_df

    .select(
        "store_id"
    )

    .distinct()

    .count()

)


# ------------------------------------------------------------
# FILAS QUE ESPERAMOS ELIMINAR
# ------------------------------------------------------------
#
# 28 días iniciales por tienda debido a lag_28.
#
# ============================================================

expected_removed_rows = (
    model_store_count
    *
    28
)


expected_model_rows = (
    original_rows
    -
    expected_removed_rows
)


# ============================================================
# VALIDAMOS DUPLICADOS DEL DATASET FINAL
# ============================================================

duplicate_model_rows = (

    ml_model_df

    .groupBy(
        "store_id",
        "sale_date"
    )

    .count()

    .filter(
        F.col("count") > 1
    )

    .count()

)


# ============================================================
# VALIDAMOS TARGET
# ============================================================

null_target_rows = (

    ml_model_df

    .filter(
        F.col("net_sales").isNull()
    )

    .count()

)


# ============================================================
# FEATURES PRINCIPALES
# ============================================================

feature_columns = [

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
# NULLS EN FEATURES
# ============================================================

null_checks = []


for feature in feature_columns:

    null_count = (

        ml_model_df

        .filter(
            F.col(feature).isNull()
        )

        .count()

    )

    null_checks.append(
        (
            feature,
            null_count
        )
    )


total_feature_nulls = sum(
    null_count
    for _, null_count in null_checks
)


# ============================================================
# RESUMEN
# ============================================================

print("=" * 80)
print("VALIDACIÓN DEL FEATURE ENGINEERING")
print("=" * 80)

print(
    f"Tiendas:                     {model_store_count:,}"
)

print(
    f"Registros dataset original:  {original_rows:,}"
)

print(
    f"Registros disponibles ML:    {model_rows:,}"
)

print(
    f"Registros eliminados:        {removed_rows:,}"
)

print(
    f"Registros a eliminar esper.: {expected_removed_rows:,}"
)

print(
    f"Registros ML esperados:      {expected_model_rows:,}"
)

print(
    f"Duplicados tienda-día:       {duplicate_model_rows:,}"
)

print(
    f"Targets NULL:                {null_target_rows:,}"
)


print()
print("=" * 80)
print("NULLS EN FEATURES")
print("=" * 80)


for feature, null_count in null_checks:

    status = (
        "OK"
        if null_count == 0
        else "ERROR"
    )

    print(
        f"{status:8} | "
        f"{feature:22} | "
        f"{null_count:,}"
    )


# ============================================================
# VALIDACIÓN FINAL
# ============================================================

validation_errors = []


if removed_rows != expected_removed_rows:

    validation_errors.append(
        f"Se han eliminado {removed_rows} registros, "
        f"pero se esperaban {expected_removed_rows}"
    )


if model_rows != expected_model_rows:

    validation_errors.append(
        f"El dataset ML contiene {model_rows} registros, "
        f"pero se esperaban {expected_model_rows}"
    )


if duplicate_model_rows > 0:

    validation_errors.append(
        f"Existen {duplicate_model_rows} claves "
        "store_id + sale_date duplicadas"
    )


if null_target_rows > 0:

    validation_errors.append(
        f"Existen {null_target_rows} registros con "
        "net_sales NULL"
    )


if total_feature_nulls > 0:

    validation_errors.append(
        f"Existen {total_feature_nulls} valores NULL "
        "en las features utilizadas"
    )


# ============================================================
# RESULTADO
# ============================================================

if validation_errors:

    print()
    print(
        "ERROR - Feature Engineering no válido"
    )

    for error in validation_errors:

        print(
            " -",
            error
        )

    raise RuntimeError(
        " | ".join(
            validation_errors
        )
    )


else:

    print()
    print(
        "OK - Feature Engineering validado correctamente"
    )

    print(
        "OK - Histórico de 28 días respetado"
    )

    print(
        "OK - No existen NULLs en las features"
    )

    print(
        "OK - No existen duplicados tienda-día"
    )

    print(
        "OK - Dataset preparado para split temporal"
    )


# ============================================================
# 03.07 INSPECCIÓN DEL RESULTADO
# ============================================================

display(

    ml_model_df

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
        "lag1_vs_mean7",

        "day_of_week",
        "month",
        "is_weekend"

    )

    .orderBy(
        "store_id",
        "sale_date"
    )

    .limit(
        100
    )

)

# COMMAND ----------

# MAGIC %md
# MAGIC # 04. SPLIT TEMPORAL: TRAIN / VALIDATION / TEST
# MAGIC
# MAGIC ## Objetivo
# MAGIC
# MAGIC Dividir el dataset preparado para Machine Learning en tres periodos temporales independientes:
# MAGIC
# MAGIC - **Train** → entrenamiento del modelo
# MAGIC - **Validation** → comparación y ajuste de modelos
# MAGIC - **Test** → evaluación final
# MAGIC
# MAGIC Al tratarse de un problema de forecasting, no utilizamos una división aleatoria.
# MAGIC
# MAGIC La separación debe respetar el orden temporal para evitar **Data Leakage** y simular correctamente un escenario real de predicción.
# MAGIC
# MAGIC ## Estrategia temporal
# MAGIC
# MAGIC | Dataset | Periodo |
# MAGIC |---|---|
# MAGIC | Train | 29/01/2024 → 31/12/2025 |
# MAGIC | Validation | 01/01/2026 → 30/04/2026 |
# MAGIC | Test | 01/05/2026 → 31/07/2026 |
# MAGIC
# MAGIC El modelo aprenderá exclusivamente del pasado y será evaluado sobre periodos posteriores que no ha utilizado durante el entrenamiento.

# COMMAND ----------

# DBTITLE 1,04.01 SPLIT TEMPORAL TRAIN / VALIDATION / TEST
# ============================================================
# 04.01 SPLIT TEMPORAL TRAIN / VALIDATION / TEST / OUT-OF-TIME
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Dividir el dataset de Machine Learning respetando
# estrictamente el orden temporal.
#
# No utilizamos randomSplit porque estamos trabajando
# con un problema de forecasting.
#
# PERIODOS
# ------------------------------------------------------------
#
# TRAIN
#   histórico disponible -> 2025-12-31
#
# VALIDATION
#   2026-01-01 -> 2026-04-30
#
# TEST
#   2026-05-01 -> 2026-07-31
#
# OUT-OF-TIME
#   2026-08-01 -> última fecha disponible
#
# IMPORTANTE
# ------------------------------------------------------------
#
# El periodo OUT-OF-TIME NO participa en:
#
# - entrenamiento
# - selección de hiperparámetros
# - selección del modelo
#
# Representa datos futuros que el modelo no ha utilizado
# durante su desarrollo.
#
# ============================================================


from pyspark.sql import functions as F


# ============================================================
# 04.01.01 FECHAS DE CORTE
# ============================================================

train_end = "2025-12-31"

validation_start = "2026-01-01"
validation_end = "2026-04-30"

test_start = "2026-05-01"
test_end = "2026-07-31"

oot_start = "2026-08-01"


# ============================================================
# 04.01.02 ÚLTIMA FECHA DISPONIBLE
# ============================================================
#
# No hardcodeamos el final del periodo OUT-OF-TIME.
#
# De esta forma, si posteriormente llegan nuevos datos,
# podremos detectarlos explícitamente.
#
# ============================================================

max_available_date = (

    ml_model_df

    .agg(
        F.max("sale_date").alias("max_date")
    )

    .first()["max_date"]

)


# ============================================================
# 04.01.03 TRAIN
# ============================================================

train_df = (

    ml_model_df

    .filter(
        F.col("sale_date")
        <=
        F.lit(train_end)
    )

)


# ============================================================
# 04.01.04 VALIDATION
# ============================================================

validation_df = (

    ml_model_df

    .filter(

        (F.col("sale_date") >= F.lit(validation_start))

        &

        (F.col("sale_date") <= F.lit(validation_end))

    )

)


# ============================================================
# 04.01.05 TEST
# ============================================================

test_df = (

    ml_model_df

    .filter(

        (F.col("sale_date") >= F.lit(test_start))

        &

        (F.col("sale_date") <= F.lit(test_end))

    )

)


# ============================================================
# 04.01.06 OUT-OF-TIME
# ============================================================
#
# Datos posteriores al periodo TEST.
#
# Este dataset queda completamente separado del proceso
# de entrenamiento y selección del modelo.
#
# ============================================================

oot_df = (

    ml_model_df

    .filter(
        F.col("sale_date")
        >=
        F.lit(oot_start)
    )

)


# ============================================================
# 04.01.07 CONTEOS
# ============================================================

total_rows = ml_model_df.count()

train_rows = train_df.count()

validation_rows = validation_df.count()

test_rows = test_df.count()

oot_rows = oot_df.count()


split_rows = (
    train_rows
    +
    validation_rows
    +
    test_rows
    +
    oot_rows
)


# ============================================================
# 04.01.08 RANGOS REALES
# ============================================================

def get_date_range(df):

    row = (

        df

        .agg(

            F.min("sale_date").alias("min_date"),

            F.max("sale_date").alias("max_date")

        )

        .first()

    )

    return (
        row["min_date"],
        row["max_date"]
    )


train_min, train_max = get_date_range(train_df)

validation_min, validation_max = get_date_range(validation_df)

test_min, test_max = get_date_range(test_df)

oot_min, oot_max = get_date_range(oot_df)


# ============================================================
# 04.01.09 VALIDACIÓN DE SOLAPAMIENTOS
# ============================================================

train_validation_overlap = (

    train_df

    .select(
        "store_id",
        "sale_date"
    )

    .join(

        validation_df.select(
            "store_id",
            "sale_date"
        ),

        [
            "store_id",
            "sale_date"
        ],

        "inner"

    )

    .count()

)


validation_test_overlap = (

    validation_df

    .select(
        "store_id",
        "sale_date"
    )

    .join(

        test_df.select(
            "store_id",
            "sale_date"
        ),

        [
            "store_id",
            "sale_date"
        ],

        "inner"

    )

    .count()

)


test_oot_overlap = (

    test_df

    .select(
        "store_id",
        "sale_date"
    )

    .join(

        oot_df.select(
            "store_id",
            "sale_date"
        ),

        [
            "store_id",
            "sale_date"
        ],

        "inner"

    )

    .count()

)


# ============================================================
# 04.01.10 RESUMEN
# ============================================================

print("=" * 80)
print("SPLIT TEMPORAL TRAIN / VALIDATION / TEST / OUT-OF-TIME")
print("=" * 80)

print(
    f"Última fecha disponible: {max_available_date}"
)

print()

print(
    f"TRAIN:      {train_rows:,} registros "
    f"| {train_min} -> {train_max}"
)

print(
    f"VALIDATION: {validation_rows:,} registros "
    f"| {validation_min} -> {validation_max}"
)

print(
    f"TEST:       {test_rows:,} registros "
    f"| {test_min} -> {test_max}"
)

print(
    f"OUT-OF-TIME:{oot_rows:,} registros "
    f"| {oot_min} -> {oot_max}"
)

print()

print(
    f"Dataset ML total:        {total_rows:,}"
)

print(
    f"Suma de los splits:      {split_rows:,}"
)

print(
    f"Diferencia:              {total_rows - split_rows:,}"
)

print()

print(
    f"Overlap TRAIN/VALIDATION: {train_validation_overlap:,}"
)

print(
    f"Overlap VALIDATION/TEST:  {validation_test_overlap:,}"
)

print(
    f"Overlap TEST/OOT:         {test_oot_overlap:,}"
)


# ============================================================
# 04.01.11 VALIDACIÓN FINAL
# ============================================================

validation_errors = []


# ------------------------------------------------------------
# Todos los registros deben pertenecer exactamente
# a uno de los cuatro periodos.
# ------------------------------------------------------------

if split_rows != total_rows:

    validation_errors.append(

        f"La suma de los splits ({split_rows}) "
        f"no coincide con el dataset ML ({total_rows})"

    )


# ------------------------------------------------------------
# Ningún periodo puede estar vacío.
# ------------------------------------------------------------

if train_rows == 0:

    validation_errors.append(
        "TRAIN está vacío"
    )


if validation_rows == 0:

    validation_errors.append(
        "VALIDATION está vacío"
    )


if test_rows == 0:

    validation_errors.append(
        "TEST está vacío"
    )


if oot_rows == 0:

    validation_errors.append(
        "OUT-OF-TIME está vacío"
    )


# ------------------------------------------------------------
# No puede existir solapamiento temporal.
# ------------------------------------------------------------

if train_validation_overlap > 0:

    validation_errors.append(
        "Existe solapamiento entre TRAIN y VALIDATION"
    )


if validation_test_overlap > 0:

    validation_errors.append(
        "Existe solapamiento entre VALIDATION y TEST"
    )


if test_oot_overlap > 0:

    validation_errors.append(
        "Existe solapamiento entre TEST y OUT-OF-TIME"
    )


# ------------------------------------------------------------
# Validamos el orden cronológico.
# ------------------------------------------------------------

if train_max >= validation_min:

    validation_errors.append(
        "TRAIN y VALIDATION no respetan el orden temporal"
    )


if validation_max >= test_min:

    validation_errors.append(
        "VALIDATION y TEST no respetan el orden temporal"
    )


if test_max >= oot_min:

    validation_errors.append(
        "TEST y OUT-OF-TIME no respetan el orden temporal"
    )


# ============================================================
# RESULTADO
# ============================================================

if validation_errors:

    print()
    print(
        "ERROR - Split temporal no válido"
    )

    for error in validation_errors:

        print(
            " -",
            error
        )

    raise RuntimeError(
        " | ".join(
            validation_errors
        )
    )


else:

    print()
    print(
        "OK - Split temporal validado correctamente"
    )

    print(
        "OK - Todos los registros están asignados"
    )

    print(
        "OK - No existen solapamientos"
    )

    print(
        "OK - Orden cronológico respetado"
    )

    print(
        "OK - OUT-OF-TIME aislado del desarrollo del modelo"
    )

# COMMAND ----------

# DBTITLE 1,04.02 VALIDACIÓN DEL SPLIT TEMPORAL
# ============================================================
# 04.02 VALIDACIÓN DEL SPLIT TEMPORAL
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Validar que:
#
# - TRAIN
# - VALIDATION
# - TEST
# - OUT-OF-TIME
#
# cumplen correctamente la separación temporal.
#
# Validaremos:
#
# - número de registros
# - rango temporal
# - cobertura de tiendas
# - orden cronológico
# - ausencia de solapamientos
# - conservación del 100 % de los registros
#
# IMPORTANTE
# ------------------------------------------------------------
#
# No hardcodeamos el número de tiendas.
#
# La referencia se obtiene dinámicamente desde ml_model_df.
#
# ============================================================


from pyspark.sql import functions as F


# ============================================================
# 04.02.01 FUNCIÓN DE RESUMEN
# ============================================================

def validate_temporal_dataset(df, dataset_name):

    summary = (

        df

        .agg(

            F.count("*").alias("records"),

            F.min("sale_date").alias("min_date"),

            F.max("sale_date").alias("max_date"),

            F.countDistinct("store_id").alias("stores")

        )

        .first()

    )

    print(

        f"{dataset_name:12} | "
        f"Registros: {summary['records']:>6,} | "
        f"Tiendas: {summary['stores']:>2} | "
        f"{summary['min_date']} -> {summary['max_date']}"

    )

    return summary


# ============================================================
# 04.02.02 NÚMERO ESPERADO DE TIENDAS
# ============================================================

expected_stores = (

    ml_model_df

    .select(
        "store_id"
    )

    .distinct()

    .count()

)


# ============================================================
# 04.02.03 RESUMEN DE LOS CUATRO PERIODOS
# ============================================================

print("=" * 90)
print("VALIDACIÓN DEL SPLIT TEMPORAL")
print("=" * 90)


train_summary = validate_temporal_dataset(
    train_df,
    "TRAIN"
)


validation_summary = validate_temporal_dataset(
    validation_df,
    "VALIDATION"
)


test_summary = validate_temporal_dataset(
    test_df,
    "TEST"
)


oot_summary = validate_temporal_dataset(
    oot_df,
    "OUT-OF-TIME"
)


# ============================================================
# 04.02.04 VALIDACIÓN DEL ORDEN TEMPORAL
# ============================================================

train_validation_ok = (

    train_summary["max_date"]
    <
    validation_summary["min_date"]

)


validation_test_ok = (

    validation_summary["max_date"]
    <
    test_summary["min_date"]

)


test_oot_ok = (

    test_summary["max_date"]
    <
    oot_summary["min_date"]

)


print()
print("=" * 90)
print("VALIDACIÓN DE ORDEN TEMPORAL")
print("=" * 90)


print(

    "OK"
    if train_validation_ok
    else "ERROR",

    "| TRAIN termina antes de VALIDATION"

)


print(

    "OK"
    if validation_test_ok
    else "ERROR",

    "| VALIDATION termina antes de TEST"

)


print(

    "OK"
    if test_oot_ok
    else "ERROR",

    "| TEST termina antes de OUT-OF-TIME"

)


# ============================================================
# 04.02.05 VALIDACIÓN DE COBERTURA DE TIENDAS
# ============================================================

print()
print("=" * 90)
print("VALIDACIÓN DE COBERTURA DE TIENDAS")
print("=" * 90)

print(
    f"Tiendas esperadas: {expected_stores}"
)


all_store_coverage_ok = True


for name, summary in [

    ("TRAIN", train_summary),

    ("VALIDATION", validation_summary),

    ("TEST", test_summary),

    ("OUT-OF-TIME", oot_summary)

]:

    store_ok = (
        summary["stores"]
        ==
        expected_stores
    )


    if not store_ok:

        all_store_coverage_ok = False


    status = (
        "OK"
        if store_ok
        else "ERROR"
    )


    print(

        f"{status:5} | "
        f"{name:12} | "
        f"{summary['stores']} tiendas"

    )


# ============================================================
# 04.02.06 VALIDACIÓN DE CONSERVACIÓN DE REGISTROS
# ============================================================

split_total = (

    train_summary["records"]

    +

    validation_summary["records"]

    +

    test_summary["records"]

    +

    oot_summary["records"]

)


original_total = (
    ml_model_df
    .count()
)


records_ok = (
    split_total
    ==
    original_total
)


print()
print("=" * 90)
print("VALIDACIÓN DE REGISTROS")
print("=" * 90)


print(
    f"Dataset ML original: {original_total:,}"
)


print(
    f"Suma de los splits:  {split_total:,}"
)


print(
    f"Diferencia:          {original_total - split_total:,}"
)


print(

    "OK | No se han perdido registros"

    if records_ok

    else

    "ERROR | Los recuentos no coinciden"

)


# ============================================================
# 04.02.07 VALIDACIÓN DE SOLAPAMIENTOS
# ============================================================
#
# Aunque el orden temporal ya debería impedirlos,
# hacemos una comprobación explícita utilizando
# store_id + sale_date.
#
# ============================================================

def count_overlap(df_left, df_right):

    return (

        df_left

        .select(
            "store_id",
            "sale_date"
        )

        .join(

            df_right.select(
                "store_id",
                "sale_date"
            ),

            [
                "store_id",
                "sale_date"
            ],

            "inner"

        )

        .count()

    )


train_validation_overlap = count_overlap(
    train_df,
    validation_df
)


validation_test_overlap = count_overlap(
    validation_df,
    test_df
)


test_oot_overlap = count_overlap(
    test_df,
    oot_df
)


overlap_ok = (

    train_validation_overlap == 0

    and

    validation_test_overlap == 0

    and

    test_oot_overlap == 0

)


print()
print("=" * 90)
print("VALIDACIÓN DE SOLAPAMIENTOS")
print("=" * 90)


print(
    f"TRAIN / VALIDATION: {train_validation_overlap:,}"
)


print(
    f"VALIDATION / TEST:  {validation_test_overlap:,}"
)


print(
    f"TEST / OUT-OF-TIME: {test_oot_overlap:,}"
)


print(

    "OK | No existen solapamientos"

    if overlap_ok

    else

    "ERROR | Existen solapamientos entre datasets"

)


# ============================================================
# 04.02.08 VALIDACIÓN FINAL
# ============================================================

validation_errors = []


if not train_validation_ok:

    validation_errors.append(
        "TRAIN y VALIDATION no respetan el orden temporal"
    )


if not validation_test_ok:

    validation_errors.append(
        "VALIDATION y TEST no respetan el orden temporal"
    )


if not test_oot_ok:

    validation_errors.append(
        "TEST y OUT-OF-TIME no respetan el orden temporal"
    )


if not all_store_coverage_ok:

    validation_errors.append(
        "Algún periodo no contiene todas las tiendas esperadas"
    )


if not records_ok:

    validation_errors.append(
        "La suma de los splits no coincide con ml_model_df"
    )


if not overlap_ok:

    validation_errors.append(
        "Existen registros solapados entre periodos"
    )


# ============================================================
# RESULTADO
# ============================================================

if validation_errors:

    print()
    print("=" * 90)
    print("ERROR - VALIDACIÓN DEL SPLIT TEMPORAL")
    print("=" * 90)

    for error in validation_errors:

        print(
            " -",
            error
        )

    raise RuntimeError(
        " | ".join(
            validation_errors
        )
    )


else:

    print()
    print("=" * 90)
    print("VALIDACIÓN FINAL")
    print("=" * 90)

    print(
        "OK - Split temporal completamente validado"
    )

    print(
        "OK - Cobertura de tiendas correcta"
    )

    print(
        "OK - No se han perdido registros"
    )

    print(
        "OK - No existen solapamientos"
    )

    print(
        "OK - OUT-OF-TIME permanece aislado"
    )

    print(
        "OK - Dataset preparado para Baseline Model"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC # 05. BASELINE MODEL
# MAGIC
# MAGIC ## Objetivo
# MAGIC
# MAGIC Establecer un modelo de referencia sencillo contra el que comparar posteriormente los modelos de Machine Learning.
# MAGIC
# MAGIC En problemas de forecasting no basta con obtener buenas métricas. El modelo debe demostrar que aporta valor frente a una estrategia simple.
# MAGIC
# MAGIC ## Baseline seleccionado: Seasonal Naive
# MAGIC
# MAGIC Utilizaremos como predicción las ventas observadas en la misma tienda **7 días antes**:
# MAGIC
# MAGIC **Predicción(t) = Ventas(t - 7)**
# MAGIC
# MAGIC Esta estrategia utiliza la feature `lag_7` y tiene sentido en un contexto Retail porque permite capturar de forma básica la estacionalidad semanal.
# MAGIC
# MAGIC Ejemplo:
# MAGIC
# MAGIC Ventas del lunes anterior → Predicción para este lunes
# MAGIC
# MAGIC ## Evaluación
# MAGIC
# MAGIC El Baseline se evaluará inicialmente sobre el conjunto de **Validation**, manteniendo el conjunto Test reservado para la evaluación final.
# MAGIC
# MAGIC Utilizaremos tres métricas:
# MAGIC
# MAGIC - **MAE** → Error absoluto medio.
# MAGIC - **RMSE** → Penaliza especialmente los errores grandes.
# MAGIC - **WAPE** → Error absoluto total respecto al volumen total de ventas.
# MAGIC
# MAGIC Estas métricas constituirán la referencia mínima que deberán superar los modelos de Machine Learning posteriores.

# COMMAND ----------

# DBTITLE 1,05.01 BASELINE MODEL - SEASONAL NAIVE
# ============================================================
# 05.01 BASELINE MODEL - SEASONAL NAIVE
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Crear una predicción de referencia utilizando las ventas
# observadas 7 días antes en cada tienda.
#
# REAL:
#     net_sales
#
# PREDICCIÓN:
#     lag_7
#
# El baseline se evalúa únicamente sobre VALIDATION.
#
# TEST:
#     reservado para la evaluación final del modelo elegido.
#
# OUT-OF-TIME:
#     reservado para datos futuros posteriores al desarrollo
#     y selección del modelo.
#
# ============================================================


from pyspark.sql import functions as F


# ============================================================
# 05.01.01 CREACIÓN DE LAS PREDICCIONES BASELINE
# ============================================================

baseline_validation_df = (

    validation_df

    .select(

        "sale_date",

        "store_id",

        F.col(
            "net_sales"
        ).alias(
            "actual"
        ),

        F.col(
            "lag_7"
        ).alias(
            "prediction"
        )

    )

)


# ============================================================
# 05.01.02 VALIDACIONES PREVIAS
# ============================================================

validation_rows = (
    validation_df
    .count()
)


baseline_rows = (
    baseline_validation_df
    .count()
)


null_actual_rows = (

    baseline_validation_df

    .filter(
        F.col("actual").isNull()
    )

    .count()

)


null_prediction_rows = (

    baseline_validation_df

    .filter(
        F.col("prediction").isNull()
    )

    .count()

)


duplicate_rows = (

    baseline_validation_df

    .groupBy(
        "store_id",
        "sale_date"
    )

    .count()

    .filter(
        F.col("count") > 1
    )

    .count()

)


# ============================================================
# 05.01.03 CÁLCULO DEL ERROR POR OBSERVACIÓN
# ============================================================

baseline_validation_df = (

    baseline_validation_df

    .withColumn(

        "error",

        F.col("actual")
        -
        F.col("prediction")

    )

    .withColumn(

        "absolute_error",

        F.abs(
            F.col("error")
        )

    )

    .withColumn(

        "squared_error",

        F.pow(
            F.col("error"),
            2
        )

    )

)


# ============================================================
# 05.01.04 CÁLCULO DE MÉTRICAS
# ============================================================
#
# MAE
# ------------------------------------------------------------
# Error absoluto medio.
#
# RMSE
# ------------------------------------------------------------
# Penaliza más los errores grandes.
#
# WAPE
# ------------------------------------------------------------
#
#     SUM(|actual - prediction|)
#     -------------------------- × 100
#          SUM(|actual|)
#
# Resulta especialmente útil para interpretar el error
# respecto al volumen total de ventas.
#
# ============================================================

baseline_metrics = (

    baseline_validation_df

    .agg(

        F.avg(
            "absolute_error"
        ).alias(
            "mae"
        ),

        F.sqrt(

            F.avg(
                "squared_error"
            )

        ).alias(
            "rmse"
        ),

        F.sum(
            "absolute_error"
        ).alias(
            "total_absolute_error"
        ),

        F.sum(
            F.abs("actual")
        ).alias(
            "total_actual"
        )

    )

    .first()

)


baseline_mae = baseline_metrics["mae"]

baseline_rmse = baseline_metrics["rmse"]

baseline_total_absolute_error = (
    baseline_metrics["total_absolute_error"]
)

baseline_total_actual = (
    baseline_metrics["total_actual"]
)


# ------------------------------------------------------------
# WAPE
# ------------------------------------------------------------
#
# Evitamos una división por cero.
#
# ============================================================

if (
    baseline_total_actual is not None
    and
    baseline_total_actual > 0
):

    baseline_wape = (

        baseline_total_absolute_error
        /
        baseline_total_actual
        *
        100

    )

else:

    baseline_wape = None


# ============================================================
# 05.01.05 VALIDACIÓN DE MÉTRICAS
# ============================================================

validation_errors = []


if baseline_rows != validation_rows:

    validation_errors.append(

        f"El baseline contiene {baseline_rows} registros, "
        f"pero VALIDATION contiene {validation_rows}"

    )


if null_actual_rows > 0:

    validation_errors.append(

        f"Existen {null_actual_rows} valores NULL "
        "en actual"

    )


if null_prediction_rows > 0:

    validation_errors.append(

        f"Existen {null_prediction_rows} valores NULL "
        "en prediction"

    )


if duplicate_rows > 0:

    validation_errors.append(

        f"Existen {duplicate_rows} claves "
        "store_id + sale_date duplicadas"

    )


if baseline_mae is None:

    validation_errors.append(
        "No se ha podido calcular MAE"
    )


if baseline_rmse is None:

    validation_errors.append(
        "No se ha podido calcular RMSE"
    )


if baseline_wape is None:

    validation_errors.append(

        "No se ha podido calcular WAPE porque "
        "el volumen real agregado es 0"

    )


# ============================================================
# 05.01.06 RESULTADOS
# ============================================================

print("=" * 80)
print("BASELINE MODEL - SEASONAL NAIVE (LAG 7)")
print("=" * 80)


print(
    f"Registros VALIDATION: {validation_rows:,}"
)


print(
    f"Registros evaluados:  {baseline_rows:,}"
)


print(
    f"Actual NULL:           {null_actual_rows:,}"
)


print(
    f"Prediction NULL:       {null_prediction_rows:,}"
)


print(
    f"Duplicados:            {duplicate_rows:,}"
)


print()
print("=" * 80)
print("MÉTRICAS BASELINE")
print("=" * 80)


if baseline_mae is not None:

    print(
        f"MAE:  {baseline_mae:,.2f}"
    )


if baseline_rmse is not None:

    print(
        f"RMSE: {baseline_rmse:,.2f}"
    )


if baseline_wape is not None:

    print(
        f"WAPE: {baseline_wape:,.2f}%"
    )


# ============================================================
# 05.01.07 RESULTADO FINAL
# ============================================================

if validation_errors:

    print()
    print("=" * 80)
    print("ERROR - BASELINE NO VÁLIDO")
    print("=" * 80)

    for error in validation_errors:

        print(
            " -",
            error
        )

    raise RuntimeError(
        " | ".join(
            validation_errors
        )
    )


else:

    print()
    print("=" * 80)
    print("VALIDACIÓN FINAL")
    print("=" * 80)

    print(
        "OK - Baseline Seasonal Naive calculado correctamente"
    )

    print(
        "OK - Todos los registros de VALIDATION han sido evaluados"
    )

    print(
        "OK - No existen NULLs en actual ni prediction"
    )

    print(
        "OK - No existen duplicados tienda-día"
    )

    print(
        "OK - TEST permanece reservado"
    )

    print(
        "OK - OUT-OF-TIME permanece aislado"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC # 06. PRIMER MODELO ML - RANDOM FOREST REGRESSOR
# MAGIC
# MAGIC ## Objetivo
# MAGIC
# MAGIC Entrenar el primer modelo de Machine Learning capaz de predecir las ventas diarias de cada tienda.
# MAGIC
# MAGIC Utilizaremos un **Random Forest Regressor**.
# MAGIC
# MAGIC Random Forest combina múltiples árboles de decisión para generar una predicción final más robusta que la obtenida mediante un único árbol.
# MAGIC
# MAGIC ## Variable objetivo
# MAGIC
# MAGIC `net_sales`
# MAGIC
# MAGIC Representa las ventas netas de una tienda para un día determinado.
# MAGIC
# MAGIC ## Features utilizadas
# MAGIC
# MAGIC ### Información temporal conocida
# MAGIC
# MAGIC - `year`
# MAGIC - `month`
# MAGIC - `day`
# MAGIC - `day_of_week`
# MAGIC - `week_of_year`
# MAGIC - `is_weekend`
# MAGIC
# MAGIC ### Histórico de ventas
# MAGIC
# MAGIC - `lag_1`
# MAGIC - `lag_7`
# MAGIC - `lag_14`
# MAGIC - `lag_28`
# MAGIC - `rolling_mean_7`
# MAGIC - `rolling_mean_28`
# MAGIC - `rolling_std_7`
# MAGIC - `diff_7`
# MAGIC - `lag1_vs_mean7`
# MAGIC
# MAGIC ### Información de tienda
# MAGIC
# MAGIC - `store_id`
# MAGIC
# MAGIC ## Prevención de Data Leakage
# MAGIC
# MAGIC No utilizaremos como features:
# MAGIC
# MAGIC - `gross_sales`
# MAGIC - `total_discount`
# MAGIC - `total_units`
# MAGIC - `total_tickets`
# MAGIC
# MAGIC Estas variables pertenecen al mismo día que queremos predecir y no serían conocidas en el momento real de realizar el forecast.
# MAGIC
# MAGIC ## Estrategia de evaluación
# MAGIC
# MAGIC El modelo se entrenará exclusivamente con **Train**.
# MAGIC
# MAGIC Las predicciones se realizarán sobre **Validation** y se compararán contra nuestro baseline:
# MAGIC
# MAGIC **Seasonal Naive (`lag_7`)**
# MAGIC
# MAGIC Baseline actual:
# MAGIC
# MAGIC - MAE: **2.970,72**
# MAGIC - RMSE: **3.808,58**
# MAGIC - WAPE: **49,18 %**
# MAGIC
# MAGIC El conjunto Test seguirá completamente reservado.

# COMMAND ----------

# DBTITLE 1,06.01 RANDOM FOREST REGRESSOR
# ============================================================
# 06.01 RANDOM FOREST REGRESSOR
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Entrenar nuestro primer modelo de Machine Learning utilizando:
#
#       TRAIN
#
# y evaluarlo posteriormente utilizando:
#
#       VALIDATION
#
# TEST permanece completamente reservado.
#
# OUT-OF-TIME permanece aislado del desarrollo del modelo.
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
# 06.01.01 DEFINICIÓN DE FEATURES
# ============================================================


# ------------------------------------------------------------
# FEATURES NUMÉRICAS
# ------------------------------------------------------------
#
# Todas estas variables deben ser conocidas antes del día
# que queremos predecir.
#
# ------------------------------------------------------------

numeric_features = [

    # Variables temporales
    "year",
    "month",
    "day",
    "day_of_week",
    "week_of_year",
    "is_weekend",

    # Histórico de ventas
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",

    # Comportamiento reciente
    "rolling_mean_7",
    "rolling_mean_28",
    "rolling_std_7",

    # Tendencias
    "lag1_minus_lag7",
    "lag1_vs_mean7"
]


# ------------------------------------------------------------
# FEATURE CATEGÓRICA
# ------------------------------------------------------------
#
# Cada tienda puede presentar un comportamiento estructural
# diferente.
#
# Incorporamos store_id como variable categórica para que
# el modelo pueda capturar diferencias entre tiendas.
#
# ------------------------------------------------------------

categorical_features = [
    "store_id"
]


# ============================================================
# 06.01.02 VALIDACIÓN PREVIA DE COLUMNAS
# ============================================================
#
# Antes de construir el Pipeline comprobamos que todas las
# features necesarias existen realmente en TRAIN.
#
# ============================================================

required_columns = (

    numeric_features

    +

    categorical_features

    +

    [
        "net_sales"
    ]
)


missing_columns = [

    column

    for column in required_columns

    if column not in train_df.columns
]


if missing_columns:

    raise ValueError(

        "Faltan columnas necesarias para entrenar el modelo: "
        +
        ", ".join(missing_columns)

    )


print("=" * 80)
print("VALIDACIÓN DE FEATURES - RANDOM FOREST")
print("=" * 80)

print(
    f"Features numéricas:   {len(numeric_features)}"
)

print(
    f"Features categóricas: {len(categorical_features)}"
)

print(
    "OK - Todas las columnas necesarias existen"
)


# ============================================================
# 06.01.03 VALIDACIÓN DE NULLS EN TRAIN
# ============================================================
#
# El Feature Engineering anterior debería haber eliminado
# todos los NULLs de las features.
#
# Lo verificamos de nuevo antes de entrenar.
#
# ============================================================

null_expressions = [

    F.sum(

        F.when(
            F.col(column).isNull(),
            1
        )

        .otherwise(
            0
        )

    ).alias(column)

    for column in required_columns

]


null_summary = (

    train_df

    .agg(
        *null_expressions
    )

    .first()

)


columns_with_nulls = {

    column: null_summary[column]

    for column in required_columns

    if null_summary[column] > 0

}


if columns_with_nulls:

    print()
    print(
        "ERROR - Se han detectado NULLs:"
    )

    for column, null_count in columns_with_nulls.items():

        print(
            f" - {column}: {null_count:,}"
        )

    raise ValueError(
        "TRAIN contiene NULLs en las features o en el target"
    )


print(
    "OK - TRAIN no contiene NULLs en features ni target"
)


# ============================================================
# 06.01.04 STRING INDEXER
# ============================================================
#
# Spark ML no trabaja directamente con strings.
#
# Transformamos:
#
#       S001
#       S002
#       S003
#
# en índices numéricos.
#
# handleInvalid="keep"
#
# permite manejar categorías no vistas durante una futura
# transformación.
#
# ============================================================

store_indexer = StringIndexer(

    inputCol="store_id",

    outputCol="store_id_index",

    handleInvalid="keep"

)


# ============================================================
# 06.01.05 ONE HOT ENCODING
# ============================================================
#
# No queremos que el modelo interprete:
#
#       S003 > S002 > S001
#
# porque store_id no representa una magnitud ordinal.
#
# OneHotEncoder crea una representación categórica.
#
# ============================================================

store_encoder = OneHotEncoder(

    inputCol="store_id_index",

    outputCol="store_id_encoded",

    handleInvalid="keep"

)


# ============================================================
# 06.01.06 VECTOR ASSEMBLER
# ============================================================
#
# Spark ML requiere que todas las variables predictoras
# estén agrupadas en una única columna vectorial:
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
# 06.01.07 RANDOM FOREST REGRESSOR
# ============================================================
#
# PRIMER MODELO
# ------------------------------------------------------------
#
# Todavía NO hacemos tuning.
#
# El objetivo es obtener un primer modelo ML razonable y
# compararlo con el baseline Seasonal Naive.
#
# CONFIGURACIÓN
# ------------------------------------------------------------
#
# numTrees = 60
#
#       Número moderado de árboles.
#
# maxDepth = 8
#
#       Limita la complejidad de cada árbol.
#
# minInstancesPerNode = 3
#
#       Evita nodos excesivamente específicos.
#
# seed = 42
#
#       Permite reproducibilidad.
#
# Esta configuración también reduce el consumo de recursos
# frente a un Random Forest más grande.
#
# ============================================================

rf = RandomForestRegressor(

    featuresCol="features",

    labelCol="net_sales",

    predictionCol="prediction",

    numTrees=60,

    maxDepth=8,

    minInstancesPerNode=3,

    seed=42

)


# ============================================================
# 06.01.08 PIPELINE DE MACHINE LEARNING
# ============================================================
#
# Flujo:
#
# store_id
#     ↓
# StringIndexer
#     ↓
# OneHotEncoder
#     ↓
# VectorAssembler
#     ↓
# RandomForest
#
# El StringIndexer y el OneHotEncoder se ajustarán
# exclusivamente utilizando TRAIN.
#
# ============================================================

rf_pipeline = Pipeline(

    stages=[

        store_indexer,

        store_encoder,

        assembler,

        rf

    ]

)


# ============================================================
# 06.01.09 RESUMEN ANTES DEL ENTRENAMIENTO
# ============================================================

train_rows_rf = (
    train_df
    .count()
)


train_stores_rf = (

    train_df

    .select(
        "store_id"
    )

    .distinct()

    .count()

)


train_dates_rf = (

    train_df

    .agg(

        F.min("sale_date").alias("min_date"),

        F.max("sale_date").alias("max_date")

    )

    .first()

)


print()
print("=" * 80)
print("DATASET DE ENTRENAMIENTO")
print("=" * 80)


print(
    f"Registros: {train_rows_rf:,}"
)


print(
    f"Tiendas:   {train_stores_rf:,}"
)


print(
    f"Periodo:   "
    f"{train_dates_rf['min_date']} "
    f"-> "
    f"{train_dates_rf['max_date']}"
)


# ============================================================
# 06.01.10 ENTRENAMIENTO
# ============================================================
#
# IMPORTANTE
# ------------------------------------------------------------
#
# Únicamente utilizamos TRAIN.
#
# VALIDATION:
#       se utilizará posteriormente para evaluar el modelo.
#
# TEST:
#       continúa completamente reservado.
#
# OUT-OF-TIME:
#       continúa completamente aislado.
#
# ============================================================

print()
print("=" * 80)
print("ENTRENAMIENTO RANDOM FOREST")
print("=" * 80)


rf_model = rf_pipeline.fit(
    train_df
)


print(
    "OK - Random Forest entrenado correctamente"
)


# ============================================================
# 06.01.11 VALIDACIÓN FINAL
# ============================================================

if rf_model is None:

    raise RuntimeError(
        "El Pipeline no ha generado un modelo entrenado"
    )


print()
print("=" * 80)
print("VALIDACIÓN FINAL")
print("=" * 80)

print(
    "OK - Pipeline creado correctamente"
)

print(
    "OK - Random Forest entrenado únicamente con TRAIN"
)

print(
    "OK - VALIDATION no ha participado en el entrenamiento"
)

print(
    "OK - TEST permanece reservado"
)

print(
    "OK - OUT-OF-TIME permanece aislado"
)

print(
    "OK - Modelo preparado para evaluación en VALIDATION"
)

# COMMAND ----------

# DBTITLE 1,06.02 EVALUACIÓN RANDOM FOREST SOBRE VALIDATION
# ============================================================
# 06.02 EVALUACIÓN RANDOM FOREST SOBRE VALIDATION
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Evaluar el Random Forest exclusivamente sobre VALIDATION
# y compararlo contra el Seasonal Naive Baseline.
#
# Métricas:
#
# - MAE
# - RMSE
# - WAPE
#
# IMPORTANTE
# ------------------------------------------------------------
#
# TEST y OUT-OF-TIME continúan completamente reservados.
#
# Las métricas del Baseline NO se hardcodean.
# Utilizamos directamente las calculadas en 05.01.
#
# ============================================================


from pyspark.sql import functions as F


# ============================================================
# 06.02.01 GENERAMOS PREDICCIONES
# ============================================================

rf_validation_predictions_df = (

    rf_model

    .transform(
        validation_df
    )

    .select(

        "sale_date",

        "store_id",

        F.col(
            "net_sales"
        ).alias(
            "actual"
        ),

        F.col(
            "prediction"
        )

    )

)


# ============================================================
# 06.02.02 VALIDACIÓN DE LAS PREDICCIONES
# ============================================================

validation_rows_rf = (
    validation_df
    .count()
)


prediction_rows_rf = (
    rf_validation_predictions_df
    .count()
)


null_actual_rf = (

    rf_validation_predictions_df

    .filter(
        F.col("actual").isNull()
    )

    .count()

)


null_prediction_rf = (

    rf_validation_predictions_df

    .filter(
        F.col("prediction").isNull()
    )

    .count()

)


duplicate_predictions_rf = (

    rf_validation_predictions_df

    .groupBy(
        "store_id",
        "sale_date"
    )

    .count()

    .filter(
        F.col("count") > 1
    )

    .count()

)


# ============================================================
# 06.02.03 CÁLCULO DEL ERROR
# ============================================================

rf_validation_predictions_df = (

    rf_validation_predictions_df

    .withColumn(

        "error",

        F.col("actual")
        -
        F.col("prediction")

    )

    .withColumn(

        "absolute_error",

        F.abs(
            F.col("error")
        )

    )

    .withColumn(

        "squared_error",

        F.pow(
            F.col("error"),
            2
        )

    )

)


# ============================================================
# 06.02.04 MÉTRICAS RANDOM FOREST
# ============================================================

rf_metrics = (

    rf_validation_predictions_df

    .agg(

        # MAE
        F.avg(
            "absolute_error"
        ).alias(
            "mae"
        ),

        # RMSE
        F.sqrt(

            F.avg(
                "squared_error"
            )

        ).alias(
            "rmse"
        ),

        # Numerador WAPE
        F.sum(
            "absolute_error"
        ).alias(
            "total_absolute_error"
        ),

        # Denominador WAPE
        F.sum(
            F.abs("actual")
        ).alias(
            "total_actual"
        )

    )

    .first()

)


# ============================================================
# 06.02.05 NORMALIZACIÓN DE TIPOS
# ============================================================
#
# Spark puede devolver agregaciones como DecimalType.
#
# Python no permite operar directamente entre:
#
#       Decimal
#       float
#
# Por tanto, convertimos TODAS las métricas a float antes
# de realizar cálculos posteriores.
#
# ============================================================

rf_mae = (

    float(rf_metrics["mae"])

    if rf_metrics["mae"] is not None

    else None

)


rf_rmse = (

    float(rf_metrics["rmse"])

    if rf_metrics["rmse"] is not None

    else None

)


rf_total_absolute_error = (

    float(
        rf_metrics["total_absolute_error"]
    )

    if rf_metrics["total_absolute_error"] is not None

    else None

)


rf_total_actual = (

    float(
        rf_metrics["total_actual"]
    )

    if rf_metrics["total_actual"] is not None

    else None

)


# ------------------------------------------------------------
# NORMALIZAMOS TAMBIÉN LAS MÉTRICAS DEL BASELINE
# ------------------------------------------------------------
#
# baseline_mae
# baseline_rmse
# baseline_wape
#
# proceden del bloque 05.01.
#
# ============================================================

baseline_mae = (

    float(baseline_mae)

    if baseline_mae is not None

    else None

)


baseline_rmse = (

    float(baseline_rmse)

    if baseline_rmse is not None

    else None

)


baseline_wape = (

    float(baseline_wape)

    if baseline_wape is not None

    else None

)


# ============================================================
# 06.02.06 CÁLCULO DE WAPE
# ============================================================

if (
    rf_total_actual is not None
    and
    rf_total_actual > 0
    and
    rf_total_absolute_error is not None
):

    rf_wape = (

        rf_total_absolute_error
        /
        rf_total_actual
        *
        100.0

    )

else:

    rf_wape = None


# ============================================================
# 06.02.07 VALIDACIONES TÉCNICAS
# ============================================================

validation_errors = []


if prediction_rows_rf != validation_rows_rf:

    validation_errors.append(

        f"Se esperaban {validation_rows_rf} predicciones "
        f"y se han obtenido {prediction_rows_rf}"

    )


if null_actual_rf > 0:

    validation_errors.append(

        f"Existen {null_actual_rf} valores NULL "
        "en actual"

    )


if null_prediction_rf > 0:

    validation_errors.append(

        f"Existen {null_prediction_rf} valores NULL "
        "en prediction"

    )


if duplicate_predictions_rf > 0:

    validation_errors.append(

        f"Existen {duplicate_predictions_rf} claves "
        "store_id + sale_date duplicadas"

    )


if rf_mae is None:

    validation_errors.append(
        "No se ha podido calcular MAE"
    )


if rf_rmse is None:

    validation_errors.append(
        "No se ha podido calcular RMSE"
    )


if rf_wape is None:

    validation_errors.append(
        "No se ha podido calcular WAPE"
    )


if baseline_mae is None:

    validation_errors.append(
        "No existe baseline_mae calculado"
    )


if baseline_rmse is None:

    validation_errors.append(
        "No existe baseline_rmse calculado"
    )


if baseline_wape is None:

    validation_errors.append(
        "No existe baseline_wape calculado"
    )


# ============================================================
# 06.02.08 RESULTADOS RANDOM FOREST
# ============================================================

print("=" * 80)
print("RANDOM FOREST - VALIDATION")
print("=" * 80)


print(
    f"Registros VALIDATION: {validation_rows_rf:,}"
)


print(
    f"Predicciones:         {prediction_rows_rf:,}"
)


print(
    f"Actual NULL:          {null_actual_rf:,}"
)


print(
    f"Prediction NULL:      {null_prediction_rf:,}"
)


print(
    f"Duplicados:           {duplicate_predictions_rf:,}"
)


print()
print("=" * 80)
print("MÉTRICAS RANDOM FOREST")
print("=" * 80)


if rf_mae is not None:

    print(
        f"MAE:  {rf_mae:,.2f}"
    )


if rf_rmse is not None:

    print(
        f"RMSE: {rf_rmse:,.2f}"
    )


if rf_wape is not None:

    print(
        f"WAPE: {rf_wape:,.2f}%"
    )


# ============================================================
# 06.02.09 COMPARACIÓN CONTRA BASELINE
# ============================================================

print()
print("=" * 80)
print("COMPARACIÓN CONTRA BASELINE")
print("=" * 80)


# ============================================================
# MEJORAS RELATIVAS
# ============================================================
#
# Un valor positivo significa que Random Forest mejora
# respecto al Baseline.
#
# ============================================================

mae_improvement = (

    (
        baseline_mae
        -
        rf_mae
    )

    /
    baseline_mae

    *
    100.0

)


rmse_improvement = (

    (
        baseline_rmse
        -
        rf_rmse
    )

    /
    baseline_rmse

    *
    100.0

)


wape_improvement = (

    (
        baseline_wape
        -
        rf_wape
    )

    /
    baseline_wape

    *
    100.0

)


# ============================================================
# TABLA DE COMPARACIÓN
# ============================================================

print(

    f"{'Métrica':10} | "
    f"{'Baseline':>14} | "
    f"{'Random Forest':>14} | "
    f"{'Mejora':>10}"

)


print("-" * 68)


print(

    f"{'MAE':10} | "
    f"{baseline_mae:>14,.2f} | "
    f"{rf_mae:>14,.2f} | "
    f"{mae_improvement:>9.2f}%"

)


print(

    f"{'RMSE':10} | "
    f"{baseline_rmse:>14,.2f} | "
    f"{rf_rmse:>14,.2f} | "
    f"{rmse_improvement:>9.2f}%"

)


print(

    f"{'WAPE':10} | "
    f"{baseline_wape:>13,.2f}% | "
    f"{rf_wape:>13,.2f}% | "
    f"{wape_improvement:>9.2f}%"

)


# ============================================================
# 06.02.10 ¿SUPERA RANDOM FOREST AL BASELINE?
# ============================================================

rf_beats_baseline_mae = (
    rf_mae
    <
    baseline_mae
)


rf_beats_baseline_rmse = (
    rf_rmse
    <
    baseline_rmse
)


rf_beats_baseline_wape = (
    rf_wape
    <
    baseline_wape
)


rf_beats_baseline_all = (

    rf_beats_baseline_mae

    and

    rf_beats_baseline_rmse

    and

    rf_beats_baseline_wape

)


# ============================================================
# 06.02.11 RESULTADO FINAL
# ============================================================

if validation_errors:

    print()
    print("=" * 80)
    print("ERROR - EVALUACIÓN RANDOM FOREST NO VÁLIDA")
    print("=" * 80)

    for error in validation_errors:

        print(
            " -",
            error
        )

    raise RuntimeError(
        " | ".join(
            validation_errors
        )
    )


else:

    print()
    print("=" * 80)
    print("VALIDACIÓN FINAL")
    print("=" * 80)

    print(
        "OK - Predicciones generadas correctamente"
    )

    print(
        "OK - Todos los registros de VALIDATION han sido evaluados"
    )

    print(
        "OK - No existen NULLs ni duplicados"
    )


    if rf_beats_baseline_all:

        print(
            "OK - Random Forest supera al Baseline "
            "en MAE, RMSE y WAPE"
        )

    else:

        print(
            "INFO - Random Forest no supera al Baseline "
            "en todas las métricas"
        )


    print(
        "OK - TEST permanece reservado"
    )

    print(
        "OK - OUT-OF-TIME permanece aislado"
    )

# COMMAND ----------

# DBTITLE 1,06.03 FEATURE IMPORTANCE - RANDOM FOREST
# ============================================================
# 06.03 FEATURE IMPORTANCE - RANDOM FOREST
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Analizar qué variables están teniendo mayor peso dentro
# del Random Forest.
#
# Esto nos permite:
#
# - interpretar el modelo
# - comprobar si las features tienen sentido
# - detectar variables poco útiles
# - preparar posteriores iteraciones del modelo
#
# IMPORTANTE
# ------------------------------------------------------------
#
# store_id ha sido transformado mediante OneHotEncoder.
#
# Por tanto, dentro del vector "features" existirán varias
# posiciones asociadas a la tienda.
#
# En este bloque recuperamos las importancias INDIVIDUALES.
#
# En 06.04 las agruparemos nuevamente bajo "store_id".
#
# ============================================================


import pandas as pd


# ============================================================
# 06.03.01 RECUPERAMOS EL MODELO RANDOM FOREST
# ============================================================
#
# Pipeline:
#
# 0 -> StringIndexer
# 1 -> OneHotEncoder
# 2 -> VectorAssembler
# 3 -> RandomForestRegressor
#
# ============================================================

if "rf_model" not in globals():

    raise RuntimeError(
        "rf_model no existe. "
        "Ejecuta primero el bloque 06.01."
    )


rf_stage_model = (
    rf_model
    .stages[-1]
)


# ============================================================
# 06.03.02 RECUPERAMOS FEATURE IMPORTANCES
# ============================================================

feature_importances = (

    rf_stage_model

    .featureImportances

    .toArray()

)


number_of_importances = len(
    feature_importances
)


# ============================================================
# 06.03.03 RECUPERAMOS METADATA DEL VECTOR FEATURES
# ============================================================
#
# VectorAssembler guarda información sobre cada posición
# del vector resultante.
#
# Recuperamos esa metadata para reconstruir la relación:
#
#       índice del vector
#               ↓
#       nombre de feature
#
# ============================================================

transformed_sample_df = (

    rf_model

    .transform(
        train_df.limit(1)
    )

)


features_metadata = (

    transformed_sample_df

    .schema[
        "features"
    ]

    .metadata

)


# ============================================================
# 06.03.04 VALIDAMOS METADATA
# ============================================================

if "ml_attr" not in features_metadata:

    raise RuntimeError(
        "No existe metadata ml_attr en la columna features."
    )


ml_attr_metadata = (
    features_metadata[
        "ml_attr"
    ]
)


if "attrs" not in ml_attr_metadata:

    raise RuntimeError(
        "No existen atributos de features en la metadata."
    )


attrs = (
    ml_attr_metadata[
        "attrs"
    ]
)


# ============================================================
# 06.03.05 EXTRAEMOS NOMBRES DE FEATURES
# ============================================================
#
# Spark puede separar los atributos en distintos grupos:
#
# - numeric
# - binary
# - nominal
#
# Recorremos todos los grupos y conservamos:
#
#       idx
#       name
#
# Después ordenamos por idx.
#
# ============================================================

feature_name_pairs = []


for attr_type in attrs:

    for attr in attrs[attr_type]:

        if (
            "idx" in attr
            and
            "name" in attr
        ):

            feature_name_pairs.append(
                (
                    int(attr["idx"]),
                    str(attr["name"])
                )
            )


feature_name_pairs = sorted(

    feature_name_pairs,

    key=lambda x: x[0]

)


feature_names = [

    name

    for idx, name in feature_name_pairs

]


number_of_feature_names = len(
    feature_names
)


# ============================================================
# 06.03.06 VALIDACIÓN NOMBRES VS IMPORTANCIAS
# ============================================================

print("=" * 80)
print("VALIDACIÓN FEATURE IMPORTANCE")
print("=" * 80)


print(
    f"Importancias RF:       {number_of_importances:,}"
)


print(
    f"Nombres recuperados:   {number_of_feature_names:,}"
)


if number_of_feature_names != number_of_importances:

    raise RuntimeError(

        "El número de nombres de features "
        f"({number_of_feature_names}) no coincide con "
        f"el número de importancias "
        f"({number_of_importances})."

    )


print(
    "OK - Cada importancia tiene una feature asociada"
)


# ============================================================
# 06.03.07 VALIDAMOS ÍNDICES DEL VECTOR
# ============================================================
#
# Los índices deberían formar:
#
#       0, 1, 2, ..., N-1
#
# ============================================================

feature_indices = [

    idx

    for idx, name in feature_name_pairs

]


expected_indices = list(
    range(number_of_importances)
)


indices_ok = (
    feature_indices
    ==
    expected_indices
)


if not indices_ok:

    raise RuntimeError(
        "Los índices recuperados de la metadata "
        "no coinciden con las posiciones del vector."
    )


print(
    "OK - Índices del vector correctamente ordenados"
)


# ============================================================
# 06.03.08 CONSTRUIMOS TABLA DE IMPORTANCIAS
# ============================================================

feature_importance_df = pd.DataFrame(

    {
        "feature": feature_names,

        "importance": feature_importances
    }

)


# ============================================================
# 06.03.09 ORDENAMOS POR IMPORTANCIA
# ============================================================

feature_importance_df = (

    feature_importance_df

    .sort_values(
        "importance",
        ascending=False
    )

    .reset_index(
        drop=True
    )

)


# ============================================================
# 06.03.10 IMPORTANCIA EN PORCENTAJE
# ============================================================

feature_importance_df[
    "importance_pct"
] = (

    feature_importance_df[
        "importance"
    ]

    *

    100.0

)


# ============================================================
# 06.03.11 VALIDAMOS SUMA DE IMPORTANCIAS
# ============================================================
#
# Las featureImportances de Spark Random Forest deberían
# sumar aproximadamente 1.
#
# ============================================================

total_importance = float(

    feature_importance_df[
        "importance"
    ].sum()

)


importance_sum_ok = (

    abs(
        total_importance
        -
        1.0
    )

    <
    0.0001

)


print(
    f"Suma importancias:     {total_importance:.6f}"
)


if not importance_sum_ok:

    raise RuntimeError(

        "La suma de las feature importances "
        f"es {total_importance:.6f} y debería ser "
        "aproximadamente 1.0."

    )


print(
    "OK - La suma de importancias es aproximadamente 100%"
)


# ============================================================
# 06.03.12 COMPROBAMOS FEATURES DE STORE_ID
# ============================================================

store_features_count = (

    feature_importance_df[
        "feature"
    ]

    .astype(str)

    .str.startswith(
        "store_id_encoded"
    )

    .sum()

)


print(
    f"Features derivadas de store_id: "
    f"{store_features_count:,}"
)


if store_features_count == 0:

    print(
        "INFO - No se han identificado nombres "
        "store_id_encoded en la metadata."
    )

else:

    print(
        "OK - Features OneHotEncoded de store_id identificadas"
    )


# ============================================================
# 06.03.13 VALIDACIÓN FINAL
# ============================================================

print()
print("=" * 80)
print("VALIDACIÓN FINAL")
print("=" * 80)


print(
    "OK - Feature importances recuperadas correctamente"
)


print(
    "OK - Correspondencia feature/importancia validada"
)


print(
    "OK - feature_importance_df creado correctamente"
)


print(
    "OK - Dataset preparado para agrupación por store_id"
)


# ============================================================
# 06.03.14 MOSTRAMOS TOP 20
# ============================================================

feature_importance_spark_df = (

    spark.createDataFrame(
        feature_importance_df
    )

)


display(

    feature_importance_spark_df

    .orderBy(
        F.desc("importance")
    )

    .limit(20)

)

# COMMAND ----------

# DBTITLE 1,06.04 IMPORTANCIA AGRUPADA DE STORE_ID
# ============================================================
# 06.04 IMPORTANCIA AGRUPADA DE STORE_ID
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# El OneHotEncoder transforma store_id en múltiples posiciones
# dentro del vector de features.
#
# Ejemplo conceptual:
#
#       store_id_encoded_S001
#       store_id_encoded_S002
#       store_id_encoded_S003
#       ...
#
# Para interpretar correctamente la importancia de la variable
# "tienda", agruparemos todas esas posiciones bajo:
#
#       store_id
#
# y sumaremos sus importancias.
#
# ============================================================


# ============================================================
# 06.04.01 VALIDACIÓN PREVIA
# ============================================================

if "feature_importance_df" not in globals():

    raise RuntimeError(
        "feature_importance_df no existe. "
        "Ejecuta primero el bloque que calcula "
        "las importancias individuales del Random Forest."
    )


required_importance_columns = [
    "feature",
    "importance"
]


missing_importance_columns = [

    column

    for column in required_importance_columns

    if column not in feature_importance_df.columns

]


if missing_importance_columns:

    raise RuntimeError(

        "Faltan columnas en feature_importance_df: "
        +
        ", ".join(missing_importance_columns)

    )


# ============================================================
# 06.04.02 COPIA DEL DATASET ORIGINAL
# ============================================================
#
# Trabajamos sobre una copia para no modificar el DataFrame
# generado en el análisis de importancia individual.
#
# ============================================================

feature_importance_grouped_df = (
    feature_importance_df
    .copy()
)


# ============================================================
# 06.04.03 AGRUPACIÓN DE FEATURES DE STORE_ID
# ============================================================
#
# Todas las posiciones generadas por OneHotEncoder que
# empiecen por:
#
#       store_id_encoded
#
# se agrupan bajo una única variable:
#
#       store_id
#
# ============================================================

feature_importance_grouped_df[
    "feature_group"
] = (

    feature_importance_grouped_df[
        "feature"
    ]

    .apply(

        lambda feature:

        "store_id"

        if str(feature).startswith(
            "store_id_encoded"
        )

        else str(feature)

    )

)


# ============================================================
# 06.04.04 SUMA DE IMPORTANCIAS POR GRUPO
# ============================================================

feature_importance_grouped_df = (

    feature_importance_grouped_df

    .groupby(
        "feature_group",
        as_index=False
    )

    .agg(

        importance=(
            "importance",
            "sum"
        )

    )

    .sort_values(
        "importance",
        ascending=False
    )

    .reset_index(
        drop=True
    )

)


# ============================================================
# 06.04.05 VALIDACIÓN DE LA SUMA DE IMPORTANCIAS
# ============================================================
#
# En un Random Forest de Spark, las feature importances
# deberían sumar aproximadamente 1.
#
# Utilizamos tolerancia para evitar problemas derivados
# de precisión decimal.
#
# ============================================================

total_importance = float(

    feature_importance_grouped_df[
        "importance"
    ].sum()

)


importance_sum_ok = (
    abs(
        total_importance
        -
        1.0
    )
    <
    0.0001
)


# ============================================================
# 06.04.06 IMPORTANCIA EN PORCENTAJE
# ============================================================

feature_importance_grouped_df[
    "importance_pct"
] = (

    feature_importance_grouped_df[
        "importance"
    ]

    *
    100.0

)


# ============================================================
# 06.04.07 IMPORTANCIA DE STORE_ID
# ============================================================

store_id_rows = (

    feature_importance_grouped_df[
        feature_importance_grouped_df[
            "feature_group"
        ]
        ==
        "store_id"
    ]

)


if len(store_id_rows) > 0:

    store_id_importance = float(
        store_id_rows.iloc[0][
            "importance"
        ]
    )

    store_id_importance_pct = (
        store_id_importance
        *
        100.0
    )

else:

    store_id_importance = None
    store_id_importance_pct = None


# ============================================================
# 06.04.08 RESUMEN
# ============================================================

print("=" * 80)
print("IMPORTANCIA AGRUPADA DE FEATURES")
print("=" * 80)


print(
    f"Features agrupadas: "
    f"{len(feature_importance_grouped_df):,}"
)


print(
    f"Suma de importancias: "
    f"{total_importance:.6f}"
)


if store_id_importance_pct is not None:

    print(
        f"Importancia agrupada store_id: "
        f"{store_id_importance_pct:.2f}%"
    )

else:

    print(
        "INFO - No se han encontrado features "
        "derivadas de store_id"
    )


# ============================================================
# 06.04.09 VALIDACIÓN FINAL
# ============================================================

if not importance_sum_ok:

    raise RuntimeError(

        "La suma de las importancias agrupadas "
        f"es {total_importance:.6f} y debería ser "
        "aproximadamente 1.0"

    )


print()
print("=" * 80)
print("VALIDACIÓN FINAL")
print("=" * 80)


print(
    "OK - Importancias agrupadas correctamente"
)


print(
    "OK - La suma de importancias es aproximadamente 100%"
)


if store_id_importance_pct is not None:

    print(
        "OK - store_id agrupado correctamente"
    )


# ============================================================
# 06.04.10 MOSTRAMOS RESULTADO
# ============================================================

feature_importance_grouped_spark_df = (

    spark.createDataFrame(
        feature_importance_grouped_df
    )

)


display(

    feature_importance_grouped_spark_df

)

# COMMAND ----------

# MAGIC %md
# MAGIC # 07. SEGUNDO MODELO ML - GRADIENT-BOOSTED TREES REGRESSOR
# MAGIC
# MAGIC ## Objetivo
# MAGIC
# MAGIC Entrenar un segundo modelo de Machine Learning utilizando **Gradient-Boosted Trees (GBT)** y comparar su rendimiento con:
# MAGIC
# MAGIC 1. Baseline Seasonal Naive (`lag_7`)
# MAGIC 2. Random Forest Regressor
# MAGIC
# MAGIC A diferencia de Random Forest, donde los árboles se construyen de forma independiente, Gradient Boosting construye los árboles de forma secuencial.
# MAGIC
# MAGIC Cada nuevo árbol intenta corregir los errores cometidos por los árboles anteriores.
# MAGIC
# MAGIC ## Estrategia
# MAGIC
# MAGIC Se mantendrán:
# MAGIC
# MAGIC - Las mismas features
# MAGIC - El mismo dataset Train
# MAGIC - El mismo dataset Validation
# MAGIC - Las mismas métricas: MAE, RMSE y WAPE
# MAGIC
# MAGIC De esta forma podremos realizar una comparación directa entre modelos.
# MAGIC
# MAGIC ## Referencias actuales
# MAGIC
# MAGIC | Modelo | MAE | RMSE | WAPE |
# MAGIC |---|---:|---:|---:|
# MAGIC | Seasonal Naive | 2.970,72 | 3.808,58 | 49,18 % |
# MAGIC | Random Forest | 2.147,08 | 2.734,93 | 35,55 % |
# MAGIC
# MAGIC El conjunto **Test permanece reservado**.

# COMMAND ----------

# DBTITLE 1,07.01 GRADIENT-BOOSTED TREES REGRESSOR
# ============================================================
# 07.01 GRADIENT-BOOSTED TREES REGRESSOR
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Entrenar un modelo Gradient-Boosted Trees utilizando
# exactamente las mismas features que Random Forest.
#
# Esto permitirá realizar una comparación justa entre modelos.
#
# IMPORTANTE
# ------------------------------------------------------------
#
# TRAIN:
#     utilizado para entrenamiento.
#
# VALIDATION:
#     utilizado posteriormente para comparar modelos.
#
# TEST:
#     permanece reservado.
#
# OUT-OF-TIME:
#     permanece completamente aislado.
#
# ============================================================


from pyspark.ml import Pipeline
from pyspark.ml.regression import GBTRegressor
from pyspark.sql import functions as F


# ============================================================
# 07.01.01 VALIDACIÓN DE FEATURES
# ============================================================
#
# Reutilizamos exactamente:
#
# - numeric_features
# - categorical_features
#
# definidas previamente para Random Forest.
#
# ============================================================

required_gbt_columns = (

    numeric_features

    +

    categorical_features

    +

    [
        "net_sales"
    ]

)


missing_gbt_columns = [

    column

    for column in required_gbt_columns

    if column not in train_df.columns

]


if missing_gbt_columns:

    raise RuntimeError(

        "Faltan columnas necesarias para entrenar GBT: "
        +
        ", ".join(missing_gbt_columns)

    )


print("=" * 80)
print("VALIDACIÓN DE FEATURES - GRADIENT BOOSTED TREES")
print("=" * 80)

print(
    f"Features numéricas:   {len(numeric_features)}"
)

print(
    f"Features categóricas: {len(categorical_features)}"
)

print(
    "OK - GBT utilizará las mismas features que Random Forest"
)


# ============================================================
# 07.01.02 VALIDACIÓN DE NULLS
# ============================================================

null_expressions_gbt = [

    F.sum(

        F.when(
            F.col(column).isNull(),
            1
        )

        .otherwise(
            0
        )

    ).alias(column)

    for column in required_gbt_columns

]


null_summary_gbt = (

    train_df

    .agg(
        *null_expressions_gbt
    )

    .first()

)


gbt_columns_with_nulls = {

    column: null_summary_gbt[column]

    for column in required_gbt_columns

    if null_summary_gbt[column] > 0

}


if gbt_columns_with_nulls:

    print()
    print(
        "ERROR - Se han detectado NULLs:"
    )

    for column, null_count in gbt_columns_with_nulls.items():

        print(
            f" - {column}: {null_count:,}"
        )

    raise RuntimeError(
        "TRAIN contiene NULLs en features o target"
    )


print(
    "OK - TRAIN no contiene NULLs en features ni target"
)


# ============================================================
# 07.01.03 DEFINICIÓN DEL MODELO
# ============================================================
#
# maxIter = 60
#
#     Número moderado de árboles secuenciales.
#
# maxDepth = 5
#
#     Mantiene la complejidad controlada.
#
# stepSize = 0.05
#
#     Learning rate conservador.
#
# seed = 42
#
#     Permite reproducibilidad.
#
# ============================================================

gbt = GBTRegressor(

    featuresCol="features",

    labelCol="net_sales",

    predictionCol="prediction",

    maxIter=60,

    maxDepth=5,

    stepSize=0.05,

    seed=42

)


# ============================================================
# 07.01.04 PIPELINE
# ============================================================

gbt_pipeline = Pipeline(

    stages=[

        store_indexer,

        store_encoder,

        assembler,

        gbt

    ]

)


# ============================================================
# 07.01.05 DATASET DE ENTRENAMIENTO
# ============================================================

train_rows_gbt = (
    train_df
    .count()
)


train_stores_gbt = (

    train_df

    .select(
        "store_id"
    )

    .distinct()

    .count()

)


train_dates_gbt = (

    train_df

    .agg(

        F.min("sale_date").alias("min_date"),

        F.max("sale_date").alias("max_date")

    )

    .first()

)


print()
print("=" * 80)
print("DATASET DE ENTRENAMIENTO")
print("=" * 80)

print(
    f"Registros: {train_rows_gbt:,}"
)

print(
    f"Tiendas:   {train_stores_gbt:,}"
)

print(
    f"Periodo:   "
    f"{train_dates_gbt['min_date']} "
    f"-> "
    f"{train_dates_gbt['max_date']}"
)


# ============================================================
# 07.01.06 ENTRENAMIENTO
# ============================================================

print()
print("=" * 80)
print("ENTRENAMIENTO GRADIENT-BOOSTED TREES")
print("=" * 80)


gbt_model = (

    gbt_pipeline

    .fit(
        train_df
    )

)


# ============================================================
# 07.01.07 VALIDACIÓN FINAL
# ============================================================

if gbt_model is None:

    raise RuntimeError(
        "No se ha generado un modelo GBT válido"
    )


print(
    "OK - Gradient-Boosted Trees entrenado correctamente"
)

print()
print("=" * 80)
print("VALIDACIÓN FINAL")
print("=" * 80)

print(
    "OK - GBT entrenado únicamente con TRAIN"
)

print(
    "OK - VALIDATION no ha participado en el entrenamiento"
)

print(
    "OK - TEST permanece reservado"
)

print(
    "OK - OUT-OF-TIME permanece aislado"
)

print(
    "OK - Modelo preparado para evaluación en VALIDATION"
)

# COMMAND ----------

# DBTITLE 1,07.02 EVALUACIÓN GBT SOBRE VALIDATION
# ============================================================
# 07.02 EVALUACIÓN GBT SOBRE VALIDATION
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Evaluar Gradient-Boosted Trees exclusivamente sobre
# VALIDATION y compararlo con:
#
# 1. Seasonal Naive Baseline
# 2. Random Forest
#
# IMPORTANTE
# ------------------------------------------------------------
#
# TEST y OUT-OF-TIME permanecen completamente reservados.
#
# ============================================================


from pyspark.sql import functions as F


# ============================================================
# 07.02.01 GENERAMOS PREDICCIONES
# ============================================================

gbt_validation_predictions_df = (

    gbt_model

    .transform(
        validation_df
    )

    .select(

        "sale_date",

        "store_id",

        F.col(
            "net_sales"
        ).alias(
            "actual"
        ),

        F.col(
            "prediction"
        ).alias(
            "prediction"
        )

    )

)


# ============================================================
# 07.02.02 VALIDACIONES TÉCNICAS
# ============================================================

validation_rows_gbt = (
    validation_df
    .count()
)


prediction_rows_gbt = (
    gbt_validation_predictions_df
    .count()
)


null_actual_gbt = (

    gbt_validation_predictions_df

    .filter(
        F.col("actual").isNull()
    )

    .count()

)


null_prediction_gbt = (

    gbt_validation_predictions_df

    .filter(
        F.col("prediction").isNull()
    )

    .count()

)


duplicate_rows_gbt = (

    gbt_validation_predictions_df

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


print("=" * 80)
print("VALIDACIÓN TÉCNICA - GBT")
print("=" * 80)

print(
    f"Registros VALIDATION: {validation_rows_gbt:,}"
)

print(
    f"Predicciones:         {prediction_rows_gbt:,}"
)

print(
    f"Actual NULL:          {null_actual_gbt:,}"
)

print(
    f"Prediction NULL:      {null_prediction_gbt:,}"
)

print(
    f"Duplicados:           {duplicate_rows_gbt:,}"
)


validation_errors_gbt = []


if prediction_rows_gbt != validation_rows_gbt:

    validation_errors_gbt.append(
        "El número de predicciones no coincide con VALIDATION."
    )


if null_actual_gbt > 0:

    validation_errors_gbt.append(
        "Existen valores NULL en actual."
    )


if null_prediction_gbt > 0:

    validation_errors_gbt.append(
        "Existen valores NULL en prediction."
    )


if duplicate_rows_gbt > 0:

    validation_errors_gbt.append(
        "Existen duplicados por sale_date + store_id."
    )


# ============================================================
# 07.02.03 CALCULAMOS ERRORES
# ============================================================

gbt_validation_predictions_df = (

    gbt_validation_predictions_df

    .withColumn(

        "error",

        F.col("actual")
        -
        F.col("prediction")

    )

    .withColumn(

        "absolute_error",

        F.abs(
            F.col("error")
        )

    )

    .withColumn(

        "squared_error",

        F.pow(
            F.col("error"),
            2
        )

    )

)


# ============================================================
# 07.02.04 AGREGAMOS MÉTRICAS
# ============================================================

gbt_metrics_row = (

    gbt_validation_predictions_df

    .agg(

        F.avg(
            "absolute_error"
        ).alias(
            "mae"
        ),

        F.sqrt(

            F.avg(
                "squared_error"
            )

        ).alias(
            "rmse"
        ),

        F.sum(
            "absolute_error"
        ).alias(
            "total_absolute_error"
        ),

        F.sum(

            F.abs(
                F.col("actual")
            )

        ).alias(
            "total_actual"
        )

    )

    .first()

)


# ============================================================
# 07.02.05 CONVERTIMOS MÉTRICAS A FLOAT
# ============================================================
#
# Evitamos problemas entre Decimal y float en Spark.
#
# ============================================================

gbt_mae = float(
    gbt_metrics_row["mae"]
)


gbt_rmse = float(
    gbt_metrics_row["rmse"]
)


gbt_total_absolute_error = float(
    gbt_metrics_row["total_absolute_error"]
)


gbt_total_actual = float(
    gbt_metrics_row["total_actual"]
)


if gbt_total_actual == 0:

    raise RuntimeError(
        "No se puede calcular WAPE porque "
        "la suma absoluta del target es 0."
    )


gbt_wape = (

    gbt_total_absolute_error

    /

    gbt_total_actual

    *

    100.0

)


# ============================================================
# 07.02.06 RECUPERAMOS MÉTRICAS ANTERIORES
# ============================================================
#
# Convertimos también a float las métricas del Baseline
# y Random Forest.
#
# ============================================================

required_previous_metrics = [

    "baseline_mae",
    "baseline_rmse",
    "baseline_wape",

    "rf_mae",
    "rf_rmse",
    "rf_wape"

]


missing_previous_metrics = [

    metric

    for metric in required_previous_metrics

    if metric not in globals()

]


if missing_previous_metrics:

    raise RuntimeError(

        "Faltan métricas de modelos anteriores: "
        +
        ", ".join(missing_previous_metrics)

    )


baseline_mae_float = float(
    baseline_mae
)

baseline_rmse_float = float(
    baseline_rmse
)

baseline_wape_float = float(
    baseline_wape
)


rf_mae_float = float(
    rf_mae
)

rf_rmse_float = float(
    rf_rmse
)

rf_wape_float = float(
    rf_wape
)


# ============================================================
# 07.02.07 RESULTADOS GBT
# ============================================================

print()
print("=" * 80)
print("GRADIENT-BOOSTED TREES - VALIDATION")
print("=" * 80)

print(
    f"MAE:  {gbt_mae:,.2f}"
)

print(
    f"RMSE: {gbt_rmse:,.2f}"
)

print(
    f"WAPE: {gbt_wape:,.2f}%"
)


# ============================================================
# 07.02.08 COMPARACIÓN DE MODELOS
# ============================================================

print()
print("=" * 80)
print("COMPARACIÓN SOBRE VALIDATION")
print("=" * 80)

print(
    f"{'Modelo':<22}"
    f"{'MAE':>14}"
    f"{'RMSE':>14}"
    f"{'WAPE':>14}"
)

print("-" * 64)

print(
    f"{'Seasonal Naive':<22}"
    f"{baseline_mae_float:>14,.2f}"
    f"{baseline_rmse_float:>14,.2f}"
    f"{baseline_wape_float:>13,.2f}%"
)

print(
    f"{'Random Forest':<22}"
    f"{rf_mae_float:>14,.2f}"
    f"{rf_rmse_float:>14,.2f}"
    f"{rf_wape_float:>13,.2f}%"
)

print(
    f"{'Gradient Boosting':<22}"
    f"{gbt_mae:>14,.2f}"
    f"{gbt_rmse:>14,.2f}"
    f"{gbt_wape:>13,.2f}%"
)


# ============================================================
# 07.02.09 MEJORA GBT VS BASELINE
# ============================================================

gbt_vs_baseline_mae_improvement = (

    (
        baseline_mae_float
        -
        gbt_mae
    )

    /

    baseline_mae_float

    *

    100.0

)


gbt_vs_baseline_rmse_improvement = (

    (
        baseline_rmse_float
        -
        gbt_rmse
    )

    /

    baseline_rmse_float

    *

    100.0

)


gbt_vs_baseline_wape_improvement = (

    (
        baseline_wape_float
        -
        gbt_wape
    )

    /

    baseline_wape_float

    *

    100.0

)


# ============================================================
# 07.02.10 MEJORA GBT VS RANDOM FOREST
# ============================================================

gbt_vs_rf_mae_improvement = (

    (
        rf_mae_float
        -
        gbt_mae
    )

    /

    rf_mae_float

    *

    100.0

)


gbt_vs_rf_rmse_improvement = (

    (
        rf_rmse_float
        -
        gbt_rmse
    )

    /

    rf_rmse_float

    *

    100.0

)


gbt_vs_rf_wape_improvement = (

    (
        rf_wape_float
        -
        gbt_wape
    )

    /

    rf_wape_float

    *

    100.0

)


print()
print("=" * 80)
print("MEJORA GBT VS BASELINE")
print("=" * 80)

print(
    f"MAE:  {gbt_vs_baseline_mae_improvement:+.2f}%"
)

print(
    f"RMSE: {gbt_vs_baseline_rmse_improvement:+.2f}%"
)

print(
    f"WAPE: {gbt_vs_baseline_wape_improvement:+.2f}%"
)


print()
print("=" * 80)
print("MEJORA GBT VS RANDOM FOREST")
print("=" * 80)

print(
    f"MAE:  {gbt_vs_rf_mae_improvement:+.2f}%"
)

print(
    f"RMSE: {gbt_vs_rf_rmse_improvement:+.2f}%"
)

print(
    f"WAPE: {gbt_vs_rf_wape_improvement:+.2f}%"
)


# ============================================================
# 07.02.11 DETERMINAMOS GANADOR EN VALIDATION
# ============================================================

model_results_validation = {

    "Seasonal Naive": {
        "mae": baseline_mae_float,
        "rmse": baseline_rmse_float,
        "wape": baseline_wape_float
    },

    "Random Forest": {
        "mae": rf_mae_float,
        "rmse": rf_rmse_float,
        "wape": rf_wape_float
    },

    "Gradient Boosting": {
        "mae": gbt_mae,
        "rmse": gbt_rmse,
        "wape": gbt_wape
    }

}


best_mae_model = min(

    model_results_validation,

    key=lambda model:
        model_results_validation[model]["mae"]

)


best_rmse_model = min(

    model_results_validation,

    key=lambda model:
        model_results_validation[model]["rmse"]

)


best_wape_model = min(

    model_results_validation,

    key=lambda model:
        model_results_validation[model]["wape"]

)


print()
print("=" * 80)
print("MEJOR MODELO POR MÉTRICA")
print("=" * 80)

print(
    f"MAE:  {best_mae_model}"
)

print(
    f"RMSE: {best_rmse_model}"
)

print(
    f"WAPE: {best_wape_model}"
)


# ============================================================
# 07.02.12 VALIDACIÓN FINAL
# ============================================================

if validation_errors_gbt:

    print()
    print("=" * 80)
    print("ERRORES DE VALIDACIÓN")
    print("=" * 80)

    for error in validation_errors_gbt:

        print(
            f"ERROR - {error}"
        )

    raise RuntimeError(
        "La evaluación del GBT no ha superado "
        "las validaciones técnicas."
    )


print()
print("=" * 80)
print("VALIDACIÓN FINAL")
print("=" * 80)

print(
    "OK - Predicciones GBT generadas correctamente"
)

print(
    "OK - No existen NULLs ni duplicados"
)

print(
    "OK - Métricas calculadas correctamente"
)

print(
    "OK - Comparación realizada sobre el mismo VALIDATION"
)

print(
    "OK - TEST permanece reservado"
)

print(
    "OK - OUT-OF-TIME permanece aislado"
)

# COMMAND ----------

# MAGIC %md
# MAGIC # 08. TUNING DE HIPERPARÁMETROS - RANDOM FOREST
# MAGIC
# MAGIC ## Objetivo
# MAGIC
# MAGIC Optimizar el modelo Random Forest que actualmente presenta el mejor rendimiento sobre Validation.
# MAGIC
# MAGIC Modelo actual:
# MAGIC
# MAGIC - MAE: **2.147,08**
# MAGIC - RMSE: **2.734,93**
# MAGIC - WAPE: **35,55 %**
# MAGIC
# MAGIC El objetivo será comprobar si distintas configuraciones de Random Forest consiguen mejorar estas métricas.
# MAGIC
# MAGIC ## Hiperparámetros analizados
# MAGIC
# MAGIC - `numTrees`: número de árboles del bosque.
# MAGIC - `maxDepth`: profundidad máxima de cada árbol.
# MAGIC - `minInstancesPerNode`: número mínimo de observaciones necesarias en un nodo.
# MAGIC - `featureSubsetStrategy`: proporción de features consideradas en cada división.
# MAGIC
# MAGIC No realizaremos una búsqueda exhaustiva.
# MAGIC
# MAGIC Se probará un conjunto reducido de configuraciones para equilibrar:
# MAGIC
# MAGIC **calidad del modelo ↔ tiempo de entrenamiento**
# MAGIC
# MAGIC ## Evaluación
# MAGIC
# MAGIC Todos los modelos se entrenarán únicamente con **Train**.
# MAGIC
# MAGIC La selección del mejor modelo se realizará utilizando **Validation**.
# MAGIC
# MAGIC El conjunto **Test permanece completamente reservado** para la evaluación final.

# COMMAND ----------

# DBTITLE 1,08.01 DEFINICIÓN DEL ESPACIO DE HIPERPARÁMETROS
# ============================================================
# 08.01 DEFINICIÓN DEL ESPACIO DE HIPERPARÁMETROS
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Definir un conjunto reducido de configuraciones de
# Random Forest para optimizar el modelo utilizando:
#
# - TRAIN      -> entrenamiento
# - VALIDATION -> selección del mejor modelo
#
# TEST y OUT-OF-TIME permanecen reservados.
#
# IMPORTANTE
# ------------------------------------------------------------
#
# Trabajamos en Databricks Serverless, por lo que evitamos
# configuraciones excesivamente grandes que puedan provocar:
#
# - mayor consumo de memoria
# - tiempos de entrenamiento innecesarios
# - problemas de estabilidad durante el tuning
#
# El modelo actual de referencia utiliza:
#
# numTrees = 60
# maxDepth = 8
# minInstancesPerNode = 3
# featureSubsetStrategy = "auto"
#
# ============================================================


# ============================================================
# 08.01.01 CONFIGURACIONES RANDOM FOREST
# ============================================================

rf_configs = [

    # --------------------------------------------------------
    # RF_01 - MODELO ACTUAL DE REFERENCIA
    # --------------------------------------------------------
    #
    # Es exactamente la configuración utilizada en 06.01.
    #
    # Nos sirve como benchmark para saber si alguna de las
    # nuevas configuraciones realmente mejora el modelo.
    #
    # --------------------------------------------------------

    {
        "config_id": "RF_01_BASE",

        "numTrees": 60,

        "maxDepth": 8,

        "minInstancesPerNode": 3,

        "featureSubsetStrategy": "auto"
    },


    # --------------------------------------------------------
    # RF_02 - MÁS ÁRBOLES
    # --------------------------------------------------------
    #
    # Aumentamos ligeramente el número de árboles.
    #
    # Objetivo:
    # comprobar si una mayor estabilidad del ensemble mejora
    # las métricas sin aumentar demasiado el coste.
    #
    # --------------------------------------------------------

    {
        "config_id": "RF_02_TREES_80",

        "numTrees": 80,

        "maxDepth": 8,

        "minInstancesPerNode": 3,

        "featureSubsetStrategy": "auto"
    },


    # --------------------------------------------------------
    # RF_03 - MENOR PROFUNDIDAD
    # --------------------------------------------------------
    #
    # Árboles más simples.
    #
    # Puede:
    #
    # - reducir overfitting
    # - acelerar entrenamiento
    # - aumentar generalización
    #
    # --------------------------------------------------------

    {
        "config_id": "RF_03_DEPTH_6",

        "numTrees": 60,

        "maxDepth": 6,

        "minInstancesPerNode": 3,

        "featureSubsetStrategy": "auto"
    },


    # --------------------------------------------------------
    # RF_04 - MAYOR PROFUNDIDAD
    # --------------------------------------------------------
    #
    # Permitimos algo más de complejidad que el modelo base,
    # pero sin volver a profundidades demasiado agresivas.
    #
    # --------------------------------------------------------

    {
        "config_id": "RF_04_DEPTH_10",

        "numTrees": 60,

        "maxDepth": 10,

        "minInstancesPerNode": 3,

        "featureSubsetStrategy": "auto"
    },


    # --------------------------------------------------------
    # RF_05 - MAYOR REGULARIZACIÓN
    # --------------------------------------------------------
    #
    # Exigimos más observaciones para poder crear nuevos nodos.
    #
    # Esto puede ayudar si el modelo actual está aprendiendo
    # patrones demasiado específicos del TRAIN.
    #
    # --------------------------------------------------------

    {
        "config_id": "RF_05_MIN_NODE_5",

        "numTrees": 60,

        "maxDepth": 8,

        "minInstancesPerNode": 5,

        "featureSubsetStrategy": "auto"
    },


    # --------------------------------------------------------
    # RF_06 - SUBCONJUNTO SQRT DE FEATURES
    # --------------------------------------------------------
    #
    # En cada split se evalúa únicamente una fracción de las
    # features.
    #
    # Esto puede aumentar la diversidad entre árboles.
    #
    # --------------------------------------------------------

    {
        "config_id": "RF_06_SQRT",

        "numTrees": 60,

        "maxDepth": 8,

        "minInstancesPerNode": 3,

        "featureSubsetStrategy": "sqrt"
    }

]


# ============================================================
# 08.01.02 VALIDACIÓN DE CONFIGURACIONES
# ============================================================

required_config_keys = {

    "config_id",

    "numTrees",

    "maxDepth",

    "minInstancesPerNode",

    "featureSubsetStrategy"

}


config_ids = []


for config in rf_configs:

    missing_keys = (

        required_config_keys

        -

        set(
            config.keys()
        )

    )


    if missing_keys:

        raise RuntimeError(

            f"La configuración {config} "
            f"no contiene las claves requeridas: "
            f"{sorted(missing_keys)}"

        )


    config_ids.append(
        config["config_id"]
    )


# ============================================================
# 08.01.03 VALIDAMOS IDS DUPLICADOS
# ============================================================

if len(config_ids) != len(set(config_ids)):

    raise RuntimeError(
        "Existen config_id duplicados en rf_configs."
    )


# ============================================================
# 08.01.04 MOSTRAMOS CONFIGURACIONES
# ============================================================

print("=" * 80)
print("CONFIGURACIONES RANDOM FOREST")
print("=" * 80)

print(
    f"Número de configuraciones: {len(rf_configs)}"
)

print()


for config in rf_configs:

    print(
        f"{config['config_id']:<18} | "
        f"Trees: {config['numTrees']:>3} | "
        f"Depth: {config['maxDepth']:>2} | "
        f"MinNode: {config['minInstancesPerNode']:>2} | "
        f"Features: {config['featureSubsetStrategy']}"
    )


# ============================================================
# 08.01.05 VALIDACIÓN FINAL
# ============================================================

print()
print("=" * 80)
print("VALIDACIÓN FINAL")
print("=" * 80)

print(
    "OK - Configuraciones definidas correctamente"
)

print(
    "OK - No existen config_id duplicados"
)

print(
    "OK - RF_01_BASE reproduce el Random Forest actual"
)

print(
    "OK - Espacio de búsqueda limitado para Serverless"
)

print(
    "OK - TEST permanece reservado"
)

print(
    "OK - OUT-OF-TIME permanece aislado"
)

# COMMAND ----------

# DBTITLE 1,08.02 ENTRENAMIENTO Y EVALUACIÓN DE CONFIGURACIONES
# ============================================================
# 08.02 ENTRENAMIENTO Y EVALUACIÓN DE CONFIGURACIONES
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Realizar un tuning controlado del Random Forest utilizando:
#
# TRAIN      -> entrenamiento
# VALIDATION -> selección
#
# TEST y OUT-OF-TIME permanecen completamente reservados.
#
# ESTRATEGIA SERVERLESS
# ------------------------------------------------------------
#
# Spark Connect mantiene modelos ML en una caché de sesión.
#
# Para reducir el riesgo de:
#
# MODEL_CACHE_SIZE_OVERFLOW_EXCEPTION
#
# hacemos lo siguiente:
#
# 1. NO reentrenamos RF_01_BASE.
#    Ya fue entrenado y evaluado anteriormente.
#
# 2. Reutilizamos sus métricas.
#
# 3. Liberamos referencias a modelos anteriores que ya
#    no necesitamos para el tuning.
#
# 4. Entrenamos las configuraciones restantes una a una.
#
# 5. Después de cada configuración:
#       - calculamos métricas
#       - guardamos solo números
#       - eliminamos referencias Python
#       - ejecutamos garbage collection
#
# ============================================================


from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.sql import functions as F

import pandas as pd
import time
import gc


# ============================================================
# 08.02.01 VALIDAMOS MÉTRICAS DEL MODELO BASE
# ============================================================
#
# RF_01_BASE coincide exactamente con el modelo evaluado
# anteriormente:
#
# numTrees = 60
# maxDepth = 8
# minInstancesPerNode = 3
# featureSubsetStrategy = auto
#
# Por tanto, no necesitamos volver a entrenarlo.
#
# ============================================================

required_base_metrics = [
    "rf_mae",
    "rf_rmse",
    "rf_wape"
]


missing_base_metrics = [

    metric

    for metric in required_base_metrics

    if metric not in globals()

]


if missing_base_metrics:

    raise RuntimeError(

        "Faltan métricas del Random Forest base: "
        +
        ", ".join(missing_base_metrics)

    )


rf_mae_base = float(
    rf_mae
)

rf_rmse_base = float(
    rf_rmse
)

rf_wape_base = float(
    rf_wape
)


# ============================================================
# 08.02.02 REINICIAMOS RESULTADOS
# ============================================================

rf_tuning_results = []


# ============================================================
# 08.02.03 INCORPORAMOS RF_01_BASE SIN REENTRENAR
# ============================================================

base_config = next(

    (
        config

        for config in rf_configs

        if config["config_id"] == "RF_01_BASE"
    ),

    None

)


if base_config is None:

    raise RuntimeError(
        "No existe RF_01_BASE dentro de rf_configs."
    )


rf_tuning_results.append(

    {
        "config_id":
            base_config["config_id"],

        "numTrees":
            base_config["numTrees"],

        "maxDepth":
            base_config["maxDepth"],

        "minInstancesPerNode":
            base_config["minInstancesPerNode"],

        "featureSubsetStrategy":
            base_config["featureSubsetStrategy"],

        "mae":
            rf_mae_base,

        "rmse":
            rf_rmse_base,

        "wape":
            rf_wape_base,

        # No medimos tiempo porque reutilizamos
        # el modelo ya entrenado.
        "training_seconds":
            None
    }

)


print("=" * 80)
print("INICIO DEL TUNING RANDOM FOREST")
print("=" * 80)

print()
print(
    "RF_01_BASE -> métricas reutilizadas; "
    "no se vuelve a entrenar"
)

print(
    f"MAE:  {rf_mae_base:,.2f}"
)

print(
    f"RMSE: {rf_rmse_base:,.2f}"
)

print(
    f"WAPE: {rf_wape_base:,.2f}%"
)


# ============================================================
# 08.02.04 LIBERAMOS MODELOS ANTERIORES
# ============================================================
#
# A partir de este punto ya no necesitamos conservar los
# modelos RF y GBT anteriores dentro de Python.
#
# Conservamos únicamente sus métricas numéricas.
#
# IMPORTANTE:
#
# del elimina la referencia Python.
# gc.collect() solicita liberar objetos que ya no tengan
# referencias activas.
#
# ============================================================

objects_to_release = [

    "rf_model",

    "gbt_model",

    "gbt_validation_predictions_df",

    "rf_validation_predictions_df"

]


released_objects = []


for object_name in objects_to_release:

    if object_name in globals():

        del globals()[object_name]

        released_objects.append(
            object_name
        )


gc.collect()


print()
print("=" * 80)
print("LIBERACIÓN DE REFERENCIAS")
print("=" * 80)


if released_objects:

    for object_name in released_objects:

        print(
            f"OK - Referencia eliminada: {object_name}"
        )

else:

    print(
        "INFO - No había referencias anteriores que liberar"
    )


print(
    "OK - Garbage collection ejecutado"
)


# ============================================================
# 08.02.05 CONFIGURACIONES A ENTRENAR
# ============================================================
#
# Excluimos RF_01_BASE porque ya tenemos sus resultados.
#
# ============================================================

configs_to_train = [

    config

    for config in rf_configs

    if config["config_id"] != "RF_01_BASE"

]


print()
print(
    f"Configuraciones nuevas a entrenar: "
    f"{len(configs_to_train)}"
)


# ============================================================
# 08.02.06 TUNING SECUENCIAL
# ============================================================

for config in configs_to_train:

    config_id = config[
        "config_id"
    ]


    print()
    print("-" * 80)

    print(
        f"Entrenando: {config_id}"
    )

    print("-" * 80)


    # --------------------------------------------------------
    # INICIAMOS CRONÓMETRO
    # --------------------------------------------------------

    start_time = time.time()


    # --------------------------------------------------------
    # DEFINIMOS RANDOM FOREST
    # --------------------------------------------------------

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

        seed=42

    )


    # --------------------------------------------------------
    # PIPELINE
    # --------------------------------------------------------

    rf_candidate_pipeline = Pipeline(

        stages=[

            store_indexer,

            store_encoder,

            assembler,

            rf_candidate

        ]

    )


    # --------------------------------------------------------
    # ENTRENAMIENTO
    # --------------------------------------------------------

    candidate_model = (

        rf_candidate_pipeline

        .fit(
            train_df
        )

    )


    # --------------------------------------------------------
    # PREDICCIÓN SOBRE VALIDATION
    # --------------------------------------------------------

    candidate_predictions_df = (

        candidate_model

        .transform(
            validation_df
        )

        .select(

            F.col(
                "net_sales"
            ).alias(
                "actual"
            ),

            F.col(
                "prediction"
            ).alias(
                "prediction"
            )

        )

        .withColumn(

            "absolute_error",

            F.abs(

                F.col("actual")

                -

                F.col("prediction")

            )

        )

        .withColumn(

            "squared_error",

            F.pow(

                F.col("actual")

                -

                F.col("prediction"),

                2

            )

        )

    )


    # --------------------------------------------------------
    # MÉTRICAS
    # --------------------------------------------------------

    candidate_metrics_row = (

        candidate_predictions_df

        .agg(

            F.avg(
                "absolute_error"
            ).alias(
                "mae"
            ),

            F.sqrt(

                F.avg(
                    "squared_error"
                )

            ).alias(
                "rmse"
            ),

            F.sum(
                "absolute_error"
            ).alias(
                "total_absolute_error"
            ),

            F.sum(

                F.abs(
                    F.col("actual")
                )

            ).alias(
                "total_actual"
            )

        )

        .first()

    )


    # --------------------------------------------------------
    # CONVERTIMOS TODO A FLOAT
    # --------------------------------------------------------

    candidate_mae = float(
        candidate_metrics_row["mae"]
    )

    candidate_rmse = float(
        candidate_metrics_row["rmse"]
    )

    candidate_total_absolute_error = float(
        candidate_metrics_row["total_absolute_error"]
    )

    candidate_total_actual = float(
        candidate_metrics_row["total_actual"]
    )


    if candidate_total_actual == 0:

        raise RuntimeError(

            f"{config_id}: no se puede calcular WAPE "
            "porque la suma absoluta del target es 0."

        )


    candidate_wape = (

        candidate_total_absolute_error

        /

        candidate_total_actual

        *

        100.0

    )


    # --------------------------------------------------------
    # TIEMPO TOTAL
    # --------------------------------------------------------

    elapsed_time = (

        time.time()

        -

        start_time

    )


    # --------------------------------------------------------
    # GUARDAMOS ÚNICAMENTE RESULTADOS NUMÉRICOS
    # --------------------------------------------------------

    rf_tuning_results.append(

        {

            "config_id":
                config_id,

            "numTrees":
                config["numTrees"],

            "maxDepth":
                config["maxDepth"],

            "minInstancesPerNode":
                config["minInstancesPerNode"],

            "featureSubsetStrategy":
                config["featureSubsetStrategy"],

            "mae":
                candidate_mae,

            "rmse":
                candidate_rmse,

            "wape":
                candidate_wape,

            "training_seconds":
                round(
                    elapsed_time,
                    2
                )

        }

    )


    # --------------------------------------------------------
    # RESULTADO
    # --------------------------------------------------------

    print(
        f"MAE:    {candidate_mae:,.2f}"
    )

    print(
        f"RMSE:   {candidate_rmse:,.2f}"
    )

    print(
        f"WAPE:   {candidate_wape:,.2f}%"
    )

    print(
        f"Tiempo: {elapsed_time:,.2f} segundos"
    )


    # ========================================================
    # LIBERAMOS EL MODELO ACTUAL
    # ========================================================

    del candidate_model
    del candidate_predictions_df
    del candidate_metrics_row
    del rf_candidate_pipeline
    del rf_candidate


    gc.collect()


    print(
        "OK - Referencias de la configuración liberadas"
    )


# ============================================================
# 08.02.07 VALIDACIÓN DEL TUNING
# ============================================================

expected_configs = len(
    rf_configs
)


obtained_configs = len(
    rf_tuning_results
)


print()
print("=" * 80)
print("VALIDACIÓN DEL TUNING")
print("=" * 80)

print(
    f"Configuraciones esperadas: {expected_configs}"
)

print(
    f"Resultados obtenidos:      {obtained_configs}"
)


if obtained_configs != expected_configs:

    raise RuntimeError(

        "No se han obtenido resultados para todas "
        "las configuraciones."

    )


# ============================================================
# 08.02.08 DATAFRAME DE RESULTADOS
# ============================================================

rf_tuning_results_df = pd.DataFrame(
    rf_tuning_results
)


rf_tuning_results_df = (

    rf_tuning_results_df

    .sort_values(

        by=[
            "wape",
            "mae",
            "rmse"
        ],

        ascending=True

    )

    .reset_index(
        drop=True
    )

)


# ============================================================
# 08.02.09 MOSTRAMOS RESULTADOS
# ============================================================

print()
print("=" * 80)
print("RESULTADOS DEL TUNING")
print("=" * 80)


display(

    spark.createDataFrame(
        rf_tuning_results_df
    )

)


# ============================================================
# 08.02.10 VALIDACIÓN FINAL
# ============================================================

print()
print("=" * 80)
print("VALIDACIÓN FINAL")
print("=" * 80)

print(
    "OK - RF_01_BASE reutilizado sin reentrenamiento"
)

print(
    f"OK - {len(configs_to_train)} configuraciones nuevas evaluadas"
)

print(
    "OK - Solo se almacenaron métricas del tuning"
)

print(
    "OK - Selección realizada exclusivamente con VALIDATION"
)

print(
    "OK - TEST permanece reservado"
)

print(
    "OK - OUT-OF-TIME permanece aislado"
)

print()
print(
    "OK - Tuning Random Forest finalizado"
)

# COMMAND ----------

# DBTITLE 1,08.03 RANKING DE CONFIGURACIONES
# ============================================================
# 08.03 RANKING DE CONFIGURACIONES
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Comparar las configuraciones evaluadas en VALIDATION
# y construir un ranking utilizando:
#
# 1. WAPE como métrica principal
# 2. MAE como segundo criterio
# 3. RMSE como tercer criterio
#
# También calculamos la mejora frente al modelo base
# RF_01_BASE para interpretar si la ganancia es realmente
# relevante o simplemente marginal.
#
# TEST y OUT-OF-TIME siguen reservados.
#
# ============================================================


import pandas as pd
import numpy as np


# ============================================================
# 08.03.01 VALIDAMOS RESULTADOS DEL TUNING
# ============================================================

if "rf_tuning_results" not in globals():

    raise RuntimeError(
        "rf_tuning_results no existe. "
        "Ejecuta primero 08.02."
    )


if len(rf_tuning_results) == 0:

    raise RuntimeError(
        "rf_tuning_results está vacío."
    )


# ============================================================
# 08.03.02 CONVERTIMOS RESULTADOS A PANDAS
# ============================================================

rf_tuning_results_df = pd.DataFrame(
    rf_tuning_results
)


required_columns = [

    "config_id",
    "numTrees",
    "maxDepth",
    "minInstancesPerNode",
    "featureSubsetStrategy",
    "mae",
    "rmse",
    "wape",
    "training_seconds"

]


missing_columns = [

    column

    for column in required_columns

    if column not in rf_tuning_results_df.columns

]


if missing_columns:

    raise RuntimeError(

        "Faltan columnas en los resultados del tuning: "
        +
        ", ".join(missing_columns)

    )


# ============================================================
# 08.03.03 RECUPERAMOS CONFIGURACIÓN BASE
# ============================================================

base_rows = (

    rf_tuning_results_df

    .loc[
        rf_tuning_results_df["config_id"]
        ==
        "RF_01_BASE"
    ]

)


if len(base_rows) != 1:

    raise RuntimeError(
        "Debe existir exactamente una configuración RF_01_BASE."
    )


base_row = (
    base_rows
    .iloc[0]
)


base_mae = float(
    base_row["mae"]
)

base_rmse = float(
    base_row["rmse"]
)

base_wape = float(
    base_row["wape"]
)


# ============================================================
# 08.03.04 CALCULAMOS MEJORA VS BASE
# ============================================================
#
# Valor positivo:
#     mejora frente al modelo base.
#
# Valor negativo:
#     empeora frente al modelo base.
#
# ============================================================

rf_tuning_results_df[
    "mae_improvement_vs_base_pct"
] = (

    (
        base_mae
        -
        rf_tuning_results_df["mae"]
    )

    /

    base_mae

    *

    100.0

)


rf_tuning_results_df[
    "rmse_improvement_vs_base_pct"
] = (

    (
        base_rmse
        -
        rf_tuning_results_df["rmse"]
    )

    /

    base_rmse

    *

    100.0

)


rf_tuning_results_df[
    "wape_improvement_vs_base_pct"
] = (

    (
        base_wape
        -
        rf_tuning_results_df["wape"]
    )

    /

    base_wape

    *

    100.0

)


# ============================================================
# 08.03.05 ORDENAMOS EL RANKING
# ============================================================
#
# Criterios:
#
# 1. WAPE
# 2. MAE
# 3. RMSE
#
# ============================================================

rf_tuning_results_df = (

    rf_tuning_results_df

    .sort_values(

        by=[
            "wape",
            "mae",
            "rmse"
        ],

        ascending=[
            True,
            True,
            True
        ]

    )

    .reset_index(
        drop=True
    )

)


# ============================================================
# 08.03.06 ASIGNAMOS RANK
# ============================================================

rf_tuning_results_df[
    "rank"
] = (

    range(
        1,
        len(rf_tuning_results_df) + 1
    )

)


# ============================================================
# 08.03.07 IDENTIFICAMOS GANADOR TÉCNICO
# ============================================================

best_config_row = (

    rf_tuning_results_df

    .iloc[0]

)


best_config_id = (
    best_config_row[
        "config_id"
    ]
)


best_wape = float(
    best_config_row[
        "wape"
    ]
)


best_mae = float(
    best_config_row[
        "mae"
    ]
)


best_rmse = float(
    best_config_row[
        "rmse"
    ]
)


# ============================================================
# 08.03.08 MEDIMOS DIFERENCIA VS BASE
# ============================================================

best_wape_absolute_gain = (

    base_wape
    -
    best_wape

)


best_wape_relative_gain = (

    (
        base_wape
        -
        best_wape
    )

    /

    base_wape

    *

    100.0

)


# ============================================================
# 08.03.09 EVALUAMOS SI LA MEJORA ES MATERIAL
# ============================================================
#
# Definimos una mejora mínima orientativa del 0.5% relativo
# en WAPE para considerarla material.
#
# Si la mejora es inferior, la trataremos como marginal.
#
# ============================================================

material_improvement_threshold_pct = 0.5


improvement_is_material = (

    best_wape_relative_gain

    >=

    material_improvement_threshold_pct

)


# ============================================================
# 08.03.10 RESUMEN
# ============================================================

print("=" * 80)
print("RANKING RANDOM FOREST - VALIDATION")
print("=" * 80)

print(
    f"Mejor configuración técnica: {best_config_id}"
)

print(
    f"WAPE ganador:               {best_wape:.4f}%"
)

print(
    f"WAPE modelo base:           {base_wape:.4f}%"
)

print(
    f"Ganancia absoluta WAPE:     "
    f"{best_wape_absolute_gain:.4f} puntos"
)

print(
    f"Ganancia relativa WAPE:     "
    f"{best_wape_relative_gain:.4f}%"
)


if best_config_id == "RF_01_BASE":

    selection_message = (
        "El modelo base sigue siendo la mejor configuración."
    )

elif improvement_is_material:

    selection_message = (
        f"{best_config_id} presenta una mejora material "
        "frente al modelo base."
    )

else:

    selection_message = (
        f"{best_config_id} obtiene el mejor WAPE, "
        "pero la mejora frente al modelo base es marginal."
    )


print()
print(
    selection_message
)


# ============================================================
# 08.03.11 ORDEN FINAL DE COLUMNAS
# ============================================================

rf_tuning_results_df = (

    rf_tuning_results_df[

        [
            "rank",
            "config_id",
            "numTrees",
            "maxDepth",
            "minInstancesPerNode",
            "featureSubsetStrategy",
            "mae",
            "rmse",
            "wape",
            "mae_improvement_vs_base_pct",
            "rmse_improvement_vs_base_pct",
            "wape_improvement_vs_base_pct",
            "training_seconds"
        ]

    ]

)


# ============================================================
# 08.03.12 MOSTRAMOS RANKING
# ============================================================

display(

    spark.createDataFrame(
        rf_tuning_results_df
    )

)


# ============================================================
# 08.03.13 VALIDACIÓN FINAL
# ============================================================

print()
print("=" * 80)
print("VALIDACIÓN FINAL")
print("=" * 80)

print(
    "OK - Ranking generado correctamente"
)

print(
    "OK - WAPE utilizado como métrica principal"
)

print(
    "OK - MAE y RMSE utilizados como desempate"
)

print(
    "OK - Mejora frente a RF_01_BASE calculada"
)

print(
    "OK - Selección realizada únicamente con VALIDATION"
)

print(
    "OK - TEST permanece reservado"
)

print(
    "OK - OUT-OF-TIME permanece aislado"
)

# COMMAND ----------

# DBTITLE 1,08.04 SELECCIÓN DEL MEJOR MODELO
# ============================================================
# 08.04 SELECCIÓN Y ENTRENAMIENTO DEL MODELO FINAL
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Seleccionar la configuración final del Random Forest
# utilizando exclusivamente los resultados obtenidos sobre
# VALIDATION.
#
# Después de seleccionar los hiperparámetros:
#
# TRAIN + VALIDATION
#         ↓
# entrenamiento del modelo final
#         ↓
# TEST
# evaluación independiente posterior
#
# OUT-OF-TIME permanece completamente aislado.
#
#
# CRITERIO DE SELECCIÓN
# ------------------------------------------------------------
#
# El ganador técnico del tuning fue:
#
# RF_04_DEPTH_10
#
# Sin embargo, su mejora frente a RF_01_BASE fue marginal:
#
# ~0.0268 % relativo en WAPE.
#
# Por tanto, aplicamos el criterio definido en 08.03:
#
# - si la mejora es material -> ganador técnico
# - si la mejora es marginal -> modelo base más simple
#
# ============================================================


from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.sql import functions as F

import time
import gc


# ============================================================
# 08.04.01 VALIDACIONES PREVIAS
# ============================================================

required_variables = [

    "rf_tuning_results_df",
    "rf_configs",
    "best_config_id",
    "improvement_is_material",
    "train_df",
    "validation_df"

]


missing_variables = [

    variable

    for variable in required_variables

    if variable not in globals()

]


if missing_variables:

    raise RuntimeError(

        "Faltan variables necesarias para 08.04: "
        +
        ", ".join(missing_variables)

    )


# ============================================================
# 08.04.02 SELECCIÓN DE CONFIGURACIÓN
# ============================================================
#
# Si la mejora del ganador técnico es suficientemente
# relevante, utilizamos ese modelo.
#
# Si la mejora es marginal, preferimos RF_01_BASE por:
#
# - menor complejidad
# - menor profundidad
# - comportamiento prácticamente idéntico
# - mayor simplicidad operativa
#
# ============================================================

if improvement_is_material:

    selected_rf_config_id = (
        best_config_id
    )

    selection_reason = (
        "La mejora sobre RF_01_BASE es material."
    )

else:

    selected_rf_config_id = (
        "RF_01_BASE"
    )

    selection_reason = (
        "La mejora del ganador técnico es marginal; "
        "se mantiene RF_01_BASE por simplicidad."
    )


# ============================================================
# 08.04.03 RECUPERAMOS HIPERPARÁMETROS
# ============================================================

selected_rf_config = next(

    (

        config

        for config in rf_configs

        if config["config_id"]
        ==
        selected_rf_config_id

    ),

    None

)


if selected_rf_config is None:

    raise RuntimeError(

        f"No se encuentra {selected_rf_config_id} "
        "dentro de rf_configs."

    )


# ============================================================
# 08.04.04 RECUPERAMOS MÉTRICAS EN VALIDATION
# ============================================================

selected_validation_rows = (

    rf_tuning_results_df

    .loc[

        rf_tuning_results_df["config_id"]
        ==
        selected_rf_config_id

    ]

)


if len(selected_validation_rows) != 1:

    raise RuntimeError(

        "Debe existir exactamente una fila para "
        f"{selected_rf_config_id}."

    )


selected_validation_result = (

    selected_validation_rows

    .iloc[0]

)


selected_validation_mae = float(
    selected_validation_result["mae"]
)

selected_validation_rmse = float(
    selected_validation_result["rmse"]
)

selected_validation_wape = float(
    selected_validation_result["wape"]
)


# ============================================================
# 08.04.05 MOSTRAMOS DECISIÓN
# ============================================================

print("=" * 80)
print("SELECCIÓN DEL MODELO FINAL")
print("=" * 80)

print(
    f"Ganador técnico tuning: "
    f"{best_config_id}"
)

print(
    f"Mejora material:        "
    f"{improvement_is_material}"
)

print()

print(
    f"Configuración elegida:  "
    f"{selected_rf_config_id}"
)

print(
    f"Motivo:                 "
    f"{selection_reason}"
)

print()

print(
    f"numTrees:               "
    f"{selected_rf_config['numTrees']}"
)

print(
    f"maxDepth:               "
    f"{selected_rf_config['maxDepth']}"
)

print(
    f"minInstancesPerNode:    "
    f"{selected_rf_config['minInstancesPerNode']}"
)

print(
    f"featureSubsetStrategy:  "
    f"{selected_rf_config['featureSubsetStrategy']}"
)

print()

print(
    "Métricas obtenidas durante selección en VALIDATION:"
)

print(
    f"MAE:                    "
    f"{selected_validation_mae:,.2f}"
)

print(
    f"RMSE:                   "
    f"{selected_validation_rmse:,.2f}"
)

print(
    f"WAPE:                   "
    f"{selected_validation_wape:,.4f}%"
)


# ============================================================
# 08.04.06 CONSTRUIMOS DATASET FINAL DE ENTRENAMIENTO
# ============================================================
#
# Una vez cerrada la selección de hiperparámetros,
# VALIDATION deja de utilizarse para elegir modelos.
#
# Podemos incorporar TRAIN + VALIDATION para entrenar
# el modelo que posteriormente evaluaremos sobre TEST.
#
# TEST NO participa en este entrenamiento.
#
# ============================================================

final_training_df = (

    train_df

    .unionByName(
        validation_df
    )

)


final_training_rows = (
    final_training_df
    .count()
)


expected_final_training_rows = (

    train_df.count()

    +

    validation_df.count()

)


if final_training_rows != expected_final_training_rows:

    raise RuntimeError(

        "TRAIN + VALIDATION no conserva "
        "el número esperado de registros."

    )


# ============================================================
# 08.04.07 VALIDAMOS PERIODO TEMPORAL
# ============================================================

final_training_dates = (

    final_training_df

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


print()
print("=" * 80)
print("DATASET FINAL DE ENTRENAMIENTO")
print("=" * 80)

print(
    f"Registros: "
    f"{final_training_rows:,}"
)

print(
    f"Periodo:   "
    f"{final_training_dates['min_date']} "
    f"-> "
    f"{final_training_dates['max_date']}"
)


# ============================================================
# 08.04.08 DEFINIMOS RANDOM FOREST FINAL
# ============================================================

final_rf = RandomForestRegressor(

    featuresCol="features",

    labelCol="net_sales",

    predictionCol="prediction",

    numTrees=
        selected_rf_config[
            "numTrees"
        ],

    maxDepth=
        selected_rf_config[
            "maxDepth"
        ],

    minInstancesPerNode=
        selected_rf_config[
            "minInstancesPerNode"
        ],

    featureSubsetStrategy=
        selected_rf_config[
            "featureSubsetStrategy"
        ],

    seed=42

)


# ============================================================
# 08.04.09 PIPELINE FINAL
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
# 08.04.10 ENTRENAMIENTO FINAL
# ============================================================

print()
print("=" * 80)
print("ENTRENAMIENTO DEL RANDOM FOREST FINAL")
print("=" * 80)


training_start_time = (
    time.time()
)


best_rf_model = (

    final_rf_pipeline

    .fit(
        final_training_df
    )

)


final_training_seconds = (

    time.time()

    -

    training_start_time

)


if best_rf_model is None:

    raise RuntimeError(
        "No se ha generado el modelo Random Forest final."
    )


print(
    "OK - Random Forest final entrenado correctamente"
)

print(
    f"Tiempo: "
    f"{final_training_seconds:,.2f} segundos"
)


# ============================================================
# 08.04.11 LIMPIAMOS REFERENCIAS AUXILIARES
# ============================================================

del final_rf_pipeline
del final_rf

gc.collect()


# ============================================================
# 08.04.12 VALIDACIÓN FINAL
# ============================================================

print()
print("=" * 80)
print("VALIDACIÓN FINAL")
print("=" * 80)

print(
    f"OK - Configuración seleccionada: "
    f"{selected_rf_config_id}"
)

print(
    "OK - Hiperparámetros seleccionados exclusivamente "
    "mediante VALIDATION"
)

print(
    "OK - Modelo final reentrenado con TRAIN + VALIDATION"
)

print(
    "OK - TEST no ha participado en entrenamiento ni selección"
)

print(
    "OK - OUT-OF-TIME permanece completamente aislado"
)

print(
    "OK - best_rf_model preparado para evaluación final en TEST"
)

# COMMAND ----------

# DBTITLE 1,08.05 COMPARACIÓN GLOBAL EN VALIDATION
# ============================================================
# 08.05 COMPARACIÓN GLOBAL EN VALIDATION
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Resumir el rendimiento de los modelos evaluados durante
# la fase de selección utilizando exclusivamente VALIDATION.
#
# Comparamos:
#
# - Seasonal Naive
# - Random Forest base
# - Gradient Boosting
# - Mejor configuración técnica del tuning
#
# IMPORTANTE
# ------------------------------------------------------------
#
# El ganador técnico del tuning no tiene por qué coincidir
# con el modelo finalmente seleccionado.
#
# En nuestro criterio:
#
# - si la mejora es material -> usamos el ganador técnico
# - si la mejora es marginal -> mantenemos el modelo base
#
# TEST y OUT-OF-TIME siguen fuera de esta comparación.
#
# ============================================================


# ============================================================
# 08.05.01 VALIDAMOS VARIABLES NECESARIAS
# ============================================================

required_variables = [

    "baseline_mae",
    "baseline_rmse",
    "baseline_wape",

    "rf_mae",
    "rf_rmse",
    "rf_wape",

    "gbt_mae",
    "gbt_rmse",
    "gbt_wape",

    "rf_tuning_results_df",

    "best_config_id",

    "selected_rf_config_id",

    "improvement_is_material"

]


missing_variables = [

    variable

    for variable in required_variables

    if variable not in globals()

]


if missing_variables:

    raise RuntimeError(

        "Faltan variables necesarias para 08.05: "
        +
        ", ".join(missing_variables)

    )


# ============================================================
# 08.05.02 NORMALIZAMOS MÉTRICAS A FLOAT
# ============================================================

baseline_mae_final = float(
    baseline_mae
)

baseline_rmse_final = float(
    baseline_rmse
)

baseline_wape_final = float(
    baseline_wape
)


rf_mae_final = float(
    rf_mae
)

rf_rmse_final = float(
    rf_rmse
)

rf_wape_final = float(
    rf_wape
)


gbt_mae_final = float(
    gbt_mae
)

gbt_rmse_final = float(
    gbt_rmse
)

gbt_wape_final = float(
    gbt_wape
)


# ============================================================
# 08.05.03 RECUPERAMOS GANADOR TÉCNICO DEL TUNING
# ============================================================

best_tuned_rows = (

    rf_tuning_results_df

    .loc[
        rf_tuning_results_df["config_id"]
        ==
        best_config_id
    ]

)


if len(best_tuned_rows) != 1:

    raise RuntimeError(

        "Debe existir exactamente una fila para "
        f"{best_config_id}."

    )


best_tuned_result = (

    best_tuned_rows

    .iloc[0]

)


best_tuned_mae = float(
    best_tuned_result["mae"]
)

best_tuned_rmse = float(
    best_tuned_result["rmse"]
)

best_tuned_wape = float(
    best_tuned_result["wape"]
)


# ============================================================
# 08.05.04 COMPARACIÓN GLOBAL
# ============================================================

print("=" * 80)
print("COMPARACIÓN FINAL DE MODELOS - VALIDATION")
print("=" * 80)


print(
    f"{'Modelo':28} | "
    f"{'MAE':>10} | "
    f"{'RMSE':>10} | "
    f"{'WAPE':>9}"
)

print("-" * 72)


print(
    f"{'Seasonal Naive':28} | "
    f"{baseline_mae_final:>10,.2f} | "
    f"{baseline_rmse_final:>10,.2f} | "
    f"{baseline_wape_final:>8,.2f}%"
)


print(
    f"{'Random Forest Base':28} | "
    f"{rf_mae_final:>10,.2f} | "
    f"{rf_rmse_final:>10,.2f} | "
    f"{rf_wape_final:>8,.2f}%"
)


print(
    f"{'Gradient Boosting':28} | "
    f"{gbt_mae_final:>10,.2f} | "
    f"{gbt_rmse_final:>10,.2f} | "
    f"{gbt_wape_final:>8,.2f}%"
)


print(
    f"{best_config_id:28} | "
    f"{best_tuned_mae:>10,.2f} | "
    f"{best_tuned_rmse:>10,.2f} | "
    f"{best_tuned_wape:>8,.2f}%"
)


# ============================================================
# 08.05.05 MEJORA TUNING VS RANDOM FOREST BASE
# ============================================================

tuning_wape_absolute_gain = (

    rf_wape_final

    -

    best_tuned_wape

)


tuning_wape_relative_gain = (

    tuning_wape_absolute_gain

    /

    rf_wape_final

    *

    100.0

)


print()
print("=" * 80)
print("IMPACTO DEL TUNING")
print("=" * 80)

print(
    f"Random Forest base WAPE:   "
    f"{rf_wape_final:.4f}%"
)

print(
    f"Mejor tuning WAPE:         "
    f"{best_tuned_wape:.4f}%"
)

print(
    f"Ganancia absoluta:         "
    f"{tuning_wape_absolute_gain:.4f} puntos"
)

print(
    f"Ganancia relativa:         "
    f"{tuning_wape_relative_gain:.4f}%"
)


# ============================================================
# 08.05.06 MODELO FINAL SELECCIONADO
# ============================================================

print()
print("=" * 80)
print("DECISIÓN FINAL DE SELECCIÓN")
print("=" * 80)

print(
    f"Ganador técnico tuning:    "
    f"{best_config_id}"
)

print(
    f"Mejora material:           "
    f"{improvement_is_material}"
)

print(
    f"Modelo finalmente elegido: "
    f"{selected_rf_config_id}"
)


if selected_rf_config_id == best_config_id:

    print(
        "DECISIÓN - Se adopta el ganador técnico del tuning."
    )

else:

    print(
        "DECISIÓN - No se adopta el ganador técnico porque "
        "la mejora es marginal."
    )

    print(
        "DECISIÓN - Se mantiene RF_01_BASE por simplicidad "
        "y menor complejidad."
    )


# ============================================================
# 08.05.07 VALIDACIÓN FINAL
# ============================================================

print()
print("=" * 80)
print("VALIDACIÓN FINAL")
print("=" * 80)

print(
    "OK - Baseline incluido en la comparación"
)

print(
    "OK - Random Forest base incluido"
)

print(
    "OK - Gradient Boosting incluido"
)

print(
    "OK - Mejor configuración del tuning incluida"
)

print(
    "OK - Decisión final documentada"
)

print(
    "OK - Comparación realizada únicamente sobre VALIDATION"
)

print(
    "OK - TEST permanece reservado"
)

print(
    "OK - OUT-OF-TIME permanece aislado"
)

# COMMAND ----------

# DBTITLE 1,08. MATERIALIZACIÓN DEL DATASET DE FEATURES ML
# ============================================================
# 08. MATERIALIZACIÓN DEL DATASET DE FEATURES ML
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Persistir el dataset preparado para Machine Learning como
# tabla Delta dentro del schema específico de ML.
#
# Hasta este punto ml_model_df existe como DataFrame dentro
# de la sesión Spark.
#
# Al materializarlo:
#
# - evitamos reconstruir todo el Feature Engineering
# - desacoplamos Data Engineering de Model Training
# - permitimos reutilizar el dataset desde otros notebooks
# - facilitamos entrenamiento, inferencia y monitoring
# - mantenemos persistencia entre sesiones
#
# IMPORTANTE
# ------------------------------------------------------------
#
# NO utilizamos conteos hardcodeados.
#
# La tabla persistida debe contener exactamente los mismos
# registros que ml_model_df en el momento de escritura.
#
# ============================================================


from pyspark.sql import functions as F


# ============================================================
# 08.01 VALIDAMOS DATASET ORIGEN
# ============================================================

if "ml_model_df" not in globals():

    raise RuntimeError(
        "ml_model_df no existe. "
        "Ejecuta primero el Feature Engineering."
    )


source_rows = (
    ml_model_df
    .count()
)


source_duplicates = (

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


source_null_target = (

    ml_model_df

    .filter(
        F.col("net_sales").isNull()
    )

    .count()

)


source_dates = (

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


print("=" * 80)
print("VALIDACIÓN DATASET ML ORIGEN")
print("=" * 80)

print(
    f"Registros:       {source_rows:,}"
)

print(
    f"Duplicados:      {source_duplicates:,}"
)

print(
    f"Target NULL:     {source_null_target:,}"
)

print(
    f"Periodo:         "
    f"{source_dates['min_date']} "
    f"-> "
    f"{source_dates['max_date']}"
)


if source_rows == 0:

    raise RuntimeError(
        "ml_model_df está vacío."
    )


if source_duplicates > 0:

    raise RuntimeError(
        "ml_model_df contiene duplicados por sale_date + store_id."
    )


if source_null_target > 0:

    raise RuntimeError(
        "ml_model_df contiene NULLs en net_sales."
    )


print(
    "OK - Dataset origen válido"
)


# ============================================================
# 08.02 CREAMOS SCHEMA MACHINE LEARNING
# ============================================================

spark.sql(
    """
    CREATE SCHEMA IF NOT EXISTS retail_analytics.5_ml
    """
)


# ============================================================
# 08.03 TABLA DESTINO
# ============================================================

ml_features_table = (
    "retail_analytics.5_ml.daily_store_features"
)


# ============================================================
# 08.04 MATERIALIZAMOS DATASET
# ============================================================
#
# Utilizamos overwrite porque esta tabla representa el
# snapshot completo y actualizado del Feature Engineering.
#
# ============================================================

(

    ml_model_df

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
        ml_features_table
    )

)


# ============================================================
# 08.05 RECARGAMOS TABLA MATERIALIZADA
# ============================================================
#
# Validamos leyendo de nuevo desde el catálogo.
#
# De esta forma comprobamos realmente la tabla persistida,
# no el DataFrame que permanece en memoria.
#
# ============================================================

ml_features_persisted_df = (

    spark.table(
        ml_features_table
    )

)


persisted_rows = (
    ml_features_persisted_df
    .count()
)


persisted_duplicates = (

    ml_features_persisted_df

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


persisted_null_target = (

    ml_features_persisted_df

    .filter(
        F.col("net_sales").isNull()
    )

    .count()

)


persisted_dates = (

    ml_features_persisted_df

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
# 08.06 COMPARACIÓN ORIGEN VS TABLA
# ============================================================

row_difference = (

    persisted_rows

    -

    source_rows

)


date_range_ok = (

    persisted_dates["min_date"]
    ==
    source_dates["min_date"]

    and

    persisted_dates["max_date"]
    ==
    source_dates["max_date"]

)


# ============================================================
# 08.07 RESULTADO
# ============================================================

print()
print("=" * 80)
print("DATASET ML MATERIALIZADO")
print("=" * 80)

print(
    f"Tabla:           {ml_features_table}"
)

print(
    f"Origen:          {source_rows:,}"
)

print(
    f"Persistidos:     {persisted_rows:,}"
)

print(
    f"Diferencia:      {row_difference:,}"
)

print(
    f"Duplicados:      {persisted_duplicates:,}"
)

print(
    f"Target NULL:     {persisted_null_target:,}"
)

print(
    f"Periodo:         "
    f"{persisted_dates['min_date']} "
    f"-> "
    f"{persisted_dates['max_date']}"
)


# ============================================================
# 08.08 VALIDACIÓN FINAL
# ============================================================

validation_errors = []


if persisted_rows != source_rows:

    validation_errors.append(
        "El número de registros persistidos "
        "no coincide con ml_model_df."
    )


if persisted_duplicates > 0:

    validation_errors.append(
        "La tabla persistida contiene duplicados "
        "por sale_date + store_id."
    )


if persisted_null_target > 0:

    validation_errors.append(
        "La tabla persistida contiene NULLs en net_sales."
    )


if not date_range_ok:

    validation_errors.append(
        "El rango de fechas de la tabla persistida "
        "no coincide con el dataset origen."
    )


if validation_errors:

    print()
    print("=" * 80)
    print("ERRORES DE VALIDACIÓN")
    print("=" * 80)

    for error in validation_errors:

        print(
            f"ERROR - {error}"
        )

    raise RuntimeError(
        "La materialización del dataset ML "
        "no ha superado las validaciones."
    )


print()
print("=" * 80)
print("VALIDACIÓN FINAL")
print("=" * 80)

print(
    "OK - Dataset ML persistido correctamente"
)

print(
    "OK - Conteo origen y destino coinciden"
)

print(
    "OK - No existen duplicados"
)

print(
    "OK - Target sin NULLs"
)

print(
    "OK - Rango temporal conservado"
)

print(
    f"OK - Tabla disponible: {ml_features_table}"
)