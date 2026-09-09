# Databricks notebook source
# COMMAND ----------

# Catálogo recibido desde Databricks Asset Bundles.
# DEV  -> retail_analytics_dev
# PROD -> retail_analytics

dbutils.widgets.text("catalog", "retail_analytics")
CATALOG = dbutils.widgets.get("catalog")

print(f"Environment catalog: {CATALOG}")

# DBTITLE 1,00. CONFIGURACIÓN Y LECTURA DE TABLAS SILVER
# ============================================================
# 00. CONFIGURACIÓN Y LECTURA DE TABLAS SILVER
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Este notebook construirá la capa Gold del proyecto.
#
# A diferencia de Silver, donde limpiábamos y estandarizábamos
# cada entidad, Gold estará orientada al consumo analítico.
#
# Vamos a construir un modelo dimensional tipo STAR SCHEMA:
#
#
#                     dim_customer
#                          |
#                          |
# dim_date ----------- fact_sales ----------- dim_product
#                          |
#                          |
#                      dim_store
#
#
# La tabla central será fact_sales.
#
# Las dimensiones aportarán el contexto necesario para analizar
# las ventas por:
#
# - cliente
# - producto
# - tienda
# - fecha
#
# IMPORTANTE:
#
# Gold NO leerá directamente de Bronze.
#
# El flujo correcto será:
#
# Landing
#    ↓
# Bronze
#    ↓
# Silver
#    ↓
# Gold
#
# De esta forma Gold trabaja exclusivamente con datos que ya
# han pasado nuestras reglas de Data Quality.
#
# ============================================================


# ============================================================
# 1. DEFINICIÓN DE LAS TABLAS SILVER DE ORIGEN
# ============================================================
#
# Guardamos los nombres completos de las tablas en un
# diccionario para evitar repetir strings por todo el notebook.
#
# Namespace de Unity Catalog:
#
#       catalog.schema.table
#
# ============================================================

silver_tables = {

    "sales":
        f"{CATALOG}.2_silver.sales",

    "customers":
        f"{CATALOG}.2_silver.customers",

    "products":
        f"{CATALOG}.2_silver.products",

    "stores":
        f"{CATALOG}.2_silver.stores"
}


# ============================================================
# 2. DEFINICIÓN DE LAS TABLAS GOLD DE DESTINO
# ============================================================
#
# En Gold utilizamos una nomenclatura dimensional:
#
# fact_
#     identifica tablas de hechos.
#
# dim_
#     identifica dimensiones.
#
# ============================================================

gold_tables = {

    "fact_sales":
        f"{CATALOG}.3_gold.fact_sales",

    "dim_customer":
        f"{CATALOG}.3_gold.dim_customer",

    "dim_product":
        f"{CATALOG}.3_gold.dim_product",

    "dim_store":
        f"{CATALOG}.3_gold.dim_store",

    "dim_date":
        f"{CATALOG}.3_gold.dim_date"
}


# ============================================================
# 3. VALIDACIÓN DE EXISTENCIA DE LAS FUENTES SILVER
# ============================================================
#
# Antes de leer ningún DataFrame comprobamos que todas las
# tablas necesarias existan realmente en Unity Catalog.
#
# Esto permite que el Job falle de forma controlada si una
# dependencia upstream no está disponible.
#
# ============================================================

silver_source_errors = []


print()
print("=" * 80)
print("VALIDACIÓN DE FUENTES SILVER")
print("=" * 80)


for table_name, full_table_name in silver_tables.items():

    table_exists = spark.catalog.tableExists(
        full_table_name
    )


    if not table_exists:

        silver_source_errors.append(
            f"No existe la tabla Silver: {full_table_name}"
        )

        print(
            f"ERROR | "
            f"{table_name:10} | "
            f"Tabla no encontrada"
        )

    else:

        print(
            f"OK    | "
            f"{table_name:10} | "
            f"{full_table_name}"
        )


# ============================================================
# 4. CONTROL DE EXISTENCIA
# ============================================================

if silver_source_errors:

    print()
    print("ERRORES DETECTADOS")
    print("-" * 80)

    for error in silver_source_errors:

        print(
            f"- {error}"
        )


    raise RuntimeError(
        "No están disponibles todas las fuentes Silver "
        "necesarias para construir Gold"
    )


# ============================================================
# 5. LECTURA DE LAS TABLAS SILVER
# ============================================================
#
# Una vez confirmada su existencia, leemos las cuatro tablas.
#
# spark.table() devuelve un DataFrame directamente desde
# Unity Catalog.
#
# ============================================================

sales_silver_df = spark.table(
    silver_tables["sales"]
)


customers_silver_df = spark.table(
    silver_tables["customers"]
)


products_silver_df = spark.table(
    silver_tables["products"]
)


stores_silver_df = spark.table(
    silver_tables["stores"]
)


# ============================================================
# 6. RECUENTOS ACTUALES DE LAS FUENTES
# ============================================================
#
# IMPORTANTE:
#
# NO utilizamos valores hardcodeados.
#
# Anteriormente el notebook esperaba:
#
# sales      -> 536.340
# customers  -> 10.000
# products   -> 1.000
# stores     -> 31
#
# SALES crece de manera incremental, por lo que ese enfoque
# deja de ser válido en cuanto llegan nuevos datos.
#
# Guardamos los recuentos reales para poder reutilizarlos
# posteriormente en las validaciones de Gold.
#
# ============================================================

silver_source_counts = {

    "sales":
        sales_silver_df.count(),

    "customers":
        customers_silver_df.count(),

    "products":
        products_silver_df.count(),

    "stores":
        stores_silver_df.count()
}


# ============================================================
# 7. VALIDACIÓN DE TABLAS VACÍAS
# ============================================================

empty_silver_tables = []


print()
print("=" * 80)
print("VOLUMEN ACTUAL DE FUENTES SILVER")
print("=" * 80)


for table_name, row_count in silver_source_counts.items():

    if row_count == 0:

        status = "ERROR"

        empty_silver_tables.append(
            table_name
        )

    else:

        status = "OK"


    print(
        f"{status:5} | "
        f"{table_name:10} | "
        f"{row_count:>10,} registros"
    )


# ============================================================
# 8. CONTROL DE TABLAS VACÍAS
# ============================================================

if empty_silver_tables:

    raise RuntimeError(

        "Existen tablas Silver vacías: "
        +
        ", ".join(
            empty_silver_tables
        )
    )


# ============================================================
# 9. VALIDACIONES ESTRUCTURALES BÁSICAS
# ============================================================
#
# Además de comprobar que existen registros, verificamos las
# claves principales de las fuentes que Gold necesita.
#
# ============================================================


required_columns = {

    "sales": {
        "sale_id",
        "line_id",
        "sale_date",
        "customer_id",
        "product_id",
        "store_id",
        "quantity",
        "unit_price",
        "discount_pct",
        "gross_amount",
        "discount_amount",
        "net_amount",
        "sales_channel",
        "payment_method"
    },

    "customers": {
        "customer_id"
    },

    "products": {
        "product_id"
    },

    "stores": {
        "store_id"
    }
}


missing_columns_errors = []


for table_name, expected_columns in required_columns.items():

    actual_columns = set(

        spark.table(
            silver_tables[
                table_name
            ]
        ).columns
    )


    missing_columns = (

        expected_columns
        -
        actual_columns
    )


    if missing_columns:

        missing_columns_errors.append(

            f"{table_name}: "
            f"{sorted(missing_columns)}"
        )


# ============================================================
# 10. CONTROL DE SCHEMA
# ============================================================

if missing_columns_errors:

    print()
    print("=" * 80)
    print("ERRORES DE SCHEMA")
    print("=" * 80)


    for error in missing_columns_errors:

        print(
            f"- {error}"
        )


    raise RuntimeError(
        "Las fuentes Silver no cumplen "
        "el contrato esperado por Gold"
    )


# ============================================================
# 11. RESULTADO FINAL
# ============================================================

print()
print("=" * 80)

print(
    "OK - Configuración y lectura de fuentes Silver "
    "completadas correctamente"
)

print("=" * 80)


print()
print("RECUENTOS DISPONIBLES PARA GOLD")
print("-" * 80)

print(
    f"SALES:     "
    f"{silver_source_counts['sales']:,}"
)

print(
    f"CUSTOMERS: "
    f"{silver_source_counts['customers']:,}"
)

print(
    f"PRODUCTS:  "
    f"{silver_source_counts['products']:,}"
)

print(
    f"STORES:    "
    f"{silver_source_counts['stores']:,}"
)

# COMMAND ----------

# DBTITLE 1,01. DEFINICIÓN DE GRANULARIDAD Y DISEÑO DEL MODELO GOLD
# ============================================================
# 01. DEFINICIÓN DE GRANULARIDAD Y DISEÑO DEL MODELO GOLD
# ============================================================
#
# Antes de construir físicamente las tablas Gold definimos
# el modelo analítico que queremos obtener.
#
# Esto es importante porque la capa Gold ya no está orientada
# simplemente a almacenar datos limpios.
#
# Gold debe estar diseñada para:
#
# - análisis
# - reporting
# - dashboards
# - SQL Views
# - futuros modelos de Machine Learning
#
# ============================================================


# ------------------------------------------------------------
# GRANULARIDAD DE FACT_SALES
# ------------------------------------------------------------
#
# Una fila de fact_sales representará:
#
#     UNA LÍNEA DE TICKET
#
# Clave natural de la fila:
#
#     sale_id + line_id
#
# Ejemplo:
#
# Ticket T000001:
#
# sale_id    line_id    product_id
# --------------------------------
# T000001       1          P001
# T000001       2          P010
# T000001       3          P023
#
# El ticket tiene 3 líneas y, por tanto,
# 3 registros en fact_sales.
#
# Esto permitirá calcular:
#
# Número de tickets:
#
#     COUNT(DISTINCT sale_id)
#
# Número de líneas:
#
#     COUNT(*)
#
# Unidades:
#
#     SUM(quantity)
#
# Ventas:
#
#     SUM(net_amount)
#
# ------------------------------------------------------------


fact_sales_grain = (
    "1 fila = 1 línea de ticket "
    "(sale_id + line_id)"
)


# ------------------------------------------------------------
# DIMENSIONES DEL MODELO
# ------------------------------------------------------------
#
# dim_customer
#     Información descriptiva del cliente.
#
# dim_product
#     Información descriptiva del producto.
#
# dim_store
#     Información descriptiva de la tienda.
#
# dim_date
#     Dimensión temporal para análisis por fechas.
#
gold_dimensions = [
    "dim_customer",
    "dim_product",
    "dim_store",
    "dim_date"
]


# ------------------------------------------------------------
# MEDIDAS PRINCIPALES DE FACT_SALES
# ------------------------------------------------------------
#
# Estas columnas podrán agregarse posteriormente en:
#
# SQL
# dashboards
# vistas
# modelos analíticos
#
fact_sales_measures = [
    "quantity",
    "unit_price",
    "discount_pct",
    "gross_amount",
    "discount_amount",
    "net_amount"
]


# ------------------------------------------------------------
# CLAVES DE FACT_SALES
# ------------------------------------------------------------
#
# La fact se relacionará con las dimensiones mediante
# las siguientes claves.
#
# Para customer_id debemos recordar que puede ser NULL,
# porque existen ventas anónimas.
#
fact_sales_dimension_keys = [
    "date_key",
    "customer_id",
    "product_id",
    "store_id"
]


# ------------------------------------------------------------
# RESUMEN DEL DISEÑO
# ------------------------------------------------------------

print("MODELO GOLD")
print("=" * 70)

print(
    "\nGranularidad fact_sales:"
)

print(
    fact_sales_grain
)

print(
    "\nDimensiones:"
)

for dimension in gold_dimensions:

    print(
        f" - {dimension}"
    )


print(
    "\nMedidas principales:"
)

for measure in fact_sales_measures:

    print(
        f" - {measure}"
    )


print(
    "\nClaves hacia dimensiones:"
)

for key in fact_sales_dimension_keys:

    print(
        f" - {key}"
    )

# COMMAND ----------

# DBTITLE 1,02. CREACIÓN DE DIM_DATE
# ============================================================
# 02. CREACIÓN DE DIM_DATE
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Crear una dimensión calendario independiente que permita
# analizar las ventas por:
#
# - día
# - mes
# - trimestre
# - año
# - semana
# - día de la semana
#
# Esta dimensión será especialmente útil posteriormente para:
#
# - SQL Views
# - dashboards
# - análisis temporal
# - forecasting
#
# ============================================================


from pyspark.sql import functions as F


# ------------------------------------------------------------
# OBTENEMOS EL RANGO TEMPORAL REAL DE SALES
# ------------------------------------------------------------
#
# En lugar de escribir manualmente:
#
# 2024-01-01
# 2025-12-31
#
# obtenemos las fechas directamente de Silver.
#
# Así la dimensión se adapta automáticamente si en el futuro
# cargamos datos de 2026.
#
date_range = (
    sales_silver_df
    .select(
        F.min("sale_date").alias("min_date"),
        F.max("sale_date").alias("max_date")
    )
    .collect()[0]
)


min_date = date_range["min_date"]
max_date = date_range["max_date"]


print(
    "Fecha mínima:",
    min_date
)

print(
    "Fecha máxima:",
    max_date
)

# COMMAND ----------

# DBTITLE 1,02.01 GENERACIÓN DEL CALENDARIO
# ============================================================
# 02.01 GENERACIÓN DEL CALENDARIO
# ============================================================
#
# sequence() genera una secuencia entre min_date y max_date.
#
# explode() convierte esa secuencia en:
#
# una fila = una fecha
#
# ============================================================


dim_date_df = (

    spark.range(1)

    # --------------------------------------------------------
    # GENERAMOS TODAS LAS FECHAS
    # --------------------------------------------------------

    .select(
        F.explode(
            F.sequence(
                F.lit(min_date),
                F.lit(max_date),
                F.expr("INTERVAL 1 DAY")
            )
        ).alias("date")
    )


    # ========================================================
    # CLAVE DE LA DIMENSIÓN
    # ========================================================
    #
    # Construimos una date_key numérica:
    #
    # 2024-01-01 -> 20240101
    #
    # Este formato es habitual en modelos dimensionales.
    #
    # La fact utilizará esta clave para relacionarse
    # con dim_date.
    # ========================================================

    .withColumn(
        "date_key",
        F.date_format(
            F.col("date"),
            "yyyyMMdd"
        ).cast("int")
    )


    # ========================================================
    # ATRIBUTOS DE CALENDARIO
    # ========================================================

    .withColumn(
        "year",
        F.year("date")
    )

    .withColumn(
        "quarter",
        F.quarter("date")
    )

    .withColumn(
        "month",
        F.month("date")
    )

    .withColumn(
        "month_name",
        F.date_format(
            F.col("date"),
            "MMMM"
        )
    )

    .withColumn(
        "year_month",
        F.date_format(
            F.col("date"),
            "yyyy-MM"
        )
    )

    .withColumn(
        "week_of_year",
        F.weekofyear("date")
    )

    .withColumn(
        "day_of_month",
        F.dayofmonth("date")
    )

    .withColumn(
        "day_of_week",
        F.dayofweek("date")
    )

    .withColumn(
        "day_name",
        F.date_format(
            F.col("date"),
            "EEEE"
        )
    )

    # --------------------------------------------------------
    # WEEKEND
    # --------------------------------------------------------
    #
    # dayofweek():
    #
    # 1 = Sunday
    # 7 = Saturday
    #
    # Generamos un indicador booleano para facilitar
    # análisis posteriores.
    #
    .withColumn(
        "is_weekend",
        F.col("day_of_week").isin(
            1,
            7
        )
    )


    # --------------------------------------------------------
    # ORDEN DE COLUMNAS
    # --------------------------------------------------------

    .select(
        "date_key",
        "date",
        "year",
        "quarter",
        "month",
        "month_name",
        "year_month",
        "week_of_year",
        "day_of_month",
        "day_of_week",
        "day_name",
        "is_weekend"
    )
)

# COMMAND ----------

# DBTITLE 1,02.02 VALIDACIÓN DE DIM_DATE
# ============================================================
# 02.02 VALIDACIÓN DE DIM_DATE
# ============================================================

display(
    dim_date_df
    .orderBy("date")
    .limit(20)
)


print(
    "Número de fechas:",
    dim_date_df.count()
)

# COMMAND ----------

# DBTITLE 1,02.03 VALIDACIONES DE DIM_DATE
# ============================================================
# 02.03 VALIDACIONES DE DIM_DATE
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Validar que la dimensión temporal:
#
# - tenga una fila por cada fecha del periodo real
# - no contenga date_key duplicadas
# - no contenga fechas duplicadas
# - no tenga NULL en columnas clave
#
# IMPORTANTE
# ------------------------------------------------------------
#
# NO utilizamos un número fijo de días.
#
# El número esperado se calcula dinámicamente a partir de:
#
#       min_date
#       max_date
#
# obtenidos previamente desde SALES Silver.
#
# De esta forma la validación seguirá funcionando cuando
# lleguen nuevos meses o nuevos años.
#
# ============================================================


from pyspark.sql import functions as F


# ============================================================
# 1. DUPLICADOS EN DATE_KEY
# ============================================================

duplicated_date_keys = (

    dim_date_df

    .groupBy(
        "date_key"
    )

    .count()

    .filter(
        F.col("count") > 1
    )

    .count()
)


# ============================================================
# 2. DUPLICADOS EN DATE
# ============================================================

duplicated_dates = (

    dim_date_df

    .groupBy(
        "date"
    )

    .count()

    .filter(
        F.col("count") > 1
    )

    .count()
)


# ============================================================
# 3. NULLS EN COLUMNAS CLAVE
# ============================================================

null_key_values = (

    dim_date_df

    .filter(

        F.col("date_key").isNull()

        |

        F.col("date").isNull()

        |

        F.col("year").isNull()

        |

        F.col("month").isNull()
    )

    .count()
)


# ============================================================
# 4. RECUENTO REAL DE DIM_DATE
# ============================================================

date_count = (
    dim_date_df
    .count()
)


# ============================================================
# 5. RECUENTO ESPERADO DINÁMICO
# ============================================================
#
# datediff(max_date, min_date) devuelve el número de días
# entre ambas fechas.
#
# Sumamos 1 porque ambas fechas deben incluirse:
#
# 2026-08-01 -> 2026-08-01 = 1 día
#
# ============================================================

expected_date_count = (

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
        .alias(
            "expected_date_count"
        )
    )

    .first()[
        "expected_date_count"
    ]
)


# ============================================================
# 6. DIFERENCIA DE RECUENTO
# ============================================================

date_count_difference = abs(
    date_count
    -
    expected_date_count
)


# ============================================================
# 7. RESULTADOS
# ============================================================

dim_date_validation = {

    "date_key duplicada":
        duplicated_date_keys,

    "date duplicada":
        duplicated_dates,

    "NULL en columnas clave":
        null_key_values,

    "Diferencia vs rango temporal esperado":
        date_count_difference
}


print()
print("=" * 80)
print("VALIDACIÓN DE DIM_DATE")
print("=" * 80)

print(
    f"Periodo: "
    f"{min_date} -> {max_date}"
)

print(
    f"Fechas esperadas: "
    f"{expected_date_count:,}"
)

print(
    f"Fechas generadas: "
    f"{date_count:,}"
)

print()
print("-" * 80)


dim_date_errors = []


for rule, errors in dim_date_validation.items():

    status = (
        "OK"
        if errors == 0
        else "ERROR"
    )


    print(
        f"{status:5} | "
        f"{rule:40} | "
        f"{errors:,}"
    )


    if errors != 0:

        dim_date_errors.append(
            f"{rule}: {errors:,}"
        )


# ============================================================
# 8. CONTROL FINAL
# ============================================================

if dim_date_errors:

    print()
    print("ERRORES DETECTADOS")

    for error in dim_date_errors:

        print(
            f"- {error}"
        )


    raise RuntimeError(
        "La validación de DIM_DATE "
        "ha detectado inconsistencias"
    )


print()
print(
    "OK - DIM_DATE generada correctamente "
    "para todo el periodo disponible"
)

# COMMAND ----------

# DBTITLE 1,02.04 ESCRITURA DE DIM_DATE EN GOLD
# ============================================================
# 02.04 ESCRITURA DE DIM_DATE EN GOLD
# ============================================================

dim_date_table = (
    f"{CATALOG}.3_gold.dim_date"
)


(
    dim_date_df

    .write

    .format("delta")

    .mode("overwrite")

    .option(
        "overwriteSchema",
        "true"
    )

    .saveAsTable(
        dim_date_table
    )
)

# COMMAND ----------

# DBTITLE 1,02.05 VALIDACIÓN DE DIM_DATE MATERIALIZADA
# ============================================================
# 02.05 VALIDACIÓN DE DIM_DATE MATERIALIZADA
# ============================================================

dim_date_gold_count = spark.sql(f"""
    SELECT COUNT(*)
    FROM {CATALOG}.`3_gold`.dim_date
""").collect()[0][0]


print(
    "Filas en dim_date:",
    dim_date_gold_count
)

# COMMAND ----------

# DBTITLE 1,03. CREACIÓN DE DIM_CUSTOMER
# ============================================================
# 03. CREACIÓN DE DIM_CUSTOMER
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Construir la dimensión de clientes que utilizaremos
# posteriormente desde fact_sales.
#
# Silver ya contiene los datos:
#
# - limpios
# - tipados
# - normalizados
# - validados
#
# Por tanto, Gold NO vuelve a limpiar el dato.
#
# Aquí seleccionamos únicamente los atributos que tienen
# valor analítico para la dimensión.
# ============================================================


from pyspark.sql import functions as F


# ------------------------------------------------------------
# CONSTRUCCIÓN DE DIM_CUSTOMER
# ------------------------------------------------------------

dim_customer_df = (

    customers_silver_df

    .select(

        # ----------------------------------------------------
        # CLAVE DE NEGOCIO
        # ----------------------------------------------------
        #
        # customer_id será la clave utilizada por fact_sales
        # para relacionarse con esta dimensión.
        #
        F.col("customer_id"),


        # ----------------------------------------------------
        # ATRIBUTOS DESCRIPTIVOS
        # ----------------------------------------------------

        F.col("customer_name"),

        F.col("email"),

        F.col("city"),

        F.col("region"),

        F.col("registration_date"),

        F.col("customer_segment")
    )
)

# COMMAND ----------

# DBTITLE 1,03.01 INSPECCIÓN DE DIM_CUSTOMER
# ============================================================
# 03.01 INSPECCIÓN DE DIM_CUSTOMER
# ============================================================

display(
    dim_customer_df
    .orderBy("customer_id")
    .limit(20)
)

# COMMAND ----------

# DBTITLE 1,03.02 VALIDACIONES DE DIM_CUSTOMER
# ============================================================
# 03.02 VALIDACIONES DE DIM_CUSTOMER
# ============================================================
#
# Comprobamos:
#
# - 10.000 clientes
# - customer_id no NULL
# - customer_id único
# - segmentos válidos
# ============================================================


# ------------------------------------------------------------
# RECUENTO
# ------------------------------------------------------------

customer_count = (
    dim_customer_df.count()
)


# ------------------------------------------------------------
# CUSTOMER_ID NULL
# ------------------------------------------------------------

null_customer_ids = (
    dim_customer_df
    .filter(
        F.col("customer_id").isNull()
    )
    .count()
)


# ------------------------------------------------------------
# CUSTOMER_ID DUPLICADO
# ------------------------------------------------------------

duplicated_customer_ids = (

    dim_customer_df

    .groupBy("customer_id")

    .count()

    .filter(
        F.col("count") > 1
    )

    .count()
)


# ------------------------------------------------------------
# SEGMENTOS INVÁLIDOS
# ------------------------------------------------------------

invalid_customer_segments = (

    dim_customer_df

    .filter(
        ~F.col("customer_segment").isin(
            "STANDARD",
            "PREMIUM",
            "VIP"
        )
        |
        F.col("customer_segment").isNull()
    )

    .count()
)


# ------------------------------------------------------------
# RESULTADOS
# ------------------------------------------------------------

dim_customer_validation = {

    "Diferencia vs 10.000 clientes":
        abs(customer_count - 10000),

    "customer_id NULL":
        null_customer_ids,

    "customer_id duplicado":
        duplicated_customer_ids,

    "customer_segment inválido":
        invalid_customer_segments
}


for rule, errors in dim_customer_validation.items():

    status = (
        "OK"
        if errors == 0
        else "ERROR"
    )

    print(
        f"{status:5} | "
        f"{rule:40} | "
        f"{errors:,}"
    )

# COMMAND ----------

# DBTITLE 1,03.03 ESCRITURA DE DIM_CUSTOMER EN GOLD
# ============================================================
# 03.03 ESCRITURA DE DIM_CUSTOMER EN GOLD
# ============================================================

dim_customer_table = (
    f"{CATALOG}.3_gold.dim_customer"
)


(
    dim_customer_df

    .write

    .format("delta")

    .mode("overwrite")

    .option(
        "overwriteSchema",
        "true"
    )

    .saveAsTable(
        dim_customer_table
    )
)

# COMMAND ----------

# DBTITLE 1,03.04 VALIDACIÓN DE DIM_CUSTOMER MATERIALIZADA
# ============================================================
# 03.04 VALIDACIÓN DE DIM_CUSTOMER MATERIALIZADA
# ============================================================

dim_customer_gold_count = spark.sql(f"""
    SELECT COUNT(*)
    FROM {CATALOG}.`3_gold`.dim_customer
""").collect()[0][0]


print(
    "Filas en dim_customer:",
    dim_customer_gold_count
)

# COMMAND ----------

# DBTITLE 1,04. CREACIÓN DE DIM_PRODUCT
# ============================================================
# 04. CREACIÓN DE DIM_PRODUCT
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Construir la dimensión de productos para analizar ventas por:
#
# - producto
# - categoría
# - subcategoría
# - marca
# - estado activo/inactivo
#
# Los datos ya están limpios y tipados en Silver, por lo que
# aquí no repetimos transformaciones de Data Quality.
# ============================================================


from pyspark.sql import functions as F


# ------------------------------------------------------------
# CONSTRUCCIÓN DE DIM_PRODUCT
# ------------------------------------------------------------

dim_product_df = (

    products_silver_df

    .select(

        # ----------------------------------------------------
        # CLAVE DE NEGOCIO
        # ----------------------------------------------------

        F.col("product_id"),


        # ----------------------------------------------------
        # ATRIBUTOS DESCRIPTIVOS
        # ----------------------------------------------------

        F.col("product_name"),

        F.col("category"),

        F.col("subcategory"),

        F.col("brand"),


        # ----------------------------------------------------
        # ATRIBUTOS ECONÓMICOS
        # ----------------------------------------------------
        #
        # Los mantenemos porque pueden ser útiles después
        # para análisis de margen o pricing.
        #
        F.col("unit_cost"),

        F.col("base_price"),


        # ----------------------------------------------------
        # ESTADO
        # ----------------------------------------------------

        F.col("active")
    )
)

# COMMAND ----------

# DBTITLE 1,04.01 INSPECCIÓN DE DIM_PRODUCT
# ============================================================
# 04.01 INSPECCIÓN DE DIM_PRODUCT
# ============================================================

display(
    dim_product_df
    .orderBy("product_id")
    .limit(20)
)

# COMMAND ----------

# DBTITLE 1,04.02 VALIDACIONES DE DIM_PRODUCT
# ============================================================
# 04.02 VALIDACIONES DE DIM_PRODUCT
# ============================================================

# ------------------------------------------------------------
# RECUENTO
# ------------------------------------------------------------

product_count = (
    dim_product_df.count()
)


# ------------------------------------------------------------
# PRODUCT_ID NULL
# ------------------------------------------------------------

null_product_ids = (
    dim_product_df
    .filter(
        F.col("product_id").isNull()
    )
    .count()
)


# ------------------------------------------------------------
# PRODUCT_ID DUPLICADO
# ------------------------------------------------------------

duplicated_product_ids = (

    dim_product_df

    .groupBy("product_id")

    .count()

    .filter(
        F.col("count") > 1
    )

    .count()
)


# ------------------------------------------------------------
# PRECIOS / COSTES INVÁLIDOS
# ------------------------------------------------------------

invalid_product_prices = (

    dim_product_df

    .filter(
        F.col("unit_cost").isNull()
        |
        F.col("base_price").isNull()
        |
        (F.col("unit_cost") <= 0)
        |
        (F.col("base_price") <= 0)
        |
        (
            F.col("base_price")
            <=
            F.col("unit_cost")
        )
    )

    .count()
)


# ------------------------------------------------------------
# RESULTADOS
# ------------------------------------------------------------

dim_product_validation = {

    "Diferencia vs 1.000 productos":
        abs(product_count - 1000),

    "product_id NULL":
        null_product_ids,

    "product_id duplicado":
        duplicated_product_ids,

    "precio/coste inválido":
        invalid_product_prices
}


for rule, errors in dim_product_validation.items():

    status = (
        "OK"
        if errors == 0
        else "ERROR"
    )

    print(
        f"{status:5} | "
        f"{rule:40} | "
        f"{errors:,}"
    )

# COMMAND ----------

# DBTITLE 1,04.03 ESCRITURA DE DIM_PRODUCT EN GOLD
# ============================================================
# 04.03 ESCRITURA DE DIM_PRODUCT EN GOLD
# ============================================================

dim_product_table = (
    f"{CATALOG}.3_gold.dim_product"
)


(
    dim_product_df

    .write

    .format("delta")

    .mode("overwrite")

    .option(
        "overwriteSchema",
        "true"
    )

    .saveAsTable(
        dim_product_table
    )
)

# COMMAND ----------

# DBTITLE 1,04.04 VALIDACIÓN DE DIM_PRODUCT MATERIALIZADA
# ============================================================
# 04.04 VALIDACIÓN DE DIM_PRODUCT MATERIALIZADA
# ============================================================

dim_product_gold_count = spark.sql(f"""
    SELECT COUNT(*)
    FROM {CATALOG}.`3_gold`.dim_product
""").collect()[0][0]


print(
    "Filas en dim_product:",
    dim_product_gold_count
)

# COMMAND ----------

# DBTITLE 1,05. CREACIÓN DE DIM_STORE
# ============================================================
# 05. CREACIÓN DE DIM_STORE
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Construir la dimensión de tiendas para analizar ventas por:
#
# - tienda
# - ciudad
# - región
# - tipo de tienda
#
# La limpieza y normalización ya se realizó en Silver.
# Por ejemplo:
#
# BCN -> Barcelona
#
# Gold simplemente modela la entidad para consumo analítico.
# ============================================================


from pyspark.sql import functions as F


# ------------------------------------------------------------
# CONSTRUCCIÓN DE DIM_STORE
# ------------------------------------------------------------

dim_store_df = (

    stores_silver_df

    .select(

        # Clave de negocio.
        F.col("store_id"),

        # Atributos descriptivos.
        F.col("store_name"),

        F.col("city"),

        F.col("region"),

        F.col("opening_date"),

        F.col("store_type")
    )
)

# COMMAND ----------

# DBTITLE 1,05.01 INSPECCIÓN DE DIM_STORE
# ============================================================
# 05.01 INSPECCIÓN DE DIM_STORE
# ============================================================

display(
    dim_store_df
    .orderBy("store_id")
)

# COMMAND ----------

# DBTITLE 1,05.02 VALIDACIONES DE DIM_STORE
# ============================================================
# 05.02 VALIDACIONES DE DIM_STORE
# ============================================================


# Número total de tiendas.
store_count = (
    dim_store_df.count()
)


# ------------------------------------------------------------
# STORE_ID NULL
# ------------------------------------------------------------

null_store_ids = (
    dim_store_df
    .filter(
        F.col("store_id").isNull()
    )
    .count()
)


# ------------------------------------------------------------
# STORE_ID DUPLICADO
# ------------------------------------------------------------

duplicated_store_ids = (

    dim_store_df

    .groupBy("store_id")

    .count()

    .filter(
        F.col("count") > 1
    )

    .count()
)


# ------------------------------------------------------------
# STORE TYPE INVÁLIDO
# ------------------------------------------------------------

invalid_store_types = (

    dim_store_df

    .filter(
        ~F.col("store_type").isin(
            "PHYSICAL",
            "ONLINE"
        )
        |
        F.col("store_type").isNull()
    )

    .count()
)


# ------------------------------------------------------------
# BCN SIN NORMALIZAR
# ------------------------------------------------------------
#
# Esta validación confirma que la corrección realizada
# previamente en Silver sigue presente en Gold.
#
remaining_bcn = (

    dim_store_df

    .filter(
        F.upper(
            F.trim(
                F.col("city")
            )
        ) == "BCN"
    )

    .count()
)


# ------------------------------------------------------------
# RESULTADOS
# ------------------------------------------------------------

dim_store_validation = {

    "Diferencia vs 31 tiendas":
        abs(store_count - 31),

    "store_id NULL":
        null_store_ids,

    "store_id duplicado":
        duplicated_store_ids,

    "store_type inválido":
        invalid_store_types,

    "BCN sin normalizar":
        remaining_bcn
}


for rule, errors in dim_store_validation.items():

    status = (
        "OK"
        if errors == 0
        else "ERROR"
    )

    print(
        f"{status:5} | "
        f"{rule:35} | "
        f"{errors:,}"
    )

# COMMAND ----------

# DBTITLE 1,05.03 ESCRITURA DE DIM_STORE EN GOLD
# ============================================================
# 05.03 ESCRITURA DE DIM_STORE EN GOLD
# ============================================================

dim_store_table = (
    f"{CATALOG}.3_gold.dim_store"
)


(
    dim_store_df

    .write

    .format("delta")

    .mode("overwrite")

    .option(
        "overwriteSchema",
        "true"
    )

    .saveAsTable(
        dim_store_table
    )
)

# COMMAND ----------

# DBTITLE 1,05.04 VALIDACIÓN DE DIM_STORE MATERIALIZADA
# ============================================================
# 05.04 VALIDACIÓN DE DIM_STORE MATERIALIZADA
# ============================================================

dim_store_gold_count = spark.sql(f"""
    SELECT COUNT(*)
    FROM {CATALOG}.`3_gold`.dim_store
""").collect()[0][0]


print(
    "Filas en dim_store:",
    dim_store_gold_count
)

# COMMAND ----------

# DBTITLE 1,06.00 IDENTIFICACIÓN DE REGISTROS INCREMENTALES DE FACT_SALES
# ============================================================
# 06.00 IDENTIFICACIÓN DE REGISTROS INCREMENTALES DE FACT_SALES
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Evitar reconstruir completamente fact_sales cada vez que
# llegan nuevas ventas a la capa Silver.
#
# Utilizamos como clave natural:
#
#       sale_id + line_id
#
# Compararemos las claves existentes en Gold contra SALES Silver.
#
# Solo continuarán hacia la construcción de fact_sales aquellos
# registros que todavía NO existan en Gold.
#
# ESCENARIOS
# ------------------------------------------------------------
#
# Primera ejecución:
#   fact_sales no existe -> procesamos todo SALES Silver.
#
# Ejecuciones posteriores:
#   fact_sales existe -> procesamos únicamente registros nuevos.
#
# IMPORTANTE
# ------------------------------------------------------------
#
# Este enfoque supone que las líneas de venta son append-only:
#
# - una línea histórica no cambia
# - las nuevas cargas añaden nuevas líneas
#
# ============================================================


from pyspark.sql import functions as F


# ------------------------------------------------------------
# TABLA GOLD DESTINO
# ------------------------------------------------------------

fact_sales_table = (
    f"{CATALOG}.3_gold.fact_sales"
)


# ------------------------------------------------------------
# COMPROBAMOS SI FACT_SALES YA EXISTE
# ------------------------------------------------------------

if spark.catalog.tableExists(
    fact_sales_table
):

    # Recuperamos únicamente las claves existentes en Gold.
    #
    # No necesitamos leer todas las medidas ni atributos para
    # determinar qué líneas ya han sido procesadas.

    fact_existing_keys_df = (

        spark.table(
            fact_sales_table
        )

        .select(
            "sale_id",
            "line_id"
        )
    )


    # --------------------------------------------------------
    # LEFT ANTI JOIN
    # --------------------------------------------------------
    #
    # Nos quedamos exclusivamente con las líneas de Silver
    # cuya clave:
    #
    #       sale_id + line_id
    #
    # todavía no existe en fact_sales.
    #
    # --------------------------------------------------------

    sales_silver_incremental_df = (

        sales_silver_df.alias(
            "silver"
        )

        .join(

            fact_existing_keys_df.alias(
                "gold"
            ),

            on=[
                "sale_id",
                "line_id"
            ],

            how="left_anti"
        )
    )


else:

    # --------------------------------------------------------
    # PRIMERA EJECUCIÓN
    # --------------------------------------------------------
    #
    # Si fact_sales todavía no existe, todo SALES Silver
    # constituye el primer batch de carga.
    #
    # --------------------------------------------------------

    sales_silver_incremental_df = (
        sales_silver_df
    )


# ------------------------------------------------------------
# CONTAMOS EL BATCH QUE REALMENTE NECESITA GOLD
# ------------------------------------------------------------

gold_incremental_rows = (

    sales_silver_incremental_df
    .count()
)


print(
    "Registros nuevos a procesar en fact_sales:",
    gold_incremental_rows
)

# COMMAND ----------

# DBTITLE 1,06. CREACIÓN DE FACT_SALES
# ============================================================
# 06. CREACIÓN INCREMENTAL DE FACT_SALES
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Construir únicamente el nuevo batch de la tabla de hechos
# principal del modelo Gold.
#
# GRANULARIDAD
# ------------------------------------------------------------
#
#     1 fila = 1 línea de ticket
#
# Clave natural:
#
#     sale_id + line_id
#
# DIFERENCIA RESPECTO A LA VERSIÓN ANTERIOR
# ------------------------------------------------------------
#
# Antes:
#
#     fact_sales_df partía de sales_silver_df completo.
#
# Ahora:
#
#     fact_sales_df parte de sales_silver_incremental_df.
#
# Por tanto, solo transformamos las líneas que todavía no
# existen en fact_sales.
#
# ============================================================


from pyspark.sql import functions as F


fact_sales_df = (

    sales_silver_incremental_df


    # ========================================================
    # DATE KEY
    # ========================================================
    #
    # Construimos la misma clave utilizada por dim_date:
    #
    #       2026-05-07 -> 20260507
    #
    # ========================================================

    .withColumn(
        "date_key",

        F.date_format(
            F.col(
                "sale_date"
            ),
            "yyyyMMdd"
        )

        .cast(
            "int"
        )
    )


    # ========================================================
    # SELECCIÓN FINAL DE COLUMNAS
    # ========================================================

    .select(

        # ----------------------------------------------------
        # CLAVE DE LA TRANSACCIÓN
        # ----------------------------------------------------
        #
        # sale_id permanece como dimensión degenerada.
        #
        # ----------------------------------------------------

        F.col(
            "sale_id"
        ),

        F.col(
            "line_id"
        ),


        # ----------------------------------------------------
        # CLAVES HACIA DIMENSIONES
        # ----------------------------------------------------

        F.col(
            "date_key"
        ),

        F.col(
            "customer_id"
        ),

        F.col(
            "product_id"
        ),

        F.col(
            "store_id"
        ),


        # ----------------------------------------------------
        # MEDIDAS
        # ----------------------------------------------------

        F.col(
            "quantity"
        ),

        F.col(
            "unit_price"
        ),

        F.col(
            "discount_pct"
        ),

        F.col(
            "gross_amount"
        ),

        F.col(
            "discount_amount"
        ),

        F.col(
            "net_amount"
        ),


        # ----------------------------------------------------
        # ATRIBUTOS DE LA TRANSACCIÓN
        # ----------------------------------------------------
        #
        # sales_channel y payment_method se conservan
        # directamente dentro de la fact.
        #
        # ----------------------------------------------------

        F.col(
            "sales_channel"
        ),

        F.col(
            "payment_method"
        )
    )
)


# ------------------------------------------------------------
# RESULTADO DEL BATCH
# ------------------------------------------------------------

print(
    "Registros construidos para el batch Gold:",
    fact_sales_df.count()
)

# COMMAND ----------

# DBTITLE 1,06.01 INSPECCIÓN DEL BATCH INCREMENTAL DE FACT_SALES
# ============================================================
# 06.01 INSPECCIÓN DEL BATCH INCREMENTAL DE FACT_SALES
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Inspeccionar visualmente el nuevo batch de fact_sales
# antes de materializarlo en Gold.
#
# IMPORTANTE
# ------------------------------------------------------------
#
# fact_sales_df ya NO contiene todo el histórico.
#
# Ahora representa únicamente los registros nuevos detectados
# en el bloque 06.00.
#
# Por tanto:
#
# - si no existen nuevos registros, el DataFrame estará vacío
# - si existe un nuevo batch, veremos solo esas nuevas líneas
#
# ============================================================


# ------------------------------------------------------------
# CONTAMOS EL BATCH
# ------------------------------------------------------------

fact_batch_count = (
    fact_sales_df
    .count()
)


print(
    "Registros del batch fact_sales:",
    fact_batch_count
)


# ------------------------------------------------------------
# INSPECCIÓN VISUAL
# ------------------------------------------------------------
#
# Ordenamos por sale_id + line_id para facilitar la revisión.
#
# ------------------------------------------------------------

display(

    fact_sales_df

    .orderBy(
        "sale_id",
        "line_id"
    )

    .limit(
        20
    )
)

# COMMAND ----------

# DBTITLE 1,06.02 VALIDACIONES DEL BATCH INCREMENTAL DE FACT_SALES
# ============================================================
# 06.02 VALIDACIONES DEL BATCH INCREMENTAL DE FACT_SALES
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Validar exclusivamente el BATCH INCREMENTAL que va a ser
# incorporado a FACT_SALES.
#
# IMPORTANTE:
#
# fact_sales_df NO representa necesariamente toda la tabla
# FACT_SALES.
#
# En una ejecución incremental puede contener:
#
#   0 registros
#       -> Gold ya está sincronizado con Silver.
#
#   N registros
#       -> existen nuevas líneas de venta pendientes de cargar.
#
# Por tanto, NO debemos comparar fact_sales_df.count()
# directamente contra el número total de registros de Silver.
#
# La igualdad entre:
#
#       SALES Silver
#
# y
#
#       FACT_SALES Gold
#
# se comprobará después de materializar el batch incremental,
# en la validación final de la capa Gold.
#
# ============================================================


from pyspark.sql import functions as F


# ============================================================
# 1. RECUENTO DEL BATCH INCREMENTAL
# ============================================================

incremental_fact_count = (
    fact_sales_df
    .count()
)


print()
print("=" * 85)
print("VALIDACIÓN DEL BATCH INCREMENTAL DE FACT_SALES")
print("=" * 85)

print(
    f"Registros del batch incremental: "
    f"{incremental_fact_count:,}"
)


# ============================================================
# 2. VALIDACIÓN DE DUPLICADOS
# ============================================================
#
# La granularidad de FACT_SALES es:
#
#       sale_id + line_id
#
# Incluso dentro de un batch incremental no puede haber
# duplicados de esta clave.
#
# ============================================================

duplicated_fact_keys = (

    fact_sales_df

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
# 3. VALIDACIÓN DE CLAVES OBLIGATORIAS NULL
# ============================================================
#
# customer_id NO se considera obligatorio.
#
# Las ventas anónimas mantienen:
#
#       customer_id = NULL
#
# ============================================================

null_mandatory_keys = (

    fact_sales_df

    .filter(

        F.col("sale_id").isNull()

        |

        F.col("line_id").isNull()

        |

        F.col("date_key").isNull()

        |

        F.col("product_id").isNull()

        |

        F.col("store_id").isNull()
    )

    .count()
)


# ============================================================
# 4. VALIDACIÓN DE IMPORTES
# ============================================================

invalid_fact_amounts = (

    fact_sales_df

    .filter(

        F.col("gross_amount").isNull()

        |

        F.col("discount_amount").isNull()

        |

        F.col("net_amount").isNull()

        |

        (F.col("gross_amount") <= 0)

        |

        (F.col("discount_amount") < 0)

        |

        (F.col("net_amount") <= 0)

        |

        (
            F.col("net_amount")
            >
            F.col("gross_amount")
        )
    )

    .count()
)


# ============================================================
# 5. RESULTADOS DE DATA QUALITY
# ============================================================

fact_sales_validation = {

    "sale_id + line_id duplicada":
        duplicated_fact_keys,

    "claves obligatorias NULL":
        null_mandatory_keys,

    "importes inválidos":
        invalid_fact_amounts
}


print()
print("-" * 85)


fact_sales_errors = []


for rule, errors in fact_sales_validation.items():

    status = (
        "OK"
        if errors == 0
        else "ERROR"
    )


    print(
        f"{status:5} | "
        f"{rule:40} | "
        f"{errors:,}"
    )


    if errors != 0:

        fact_sales_errors.append(
            f"{rule}: {errors:,}"
        )


# ============================================================
# 6. CONTROL FINAL
# ============================================================

if fact_sales_errors:

    print()
    print("ERRORES DETECTADOS")
    print("-" * 85)


    for error in fact_sales_errors:

        print(
            f"- {error}"
        )


    raise RuntimeError(
        "La validación del batch incremental de FACT_SALES "
        "ha detectado inconsistencias"
    )


# ============================================================
# 7. RESULTADO
# ============================================================

print()


if incremental_fact_count == 0:

    print(
        "OK - No existen nuevos registros pendientes "
        "de incorporar a FACT_SALES"
    )

else:

    print(
        f"OK - Batch incremental validado correctamente: "
        f"{incremental_fact_count:,} registros preparados "
        f"para FACT_SALES"
    )

# COMMAND ----------

# DBTITLE 1,06.03 VALIDACIÓN DE INTEGRIDAD REFERENCIAL DEL STAR SCHEMA
# ============================================================
# 06.03 VALIDACIÓN DE INTEGRIDAD REFERENCIAL DEL STAR SCHEMA
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Comprobar que todas las claves presentes en el NUEVO batch
# de fact_sales encuentran su correspondiente registro
# en las dimensiones Gold.
#
# Validaremos:
#
# - date_key       -> dim_date
# - customer_id    -> dim_customer
# - product_id     -> dim_product
# - store_id       -> dim_store
#
# IMPORTANTE
# ------------------------------------------------------------
#
# customer_id puede ser NULL porque existen ventas anónimas.
#
# Por tanto, solo consideramos error un customer_id:
#
# - distinto de NULL
# - y no encontrado en dim_customer
#
# ============================================================


from pyspark.sql import functions as F


# ------------------------------------------------------------
# 1. LEEMOS LAS DIMENSIONES MATERIALIZADAS
# ------------------------------------------------------------
#
# Leemos desde Unity Catalog para no depender de variables
# previamente definidas en memoria.
#
# Esto hace el notebook más robusto cuando se ejecuta
# desde Lakeflow Jobs.
#
# ------------------------------------------------------------

dim_date_materialized_df = spark.table(
    f"{CATALOG}.3_gold.dim_date"
)


dim_customer_materialized_df = spark.table(
    f"{CATALOG}.3_gold.dim_customer"
)


dim_product_materialized_df = spark.table(
    f"{CATALOG}.3_gold.dim_product"
)


dim_store_materialized_df = spark.table(
    f"{CATALOG}.3_gold.dim_store"
)


# ------------------------------------------------------------
# 2. DATE_KEY NO ENCONTRADA
# ------------------------------------------------------------

invalid_date_keys = (

    fact_sales_df.alias("fact")

    .join(

        dim_date_materialized_df
        .select("date_key")
        .alias("dim"),

        on="date_key",

        how="left_anti"
    )

    .count()
)


# ------------------------------------------------------------
# 3. CUSTOMER_ID NO ENCONTRADO
# ------------------------------------------------------------

invalid_customer_keys = (

    fact_sales_df

    .filter(
        F.col("customer_id").isNotNull()
    )

    .alias("fact")

    .join(

        dim_customer_materialized_df
        .select("customer_id")
        .alias("dim"),

        on="customer_id",

        how="left_anti"
    )

    .count()
)


# ------------------------------------------------------------
# 4. PRODUCT_ID NO ENCONTRADO
# ------------------------------------------------------------

invalid_product_keys = (

    fact_sales_df.alias("fact")

    .join(

        dim_product_materialized_df
        .select("product_id")
        .alias("dim"),

        on="product_id",

        how="left_anti"
    )

    .count()
)


# ------------------------------------------------------------
# 5. STORE_ID NO ENCONTRADO
# ------------------------------------------------------------

invalid_store_keys = (

    fact_sales_df.alias("fact")

    .join(

        dim_store_materialized_df
        .select("store_id")
        .alias("dim"),

        on="store_id",

        how="left_anti"
    )

    .count()
)


# ------------------------------------------------------------
# 6. RESUMEN
# ------------------------------------------------------------

referential_integrity_results = {

    "date_key sin correspondencia":
        invalid_date_keys,

    "customer_id sin correspondencia":
        invalid_customer_keys,

    "product_id sin correspondencia":
        invalid_product_keys,

    "store_id sin correspondencia":
        invalid_store_keys
}


for rule, errors in (
    referential_integrity_results.items()
):

    status = (
        "OK"
        if errors == 0
        else "ERROR"
    )

    print(
        f"{status:5} | "
        f"{rule:40} | "
        f"{errors:,}"
    )

# COMMAND ----------

# DBTITLE 1,06.04 ESCRITURA DE FACT_SALES EN GOLD
# ============================================================
# 06.04 ESCRITURA INCREMENTAL DE FACT_SALES EN GOLD
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Incorporar únicamente las nuevas líneas de venta a fact_sales
# mediante Delta MERGE.
#
# Clave natural:
#
#       sale_id + line_id
#
# COMPORTAMIENTO
# ------------------------------------------------------------
#
# Si la clave ya existe:
#
#       no hacemos nada.
#
# Si la clave todavía no existe:
#
#       insertamos la nueva línea.
#
# Esto evita:
#
# - reconstruir toda la tabla Gold
# - reescribir cientos de miles de registros históricos
# - generar duplicados si una carga se reejecuta
#
# ============================================================


from delta.tables import DeltaTable


# ------------------------------------------------------------
# TABLA DESTINO
# ------------------------------------------------------------

fact_sales_table = (
    f"{CATALOG}.3_gold.fact_sales"
)

# Comprobamos si fact_sales ya existe en Gold.
fact_sales_exists = spark.catalog.tableExists(
    fact_sales_table
)

print(
    f"Tabla FACT_SALES existente: {fact_sales_exists}"
)

# ------------------------------------------------------------
# SOLO HACEMOS MERGE SI EXISTE UN BATCH NUEVO
# ------------------------------------------------------------

if gold_incremental_rows > 0:

    # ========================================================
    # CARGA INICIAL
    # ========================================================

    if not fact_sales_exists:

        (
            fact_sales_df
            .write
            .format("delta")
            .mode("overwrite")
            .option(
                "overwriteSchema",
                "true"
            )
            .saveAsTable(
                fact_sales_table
            )
        )

        print(
            "OK | Carga inicial de FACT_SALES completada:",
            gold_incremental_rows
        )

    # ========================================================
    # CARGA INCREMENTAL
    # ========================================================

    else:

        fact_sales_delta = DeltaTable.forName(
            spark,
            fact_sales_table
        )

        (
            fact_sales_delta
            .alias("target")

            .merge(
                fact_sales_df.alias("source"),
                """
                target.sale_id = source.sale_id
                AND
                target.line_id = source.line_id
                """
            )

            .whenNotMatchedInsertAll()

            .execute()
        )

        print(
            "OK | Registros nuevos incorporados a fact_sales:",
            gold_incremental_rows
        )
else:

    # Evitamos lanzar un MERGE completamente innecesario
    # cuando Gold ya está actualizado.

    print(
        "OK | No existen nuevos registros para incorporar a fact_sales"
    )

# COMMAND ----------

# DBTITLE 1,07. VALIDACIÓN FINAL DE LA CAPA GOLD
# ============================================================
# 07. VALIDACIÓN FINAL DE LA CAPA GOLD
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Comprobar que el modelo Gold completo está correctamente
# materializado y mantiene la estructura esperada.
#
# Validaremos:
#
# - existencia de las 5 tablas Gold
# - recuentos dinámicos y coherentes con Silver
# - recuento dinámico de DIM_DATE
# - unicidad de las claves
# - consistencia final del modelo
#
# IMPORTANTE
# ------------------------------------------------------------
#
# NO utilizamos valores hardcodeados como:
#
# dim_date   = 731
# fact_sales = 536340
#
# porque el proyecto recibe nuevas cargas incrementales.
#
# ============================================================


from pyspark.sql import functions as F


# ============================================================
# 1. TABLAS GOLD
# ============================================================

gold_tables_validation = {

    "dim_date":
        f"{CATALOG}.3_gold.dim_date",

    "dim_customer":
        f"{CATALOG}.3_gold.dim_customer",

    "dim_product":
        f"{CATALOG}.3_gold.dim_product",

    "dim_store":
        f"{CATALOG}.3_gold.dim_store",

    "fact_sales":
        f"{CATALOG}.3_gold.fact_sales"
}


# ============================================================
# 2. RECUPERAMOS RECUENTOS ESPERADOS DINÁMICOS
# ============================================================
#
# Dimensiones maestras:
#
# deben mantener el mismo número de registros que sus
# correspondientes tablas Silver.
#
# FACT_SALES:
#
# debe mantener la misma granularidad que SALES Silver.
#
# DIM_DATE:
#
# debe contener una fila por cada fecha comprendida entre
# min_date y max_date.
#
# ============================================================


expected_dim_date_count = (

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

        .alias(
            "expected_dim_date_count"
        )
    )

    .first()[
        "expected_dim_date_count"
    ]
)


expected_gold_counts = {

    "dim_date":
        expected_dim_date_count,

    "dim_customer":
        silver_source_counts[
            "customers"
        ],

    "dim_product":
        silver_source_counts[
            "products"
        ],

    "dim_store":
        silver_source_counts[
            "stores"
        ],

    "fact_sales":
        silver_source_counts[
            "sales"
        ]
}


# ============================================================
# 3. CONTENEDOR DE ERRORES
# ============================================================

gold_validation_errors = []


print()
print("=" * 90)
print("VALIDACIÓN FINAL DE LA CAPA GOLD")
print("=" * 90)


# ============================================================
# 4. VALIDACIÓN DE EXISTENCIA Y RECUENTOS
# ============================================================

for table_name, full_table_name in gold_tables_validation.items():

    print()
    print("-" * 90)
    print(
        f"TABLE: {table_name.upper()}"
    )
    print("-" * 90)


    # --------------------------------------------------------
    # EXISTENCIA
    # --------------------------------------------------------

    table_exists = (
        spark.catalog.tableExists(
            full_table_name
        )
    )


    if not table_exists:

        gold_validation_errors.append(
            f"No existe la tabla Gold: {full_table_name}"
        )

        print(
            "ERROR | Tabla no encontrada"
        )

        continue


    # --------------------------------------------------------
    # RECUENTO REAL
    # --------------------------------------------------------

    actual_count = (

        spark.table(
            full_table_name
        )

        .count()
    )


    expected_count = (
        expected_gold_counts[
            table_name
        ]
    )


    # --------------------------------------------------------
    # VALIDACIÓN DEL RECUENTO
    # --------------------------------------------------------

    count_ok = (
        actual_count
        ==
        expected_count
    )


    if not count_ok:

        gold_validation_errors.append(

            f"{table_name}: "
            f"esperados={expected_count:,}, "
            f"actuales={actual_count:,}"
        )


    print(
        f"{'OK' if count_ok else 'ERROR':5} | "
        f"Esperados: {expected_count:>10,} | "
        f"Gold: {actual_count:>10,}"
    )


# ============================================================
# 5. VALIDACIÓN DE CLAVES
# ============================================================
#
# Comprobamos unicidad de las claves principales.
#
# ============================================================


dim_date_gold_df = spark.table(
    gold_tables_validation[
        "dim_date"
    ]
)

dim_customer_gold_df = spark.table(
    gold_tables_validation[
        "dim_customer"
    ]
)

dim_product_gold_df = spark.table(
    gold_tables_validation[
        "dim_product"
    ]
)

dim_store_gold_df = spark.table(
    gold_tables_validation[
        "dim_store"
    ]
)

fact_sales_gold_df = spark.table(
    gold_tables_validation[
        "fact_sales"
    ]
)


# ============================================================
# 6. FUNCIÓN PARA CONTAR CLAVES DUPLICADAS
# ============================================================

def count_duplicate_keys(
    df,
    key_columns
):

    return (

        df

        .groupBy(
            *key_columns
        )

        .count()

        .filter(
            F.col("count") > 1
        )

        .count()
    )


# ============================================================
# 7. EJECUTAMOS VALIDACIONES DE UNICIDAD
# ============================================================

gold_key_validation = {

    "dim_date.date_key":
        count_duplicate_keys(
            dim_date_gold_df,
            ["date_key"]
        ),

    "dim_customer.customer_id":
        count_duplicate_keys(
            dim_customer_gold_df,
            ["customer_id"]
        ),

    "dim_product.product_id":
        count_duplicate_keys(
            dim_product_gold_df,
            ["product_id"]
        ),

    "dim_store.store_id":
        count_duplicate_keys(
            dim_store_gold_df,
            ["store_id"]
        ),

    "fact_sales.sale_id + line_id":
        count_duplicate_keys(
            fact_sales_gold_df,
            [
                "sale_id",
                "line_id"
            ]
        )
}


print()
print("=" * 90)
print("VALIDACIÓN DE CLAVES GOLD")
print("=" * 90)


for rule, errors in gold_key_validation.items():

    status = (
        "OK"
        if errors == 0
        else "ERROR"
    )


    print(
        f"{status:5} | "
        f"{rule:40} | "
        f"{errors:,}"
    )


    if errors != 0:

        gold_validation_errors.append(
            f"{rule}: {errors:,} duplicados"
        )


# ============================================================
# 8. RESULTADO FINAL
# ============================================================

print()
print("=" * 90)


if gold_validation_errors:

    print(
        "GOLD VALIDATION: FAILED"
    )

    print()

    for error in gold_validation_errors:

        print(
            f"- {error}"
        )


    raise RuntimeError(
        "La validación final de Gold "
        "ha detectado inconsistencias"
    )


else:

    print(
        "OK - Todas las tablas Gold "
        "han superado la validación final"
    )


print("=" * 90)

# COMMAND ----------

# DBTITLE 1,07.01 VALIDACIÓN FINAL DE CLAVES GOLD
# ============================================================
# 07.01 VALIDACIÓN FINAL DE CLAVES GOLD
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Comprobar la unicidad global de todas las claves del modelo
# Gold después de las cargas incrementales.
#
# ============================================================


from pyspark.sql import functions as F


# ------------------------------------------------------------
# CARGAMOS LAS TABLAS GOLD MATERIALIZADAS
# ------------------------------------------------------------

dim_date_gold_df = spark.table(
    f"{CATALOG}.3_gold.dim_date"
)

dim_customer_gold_df = spark.table(
    f"{CATALOG}.3_gold.dim_customer"
)

dim_product_gold_df = spark.table(
    f"{CATALOG}.3_gold.dim_product"
)

dim_store_gold_df = spark.table(
    f"{CATALOG}.3_gold.dim_store"
)

fact_sales_gold_df = spark.table(
    f"{CATALOG}.3_gold.fact_sales"
)


# ------------------------------------------------------------
# FUNCIÓN GENÉRICA PARA DETECTAR DUPLICADOS
# ------------------------------------------------------------

def count_duplicate_keys(
    df,
    key_columns
):

    return (

        df

        .groupBy(
            *key_columns
        )

        .count()

        .filter(
            F.col("count") > 1
        )

        .count()
    )


# ------------------------------------------------------------
# VALIDACIONES
# ------------------------------------------------------------

gold_key_validation = {

    "dim_date.date_key":
        count_duplicate_keys(
            dim_date_gold_df,
            ["date_key"]
        ),

    "dim_customer.customer_id":
        count_duplicate_keys(
            dim_customer_gold_df,
            ["customer_id"]
        ),

    "dim_product.product_id":
        count_duplicate_keys(
            dim_product_gold_df,
            ["product_id"]
        ),

    "dim_store.store_id":
        count_duplicate_keys(
            dim_store_gold_df,
            ["store_id"]
        ),

    "fact_sales.sale_id + line_id":
        count_duplicate_keys(
            fact_sales_gold_df,
            [
                "sale_id",
                "line_id"
            ]
        )
}


print("=" * 80)
print("VALIDACIÓN FINAL DE UNICIDAD DE CLAVES GOLD")
print("=" * 80)


for rule, errors in (
    gold_key_validation.items()
):

    status = (
        "OK"
        if errors == 0
        else "ERROR"
    )

    print(
        f"{status:5} | "
        f"{rule:35} | "
        f"{errors:,}"
    )
