# Databricks notebook source
# MAGIC %md
# MAGIC # **Bronze realizando profiling y validaciones de Data Quality**

# COMMAND ----------
# Catálogo recibido desde Databricks Asset Bundles.
# DEV  -> retail_analytics_dev
# PROD -> retail_analytics

dbutils.widgets.text("catalog", "retail_analytics")
CATALOG = dbutils.widgets.get("catalog")

print(f"Environment catalog: {CATALOG}")
# DBTITLE 1,00. CONFIGURACIÓN Y LECTURA DE TABLAS BRONZE
# ============================================================
# 01. CONFIGURACIÓN Y LECTURA DE TABLAS BRONZE
# ============================================================

# En Silver partimos de las tablas Delta almacenadas en Bronze.
#
# El objetivo de esta capa será:
#
# - analizar la calidad de los datos
# - limpiar registros incorrectos
# - estandarizar formatos
# - validar reglas de negocio
# - comprobar integridad referencial
# - generar columnas derivadas
#
# Mantendremos Bronze sin modificar para conservar siempre
# una representación fiel de los datos ingeridos.


# ------------------------------------------------------------
# DEFINICIÓN DE TABLAS BRONZE
# ------------------------------------------------------------

bronze_tables = {
    "sales": f"{CATALOG}.1_bronze.sales",
    "customers": f"{CATALOG}.1_bronze.customers",
    "products": f"{CATALOG}.1_bronze.products",
    "stores": f"{CATALOG}.1_bronze.stores"
}


# ------------------------------------------------------------
# LECTURA DE TABLAS
# ------------------------------------------------------------

sales_bronze_df = spark.table(bronze_tables["sales"])

customers_bronze_df = spark.table(
    bronze_tables["customers"]
)

products_bronze_df = spark.table(
    bronze_tables["products"]
)

stores_bronze_df = spark.table(
    bronze_tables["stores"]
)


# ------------------------------------------------------------
# VALIDACIÓN
# ------------------------------------------------------------

print("Tablas Bronze cargadas correctamente:")
print()

for table_name, full_table_name in bronze_tables.items():

    row_count = spark.table(full_table_name).count()

    print(
        f"OK | {table_name:10} | "
        f"{row_count:,} registros"
    )

# COMMAND ----------

# DBTITLE 1,01. INSPECCIÓN DE SCHEMA Y METADATA DE TABLAS BRONZE
# ============================================================
# 01. INSPECCIÓN DE SCHEMA Y METADATA DE TABLAS BRONZE
# ============================================================
#
# Antes de analizar calidad a nivel de registros,
# revisamos la estructura técnica de las tablas:
#
# - nombres de columnas
# - tipos de datos
# - nullability
# - metadata Delta
#
# Esto nos ayuda a detectar problemas estructurales antes
# de entrar en reglas de calidad de negocio.
# ============================================================


# ------------------------------------------------------------
# SCHEMA DE SALES
# ------------------------------------------------------------

print("SCHEMA | SALES")
print("=" * 70)

sales_bronze_df.printSchema()


# ------------------------------------------------------------
# DESCRIBE TABLE
# ------------------------------------------------------------
#
# DESCRIBE TABLE nos devuelve información sobre:
#
# - nombre de columna
# - tipo de dato
# - comentarios
#
# Lo ejecutamos sobre la tabla Delta registrada
# en Unity Catalog.
# ------------------------------------------------------------

display(
    spark.sql(f"""
        DESCRIBE TABLE
        {CATALOG}.`1_bronze`.sales
    """)
)

# COMMAND ----------

# DBTITLE 1,02. ESTADÍSTICAS DESCRIPTIVAS DE SALES
# ============================================================
# 02. ESTADÍSTICAS DESCRIPTIVAS DE SALES
# ============================================================

# ------------------------------------------------------------
# describe()
# ------------------------------------------------------------
#
# Calcula estadísticas básicas:
#
# count
# mean
# stddev
# min
# max
#
# Nos interesa especialmente para columnas numéricas como:
#
# line_id
# quantity
# unit_price
# discount_pct
# ------------------------------------------------------------

display(
    sales_bronze_df.select(
        "line_id",
        "quantity",
        "unit_price",
        "discount_pct"
    ).describe()
)

# COMMAND ----------

# DBTITLE 1,02.01 SUMMARY CON PERCENTILES
# ============================================================
# 02.01 SUMMARY CON PERCENTILES
# ============================================================

# summary() amplía describe() permitiendo obtener
# percentiles de las columnas numéricas.
#
# Esto ayuda a detectar:
#
# - valores extremos
# - distribuciones extrañas
# - posibles outliers
#
# 50% representa la mediana.
# ------------------------------------------------------------

display(
    sales_bronze_df.select(
        "quantity",
        "unit_price",
        "discount_pct"
    ).summary(
        "count",
        "mean",
        "stddev",
        "min",
        "25%",
        "50%",
        "75%",
        "max"
    )
)

# COMMAND ----------

# DBTITLE 1,03. PERFIL INICIAL DE DATA QUALITY
# ============================================================
# 02. PERFIL INICIAL DE DATA QUALITY
# ============================================================

# Antes de transformar los datos analizamos su calidad.
#
# Para cada tabla comprobaremos:
#
# - número total de registros
# - número de columnas
# - valores NULL por columna
#
# Esto nos permitirá identificar qué problemas debemos
# resolver posteriormente en la capa Silver.


from pyspark.sql import functions as F


# ------------------------------------------------------------
# FUNCIÓN DE PERFILADO
# ------------------------------------------------------------

def profile_table(df, table_name):

    total_rows = df.count()
    total_columns = len(df.columns)

    print("=" * 70)
    print(f"DATA QUALITY PROFILE | {table_name.upper()}")
    print("=" * 70)

    print(f"Registros : {total_rows:,}")
    print(f"Columnas  : {total_columns}")
    print()

    # Calculamos NULLs para todas las columnas en una sola agregación

    null_counts = df.select([
        F.sum(
            F.when(F.col(column).isNull(), 1).otherwise(0)
        ).alias(column)
        for column in df.columns
    ]).collect()[0]

    print("NULLS POR COLUMNA")
    print("-" * 70)

    for column in df.columns:

        null_count = null_counts[column]

        if null_count > 0:

            percentage = (
                null_count / total_rows * 100
                if total_rows > 0
                else 0
            )

            print(
                f"{column:35} "
                f"{null_count:>8,} "
                f"({percentage:.2f}%)"
            )

    print()


# ------------------------------------------------------------
# EJECUTAMOS EL PERFILADO
# ------------------------------------------------------------

profile_table(
    sales_bronze_df,
    "sales"
)

profile_table(
    customers_bronze_df,
    "customers"
)

profile_table(
    products_bronze_df,
    "products"
)

profile_table(
    stores_bronze_df,
    "stores"
)

# COMMAND ----------

# DBTITLE 1,04. VALIDACIÓN DE DUPLICADOS Y UNICIDAD DE CLAVES
# ============================================================
# 04. VALIDACIÓN DE DUPLICADOS Y UNICIDAD DE CLAVES
# ============================================================
#
# Los NULLs son solo una parte del análisis de calidad.
#
# Ahora comprobaremos si las claves que deberían ser únicas
# realmente lo son.
#
# Reglas esperadas:
#
# CUSTOMERS
#   customer_id debe ser único.
#
# PRODUCTS
#   product_id debe ser único.
#
# STORES
#   store_id debe ser único.
#
# SALES
#   sale_id NO debe ser único porque un ticket puede tener
#   varias líneas.
#
#   La combinación:
#
#       sale_id + line_id
#
#   sí debe identificar de forma única cada línea de venta.
# ============================================================


from pyspark.sql import functions as F


# ------------------------------------------------------------
# FUNCIÓN PARA VALIDAR UNA CLAVE ÚNICA
# ------------------------------------------------------------

def validate_unique_key(df, table_name, key_columns):

    # Agrupamos por las columnas que deberían formar
    # una clave única.
    #
    # Después contamos cuántas veces aparece cada combinación.
    duplicates_df = (
        df
        .groupBy(*key_columns)
        .count()
        .filter(F.col("count") > 1)
    )

    # Contamos cuántas claves duplicadas hemos encontrado.
    duplicate_keys = duplicates_df.count()


    # --------------------------------------------------------
    # MOSTRAMOS EL RESULTADO
    # --------------------------------------------------------

    if duplicate_keys == 0:

        print(
            f"OK    | {table_name:10} | "
            f"Clave {key_columns} sin duplicados"
        )

    else:

        print(
            f"ERROR | {table_name:10} | "
            f"{duplicate_keys:,} claves duplicadas "
            f"para {key_columns}"
        )


# ------------------------------------------------------------
# EJECUTAMOS LAS VALIDACIONES
# ------------------------------------------------------------

validate_unique_key(
    customers_bronze_df,
    "customers",
    ["customer_id"]
)

validate_unique_key(
    products_bronze_df,
    "products",
    ["product_id"]
)

validate_unique_key(
    stores_bronze_df,
    "stores",
    ["store_id"]
)

validate_unique_key(
    sales_bronze_df,
    "sales",
    ["sale_id", "line_id"]
)

# COMMAND ----------

# DBTITLE 1,05. VALIDACIÓN DE REGLAS DE NEGOCIO Y VALORES FUERA DE RANGO
# ============================================================
# 05. VALIDACIÓN DE REGLAS DE NEGOCIO Y VALORES FUERA DE RANGO
# ============================================================
#
# Ahora buscamos registros que, aunque tengan tipos correctos
# y claves únicas, no cumplan las reglas de negocio esperadas.
#
# IMPORTANTE:
#
# Seguimos trabajando sobre Bronze.
# Todavía NO corregimos ni eliminamos ningún registro.
#
# Solo identificamos incidencias.
# ============================================================

from pyspark.sql import functions as F


# ------------------------------------------------------------
# 1. SALES - QUANTITY
# ------------------------------------------------------------
#
# Según nuestro generador, quantity debe estar entre 1 y 5.
#
invalid_quantity = (
    sales_bronze_df
    .filter(
        (F.col("quantity") < 1)
        |
        (F.col("quantity") > 5)
        |
        F.col("quantity").isNull()
    )
)

invalid_quantity_count = invalid_quantity.count()


# ------------------------------------------------------------
# 2. SALES - UNIT PRICE
# ------------------------------------------------------------
#
# Un precio unitario debe ser estrictamente positivo.
#
invalid_unit_price = (
    sales_bronze_df
    .filter(
        (F.col("unit_price") <= 0)
        |
        F.col("unit_price").isNull()
    )
)

invalid_unit_price_count = invalid_unit_price.count()


# ------------------------------------------------------------
# 3. SALES - DISCOUNT
# ------------------------------------------------------------
#
# Nuestro modelo admite descuentos entre:
#
# 0% y 20%
#
invalid_discount = (
    sales_bronze_df
    .filter(
        (F.col("discount_pct") < 0)
        |
        (F.col("discount_pct") > 0.20)
        |
        F.col("discount_pct").isNull()
    )
)

invalid_discount_count = invalid_discount.count()


# ------------------------------------------------------------
# 4. SALES - SALES CHANNEL
# ------------------------------------------------------------
#
# Solo admitimos:
#
# PHYSICAL
# ONLINE
#
invalid_sales_channel = (
    sales_bronze_df
    .filter(
        ~F.col("sales_channel").isin(
            "PHYSICAL",
            "ONLINE"
        )
        |
        F.col("sales_channel").isNull()
    )
)

invalid_sales_channel_count = (
    invalid_sales_channel.count()
)


# ------------------------------------------------------------
# 5. SALES - ONLINE + CASH
# ------------------------------------------------------------
#
# Esta fue una regla explícita desde la generación:
#
# Una venta ONLINE nunca puede pagarse en CASH.
#
invalid_online_cash = (
    sales_bronze_df
    .filter(
        (F.col("sales_channel") == "ONLINE")
        &
        (F.col("payment_method") == "CASH")
    )
)

invalid_online_cash_count = (
    invalid_online_cash.count()
)


# ------------------------------------------------------------
# 6. PRODUCTS - COSTE Y PRECIO
# ------------------------------------------------------------
#
# Reglas:
#
# unit_cost > 0
# base_price > 0
# base_price > unit_cost
#
invalid_product_prices = (
    products_bronze_df
    .filter(
        (F.col("unit_cost") <= 0)
        |
        (F.col("base_price") <= 0)
        |
        (
            F.col("base_price")
            <=
            F.col("unit_cost")
        )
        |
        F.col("unit_cost").isNull()
        |
        F.col("base_price").isNull()
    )
)

invalid_product_prices_count = (
    invalid_product_prices.count()
)


# ------------------------------------------------------------
# MOSTRAMOS RESULTADOS
# ------------------------------------------------------------

business_rule_results = {
    "sales.quantity fuera de rango":
        invalid_quantity_count,

    "sales.unit_price inválido":
        invalid_unit_price_count,

    "sales.discount_pct inválido":
        invalid_discount_count,

    "sales.sales_channel inválido":
        invalid_sales_channel_count,

    "sales ONLINE + CASH":
        invalid_online_cash_count,

    "products precio/coste inválido":
        invalid_product_prices_count
}


for rule, errors in business_rule_results.items():

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

# DBTITLE 1,06. VALIDACIÓN DE INTEGRIDAD REFERENCIAL
# ============================================================
# 06. VALIDACIÓN DE INTEGRIDAD REFERENCIAL
# ============================================================
#
# Ahora comprobamos que las claves presentes en SALES
# tengan correspondencia en sus maestros.
#
# Reglas:
#
# sales.product_id  -> debe existir en products
# sales.store_id    -> debe existir en stores
# sales.customer_id -> debe existir en customers o ser NULL
#
# Seguimos sin transformar nada.
# Solo detectamos posibles inconsistencias.
# ============================================================

from pyspark.sql import functions as F


# ------------------------------------------------------------
# 1. SALES -> PRODUCTS
# ------------------------------------------------------------
#
# Utilizamos un LEFT ANTI JOIN.
#
# Un LEFT ANTI JOIN devuelve las filas de la tabla izquierda
# que NO tienen correspondencia en la tabla derecha.
#
# Es perfecto para encontrar claves huérfanas.
#
invalid_product_refs = (
    sales_bronze_df.alias("s")
    .join(
        products_bronze_df
        .select("product_id")
        .distinct()
        .alias("p"),
        F.col("s.product_id") == F.col("p.product_id"),
        "left_anti"
    )
)

invalid_product_refs_count = (
    invalid_product_refs.count()
)


# ------------------------------------------------------------
# 2. SALES -> STORES
# ------------------------------------------------------------

invalid_store_refs = (
    sales_bronze_df.alias("s")
    .join(
        stores_bronze_df
        .select("store_id")
        .distinct()
        .alias("st"),
        F.col("s.store_id") == F.col("st.store_id"),
        "left_anti"
    )
)

invalid_store_refs_count = (
    invalid_store_refs.count()
)


# ------------------------------------------------------------
# 3. SALES -> CUSTOMERS
# ------------------------------------------------------------
#
# Aquí debemos respetar una particularidad de negocio:
#
# customer_id puede ser NULL porque existen compras anónimas.
#
# Por tanto:
#
# - NULL        -> válido
# - ID existente -> válido
# - ID no existente -> error
#
identified_sales_df = (
    sales_bronze_df
    .filter(
        F.col("customer_id").isNotNull()
    )
)


invalid_customer_refs = (
    identified_sales_df.alias("s")
    .join(
        customers_bronze_df
        .select("customer_id")
        .distinct()
        .alias("c"),
        F.col("s.customer_id") == F.col("c.customer_id"),
        "left_anti"
    )
)

invalid_customer_refs_count = (
    invalid_customer_refs.count()
)


# ------------------------------------------------------------
# MOSTRAMOS RESULTADOS
# ------------------------------------------------------------

referential_integrity_results = {
    "sales.product_id -> products":
        invalid_product_refs_count,

    "sales.store_id -> stores":
        invalid_store_refs_count,

    "sales.customer_id -> customers":
        invalid_customer_refs_count
}


for rule, errors in referential_integrity_results.items():

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

# MAGIC %md
# MAGIC # Diseño del modelo Silver
# MAGIC
# MAGIC La filosofía será:
# MAGIC
# MAGIC BRONZE
# MAGIC Datos ingeridos + metadata técnica
# MAGIC         ↓
# MAGIC SILVER
# MAGIC Datos limpios + tipados + estandarizados + reglas de negocio
# MAGIC         ↓
# MAGIC GOLD
# MAGIC Modelo analítico / KPIs / joins / agregaciones
# MAGIC
# MAGIC Para cada tabla haría esto:
# MAGIC
# MAGIC Tabla	Transformaciones Silver
# MAGIC customers	trim de strings, tipos definitivos, validar segmento, mantener customer_id
# MAGIC products	tipos definitivos, precios válidos, normalización de textos
# MAGIC stores	corregir BCN → Barcelona, normalizar strings, tipos definitivos
# MAGIC sales	tipos definitivos, eliminar metadata innecesaria, calcular importes, columnas de fecha
# MAGIC
# MAGIC En todas eliminaría:
# MAGIC
# MAGIC _rescued_data
# MAGIC
# MAGIC porque está 100% NULL y solo tenía sentido técnico en Bronze.
# MAGIC
# MAGIC Pero mantendría:
# MAGIC
# MAGIC _ingestion_timestamp
# MAGIC _source_file
# MAGIC _source_file_modification_time
# MAGIC
# MAGIC porque siguen siendo útiles para auditoría.
# MAGIC
# MAGIC Y en sales haría una limpieza adicional. Como Auto Loader ha obtenido year y month de la estructura:
# MAGIC
# MAGIC year=2024/month=01/
# MAGIC
# MAGIC yo no confiaría en esas columnas como atributos de negocio. Las eliminaría y las volvería a derivar de sale_timestamp.
# MAGIC
# MAGIC Así garantizamos:
# MAGIC
# MAGIC sale_year  = YEAR(sale_timestamp)
# MAGIC sale_month = MONTH(sale_timestamp)
# MAGIC sale_date  = DATE(sale_timestamp)
# MAGIC Columnas calculadas de sales
# MAGIC
# MAGIC Aquí empieza la parte buena:
# MAGIC
# MAGIC gross_amount
# MAGIC     = quantity × unit_price
# MAGIC
# MAGIC
# MAGIC discount_amount
# MAGIC     = gross_amount × discount_pct
# MAGIC
# MAGIC
# MAGIC net_amount
# MAGIC     = gross_amount - discount_amount
# MAGIC
# MAGIC Ejemplo:
# MAGIC
# MAGIC quantity = 2
# MAGIC unit_price = 100
# MAGIC discount_pct = 0.10
# MAGIC
# MAGIC
# MAGIC gross_amount    = 200
# MAGIC discount_amount = 20
# MAGIC net_amount      = 180
# MAGIC
# MAGIC Y redondearemos importes a 2 decimales.
# MAGIC
# MAGIC Una decisión que no tomaría
# MAGIC
# MAGIC No sustituiría:
# MAGIC
# MAGIC customer_id = NULL
# MAGIC
# MAGIC por algo tipo:
# MAGIC
# MAGIC UNKNOWN
# MAGIC C00000
# MAGIC ANONYMOUS
# MAGIC
# MAGIC Porque ese NULL tiene significado de negocio: compra anónima. No es un error que tengamos que maquillar.

# COMMAND ----------

# DBTITLE 1,07. DEFINICIÓN DEL MODELO Y REGLAS DE LA CAPA SILVER
# ============================================================
# 07. DEFINICIÓN DEL MODELO Y REGLAS DE LA CAPA SILVER
# ============================================================
#
# Hasta este punto hemos trabajado exclusivamente sobre Bronze
# realizando profiling y validaciones de Data Quality.
#
# A partir de aquí comenzamos la construcción de Silver.
#
# OBJETIVO DE SILVER
# ------------------------------------------------------------
#
# Convertir los datos ingeridos en Bronze en datasets:
#
# - limpios
# - tipados explícitamente
# - estandarizados
# - consistentes
# - preparados para consumo analítico
#
# Mantendremos cuatro entidades:
#
# customers
# products
# stores
# sales
#
# Los joins analíticos y agregaciones se realizarán
# posteriormente en Gold.
# ============================================================


# ------------------------------------------------------------
# COLUMNAS TÉCNICAS QUE CONSERVAREMOS
# ------------------------------------------------------------
#
# Aunque Silver sea una capa de negocio más limpia,
# queremos seguir manteniendo trazabilidad hacia el origen.
#
technical_columns_to_keep = [
    "_ingestion_timestamp",
    "_source_file",
    "_source_file_modification_time"
]


# ------------------------------------------------------------
# COLUMNAS TÉCNICAS QUE ELIMINAREMOS
# ------------------------------------------------------------
#
# _rescued_data es generada por Auto Loader.
#
# En nuestro caso está 100% a NULL en todas las tablas,
# lo que significa que no existieron registros que Auto Loader
# necesitara rescatar por incompatibilidades de schema.
#
technical_columns_to_drop = [
    "_rescued_data"
]


# ------------------------------------------------------------
# TABLAS SILVER DESTINO
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# MOSTRAMOS LA CONFIGURACIÓN
# ------------------------------------------------------------

print("TABLAS SILVER")
print("=" * 70)

for table_name, full_table_name in silver_tables.items():

    print(
        f"{table_name:10} -> {full_table_name}"
    )


print("\nMETADATA CONSERVADA")

for column in technical_columns_to_keep:

    print(
        f"KEEP -> {column}"
    )


print("\nMETADATA ELIMINADA")

for column in technical_columns_to_drop:

    print(
        f"DROP -> {column}"
    )

# COMMAND ----------

# DBTITLE 1,08. TRANSFORMACIÓN DE CUSTOMERS BRONZE -> SILVER
# ============================================================
# 08. TRANSFORMACIÓN DE CUSTOMERS BRONZE -> SILVER
# ============================================================
#
# Objetivo:
#
# Construir una versión limpia y estandarizada de customers.
#
# Aplicaremos:
#
# - selección explícita de columnas
# - trim de campos de texto
# - tipos explícitos
# - normalización de segmento
# - eliminación de _rescued_data
# - conservación de metadata técnica
#
# No escribimos todavía la tabla.
# Primero construimos y validamos el DataFrame Silver.
# ============================================================


from pyspark.sql import functions as F


# ------------------------------------------------------------
# CONSTRUCCIÓN DEL DATAFRAME SILVER
# ------------------------------------------------------------

customers_silver_df = (

    customers_bronze_df

    # --------------------------------------------------------
    # SELECCIÓN Y NORMALIZACIÓN DE COLUMNAS
    # --------------------------------------------------------
    #
    # Usamos select() para definir explícitamente qué columnas
    # queremos que existan en Silver.
    #
    # Esto evita arrastrar columnas técnicas innecesarias
    # como _rescued_data.
    #
    .select(

        # Clave del cliente.
        F.trim(
            F.col("customer_id")
        ).alias("customer_id"),

        # Nombre completo.
        F.trim(
            F.col("customer_name")
        ).alias("customer_name"),

        # Email.
        #
        # Además de eliminar posibles espacios,
        # lo pasamos a minúsculas para estandarizarlo.
        F.lower(
            F.trim(
                F.col("email")
            )
        ).alias("email"),

        # Ciudad.
        F.trim(
            F.col("city")
        ).alias("city"),

        # Región.
        F.trim(
            F.col("region")
        ).alias("region"),

        # Fecha de registro.
        #
        # Aunque Bronze ya la haya inferido correctamente,
        # en Silver dejamos el contrato de tipo explícito.
        F.col(
            "registration_date"
        ).cast("date").alias(
            "registration_date"
        ),

        # Segmento.
        #
        # Convertimos a mayúsculas para garantizar
        # una representación homogénea:
        #
        # STANDARD
        # PREMIUM
        # VIP
        F.upper(
            F.trim(
                F.col("customer_segment")
            )
        ).alias("customer_segment"),

        # ----------------------------------------------------
        # METADATA TÉCNICA
        # ----------------------------------------------------
        #
        # Conservamos trazabilidad hacia Bronze y origen.
        #
        F.col("_ingestion_timestamp"),

        F.col("_source_file"),

        F.col(
            "_source_file_modification_time"
        )
    )
)

# COMMAND ----------

# DBTITLE 1,08.01 INSPECCIÓN DEL RESULTADO CUSTOMERS SILVER
# ============================================================
# 08.01 INSPECCIÓN DEL RESULTADO CUSTOMERS SILVER
# ============================================================

display(
    customers_silver_df.limit(20)
)

# COMMAND ----------

# DBTITLE 1,08.02 VALIDACIÓN DEL SCHEMA CUSTOMERS SILVER
# ============================================================
# 08.02 VALIDACIÓN DEL SCHEMA CUSTOMERS SILVER
# ============================================================

customers_silver_df.printSchema()

# COMMAND ----------

# DBTITLE 1,08.03 VALIDACIONES DE CUSTOMERS SILVER
# ============================================================
# 08.03 VALIDACIONES DE CUSTOMERS SILVER
# ============================================================
#
# Comprobamos:
#
# - mismo número de registros que Bronze
# - customer_id no NULL
# - customer_id único
# - segmento dentro del dominio esperado
# - email no NULL
#
# ============================================================


# Número de registros Bronze.
customers_bronze_count = (
    customers_bronze_df.count()
)


# Número de registros Silver.
customers_silver_count = (
    customers_silver_df.count()
)


# ------------------------------------------------------------
# CUSTOMER_ID NULL
# ------------------------------------------------------------

null_customer_ids = (
    customers_silver_df
    .filter(
        F.col("customer_id").isNull()
    )
    .count()
)


# ------------------------------------------------------------
# CUSTOMER_ID DUPLICADO
# ------------------------------------------------------------

duplicated_customer_ids = (

    customers_silver_df

    .groupBy(
        "customer_id"
    )

    .count()

    .filter(
        F.col("count") > 1
    )

    .count()
)


# ------------------------------------------------------------
# SEGMENTOS INVÁLIDOS
# ------------------------------------------------------------

invalid_segments = (

    customers_silver_df

    .filter(
        ~F.col(
            "customer_segment"
        ).isin(
            "STANDARD",
            "PREMIUM",
            "VIP"
        )
        |
        F.col(
            "customer_segment"
        ).isNull()
    )

    .count()
)


# ------------------------------------------------------------
# EMAIL NULL
# ------------------------------------------------------------

null_emails = (

    customers_silver_df

    .filter(
        F.col("email").isNull()
    )

    .count()
)


# ------------------------------------------------------------
# RESULTADOS
# ------------------------------------------------------------

customer_validation_results = {
    "Bronze vs Silver":
        customers_bronze_count
        -
        customers_silver_count,

    "customer_id NULL":
        null_customer_ids,

    "customer_id duplicado":
        duplicated_customer_ids,

    "segmento inválido":
        invalid_segments,

    "email NULL":
        null_emails
}


for rule, errors in customer_validation_results.items():

    status = (
        "OK"
        if errors == 0
        else "ERROR"
    )

    print(
        f"{status:5} | "
        f"{rule:30} | "
        f"{errors:,}"
    )

# COMMAND ----------

# DBTITLE 1,08.04 ESCRITURA DE CUSTOMERS EN SILVER
# ============================================================
# 08.04 ESCRITURA DE CUSTOMERS EN SILVER
# ============================================================
#
# Una vez validado el DataFrame, lo persistimos como tabla Delta
# gestionada dentro de Unity Catalog.
#
# En esta fase usamos una escritura batch normal, no streaming.
#
# ¿Por qué?
#
# Porque Silver se construye a partir de una tabla Bronze ya
# materializada. Más adelante podremos evolucionar este diseño
# hacia pipelines incrementales, pero para esta primera versión
# priorizamos claridad y control.
# ============================================================


# ------------------------------------------------------------
# NOMBRE DE LA TABLA DESTINO
# ------------------------------------------------------------

customers_silver_table = (
    f"{CATALOG}.2_silver.customers"
)


# ------------------------------------------------------------
# ESCRITURA DEL DATAFRAME
# ------------------------------------------------------------
#
# format("delta")
#   La tabla se almacenará en Delta Lake.
#
# mode("overwrite")
#   Para esta primera construcción de Silver reemplazamos
#   completamente la tabla si ya existiera.
#
# overwriteSchema = true
#   Permite actualizar el schema de la tabla destino si
#   cambiara durante el desarrollo.
#
# saveAsTable(...)
#   Registra la tabla directamente en Unity Catalog.
#
(
    customers_silver_df
    .write
    .format("delta")
    .mode("overwrite")
    .option(
        "overwriteSchema",
        "true"
    )
    .saveAsTable(
        customers_silver_table
    )
)

# COMMAND ----------

# DBTITLE 1,08.05 VALIDACIÓN DE CUSTOMERS SILVER MATERIALIZADA
# ============================================================
# 08.05 VALIDACIÓN DE CUSTOMERS SILVER MATERIALIZADA
# ============================================================

customers_silver_table_count = spark.sql(f"""
    SELECT COUNT(*)
    FROM {CATALOG}.`2_silver`.customers
""").collect()[0][0]


print(
    "Filas en Customers Silver:",
    customers_silver_table_count
)

# COMMAND ----------

# DBTITLE 1,09. TRANSFORMACIÓN DE PRODUCTS BRONZE -> SILVER
# ============================================================
# 09. TRANSFORMACIÓN DE PRODUCTS BRONZE -> SILVER
# ============================================================
#
# Objetivo:
#
# Construir una versión limpia y estandarizada del maestro
# de productos.
#
# Aplicaremos:
#
# - selección explícita de columnas
# - limpieza de espacios en textos
# - normalización de category y subcategory
# - tipos explícitos para importes
# - conversión de active a BOOLEAN
# - eliminación de _rescued_data
# - conservación de metadata técnica
#
# IMPORTANTE:
#
# Primero construimos el DataFrame.
# Después lo validamos.
# Solo si supera las validaciones lo escribiremos en Silver.
# ============================================================


from pyspark.sql import functions as F


products_silver_df = (

    products_bronze_df

    .select(

        # ----------------------------------------------------
        # IDENTIFICADOR DEL PRODUCTO
        # ----------------------------------------------------

        F.trim(
            F.col("product_id")
        ).alias("product_id"),


        # ----------------------------------------------------
        # NOMBRE DEL PRODUCTO
        # ----------------------------------------------------

        F.trim(
            F.col("product_name")
        ).alias("product_name"),


        # ----------------------------------------------------
        # CATEGORÍA
        # ----------------------------------------------------
        #
        # Utilizamos initcap() para estandarizar:
        #
        # electronics -> Electronics
        # ELECTRONICS -> Electronics
        #
        F.initcap(
            F.trim(
                F.col("category")
            )
        ).alias("category"),


        # ----------------------------------------------------
        # SUBCATEGORÍA
        # ----------------------------------------------------

        F.initcap(
            F.trim(
                F.col("subcategory")
            )
        ).alias("subcategory"),


        # ----------------------------------------------------
        # MARCA
        # ----------------------------------------------------

        F.trim(
            F.col("brand")
        ).alias("brand"),


        # ----------------------------------------------------
        # COSTE UNITARIO
        # ----------------------------------------------------
        #
        # En Silver dejamos explícitamente el tipo decimal.
        #
        # DECIMAL(10,2) evita depender de la inferencia
        # realizada durante la ingestión.
        #
        F.col("unit_cost")
        .cast("decimal(10,2)")
        .alias("unit_cost"),


        # ----------------------------------------------------
        # PRECIO BASE
        # ----------------------------------------------------

        F.col("base_price")
        .cast("decimal(10,2)")
        .alias("base_price"),


        # ----------------------------------------------------
        # PRODUCTO ACTIVO
        # ----------------------------------------------------
        #
        # Convertimos explícitamente el indicador a BOOLEAN.
        #
        F.col("active")
        .cast("boolean")
        .alias("active"),


        # ----------------------------------------------------
        # METADATA TÉCNICA
        # ----------------------------------------------------

        F.col("_ingestion_timestamp"),

        F.col("_source_file"),

        F.col(
            "_source_file_modification_time"
        )
    )
)

# COMMAND ----------

# DBTITLE 1,09.01 INSPECCIÓN DEL RESULTADO PRODUCTS SILVER
# ============================================================
# 09.01 INSPECCIÓN DEL RESULTADO PRODUCTS SILVER
# ============================================================

display(
    products_silver_df.limit(20)
)

# COMMAND ----------

# DBTITLE 1,09.02 VALIDACIÓN DEL SCHEMA PRODUCTS SILVER
# ============================================================
# 09.02 VALIDACIÓN DEL SCHEMA PRODUCTS SILVER
# ============================================================

products_silver_df.printSchema()

# COMMAND ----------

# DBTITLE 1,09.03 VALIDACIONES DE PRODUCTS SILVER
# ============================================================
# 09.03 VALIDACIONES DE PRODUCTS SILVER
# ============================================================


# ------------------------------------------------------------
# RECUENTOS BRONZE VS SILVER
# ------------------------------------------------------------

products_bronze_count = (
    products_bronze_df.count()
)

products_silver_count = (
    products_silver_df.count()
)


# ------------------------------------------------------------
# PRODUCT_ID NULL
# ------------------------------------------------------------

null_product_ids = (
    products_silver_df
    .filter(
        F.col("product_id").isNull()
    )
    .count()
)


# ------------------------------------------------------------
# PRODUCT_ID DUPLICADO
# ------------------------------------------------------------

duplicated_product_ids = (

    products_silver_df

    .groupBy("product_id")

    .count()

    .filter(
        F.col("count") > 1
    )

    .count()
)


# ------------------------------------------------------------
# COSTES / PRECIOS INVÁLIDOS
# ------------------------------------------------------------
#
# Validamos:
#
# unit_cost > 0
# base_price > 0
# base_price > unit_cost
#
invalid_prices = (

    products_silver_df

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
# ACTIVE NULL
# ------------------------------------------------------------

null_active = (
    products_silver_df
    .filter(
        F.col("active").isNull()
    )
    .count()
)


# ------------------------------------------------------------
# RESULTADOS
# ------------------------------------------------------------

product_validation_results = {

    "Bronze vs Silver":
        products_bronze_count
        -
        products_silver_count,

    "product_id NULL":
        null_product_ids,

    "product_id duplicado":
        duplicated_product_ids,

    "precio/coste inválido":
        invalid_prices,

    "active NULL":
        null_active
}


for rule, errors in product_validation_results.items():

    status = (
        "OK"
        if errors == 0
        else "ERROR"
    )

    print(
        f"{status:5} | "
        f"{rule:30} | "
        f"{errors:,}"
    )

# COMMAND ----------

# DBTITLE 1,09.04 ESCRITURA DE PRODUCTS EN SILVER
# ============================================================
# 09.04 ESCRITURA DE PRODUCTS EN SILVER
# ============================================================
#
# El DataFrame ya ha superado:
#
# - validación de claves
# - validación de NULLs
# - validación de tipos
# - validación de precios y costes
#
# Por tanto, ya podemos persistirlo como tabla Delta
# gestionada dentro de Unity Catalog.
# ============================================================


# ------------------------------------------------------------
# NOMBRE DE LA TABLA DESTINO
# ------------------------------------------------------------

products_silver_table = (
    f"{CATALOG}.2_silver.products"
)


# ------------------------------------------------------------
# ESCRITURA DEL DATAFRAME
# ------------------------------------------------------------
#
# format("delta")
#   Guardamos la tabla en formato Delta.
#
# mode("overwrite")
#   En esta primera versión de Silver reconstruimos
#   completamente la tabla.
#
# overwriteSchema = true
#   Nos permite modificar el schema durante el desarrollo
#   sin tener que eliminar manualmente la tabla.
#
(
    products_silver_df
    .write
    .format("delta")
    .mode("overwrite")
    .option(
        "overwriteSchema",
        "true"
    )
    .saveAsTable(
        products_silver_table
    )
)

# COMMAND ----------

# DBTITLE 1,09.05 VALIDACIÓN DE PRODUCTS SILVER MATERIALIZADA
# ============================================================
# 09.05 VALIDACIÓN DE PRODUCTS SILVER MATERIALIZADA
# ============================================================

products_silver_table_count = spark.sql(f"""
    SELECT COUNT(*)
    FROM {CATALOG}.`2_silver`.products
""").collect()[0][0]


print(
    "Filas en Products Silver:",
    products_silver_table_count
)

# COMMAND ----------

# DBTITLE 1,10. TRANSFORMACIÓN DE STORES BRONZE -> SILVER
# ============================================================
# 10. TRANSFORMACIÓN DE STORES BRONZE -> SILVER
# ============================================================
#
# Objetivo:
#
# Construir una versión limpia y estandarizada del maestro
# de tiendas.
#
# Aquí tenemos nuestra primera corrección real de Data Quality:
#
#     BCN -> Barcelona
#
# Esta anomalía fue introducida deliberadamente durante
# la generación de los datos para poder practicar una
# transformación real en Silver.
#
# También aplicaremos:
#
# - selección explícita de columnas
# - limpieza de espacios
# - estandarización de textos
# - tipos explícitos
# - eliminación de _rescued_data
# - conservación de metadata técnica
# ============================================================


from pyspark.sql import functions as F


stores_silver_df = (

    stores_bronze_df

    .select(

        # ----------------------------------------------------
        # STORE ID
        # ----------------------------------------------------

        F.trim(
            F.col("store_id")
        ).alias("store_id"),


        # ----------------------------------------------------
        # STORE NAME
        # ----------------------------------------------------

        F.trim(
            F.col("store_name")
        ).alias("store_name"),


        # ----------------------------------------------------
        # CITY
        # ----------------------------------------------------
        #
        # Corregimos la anomalía conocida:
        #
        # BCN -> Barcelona
        #
        # otherwise() conserva el valor original para
        # cualquier otra ciudad.
        #
        F.when(
            F.upper(
                F.trim(F.col("city"))
            ) == "BCN",
            "Barcelona"
        )
        .otherwise(
            F.trim(F.col("city"))
        )
        .alias("city"),


        # ----------------------------------------------------
        # REGION
        # ----------------------------------------------------

        F.trim(
            F.col("region")
        ).alias("region"),


        # ----------------------------------------------------
        # OPENING DATE
        # ----------------------------------------------------
        #
        # Dejamos explícitamente DATE como contrato Silver.
        #
        F.col("opening_date")
        .cast("date")
        .alias("opening_date"),


        # ----------------------------------------------------
        # STORE TYPE
        # ----------------------------------------------------
        #
        # Normalizamos el dominio a mayúsculas:
        #
        # PHYSICAL
        # ONLINE
        #
        F.upper(
            F.trim(
                F.col("store_type")
            )
        ).alias("store_type"),


        # ----------------------------------------------------
        # METADATA TÉCNICA
        # ----------------------------------------------------

        F.col("_ingestion_timestamp"),

        F.col("_source_file"),

        F.col(
            "_source_file_modification_time"
        )
    )
)

# COMMAND ----------

# DBTITLE 1,10.01 VALIDACIÓN DE NORMALIZACIÓN DE CITY
# ============================================================
# 10.01 VALIDACIÓN DE NORMALIZACIÓN DE CITY
# ============================================================

display(
    stores_silver_df
    .select(
        "store_id",
        "store_name",
        "city",
        "region"
    )
    .orderBy(
        "store_id"
    )
)

# COMMAND ----------

# DBTITLE 1,10.02 VALIDACIÓN DE STORES SILVER
# ============================================================
# 10.02 VALIDACIÓN DE STORES SILVER
# ============================================================


# ------------------------------------------------------------
# RECUENTOS
# ------------------------------------------------------------

stores_bronze_count = (
    stores_bronze_df.count()
)

stores_silver_count = (
    stores_silver_df.count()
)


# ------------------------------------------------------------
# STORE_ID NULL
# ------------------------------------------------------------

null_store_ids = (
    stores_silver_df
    .filter(
        F.col("store_id").isNull()
    )
    .count()
)


# ------------------------------------------------------------
# STORE_ID DUPLICADO
# ------------------------------------------------------------

duplicated_store_ids = (

    stores_silver_df

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

    stores_silver_df

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
# BCN SIN CORREGIR
# ------------------------------------------------------------
#
# Después de la transformación no debería existir
# ningún registro cuyo city siga siendo BCN.
#
remaining_bcn = (

    stores_silver_df

    .filter(
        F.upper(
            F.trim(F.col("city"))
        ) == "BCN"
    )

    .count()
)


# ------------------------------------------------------------
# RESULTADOS
# ------------------------------------------------------------

store_validation_results = {

    "Bronze vs Silver":
        stores_bronze_count
        -
        stores_silver_count,

    "store_id NULL":
        null_store_ids,

    "store_id duplicado":
        duplicated_store_ids,

    "store_type inválido":
        invalid_store_types,

    "BCN sin corregir":
        remaining_bcn
}


for rule, errors in store_validation_results.items():

    status = (
        "OK"
        if errors == 0
        else "ERROR"
    )

    print(
        f"{status:5} | "
        f"{rule:30} | "
        f"{errors:,}"
    )

# COMMAND ----------

# DBTITLE 1,10.03 ESCRITURA DE STORES EN SILVER
# ============================================================
# 10.03 ESCRITURA DE STORES EN SILVER
# ============================================================
#
# stores_silver_df ya ha superado nuestras validaciones:
#
# - mismo número de registros que Bronze
# - store_id no NULL
# - store_id único
# - store_type dentro del dominio esperado
# - normalización BCN -> Barcelona completada
#
# Por tanto, podemos materializar el DataFrame como tabla
# Delta gestionada en Unity Catalog.
# ============================================================


# ------------------------------------------------------------
# TABLA DESTINO
# ------------------------------------------------------------

stores_silver_table = (
    f"{CATALOG}.2_silver.stores"
)


# ------------------------------------------------------------
# ESCRITURA DEL DATAFRAME
# ------------------------------------------------------------

(
    stores_silver_df

    .write

    # Persistimos utilizando Delta Lake.
    .format("delta")

    # Durante esta primera construcción de Silver
    # reconstruimos completamente la tabla.
    .mode("overwrite")

    # Permitimos que el schema se actualice durante
    # esta fase de desarrollo.
    .option(
        "overwriteSchema",
        "true"
    )

    # Registramos la tabla dentro de Unity Catalog.
    .saveAsTable(
        stores_silver_table
    )
)

# COMMAND ----------

# DBTITLE 1,10.04 VALIDACIÓN DE STORES SILVER MATERIALIZADA
# ============================================================
# 10.04 VALIDACIÓN DE STORES SILVER MATERIALIZADA
# ============================================================

stores_silver_table_count = spark.sql(f"""
    SELECT COUNT(*)
    FROM {CATALOG}.`2_silver`.stores
""").collect()[0][0]


print(
    "Filas en Stores Silver:",
    stores_silver_table_count
)

# COMMAND ----------

# DBTITLE 1,11.00 IDENTIFICACIÓN DE REGISTROS INCREMENTALES DE SALES
# ============================================================
# 11.00 IDENTIFICACIÓN DE REGISTROS INCREMENTALES DE SALES
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Evitar transformar nuevamente todo el histórico almacenado
# en Bronze cada vez que se ejecuta el pipeline.
#
# Utilizamos como clave de negocio:
#
#       sale_id + line_id
#
# Compararemos las claves existentes en Silver con Bronze.
#
# Solo continuarán hacia la transformación aquellos registros
# cuya clave todavía NO exista en Silver.
#
# ESCENARIOS
# ------------------------------------------------------------
#
# Primera ejecución:
#   Silver no existe -> procesamos todo Bronze.
#
# Ejecuciones posteriores:
#   Silver existe -> procesamos únicamente registros nuevos.
#
# IMPORTANTE
# ------------------------------------------------------------
#
# Este modelo asume que SALES es append-only.
# Es decir, las líneas históricas no se modifican después
# de haber sido incorporadas.
#
# ============================================================


from pyspark.sql import functions as F


# ------------------------------------------------------------
# TABLA SILVER DESTINO
# ------------------------------------------------------------

sales_silver_table = (
    f"{CATALOG}.2_silver.sales"
)


# ------------------------------------------------------------
# COMPROBAMOS SI SALES SILVER YA EXISTE
# ------------------------------------------------------------

if spark.catalog.tableExists(
    sales_silver_table
):

    # Recuperamos únicamente las claves ya procesadas.
    #
    # No necesitamos leer todas las columnas de Silver para
    # identificar qué registros son nuevos.

    silver_existing_keys_df = (

        spark.table(
            sales_silver_table
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
    # Conservamos únicamente registros presentes en Bronze
    # cuya combinación:
    #
    #       sale_id + line_id
    #
    # todavía no existe en Silver.
    #
    # Ejemplo:
    #
    # Bronze:
    #   T000001 / 1
    #   T000001 / 2
    #   T020001 / 1   <- nuevo
    #
    # Silver:
    #   T000001 / 1
    #   T000001 / 2
    #
    # Resultado:
    #   T020001 / 1
    #
    # --------------------------------------------------------

    sales_bronze_incremental_df = (

        sales_bronze_df.alias(
            "bronze"
        )

        .join(

            silver_existing_keys_df.alias(
                "silver"
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
    # Si todavía no existe una tabla Silver, todo Bronze
    # constituye nuestro primer batch.
    #
    # --------------------------------------------------------

    sales_bronze_incremental_df = (
        sales_bronze_df
    )


# ------------------------------------------------------------
# CONTAMOS EL BATCH A PROCESAR
# ------------------------------------------------------------

incremental_rows = (
    sales_bronze_incremental_df
    .count()
)


print(
    "Registros nuevos a procesar en Silver:",
    incremental_rows
)

# COMMAND ----------

# DBTITLE 1,11. TRANSFORMACIÓN DE SALES BRONZE -> SILVER
# ============================================================
# 11. TRANSFORMACIÓN INCREMENTAL DE SALES BRONZE -> SILVER
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Construir una versión limpia, tipada y enriquecida
# exclusivamente del nuevo batch de SALES detectado
# en el bloque 11.00.
#
# Aplicaremos:
#
# - selección explícita de columnas
# - tipos definitivos
# - conservación de customer_id NULL
# - normalización de atributos
# - derivación de atributos temporales
# - cálculo de gross_amount
# - cálculo de discount_amount
# - cálculo de net_amount
# - conservación de metadata técnica
#
# DIFERENCIA RESPECTO A LA VERSIÓN ANTERIOR
# ------------------------------------------------------------
#
# Antes:
#
#       sales_bronze_df
#
# contenía todo el histórico.
#
# Ahora:
#
#       sales_bronze_incremental_df
#
# contiene únicamente las líneas todavía no procesadas.
#
# ============================================================


from pyspark.sql import functions as F


sales_silver_df = (

    sales_bronze_incremental_df


    # ========================================================
    # SELECCIÓN Y TIPADO
    # ========================================================

    .select(

        # ----------------------------------------------------
        # IDENTIFICADORES
        # ----------------------------------------------------

        F.trim(
            F.col("sale_id")
        ).alias(
            "sale_id"
        ),


        F.col(
            "line_id"
        )
        .cast(
            "int"
        )
        .alias(
            "line_id"
        ),


        # ----------------------------------------------------
        # FECHA / HORA
        # ----------------------------------------------------

        F.col(
            "sale_timestamp"
        )
        .cast(
            "timestamp"
        )
        .alias(
            "sale_timestamp"
        ),


        # ----------------------------------------------------
        # CLAVES DE MAESTROS
        # ----------------------------------------------------

        F.trim(
            F.col("store_id")
        ).alias(
            "store_id"
        ),


        # customer_id puede ser NULL porque admitimos
        # ventas anónimas.

        F.trim(
            F.col("customer_id")
        ).alias(
            "customer_id"
        ),


        F.trim(
            F.col("product_id")
        ).alias(
            "product_id"
        ),


        # ----------------------------------------------------
        # MEDIDAS BASE
        # ----------------------------------------------------

        F.col(
            "quantity"
        )
        .cast(
            "int"
        )
        .alias(
            "quantity"
        ),


        F.col(
            "unit_price"
        )
        .cast(
            "decimal(12,2)"
        )
        .alias(
            "unit_price"
        ),


        F.col(
            "discount_pct"
        )
        .cast(
            "decimal(5,4)"
        )
        .alias(
            "discount_pct"
        ),


        # ----------------------------------------------------
        # ATRIBUTOS DE NEGOCIO
        # ----------------------------------------------------

        F.upper(
            F.trim(
                F.col(
                    "sales_channel"
                )
            )
        ).alias(
            "sales_channel"
        ),


        F.upper(
            F.trim(
                F.col(
                    "payment_method"
                )
            )
        ).alias(
            "payment_method"
        ),


        # ----------------------------------------------------
        # METADATA TÉCNICA
        # ----------------------------------------------------

        F.col(
            "_ingestion_timestamp"
        ),


        F.col(
            "_source_file"
        ),


        F.col(
            "_source_file_modification_time"
        )
    )


    # ========================================================
    # COLUMNAS DERIVADAS DE FECHA
    # ========================================================

    .withColumn(
        "sale_date",

        F.to_date(
            F.col(
                "sale_timestamp"
            )
        )
    )


    .withColumn(
        "sale_year",

        F.year(
            F.col(
                "sale_timestamp"
            )
        )
    )


    .withColumn(
        "sale_month",

        F.month(
            F.col(
                "sale_timestamp"
            )
        )
    )


    .withColumn(
        "sale_day",

        F.dayofmonth(
            F.col(
                "sale_timestamp"
            )
        )
    )


    .withColumn(
        "sale_week",

        F.weekofyear(
            F.col(
                "sale_timestamp"
            )
        )
    )


    # ========================================================
    # GROSS AMOUNT
    # ========================================================

    .withColumn(
        "gross_amount",

        F.round(

            F.col(
                "quantity"
            )
            *
            F.col(
                "unit_price"
            ),

            2
        )

        .cast(
            "decimal(14,2)"
        )
    )


    # ========================================================
    # DISCOUNT AMOUNT
    # ========================================================

    .withColumn(
        "discount_amount",

        F.round(

            F.col(
                "gross_amount"
            )
            *
            F.col(
                "discount_pct"
            ),

            2
        )

        .cast(
            "decimal(14,2)"
        )
    )


    # ========================================================
    # NET AMOUNT
    # ========================================================

    .withColumn(
        "net_amount",

        F.round(

            F.col(
                "gross_amount"
            )
            -
            F.col(
                "discount_amount"
            ),

            2
        )

        .cast(
            "decimal(14,2)"
        )
    )
)


# ------------------------------------------------------------
# RESULTADO
# ------------------------------------------------------------

print(
    "Registros transformados en el batch Silver:",
    sales_silver_df.count()
)

# COMMAND ----------

# DBTITLE 1,11.01 INSPECCIÓN DEL BATCH INCREMENTAL SALES SILVER
# ============================================================
# 11.01 INSPECCIÓN DEL BATCH INCREMENTAL SALES SILVER
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Inspeccionar visualmente el resultado de la transformación
# incremental antes de escribirlo en Silver.
#
# ============================================================


print(
    "Registros del batch:",
    sales_silver_df.count()
)


display(

    sales_silver_df

    .orderBy(
        "sale_timestamp",
        "sale_id",
        "line_id"
    )

    .limit(
        20
    )
)

# COMMAND ----------

# DBTITLE 1,VALIDACIÓN DEL SCHEMA DEL BATCH SALES SILVER
# ============================================================
# 11.02 VALIDACIÓN DEL SCHEMA DEL BATCH SALES SILVER
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Comprobar que el batch incremental conserva el schema
# esperado para la tabla Silver.
#
# ============================================================


expected_sales_silver_columns = [

    "sale_id",
    "line_id",
    "sale_timestamp",

    "store_id",
    "customer_id",
    "product_id",

    "quantity",
    "unit_price",
    "discount_pct",

    "sales_channel",
    "payment_method",

    "_ingestion_timestamp",
    "_source_file",
    "_source_file_modification_time",

    "sale_date",
    "sale_year",
    "sale_month",
    "sale_day",
    "sale_week",

    "gross_amount",
    "discount_amount",
    "net_amount"
]


actual_sales_silver_columns = (
    sales_silver_df.columns
)


missing_columns = (

    set(
        expected_sales_silver_columns
    )

    -

    set(
        actual_sales_silver_columns
    )
)


unexpected_columns = (

    set(
        actual_sales_silver_columns
    )

    -

    set(
        expected_sales_silver_columns
    )
)


print(
    "Columnas esperadas:",
    len(
        expected_sales_silver_columns
    )
)


print(
    "Columnas obtenidas:",
    len(
        actual_sales_silver_columns
    )
)


print(
    "Columnas ausentes:",
    missing_columns
)


print(
    "Columnas inesperadas:",
    unexpected_columns
)


if (
    len(missing_columns) == 0
    and
    len(unexpected_columns) == 0
):

    print(
        "OK - Schema incremental Silver correcto"
    )

else:

    print(
        "ERROR - Revisar schema incremental Silver"
    )

# COMMAND ----------

# DBTITLE 1,VALIDACIONES DEL BATCH INCREMENTAL SALES SILVER
# ============================================================
# 11.03 VALIDACIONES DEL BATCH INCREMENTAL SALES SILVER
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Validar únicamente los nuevos registros que van a ser
# incorporados a Silver.
#
# Comprobamos:
#
# - igualdad entre batch Bronze y batch transformado
# - clave sale_id + line_id única
# - IDs obligatorios no NULL
# - quantity válida
# - unit_price válido
# - discount_pct válido
# - importes derivados coherentes
#
# ============================================================


from pyspark.sql import functions as F


# ------------------------------------------------------------
# RECUENTOS
# ------------------------------------------------------------

bronze_incremental_count = (
    sales_bronze_incremental_df
    .count()
)


silver_incremental_count = (
    sales_silver_df
    .count()
)


# ------------------------------------------------------------
# DUPLICADOS
# ------------------------------------------------------------

duplicated_sales_keys = (

    sales_silver_df

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


# ------------------------------------------------------------
# IDs OBLIGATORIOS NULL
# ------------------------------------------------------------

null_mandatory_ids = (

    sales_silver_df

    .filter(

        F.col(
            "sale_id"
        ).isNull()

        |

        F.col(
            "line_id"
        ).isNull()

        |

        F.col(
            "store_id"
        ).isNull()

        |

        F.col(
            "product_id"
        ).isNull()
    )

    .count()
)


# ------------------------------------------------------------
# QUANTITY INVÁLIDA
# ------------------------------------------------------------

invalid_quantity = (

    sales_silver_df

    .filter(
        F.col(
            "quantity"
        ) <= 0
    )

    .count()
)


# ------------------------------------------------------------
# UNIT PRICE INVÁLIDO
# ------------------------------------------------------------

invalid_unit_price = (

    sales_silver_df

    .filter(
        F.col(
            "unit_price"
        ) <= 0
    )

    .count()
)


# ------------------------------------------------------------
# DESCUENTO INVÁLIDO
# ------------------------------------------------------------

invalid_discount = (

    sales_silver_df

    .filter(

        (
            F.col(
                "discount_pct"
            ) < 0
        )

        |

        (
            F.col(
                "discount_pct"
            ) > 1
        )
    )

    .count()
)


# ------------------------------------------------------------
# IMPORTES DERIVADOS
# ------------------------------------------------------------

invalid_amounts = (

    sales_silver_df

    .filter(

        F.abs(

            F.col(
                "net_amount"
            )

            -

            (
                F.col(
                    "gross_amount"
                )
                -
                F.col(
                    "discount_amount"
                )
            )

        ) > 0.01
    )

    .count()
)


# ------------------------------------------------------------
# RESULTADOS
# ------------------------------------------------------------

validation_results = {

    "Bronze incremental vs Silver batch":
        abs(
            bronze_incremental_count
            -
            silver_incremental_count
        ),

    "sale_id + line_id duplicada":
        duplicated_sales_keys,

    "IDs obligatorios NULL":
        null_mandatory_ids,

    "quantity inválida":
        invalid_quantity,

    "unit_price inválido":
        invalid_unit_price,

    "discount_pct inválido":
        invalid_discount,

    "importes derivados inválidos":
        invalid_amounts
}


for rule, errors in (
    validation_results.items()
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

# DBTITLE 1,11.01.01 INSPECCIÓN DEL RESULTADO SALES SILVER
# ============================================================
# 11.01.01 INSPECCIÓN DEL RESULTADO SALES SILVER
# ============================================================

display(
    sales_silver_df
    .select(
        "sale_id",
        "line_id",
        "sale_timestamp",
        "sale_date",
        "sale_year",
        "sale_month",
        "store_id",
        "customer_id",
        "product_id",
        "quantity",
        "unit_price",
        "discount_pct",
        "gross_amount",
        "discount_amount",
        "net_amount",
        "sales_channel",
        "payment_method"
    )
    .limit(20)
)

# COMMAND ----------

# DBTITLE 1,11.02 VALIDACIÓN DEL SCHEMA SALES SILVER
# ============================================================
# 11.02 VALIDACIÓN DEL SCHEMA SALES SILVER
# ============================================================

sales_silver_df.printSchema()

# COMMAND ----------

# DBTITLE 1,11.03 VALIDACIONES DE SALES SILVER
# ============================================================
# 11.03 VALIDACIONES DE SALES SILVER
# ============================================================
#
# Validaremos que las transformaciones no hayan introducido
# problemas nuevos.
#
# Comprobaciones:
#
# - mismo número de registros Bronze vs Silver
# - clave sale_id + line_id única
# - IDs obligatorios no NULL
# - quantity válida
# - unit_price válido
# - discount_pct válido
# - importes coherentes
# - customer_id NULL permitido
# ============================================================


# ------------------------------------------------------------
# RECUENTOS
# ------------------------------------------------------------

sales_bronze_count = (
    sales_bronze_df.count()
)

sales_silver_count = (
    sales_silver_df.count()
)


# ------------------------------------------------------------
# IDS OBLIGATORIOS NULL
# ------------------------------------------------------------

null_mandatory_ids = (

    sales_silver_df

    .filter(
        F.col("sale_id").isNull()
        |
        F.col("line_id").isNull()
        |
        F.col("store_id").isNull()
        |
        F.col("product_id").isNull()
    )

    .count()
)


# ------------------------------------------------------------
# CLAVE DUPLICADA
# ------------------------------------------------------------

duplicated_sales_keys = (

    sales_silver_df

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


# ------------------------------------------------------------
# QUANTITY INVÁLIDA
# ------------------------------------------------------------

invalid_quantity = (

    sales_silver_df

    .filter(
        F.col("quantity").isNull()
        |
        (F.col("quantity") < 1)
        |
        (F.col("quantity") > 5)
    )

    .count()
)


# ------------------------------------------------------------
# UNIT PRICE INVÁLIDO
# ------------------------------------------------------------

invalid_unit_price = (

    sales_silver_df

    .filter(
        F.col("unit_price").isNull()
        |
        (F.col("unit_price") <= 0)
    )

    .count()
)


# ------------------------------------------------------------
# DISCOUNT INVÁLIDO
# ------------------------------------------------------------

invalid_discount = (

    sales_silver_df

    .filter(
        F.col("discount_pct").isNull()
        |
        (F.col("discount_pct") < 0)
        |
        (F.col("discount_pct") > 0.20)
    )

    .count()
)


# ------------------------------------------------------------
# IMPORTES INVÁLIDOS
# ------------------------------------------------------------
#
# Comprobamos:
#
# gross_amount > 0
#
# discount_amount >= 0
#
# net_amount > 0
#
# net_amount <= gross_amount
#
invalid_amounts = (

    sales_silver_df

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


# ------------------------------------------------------------
# RESULTADOS
# ------------------------------------------------------------

sales_validation_results = {

    "Bronze vs Silver":
        sales_bronze_count
        -
        sales_silver_count,

    "IDs obligatorios NULL":
        null_mandatory_ids,

    "clave sale_id + line_id duplicada":
        duplicated_sales_keys,

    "quantity inválida":
        invalid_quantity,

    "unit_price inválido":
        invalid_unit_price,

    "discount_pct inválido":
        invalid_discount,

    "importes derivados inválidos":
        invalid_amounts
}


for rule, errors in sales_validation_results.items():

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
# ============================================================
# DEDUPLICACIÓN SALES SILVER
# ============================================================
#
# Silver debe contener una única fila por:
#
#       sale_id + line_id
#
# Bronze conserva fielmente la fuente, incluidos posibles
# duplicados. Silver aplica la regla de unicidad.
#

sales_silver_df = (
    sales_silver_df
    .dropDuplicates(
        ["sale_id", "line_id"]
    )
)

print(
    "Registros SALES Silver tras deduplicación:",
    sales_silver_df.count()
)

# COMMAND ----------
# DBTITLE 1,11.04 ESCRITURA INCREMENTAL DE SALES EN SILVER
# ============================================================
# 11.04 ESCRITURA INCREMENTAL DE SALES EN SILVER
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Persistir SALES Silver de forma incremental.
#
# La tabla Bronze crece cada vez que Auto Loader detecta
# nuevos archivos. Por tanto, no necesitamos reconstruir
# toda la tabla Silver en cada ejecución.
#
# Estrategia:
#
#   1. Comprobar si la tabla Silver existe.
#   2. Si NO existe:
#        -> realizar carga inicial completa.
#
#   3. Si existe:
#        -> identificar únicamente registros nuevos
#           mediante:
#
#              sale_id + line_id
#
#        -> insertar solamente esos registros utilizando
#           Delta MERGE.
#
# IMPORTANTE
# ------------------------------------------------------------
#
# Este proyecto utiliza una fuente append-only:
# los archivos nuevos añaden ventas nuevas y no actualizan
# ventas históricas existentes.
#
# Por ese motivo utilizamos:
#
#       WHEN NOT MATCHED THEN INSERT
#
# y NO realizamos UPDATE de registros existentes.
#
# ============================================================


from delta.tables import DeltaTable
from pyspark.sql import functions as F


# ============================================================
# 1. TABLA DESTINO
# ============================================================

sales_silver_table = (
    f"{CATALOG}.2_silver.sales"
)


# ============================================================
# 2. COMPROBAMOS SI SALES SILVER YA EXISTE
# ============================================================

sales_silver_exists = (
    spark.catalog.tableExists(
        sales_silver_table
    )
)


print()
print("=" * 75)
print("ESCRITURA INCREMENTAL SALES SILVER")
print("=" * 75)

print(
    f"Tabla destino: {sales_silver_table}"
)

print(
    f"Tabla existente: {sales_silver_exists}"
)


# ============================================================
# 3. CARGA INICIAL
# ============================================================
#
# Si la tabla todavía no existe, realizamos la primera carga
# completa utilizando el DataFrame Silver ya transformado
# y validado.
#
# ============================================================

if not sales_silver_exists:

    initial_rows = (
        sales_silver_df
        .count()
    )

    print()
    print(
        "INFO - La tabla Silver todavía no existe."
    )

    print(
        f"Registros de carga inicial: {initial_rows:,}"
    )


    (
        sales_silver_df

        .write

        .format("delta")

        .mode("overwrite")

        .option(
            "overwriteSchema",
            "true"
        )

        .saveAsTable(
            sales_silver_table
        )
    )


    print()
    print(
        "OK - Carga inicial de SALES Silver completada."
    )


# ============================================================
# 4. CARGA INCREMENTAL
# ============================================================
#
# Si la tabla ya existe:
#
# - obtenemos únicamente sus claves actuales
# - hacemos LEFT ANTI JOIN contra el DataFrame Silver completo
# - nos quedamos únicamente con las ventas todavía no cargadas
#
# ============================================================

else:

    # --------------------------------------------------------
    # 4.1 LEEMOS LAS CLAVES EXISTENTES EN SILVER
    # --------------------------------------------------------

    existing_sales_keys_df = (

        spark.table(
            sales_silver_table
        )

        .select(
            "sale_id",
            "line_id"
        )

        .distinct()
    )


    # --------------------------------------------------------
    # 4.2 IDENTIFICAMOS NUEVOS REGISTROS
    # --------------------------------------------------------
    #
    # LEFT ANTI:
    #
    # devuelve registros del DataFrame Bronze transformado
    # cuya clave todavía NO existe en Silver.
    #
    # --------------------------------------------------------

    new_sales_silver_df = (

        sales_silver_df
        .alias("source")

        .join(

            existing_sales_keys_df
            .alias("target"),

            on=[
                "sale_id",
                "line_id"
            ],

            how="left_anti"
        )
    )


    # --------------------------------------------------------
    # 4.3 CONTAMOS REGISTROS NUEVOS
    # --------------------------------------------------------

    new_sales_count = (
        new_sales_silver_df
        .count()
    )


    print()

    print(
        f"Registros nuevos detectados: {new_sales_count:,}"
    )


    # ========================================================
    # 5. MERGE DELTA
    # ========================================================

    if new_sales_count > 0:

        # ----------------------------------------------------
        # Recuperamos la tabla Delta destino.
        # ----------------------------------------------------

        silver_delta_table = (
            DeltaTable.forName(
                spark,
                sales_silver_table
            )
        )


        # ----------------------------------------------------
        # MERGE
        # ----------------------------------------------------
        #
        # La clave de negocio/técnica de una línea de venta es:
        #
        #       sale_id + line_id
        #
        # Como la fuente es append-only:
        #
        # - MATCHED     -> no hacemos nada
        # - NOT MATCHED -> insertamos
        #
        # ----------------------------------------------------

        (
            silver_delta_table
            .alias("target")

            .merge(

                new_sales_silver_df
                .alias("source"),

                """
                target.sale_id = source.sale_id
                AND
                target.line_id = source.line_id
                """
            )

            .whenNotMatchedInsertAll()

            .execute()
        )


        print()

        print(
            f"OK - {new_sales_count:,} registros nuevos "
            "insertados en SALES Silver."
        )


    # ========================================================
    # 6. SIN DATOS NUEVOS
    # ========================================================

    else:

        print()

        print(
            "OK - No existen registros nuevos para procesar."
        )

        print(
            "SALES Silver ya estaba completamente actualizada."
        )


# ============================================================
# 7. VALIDACIÓN POST-CARGA
# ============================================================

sales_silver_table_count = (

    spark.table(
        sales_silver_table
    )

    .count()
)


print()
print("-" * 75)

print(
    f"Registros actuales SALES Silver: "
    f"{sales_silver_table_count:,}"
)

print("-" * 75)

# COMMAND ----------

# DBTITLE 1,11.05 VALIDACIÓN DE SALES SILVER MATERIALIZADA
# ============================================================
# 11.05 VALIDACIÓN DE SALES SILVER MATERIALIZADA
# ============================================================

# Contamos directamente sobre la tabla Delta ya guardada,
# no sobre el DataFrame temporal.
#
# Esto confirma que la escritura física se realizó correctamente.

sales_silver_table_count = spark.sql(f"""
    SELECT COUNT(*)
    FROM {CATALOG}.`2_silver`.sales
""").collect()[0][0]


print(
    "Filas en Sales Silver:",
    sales_silver_table_count
)

# COMMAND ----------

# DBTITLE 1,12. VALIDACIÓN FINAL DE LA CAPA SILVER
# ============================================================
# 12. VALIDACIÓN FINAL DE LA CAPA SILVER
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Validar de forma conjunta que las tablas Silver:
#
# - existen
# - contienen los registros de negocio esperados
# - mantienen las columnas técnicas obligatorias
# - NO contienen _rescued_data
#
# PARTICULARIDAD DE SALES
# ------------------------------------------------------------
#
# Bronze representa fielmente la fuente y puede contener
# registros duplicados.
#
# Silver exige una única fila por:
#
#       sale_id + line_id
#
# Por tanto, para SALES no comparamos:
#
#       COUNT(*) Bronze == COUNT(*) Silver
#
# sino:
#
#       claves únicas Bronze == claves únicas Silver
#
# ============================================================


from pyspark.sql import functions as F


# ============================================================
# 1. TABLAS
# ============================================================

tables_to_validate = {

    "customers": {
        "bronze": f"{CATALOG}.1_bronze.customers",
        "silver": f"{CATALOG}.2_silver.customers"
    },

    "products": {
        "bronze": f"{CATALOG}.1_bronze.products",
        "silver": f"{CATALOG}.2_silver.products"
    },

    "stores": {
        "bronze": f"{CATALOG}.1_bronze.stores",
        "silver": f"{CATALOG}.2_silver.stores"
    }
}


sales_bronze_table = (
    f"{CATALOG}.1_bronze.sales"
)

sales_silver_table = (
    f"{CATALOG}.2_silver.sales"
)


# ============================================================
# 2. COLUMNAS TÉCNICAS OBLIGATORIAS
# ============================================================

required_technical_columns = {
    "_ingestion_timestamp",
    "_source_file",
    "_source_file_modification_time"
}


# ============================================================
# 3. CONTENEDOR DE ERRORES
# ============================================================

silver_validation_errors = []


print()
print("=" * 85)
print("VALIDACIÓN FINAL DE LA CAPA SILVER")
print("=" * 85)


# ============================================================
# 4. VALIDACIÓN ESPECÍFICA DE SALES
# ============================================================

print()
print("-" * 85)
print("TABLE: SALES")
print("-" * 85)


# ------------------------------------------------------------
# Existencia
# ------------------------------------------------------------

if not spark.catalog.tableExists(sales_bronze_table):

    silver_validation_errors.append(
        f"No existe Bronze: {sales_bronze_table}"
    )

elif not spark.catalog.tableExists(sales_silver_table):

    silver_validation_errors.append(
        f"No existe Silver: {sales_silver_table}"
    )

else:

    sales_bronze_df_validation = spark.table(
        sales_bronze_table
    )

    sales_silver_df_validation = spark.table(
        sales_silver_table
    )


    # ========================================================
    # 4.1 FILAS FÍSICAS EN BRONZE
    # ========================================================

    bronze_sales_rows = (
        sales_bronze_df_validation
        .count()
    )


    # ========================================================
    # 4.2 CLAVES ÚNICAS EN BRONZE
    # ========================================================

    bronze_sales_unique_keys = (

        sales_bronze_df_validation

        .select(
            "sale_id",
            "line_id"
        )

        .distinct()

        .count()
    )


    # ========================================================
    # 4.3 CLAVES ÚNICAS EN SILVER
    # ========================================================

    silver_sales_rows = (
        sales_silver_df_validation
        .count()
    )


    silver_sales_unique_keys = (

        sales_silver_df_validation

        .select(
            "sale_id",
            "line_id"
        )

        .distinct()

        .count()
    )


    # ========================================================
    # 4.4 DUPLICADOS EN BRONZE
    # ========================================================

    bronze_duplicate_rows = (
        bronze_sales_rows
        -
        bronze_sales_unique_keys
    )


    # ========================================================
    # 4.5 VALIDACIÓN DE COBERTURA
    # ========================================================

    sales_count_ok = (

        bronze_sales_unique_keys
        ==
        silver_sales_unique_keys
        ==
        silver_sales_rows
    )


    if not sales_count_ok:

        silver_validation_errors.append(

            "sales: inconsistencia entre claves únicas "
            "Bronze y Silver"
        )


    print(
        f"{'OK' if sales_count_ok else 'ERROR':5} | "
        f"Bronze filas: {bronze_sales_rows:,} | "
        f"Bronze claves únicas: {bronze_sales_unique_keys:,} | "
        f"Silver: {silver_sales_rows:,}"
    )


    # --------------------------------------------------------
    # Los duplicados en Bronze se informan, pero NO provocan
    # fallo técnico porque Silver ya los controla mediante
    # la clave sale_id + line_id.
    # --------------------------------------------------------

    if bronze_duplicate_rows > 0:

        print(
            f"INFO  | Bronze contiene "
            f"{bronze_duplicate_rows:,} filas duplicadas "
            "por sale_id + line_id"
        )

    else:

        print(
            "OK    | Bronze sin duplicados de clave"
        )


    # ========================================================
    # 4.6 METADATA SALES
    # ========================================================

    sales_columns = set(
        sales_silver_df_validation.columns
    )


    missing_sales_metadata = (

        required_technical_columns
        -
        sales_columns
    )


    if missing_sales_metadata:

        silver_validation_errors.append(

            "sales: faltan columnas técnicas "
            f"{sorted(missing_sales_metadata)}"
        )


    print(
        f"{'OK' if not missing_sales_metadata else 'ERROR':5} | "
        "Metadata técnica"
    )


    # ========================================================
    # 4.7 _rescued_data
    # ========================================================

    rescued_sales_present = (
        "_rescued_data"
        in sales_columns
    )


    if rescued_sales_present:

        silver_validation_errors.append(
            "sales: _rescued_data sigue presente"
        )


    print(
        f"{'OK' if not rescued_sales_present else 'ERROR':5} | "
        "_rescued_data eliminado"
    )


# ============================================================
# 5. VALIDACIÓN DE MAESTROS
# ============================================================

for table_name, table_paths in tables_to_validate.items():

    bronze_table = table_paths["bronze"]
    silver_table = table_paths["silver"]

    print()
    print("-" * 85)
    print(f"TABLE: {table_name.upper()}")
    print("-" * 85)


    # --------------------------------------------------------
    # Existencia
    # --------------------------------------------------------

    if not spark.catalog.tableExists(bronze_table):

        silver_validation_errors.append(
            f"No existe Bronze: {bronze_table}"
        )

        continue


    if not spark.catalog.tableExists(silver_table):

        silver_validation_errors.append(
            f"No existe Silver: {silver_table}"
        )

        continue


    bronze_df_validation = spark.table(
        bronze_table
    )

    silver_df_validation = spark.table(
        silver_table
    )


    # ========================================================
    # 5.1 RECUENTOS
    # ========================================================

    bronze_count = (
        bronze_df_validation
        .count()
    )


    silver_count = (
        silver_df_validation
        .count()
    )


    count_ok = (
        bronze_count == silver_count
    )


    if not count_ok:

        silver_validation_errors.append(

            f"{table_name}: "
            f"Bronze={bronze_count:,}, "
            f"Silver={silver_count:,}"
        )


    print(
        f"{'OK' if count_ok else 'ERROR':5} | "
        f"Bronze: {bronze_count:,} | "
        f"Silver: {silver_count:,}"
    )


    # ========================================================
    # 5.2 METADATA
    # ========================================================

    silver_columns = set(
        silver_df_validation.columns
    )


    missing_technical_columns = (

        required_technical_columns
        -
        silver_columns
    )


    metadata_ok = (
        len(missing_technical_columns) == 0
    )


    if not metadata_ok:

        silver_validation_errors.append(

            f"{table_name}: faltan columnas técnicas "
            f"{sorted(missing_technical_columns)}"
        )


    print(
        f"{'OK' if metadata_ok else 'ERROR':5} | "
        "Metadata técnica"
    )


    # ========================================================
    # 5.3 _rescued_data
    # ========================================================

    rescued_data_present = (
        "_rescued_data"
        in silver_columns
    )


    rescued_ok = (
        not rescued_data_present
    )


    if not rescued_ok:

        silver_validation_errors.append(

            f"{table_name}: "
            "_rescued_data sigue presente en Silver"
        )


    print(
        f"{'OK' if rescued_ok else 'ERROR':5} | "
        "_rescued_data eliminado"
    )


# ============================================================
# 6. RESULTADO GLOBAL
# ============================================================

print()
print("=" * 85)


if silver_validation_errors:

    print(
        "SILVER VALIDATION: FAILED"
    )

    print()

    for error in silver_validation_errors:

        print(
            f"- {error}"
        )


    raise RuntimeError(
        "La validación final de Silver "
        "ha detectado inconsistencias"
    )


else:

    print(
        "OK - Todas las tablas Silver "
        "han superado la validación final"
    )


print("=" * 85)
