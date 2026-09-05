# Databricks notebook source
# DBTITLE 1,01. VALIDACIÓN DE ARCHIVOS SALES EN LA LANDING ZONE
# ============================================================
# VALIDACIÓN DE ARCHIVOS SALES EN LA LANDING ZONE
# ============================================================

# Ruta raíz donde hemos subido todos los CSV diarios de ventas.
#
# La estructura esperada es:
#
# /Volumes/retail_analytics/0_landing/source_files/sales/
# ├── year=2024/
# │   ├── month=01/
# │   ├── month=02/
# │   └── ...
# └── year=2025/
#     ├── month=01/
#     └── ...
sales_landing_path = (
    "/Volumes/retail_analytics/0_landing/source_files/sales"
)

# ------------------------------------------------------------
# dbutils.fs.ls() lista archivos y carpetas de una ruta.
#
# En este primer nivel esperamos encontrar únicamente:
# - year=2024
# - year=2025
# ------------------------------------------------------------
root_items = dbutils.fs.ls(sales_landing_path)

for item in root_items:
    print(item.path)

# COMMAND ----------

# DBTITLE 1,02. FUNCIÓN PARA CONTAR ARCHIVOS CSV DE FORMA RECURSIVA
# ============================================================
# FUNCIÓN PARA CONTAR ARCHIVOS CSV DE FORMA RECURSIVA
# ============================================================

def count_csv_files(path):
    """
    Recorre una ruta de Databricks de forma recursiva
    y devuelve el número total de archivos .csv encontrados.

    Parámetro
    ---------
    path : str
        Ruta inicial que queremos recorrer.

    Retorno
    -------
    int
        Número total de archivos CSV encontrados.
    """

    # Inicializamos el contador.
    total_csv_files = 0

    # Listamos los elementos contenidos en la ruta actual.
    items = dbutils.fs.ls(path)

    # Recorremos cada elemento encontrado.
    for item in items:

        # ----------------------------------------------------
        # Si la ruta termina en "/",
        # Databricks nos está indicando que es un directorio.
        # ----------------------------------------------------
        if item.path.endswith("/"):

            # Llamamos otra vez a la misma función
            # para recorrer esa subcarpeta.
            total_csv_files += count_csv_files(
                item.path
            )

        # ----------------------------------------------------
        # Si NO es una carpeta, comprobamos si es un CSV.
        # ----------------------------------------------------
        elif item.path.lower().endswith(".csv"):

            total_csv_files += 1

    return total_csv_files


# Ejecutamos la función sobre nuestra landing de ventas.
total_sales_files = count_csv_files(
    sales_landing_path
)

print(
    "Número total de CSV encontrados:",
    total_sales_files
)

# COMMAND ----------

# DBTITLE 1,03. INSPECCIÓN DE UN ARCHIVO CSV DE SALES
# ============================================================
# INSPECCIÓN DE UN ARCHIVO CSV DE SALES
# ============================================================

# Seleccionamos un archivo concreto de la landing.
# Usamos el primer día de 2024 simplemente porque sabemos
# que existe y nos sirve como muestra representativa.
sample_sales_file = (
    "/Volumes/retail_analytics/0_landing/source_files/sales/"
    "year=2024/month=01/sales_2024_01_01.csv"
)

# ------------------------------------------------------------
# Leemos el CSV con Spark.
#
# header = true
#   Indica que la primera fila contiene los nombres de columnas.
#
# inferSchema = true
#   Spark intentará detectar automáticamente los tipos de datos.
#
# IMPORTANTE:
# Esto lo usamos ahora únicamente para INSPECCIÓN.
# Más adelante, para Bronze, decidiremos si queremos controlar
# el schema explícitamente en lugar de depender de inferSchema.
# ------------------------------------------------------------
sample_sales_df = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(sample_sales_file)
)

# Mostramos algunas filas para inspección visual.
display(sample_sales_df.limit(10))

# COMMAND ----------

# DBTITLE 1,04. INSPECCIÓN DEL SCHEMA DETECTADO POR SPARK
# ============================================================
# INSPECCIÓN DEL SCHEMA DETECTADO POR SPARK
# ============================================================

# printSchema() nos muestra:
#
# - nombre de columna
# - tipo detectado
# - posibilidad de NULL
#
# Esto será importante para decidir cómo queremos definir
# la tabla Bronze posteriormente.
sample_sales_df.printSchema()

# COMMAND ----------

# DBTITLE 1,05. CONFIGURACIÓN DE RUTAS PARA AUTO LOADER
# ============================================================
# CONFIGURACIÓN DE RUTAS PARA AUTO LOADER
# ============================================================

# ------------------------------------------------------------
# RUTA DE ORIGEN
# ------------------------------------------------------------
#
# Esta es nuestra Landing Zone.
#
# Auto Loader vigilará esta ruta y descubrirá automáticamente
# los archivos CSV que ya existen y los nuevos archivos que
# añadamos posteriormente.
#
# Ahora mismo contiene:
#
# sales/
# ├── year=2024/
# └── year=2025/
#
sales_source_path = (
    "/Volumes/retail_analytics/0_landing/"
    "source_files/sales"
)


# ------------------------------------------------------------
# SCHEMA LOCATION
# ------------------------------------------------------------
#
# Auto Loader necesita almacenar información sobre el schema
# que ha detectado.
#
# Esto permite que Auto Loader:
#
# - recuerde el schema entre ejecuciones
# - detecte cambios futuros
# - gestione schema evolution
#
# No es una tabla.
# Es metadata técnica utilizada internamente por Auto Loader.
#
sales_schema_path = (
    "/Volumes/retail_analytics/0_landing/"
    "autoloader_metadata/schemas/sales"
)


# ------------------------------------------------------------
# CHECKPOINT LOCATION
# ------------------------------------------------------------
#
# Structured Streaming necesita un checkpoint para recordar
# qué archivos ya ha procesado.
#
# Este punto es FUNDAMENTAL.
#
# Gracias al checkpoint, si mañana añadimos:
#
# sales_2026_01_01.csv
#
# Auto Loader no volverá a procesar los 731 archivos históricos.
#
# Procesará únicamente el archivo nuevo.
#
sales_checkpoint_path = (
    "/Volumes/retail_analytics/0_landing/"
    "autoloader_metadata/checkpoints/sales"
)


# ------------------------------------------------------------
# TABLA DESTINO
# ------------------------------------------------------------
#
# Esta será nuestra tabla Delta Bronze gestionada
# mediante Unity Catalog.
#
# Namespace:
#
# catalog.schema.table
#
bronze_sales_table = (
    "retail_analytics.1_bronze.sales"
)


# Mostramos las rutas para comprobar visualmente
# que todo apunta donde esperamos.
print("SOURCE:")
print(sales_source_path)

print("\nSCHEMA LOCATION:")
print(sales_schema_path)

print("\nCHECKPOINT LOCATION:")
print(sales_checkpoint_path)

print("\nTARGET TABLE:")
print(bronze_sales_table)

# COMMAND ----------

# DBTITLE 1,06. DEFINICIÓN DEL STREAM DE AUTO LOADER PARA SALES
# ============================================================
# DEFINICIÓN DEL STREAM DE AUTO LOADER PARA SALES
# ============================================================

# Importamos las funciones necesarias.
#
# current_timestamp():
#   Añade el momento exacto en que la fila es ingerida en Bronze.
#
# col():
#   Permite referenciar columnas de Spark, incluida la columna
#   especial "_metadata" que Databricks expone al leer archivos.
from pyspark.sql.functions import (
    current_timestamp,
    col
)


# ------------------------------------------------------------
# CREAMOS EL STREAM DE INGESTIÓN
# ------------------------------------------------------------
#
# spark.readStream:
#   Inicia una lectura incremental mediante Structured Streaming.
#
# format("cloudFiles"):
#   Activa Auto Loader.
#
# Auto Loader se encargará de descubrir qué archivos existen,
# cuáles son nuevos y cuáles ya han sido procesados.
bronze_sales_stream = (

    spark.readStream

    # --------------------------------------------------------
    # ACTIVAMOS AUTO LOADER
    # --------------------------------------------------------
    .format("cloudFiles")

    # --------------------------------------------------------
    # FORMATO DE LOS ARCHIVOS DE ORIGEN
    # --------------------------------------------------------
    #
    # Nuestros archivos son CSV.
    .option(
        "cloudFiles.format",
        "csv"
    )

    # --------------------------------------------------------
    # SCHEMA LOCATION
    # --------------------------------------------------------
    #
    # Auto Loader guardará aquí la metadata relacionada con
    # el schema detectado.
    #
    # Esto permite:
    # - recordar el schema entre ejecuciones
    # - detectar cambios futuros
    # - gestionar schema evolution
    .option(
        "cloudFiles.schemaLocation",
        sales_schema_path
    )

    # --------------------------------------------------------
    # HEADER DEL CSV
    # --------------------------------------------------------
    #
    # Nuestros archivos contienen cabecera:
    #
    # sale_id,line_id,sale_timestamp,...
    .option(
        "header",
        "true"
    )

    # --------------------------------------------------------
    # INFERENCIA DE TIPOS
    # --------------------------------------------------------
    #
    # Por defecto, Auto Loader suele tratar las columnas CSV
    # como STRING.
    #
    # Con esta opción le permitimos inferir tipos como:
    #
    # integer
    # double
    # timestamp
    #
    # Esto hará que Bronze conserve tipos básicos coherentes
    # con los datos fuente.
    .option(
        "cloudFiles.inferColumnTypes",
        "true"
    )

    # --------------------------------------------------------
    # CARGAMOS LA LANDING ZONE
    # --------------------------------------------------------
    #
    # La ruta contiene:
    #
    # sales/
    # ├── year=2024/
    # │   ├── month=01/
    # │   └── ...
    # └── year=2025/
    #
    # Auto Loader recorrerá las subcarpetas automáticamente.
    .load(
        sales_source_path
    )

    # --------------------------------------------------------
    # TIMESTAMP DE INGESTIÓN
    # --------------------------------------------------------
    #
    # Añadimos el momento en que la fila entra en Bronze.
    #
    # Diferencia importante:
    #
    # sale_timestamp
    #   -> cuándo ocurrió la venta
    #
    # _ingestion_timestamp
    #   -> cuándo Databricks procesó ese registro
    .withColumn(
        "_ingestion_timestamp",
        current_timestamp()
    )

    # --------------------------------------------------------
    # ARCHIVO DE ORIGEN
    # --------------------------------------------------------
    #
    # Databricks expone una columna especial llamada:
    #
    # _metadata
    #
    # Dentro de ella encontramos información técnica del archivo.
    #
    # _metadata.file_path:
    #   ruta completa del CSV del que procede la fila.
    #
    # Esto sustituye a input_file_name(), que no está soportado
    # en este contexto con Unity Catalog.
    .withColumn(
        "_source_file",
        col("_metadata.file_path")
    )

    # --------------------------------------------------------
    # FECHA DE MODIFICACIÓN DEL ARCHIVO
    # --------------------------------------------------------
    #
    # Guardamos también la fecha/hora de modificación del CSV
    # original.
    #
    # Esto es útil para:
    # - auditoría
    # - trazabilidad
    # - debugging
    # - detectar archivos reemplazados o modificados
    .withColumn(
        "_source_file_modification_time",
        col("_metadata.file_modification_time")
    )
)

# COMMAND ----------

# DBTITLE 1,07. ESCRITURA DEL STREAM EN LA TABLA BRONZE
# ============================================================
# ESCRITURA DEL STREAM EN LA TABLA BRONZE
# ============================================================

bronze_sales_query = (

    bronze_sales_stream
    .writeStream

    # --------------------------------------------------------
    # CHECKPOINT LOCATION
    # --------------------------------------------------------
    #
    # Structured Streaming almacenará aquí su estado.
    #
    # Gracias al checkpoint, Databricks podrá recordar:
    #
    # - qué archivos ya procesó
    # - hasta dónde llegó la ingesta
    #
    # Esto es lo que permitirá que, cuando añadamos archivos
    # de 2026, no vuelva a cargar todo 2024 y 2025.
    .option(
        "checkpointLocation",
        sales_checkpoint_path
    )

    # --------------------------------------------------------
    # TRIGGER AVAILABLE NOW
    # --------------------------------------------------------
    #
    # availableNow=True significa:
    #
    # "procesa todo lo que haya disponible ahora y termina".
    #
    # Para nuestra carga histórica es ideal porque:
    #
    # - procesa los 731 CSV
    # - termina al acabar
    # - conserva el checkpoint
    #
    # Más adelante volveremos a ejecutar el mismo código
    # cuando lleguen nuevos archivos.
    .trigger(
        availableNow=True
    )

    # --------------------------------------------------------
    # TABLA DELTA DESTINO
    # --------------------------------------------------------
    #
    # Los datos se escribirán como tabla Delta gestionada
    # por Unity Catalog.
    #
    # Destino:
    #
    # retail_analytics.1_bronze.sales
    .toTable(
        bronze_sales_table
    )
)

# COMMAND ----------

# DBTITLE 1,08. VALIDACIÓN DE LA CARGA BRONZE
# ============================================================
# VALIDACIÓN DE LA CARGA BRONZE
# ============================================================

# Ejecutamos un COUNT sobre la tabla Bronze para comprobar
# que el número de filas coincide con el origen.
bronze_count = spark.sql("""
    SELECT COUNT(*)
    FROM retail_analytics.`1_bronze`.sales
""").collect()[0][0]

print(
    "Filas cargadas en Bronze:",
    bronze_count
)

# COMMAND ----------

# DBTITLE 1,09. DEFINICIÓN DE AUTO LOADER PARA CUSTOMERS
# ============================================================
# 9. DEFINICIÓN DE AUTO LOADER PARA CUSTOMERS
# ============================================================

from pyspark.sql.functions import (
    current_timestamp,
    col
)

# ------------------------------------------------------------
# RUTA DE ORIGEN
# ------------------------------------------------------------
#
# Aquí está el archivo original:
#
# customers/
# └── customers.csv
#
customers_source_path = (
    "/Volumes/retail_analytics/0_landing/"
    "source_files/customers"
)


# ------------------------------------------------------------
# SCHEMA LOCATION
# ------------------------------------------------------------
#
# Auto Loader guardará aquí la información del schema
# detectado para customers.
#
# Cada fuente debe tener SU PROPIO schemaLocation.
customers_schema_path = (
    "/Volumes/retail_analytics/0_landing/"
    "autoloader_metadata/schemas/customers"
)


# ------------------------------------------------------------
# CHECKPOINT LOCATION
# ------------------------------------------------------------
#
# Cada streaming query necesita su propio checkpoint.
#
# Esto permitirá que, si en el futuro añadimos nuevos archivos
# de customers, Auto Loader procese solamente los nuevos.
customers_checkpoint_path = (
    "/Volumes/retail_analytics/0_landing/"
    "autoloader_metadata/checkpoints/customers"
)


# ------------------------------------------------------------
# TABLA BRONZE DESTINO
# ------------------------------------------------------------
#
# Usamos backticks porque nuestro schema comienza por número.
customers_bronze_table = (
    "`retail_analytics`.`1_bronze`.`customers`"
)


# ------------------------------------------------------------
# CREACIÓN DEL STREAM
# ------------------------------------------------------------

bronze_customers_stream = (

    spark.readStream

    # Activamos Auto Loader.
    .format("cloudFiles")

    # Indicamos que el origen contiene CSV.
    .option(
        "cloudFiles.format",
        "csv"
    )

    # Ruta donde Auto Loader gestionará el schema.
    .option(
        "cloudFiles.schemaLocation",
        customers_schema_path
    )

    # El CSV tiene nombres de columnas en la primera fila.
    .option(
        "header",
        "true"
    )

    # Permitimos que Auto Loader detecte tipos como
    # DATE, INTEGER, STRING, etc.
    .option(
        "cloudFiles.inferColumnTypes",
        "true"
    )

    # Cargamos la carpeta de customers.
    .load(
        customers_source_path
    )

    # --------------------------------------------------------
    # METADATA DE INGESTIÓN
    # --------------------------------------------------------

    # Momento en el que Databricks ingiere el registro.
    .withColumn(
        "_ingestion_timestamp",
        current_timestamp()
    )

    # Archivo del que procede cada registro.
    .withColumn(
        "_source_file",
        col("_metadata.file_path")
    )

    # Fecha de modificación del archivo de origen.
    .withColumn(
        "_source_file_modification_time",
        col("_metadata.file_modification_time")
    )
)

# COMMAND ----------

# DBTITLE 1,10. ESCRITURA DE CUSTOMERS EN BRONZE
# ============================================================
# 10. ESCRITURA DE CUSTOMERS EN BRONZE
# ============================================================

bronze_customers_query = (

    bronze_customers_stream
    .writeStream

    # --------------------------------------------------------
    # CHECKPOINT
    # --------------------------------------------------------
    #
    # Aquí Structured Streaming guardará qué archivos
    # de customers ya han sido procesados.
    .option(
        "checkpointLocation",
        customers_checkpoint_path
    )

    # --------------------------------------------------------
    # AVAILABLE NOW
    # --------------------------------------------------------
    #
    # Procesa todos los archivos disponibles actualmente
    # y termina al finalizar.
    .trigger(
        availableNow=True
    )

    # --------------------------------------------------------
    # TABLA DELTA DESTINO
    # --------------------------------------------------------
    .toTable(
        customers_bronze_table
    )
)

# COMMAND ----------

# DBTITLE 1,11. VALIDACIÓN DE LA CARGA CUSTOMERS BRONZE
# ============================================================
# 11. VALIDACIÓN DE LA CARGA CUSTOMERS BRONZE
# ============================================================

# Contamos los registros almacenados en la tabla Delta Bronze.
customers_bronze_count = spark.sql("""
    SELECT COUNT(*)
    FROM retail_analytics.`1_bronze`.customers
""").collect()[0][0]


print(
    "Filas cargadas en Customers Bronze:",
    customers_bronze_count
)

# COMMAND ----------

# DBTITLE 1,12. DEFINICIÓN DE AUTO LOADER PARA PRODUCTS
# ============================================================
# 12. DEFINICIÓN DE AUTO LOADER PARA PRODUCTS
# ============================================================

# ------------------------------------------------------------
# RUTA DE ORIGEN
# ------------------------------------------------------------
#
# Aquí está nuestro archivo:
#
# products/
# └── products.csv
#
products_source_path = (
    "/Volumes/retail_analytics/0_landing/"
    "source_files/products"
)


# ------------------------------------------------------------
# SCHEMA LOCATION
# ------------------------------------------------------------
#
# Auto Loader necesita una ubicación independiente
# para gestionar el schema de products.
#
products_schema_path = (
    "/Volumes/retail_analytics/0_landing/"
    "autoloader_metadata/schemas/products"
)


# ------------------------------------------------------------
# CHECKPOINT LOCATION
# ------------------------------------------------------------
#
# Cada stream debe tener su propio checkpoint.
#
# Esto permitirá que futuras cargas de products
# sean incrementales.
#
products_checkpoint_path = (
    "/Volumes/retail_analytics/0_landing/"
    "autoloader_metadata/checkpoints/products"
)


# ------------------------------------------------------------
# TABLA DESTINO
# ------------------------------------------------------------
#
# Tabla Delta Bronze gestionada por Unity Catalog.
#
products_bronze_table = (
    "`retail_analytics`.`1_bronze`.`products`"
)


# ------------------------------------------------------------
# CREACIÓN DEL STREAM
# ------------------------------------------------------------

bronze_products_stream = (

    spark.readStream

    # Activamos Auto Loader.
    .format("cloudFiles")

    # Los archivos de origen son CSV.
    .option(
        "cloudFiles.format",
        "csv"
    )

    # Ubicación donde Auto Loader almacenará
    # información sobre el schema.
    .option(
        "cloudFiles.schemaLocation",
        products_schema_path
    )

    # Primera fila del CSV = nombres de columnas.
    .option(
        "header",
        "true"
    )

    # Permitimos inferencia de tipos.
    .option(
        "cloudFiles.inferColumnTypes",
        "true"
    )

    # Leemos la carpeta products.
    .load(
        products_source_path
    )

    # --------------------------------------------------------
    # METADATA TÉCNICA
    # --------------------------------------------------------

    # Momento de ingestión del registro.
    .withColumn(
        "_ingestion_timestamp",
        current_timestamp()
    )

    # Ruta del archivo fuente.
    .withColumn(
        "_source_file",
        col("_metadata.file_path")
    )

    # Fecha de modificación del archivo fuente.
    .withColumn(
        "_source_file_modification_time",
        col("_metadata.file_modification_time")
    )
)

# COMMAND ----------

# DBTITLE 1,13. ESCRITURA DE PRODUCTS EN BRONZE
# ============================================================
# 13. ESCRITURA DE PRODUCTS EN BRONZE
# ============================================================

bronze_products_query = (

    bronze_products_stream
    .writeStream

    # Guardamos el estado del stream.
    .option(
        "checkpointLocation",
        products_checkpoint_path
    )

    # Procesa todo lo disponible y termina.
    .trigger(
        availableNow=True
    )

    # Escribe como tabla Delta en Unity Catalog.
    .toTable(
        products_bronze_table
    )
)

# COMMAND ----------

# DBTITLE 1,14. VALIDACIÓN DE LA CARGA PRODUCTS BRONZE
# ============================================================
# 14. VALIDACIÓN DE LA CARGA PRODUCTS BRONZE
# ============================================================

products_bronze_count = spark.sql("""
    SELECT COUNT(*)
    FROM retail_analytics.`1_bronze`.products
""").collect()[0][0]

print(
    "Filas cargadas en Products Bronze:",
    products_bronze_count
)

# COMMAND ----------

# DBTITLE 1,15. DEFINICIÓN DE AUTO LOADER PARA STORES
# ============================================================
# 15. DEFINICIÓN DE AUTO LOADER PARA STORES
# ============================================================

# ------------------------------------------------------------
# RUTA DE ORIGEN
# ------------------------------------------------------------
#
# Contiene el archivo stores.csv que subimos previamente
# al Volume de Landing.
#
stores_source_path = (
    "/Volumes/retail_analytics/0_landing/"
    "source_files/stores"
)


# ------------------------------------------------------------
# SCHEMA LOCATION
# ------------------------------------------------------------
#
# Auto Loader almacenará aquí la información necesaria
# para gestionar el schema detectado de stores.
#
# Mantenemos una ubicación independiente para cada fuente.
#
stores_schema_path = (
    "/Volumes/retail_analytics/0_landing/"
    "autoloader_metadata/schemas/stores"
)


# ------------------------------------------------------------
# CHECKPOINT LOCATION
# ------------------------------------------------------------
#
# Structured Streaming utilizará este directorio para
# mantener el estado de la ingesta.
#
# Gracias al checkpoint, en futuras ejecuciones Auto Loader
# sabrá qué archivos ya ha procesado.
#
stores_checkpoint_path = (
    "/Volumes/retail_analytics/0_landing/"
    "autoloader_metadata/checkpoints/stores"
)


# ------------------------------------------------------------
# TABLA DELTA DESTINO
# ------------------------------------------------------------
#
# Nuestra tabla Bronze será:
#
# retail_analytics
#     └── 1_bronze
#           └── stores
#
stores_bronze_table = (
    "`retail_analytics`.`1_bronze`.`stores`"
)


# ------------------------------------------------------------
# CREACIÓN DEL STREAM CON AUTO LOADER
# ------------------------------------------------------------

bronze_stores_stream = (

    spark.readStream

    # cloudFiles activa Databricks Auto Loader.
    .format("cloudFiles")

    # Indicamos el formato de los archivos fuente.
    .option(
        "cloudFiles.format",
        "csv"
    )

    # Directorio utilizado por Auto Loader para
    # almacenar y gestionar el schema.
    .option(
        "cloudFiles.schemaLocation",
        stores_schema_path
    )

    # El CSV contiene una cabecera.
    .option(
        "header",
        "true"
    )

    # Permitimos que Auto Loader infiera los tipos
    # de las columnas en lugar de leer todo como STRING.
    .option(
        "cloudFiles.inferColumnTypes",
        "true"
    )

    # Carpeta que Auto Loader debe vigilar.
    .load(
        stores_source_path
    )

    # --------------------------------------------------------
    # METADATA TÉCNICA DE INGESTIÓN
    # --------------------------------------------------------

    # Timestamp en el que el registro llega a Bronze.
    .withColumn(
        "_ingestion_timestamp",
        current_timestamp()
    )

    # Archivo concreto del que procede el registro.
    .withColumn(
        "_source_file",
        col("_metadata.file_path")
    )

    # Fecha de modificación del archivo original.
    .withColumn(
        "_source_file_modification_time",
        col("_metadata.file_modification_time")
    )
)

# COMMAND ----------

# DBTITLE 1,16. ESCRITURA DE STORES EN BRONZE
# ============================================================
# 16. ESCRITURA DE STORES EN BRONZE
# ============================================================

bronze_stores_query = (

    bronze_stores_stream
    .writeStream

    # --------------------------------------------------------
    # CHECKPOINT
    # --------------------------------------------------------
    #
    # Guarda el progreso de esta ingesta concreta.
    #
    .option(
        "checkpointLocation",
        stores_checkpoint_path
    )

    # --------------------------------------------------------
    # AVAILABLE NOW
    # --------------------------------------------------------
    #
    # Procesa todos los archivos disponibles actualmente.
    #
    # Cuando termina de procesarlos, el stream se detiene.
    #
    .trigger(
        availableNow=True
    )

    # --------------------------------------------------------
    # TABLA DELTA DESTINO
    # --------------------------------------------------------
    #
    # El resultado queda registrado como tabla gestionada
    # dentro de Unity Catalog.
    #
    .toTable(
        stores_bronze_table
    )
)

# COMMAND ----------

# DBTITLE 1,17. VALIDACIÓN DE LA CARGA STORES BRONZE
# ============================================================
# 17. VALIDACIÓN DE LA CARGA STORES BRONZE
# ============================================================

# Contamos las filas que finalmente existen en la tabla
# Delta para comprobar que no hemos perdido registros
# durante la ingestión.

stores_bronze_count = spark.sql("""
    SELECT COUNT(*)
    FROM retail_analytics.`1_bronze`.stores
""").collect()[0][0]


print(
    "Filas cargadas en Stores Bronze:",
    stores_bronze_count
)

# COMMAND ----------

# DBTITLE 1,18. VALIDACIÓN FINAL DE LA CAPA BRONZE
# ============================================================
# 18. VALIDACIÓN FINAL DE LA CAPA BRONZE
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
#
# Validar que todas las tablas Bronze necesarias:
#
# - existen
# - contienen registros
#
# IMPORTANTE
# ------------------------------------------------------------
#
# No validamos SALES contra un número fijo de registros porque
# la tabla crece de forma incremental cada vez que Auto Loader
# detecta nuevos archivos.
#
# De esta forma esta validación seguirá siendo válida para:
#
# 2024
# 2025
# 2026
# y futuras cargas.
#
# ============================================================


# ============================================================
# 1. TABLAS BRONZE ESPERADAS
# ============================================================

bronze_tables = [
    "sales",
    "customers",
    "products",
    "stores"
]


# ============================================================
# 2. CONTROL GLOBAL
# ============================================================

bronze_validation_errors = []


print()
print("=" * 75)
print("VALIDACIÓN FINAL DE LA CAPA BRONZE")
print("=" * 75)


# ============================================================
# 3. VALIDAMOS CADA TABLA
# ============================================================

for table_name in bronze_tables:

    full_table_name = (
        f"retail_analytics.`1_bronze`.{table_name}"
    )


    # --------------------------------------------------------
    # 3.1 VALIDAMOS EXISTENCIA
    # --------------------------------------------------------

    table_exists = spark.catalog.tableExists(
        full_table_name
    )


    if not table_exists:

        bronze_validation_errors.append(
            f"No existe la tabla {full_table_name}"
        )

        print(
            f"ERROR | {table_name:10} | Tabla no encontrada"
        )

        continue


    # --------------------------------------------------------
    # 3.2 CONTAMOS REGISTROS
    # --------------------------------------------------------

    actual_count = (
        spark.table(full_table_name)
        .count()
    )


    # --------------------------------------------------------
    # 3.3 VALIDAMOS QUE NO ESTÉ VACÍA
    # --------------------------------------------------------

    if actual_count == 0:

        bronze_validation_errors.append(
            f"La tabla {full_table_name} está vacía"
        )

        status = "ERROR"

    else:

        status = "OK"


    # --------------------------------------------------------
    # 3.4 RESULTADO
    # --------------------------------------------------------

    print(
        f"{status:5} | "
        f"{table_name:10} | "
        f"Registros: {actual_count:,}"
    )


# ============================================================
# 4. RESULTADO GLOBAL
# ============================================================

print()
print("-" * 75)


if bronze_validation_errors:

    print("BRONZE VALIDATION: FAILED")

    print()

    for error in bronze_validation_errors:

        print(
            f"- {error}"
        )


    raise RuntimeError(
        "La validación final de Bronze ha detectado errores"
    )


else:

    print(
        "OK - Todas las tablas Bronze existen "
        "y contienen registros"
    )

# COMMAND ----------

# DBTITLE 1,19. VALIDACIÓN DE COLUMNAS TÉCNICAS EN BRONZE
# ============================================================
# 19. VALIDACIÓN DE COLUMNAS TÉCNICAS EN BRONZE
# ============================================================

# Estas son las columnas técnicas que esperamos encontrar
# en TODAS las tablas Bronze.
#
# Sirven para trazabilidad de ingestión:
#
# _ingestion_timestamp
#   -> cuándo Databricks procesó el registro
#
# _source_file
#   -> archivo del que procede el registro
#
# _source_file_modification_time
#   -> última modificación del archivo de origen
#
required_technical_columns = {
    "_ingestion_timestamp",
    "_source_file",
    "_source_file_modification_time"
}


# ------------------------------------------------------------
# TABLAS QUE QUEREMOS VALIDAR
# ------------------------------------------------------------

bronze_tables = [
    "sales",
    "customers",
    "products",
    "stores"
]


# ------------------------------------------------------------
# RECORREMOS CADA TABLA
# ------------------------------------------------------------

for table_name in bronze_tables:

    # Construimos el nombre completo de la tabla.
    full_table_name = (
        f"retail_analytics.`1_bronze`.{table_name}"
    )

    # --------------------------------------------------------
    # OBTENEMOS LAS COLUMNAS REALES
    # --------------------------------------------------------
    #
    # spark.table(...) devuelve un DataFrame.
    #
    # .columns devuelve una lista con los nombres
    # de todas sus columnas.
    #
    actual_columns = set(
        spark.table(full_table_name).columns
    )

    # --------------------------------------------------------
    # CALCULAMOS QUÉ COLUMNAS FALTAN
    # --------------------------------------------------------
    #
    # Si required - actual devuelve un conjunto vacío,
    # significa que están todas.
    #
    missing_columns = (
        required_technical_columns
        -
        actual_columns
    )

    # --------------------------------------------------------
    # MOSTRAMOS RESULTADO
    # --------------------------------------------------------

    if not missing_columns:

        print(
            f"OK    | {table_name:10} | "
            "Columnas técnicas correctas"
        )

    else:

        print(
            f"ERROR | {table_name:10} | "
            f"Faltan: {missing_columns}"
        )