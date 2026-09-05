# Databricks notebook source
# DBTITLE 1,00. CONFIGURACIÓN Y VALIDACIÓN DE LA CAPA GOLD
# MAGIC %sql
# MAGIC -- ============================================================
# MAGIC -- 00. CONFIGURACIÓN Y VALIDACIÓN DE LA CAPA GOLD
# MAGIC -- ============================================================
# MAGIC --
# MAGIC -- OBJETIVO
# MAGIC -- ------------------------------------------------------------
# MAGIC --
# MAGIC -- Este notebook construirá la capa de consumo analítico
# MAGIC -- utilizando SQL sobre el modelo Gold.
# MAGIC --
# MAGIC -- Partimos de Star Schema:
# MAGIC --
# MAGIC --                     dim_customer
# MAGIC --                          |
# MAGIC --                          |
# MAGIC -- dim_date ----------- fact_sales ----------- dim_product
# MAGIC --                          |
# MAGIC --                          |
# MAGIC --                      dim_store
# MAGIC --
# MAGIC -- A partir de estas tablas construiremos SQL Views orientadas
# MAGIC -- a casos de negocio concretos.
# MAGIC --
# MAGIC -- Estas Views servirán posteriormente como fuente para:
# MAGIC --
# MAGIC -- - análisis SQL
# MAGIC -- - KPIs
# MAGIC -- - dashboards
# MAGIC -- - exploración de negocio
# MAGIC --
# MAGIC -- ============================================================
# MAGIC
# MAGIC
# MAGIC -- ------------------------------------------------------------
# MAGIC -- SELECCIONAMOS EL CATÁLOGO Y SCHEMA GOLD
# MAGIC -- ------------------------------------------------------------
# MAGIC
# MAGIC USE CATALOG retail_analytics;
# MAGIC
# MAGIC USE SCHEMA `3_gold`;
# MAGIC
# MAGIC
# MAGIC -- ------------------------------------------------------------
# MAGIC -- COMPROBAMOS LAS TABLAS DISPONIBLES
# MAGIC -- ------------------------------------------------------------
# MAGIC
# MAGIC SHOW TABLES;

# COMMAND ----------

# DBTITLE 1,00.01 VALIDACIÓN DE RECUENTOS GOLD
# MAGIC %sql
# MAGIC -- ============================================================
# MAGIC -- 00.01 VALIDACIÓN DE RECUENTOS GOLD
# MAGIC -- ============================================================
# MAGIC
# MAGIC SELECT
# MAGIC     'fact_sales' AS table_name,
# MAGIC     COUNT(*) AS row_count
# MAGIC FROM fact_sales
# MAGIC
# MAGIC UNION ALL
# MAGIC
# MAGIC SELECT
# MAGIC     'dim_customer',
# MAGIC     COUNT(*)
# MAGIC FROM dim_customer
# MAGIC
# MAGIC UNION ALL
# MAGIC
# MAGIC SELECT
# MAGIC     'dim_product',
# MAGIC     COUNT(*)
# MAGIC FROM dim_product
# MAGIC
# MAGIC UNION ALL
# MAGIC
# MAGIC SELECT
# MAGIC     'dim_store',
# MAGIC     COUNT(*)
# MAGIC FROM dim_store
# MAGIC
# MAGIC UNION ALL
# MAGIC
# MAGIC SELECT
# MAGIC     'dim_date',
# MAGIC     COUNT(*)
# MAGIC FROM dim_date;

# COMMAND ----------

# DBTITLE 1,01. VIEW DE VENTAS DIARIAS
# MAGIC %sql
# MAGIC
# MAGIC -- ============================================================
# MAGIC -- 01. VIEW DE VENTAS DIARIAS
# MAGIC -- ============================================================
# MAGIC --
# MAGIC -- OBJETIVO
# MAGIC -- ------------------------------------------------------------
# MAGIC --
# MAGIC -- Crear una vista agregada por fecha para analizar:
# MAGIC --
# MAGIC -- - ventas netas
# MAGIC -- - ventas brutas
# MAGIC -- - descuentos
# MAGIC -- - unidades
# MAGIC -- - número de tickets
# MAGIC -- - número de líneas
# MAGIC --
# MAGIC -- IMPORTANTE
# MAGIC -- ------------------------------------------------------------
# MAGIC --
# MAGIC -- Utilizamos DIM_DATE como tabla principal y hacemos:
# MAGIC --
# MAGIC --      DIM_DATE
# MAGIC --          LEFT JOIN
# MAGIC --      FACT_SALES
# MAGIC --
# MAGIC -- De esta forma todos los días del calendario aparecen en
# MAGIC -- la vista, incluso cuando no existan ventas.
# MAGIC --
# MAGIC -- Esto es importante para:
# MAGIC --
# MAGIC -- - análisis temporal
# MAGIC -- - detección de días sin actividad
# MAGIC -- - forecasting
# MAGIC -- - Machine Learning
# MAGIC --
# MAGIC -- Además, utilizamos nombres totalmente cualificados:
# MAGIC --
# MAGIC --      catalog.schema.table
# MAGIC --
# MAGIC -- para que la celda pueda ejecutarse de forma independiente
# MAGIC -- sin depender de USE CATALOG / USE SCHEMA.
# MAGIC --
# MAGIC -- ============================================================
# MAGIC
# MAGIC
# MAGIC CREATE OR REPLACE VIEW
# MAGIC     retail_analytics.`3_gold`.vw_sales_daily
# MAGIC AS
# MAGIC
# MAGIC
# MAGIC SELECT
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- DIMENSIÓN TEMPORAL
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     d.date_key,
# MAGIC
# MAGIC     d.date,
# MAGIC
# MAGIC     d.year,
# MAGIC
# MAGIC     d.quarter,
# MAGIC
# MAGIC     d.month,
# MAGIC
# MAGIC     d.month_name,
# MAGIC
# MAGIC     d.year_month,
# MAGIC
# MAGIC     d.week_of_year,
# MAGIC
# MAGIC     d.day_name,
# MAGIC
# MAGIC     d.is_weekend,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- NÚMERO DE TICKETS
# MAGIC     -- ========================================================
# MAGIC     --
# MAGIC     -- COUNT DISTINCT ignora los NULL producidos por el
# MAGIC     -- LEFT JOIN en los días sin ventas.
# MAGIC     --
# MAGIC     -- Por tanto:
# MAGIC     --
# MAGIC     -- día con ventas    -> N tickets
# MAGIC     -- día sin ventas    -> 0 tickets
# MAGIC     --
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     COUNT(
# MAGIC         DISTINCT f.sale_id
# MAGIC     ) AS total_tickets,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- NÚMERO DE LÍNEAS DE VENTA
# MAGIC     -- ========================================================
# MAGIC     --
# MAGIC     -- NO utilizamos COUNT(*).
# MAGIC     --
# MAGIC     -- Con un LEFT JOIN, COUNT(*) devolvería 1 para un día
# MAGIC     -- sin ventas porque DIM_DATE aporta igualmente una fila.
# MAGIC     --
# MAGIC     -- COUNT(f.line_id) solo cuenta líneas reales de venta.
# MAGIC     --
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     COUNT(
# MAGIC         f.line_id
# MAGIC     ) AS total_lines,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- UNIDADES VENDIDAS
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     COALESCE(
# MAGIC
# MAGIC         SUM(
# MAGIC             f.quantity
# MAGIC         ),
# MAGIC
# MAGIC         0
# MAGIC
# MAGIC     ) AS total_units,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- VENTAS BRUTAS
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     ROUND(
# MAGIC
# MAGIC         COALESCE(
# MAGIC
# MAGIC             SUM(
# MAGIC                 f.gross_amount
# MAGIC             ),
# MAGIC
# MAGIC             0
# MAGIC
# MAGIC         ),
# MAGIC
# MAGIC         2
# MAGIC
# MAGIC     ) AS gross_sales,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- DESCUENTOS
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     ROUND(
# MAGIC
# MAGIC         COALESCE(
# MAGIC
# MAGIC             SUM(
# MAGIC                 f.discount_amount
# MAGIC             ),
# MAGIC
# MAGIC             0
# MAGIC
# MAGIC         ),
# MAGIC
# MAGIC         2
# MAGIC
# MAGIC     ) AS total_discount,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- VENTAS NETAS
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     ROUND(
# MAGIC
# MAGIC         COALESCE(
# MAGIC
# MAGIC             SUM(
# MAGIC                 f.net_amount
# MAGIC             ),
# MAGIC
# MAGIC             0
# MAGIC
# MAGIC         ),
# MAGIC
# MAGIC         2
# MAGIC
# MAGIC     ) AS net_sales,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- TICKET MEDIO
# MAGIC     -- ========================================================
# MAGIC     --
# MAGIC     -- Fórmula:
# MAGIC     --
# MAGIC     --      ventas netas / número de tickets
# MAGIC     --
# MAGIC     -- NULLIF evita una división entre cero.
# MAGIC     --
# MAGIC     -- COALESCE convierte el resultado NULL en 0 para los
# MAGIC     -- días sin ventas.
# MAGIC     --
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     COALESCE(
# MAGIC
# MAGIC         ROUND(
# MAGIC
# MAGIC             SUM(
# MAGIC                 f.net_amount
# MAGIC             )
# MAGIC
# MAGIC             /
# MAGIC
# MAGIC             NULLIF(
# MAGIC
# MAGIC                 COUNT(
# MAGIC                     DISTINCT f.sale_id
# MAGIC                 ),
# MAGIC
# MAGIC                 0
# MAGIC
# MAGIC             ),
# MAGIC
# MAGIC             2
# MAGIC
# MAGIC         ),
# MAGIC
# MAGIC         0
# MAGIC
# MAGIC     ) AS average_ticket
# MAGIC
# MAGIC
# MAGIC -- ============================================================
# MAGIC -- DIM_DATE COMO BASE DEL CALENDARIO
# MAGIC -- ============================================================
# MAGIC
# MAGIC FROM
# MAGIC     retail_analytics.`3_gold`.dim_date d
# MAGIC
# MAGIC
# MAGIC -- ============================================================
# MAGIC -- LEFT JOIN CON FACT_SALES
# MAGIC -- ============================================================
# MAGIC
# MAGIC LEFT JOIN
# MAGIC     retail_analytics.`3_gold`.fact_sales f
# MAGIC
# MAGIC     ON d.date_key = f.date_key
# MAGIC
# MAGIC
# MAGIC -- ============================================================
# MAGIC -- GRANULARIDAD
# MAGIC -- ============================================================
# MAGIC --
# MAGIC -- Una fila por día.
# MAGIC --
# MAGIC -- ============================================================
# MAGIC
# MAGIC GROUP BY
# MAGIC
# MAGIC     d.date_key,
# MAGIC
# MAGIC     d.date,
# MAGIC
# MAGIC     d.year,
# MAGIC
# MAGIC     d.quarter,
# MAGIC
# MAGIC     d.month,
# MAGIC
# MAGIC     d.month_name,
# MAGIC
# MAGIC     d.year_month,
# MAGIC
# MAGIC     d.week_of_year,
# MAGIC
# MAGIC     d.day_name,
# MAGIC
# MAGIC     d.is_weekend;

# COMMAND ----------

# DBTITLE 1,01.01 VALIDACIÓN DE VW_SALES_DAILY
# MAGIC %sql
# MAGIC -- ============================================================
# MAGIC -- 01.01 VALIDACIÓN DE VW_SALES_DAILY
# MAGIC -- ============================================================
# MAGIC
# MAGIC SELECT *
# MAGIC FROM retail_analytics.`3_gold`.vw_sales_daily
# MAGIC ORDER BY date
# MAGIC LIMIT 20;

# COMMAND ----------

# DBTITLE 1,02. VIEW DE VENTAS MENSUALES
# MAGIC %sql
# MAGIC
# MAGIC -- ============================================================
# MAGIC -- 02. VIEW DE VENTAS MENSUALES
# MAGIC -- ============================================================
# MAGIC --
# MAGIC -- OBJETIVO
# MAGIC -- ------------------------------------------------------------
# MAGIC --
# MAGIC -- Crear una vista agregada por mes para analizar la evolución
# MAGIC -- temporal del negocio.
# MAGIC --
# MAGIC -- KPIs:
# MAGIC --
# MAGIC -- - ventas netas
# MAGIC -- - ventas brutas
# MAGIC -- - descuentos
# MAGIC -- - unidades vendidas
# MAGIC -- - número de tickets
# MAGIC -- - número de líneas
# MAGIC -- - ticket medio
# MAGIC -- - clientes identificados
# MAGIC --
# MAGIC -- IMPORTANTE
# MAGIC -- ------------------------------------------------------------
# MAGIC --
# MAGIC -- Esta vista se calcula directamente desde FACT_SALES.
# MAGIC --
# MAGIC -- No agregamos VW_SALES_DAILY porque métricas como:
# MAGIC --
# MAGIC --      COUNT(DISTINCT customer_id)
# MAGIC --
# MAGIC -- no son aditivas.
# MAGIC --
# MAGIC -- Un cliente que compre en varios días del mismo mes debe
# MAGIC -- contabilizarse una sola vez a nivel mensual.
# MAGIC --
# MAGIC -- Utilizamos nombres completamente cualificados para que
# MAGIC -- la ejecución no dependa de USE CATALOG / USE SCHEMA.
# MAGIC --
# MAGIC -- ============================================================
# MAGIC
# MAGIC
# MAGIC CREATE OR REPLACE VIEW
# MAGIC     retail_analytics.`3_gold`.vw_sales_monthly
# MAGIC AS
# MAGIC
# MAGIC
# MAGIC SELECT
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- DIMENSIÓN TEMPORAL
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     d.year,
# MAGIC
# MAGIC     d.month,
# MAGIC
# MAGIC     d.month_name,
# MAGIC
# MAGIC     d.year_month,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- TICKETS
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     COUNT(
# MAGIC         DISTINCT f.sale_id
# MAGIC     ) AS total_tickets,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- LÍNEAS DE VENTA
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     COUNT(
# MAGIC         f.line_id
# MAGIC     ) AS total_lines,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- UNIDADES VENDIDAS
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     SUM(
# MAGIC         f.quantity
# MAGIC     ) AS total_units,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- CLIENTES IDENTIFICADOS
# MAGIC     -- ========================================================
# MAGIC     --
# MAGIC     -- customer_id puede ser NULL para ventas anónimas.
# MAGIC     --
# MAGIC     -- COUNT DISTINCT ignora automáticamente esos NULL.
# MAGIC     --
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     COUNT(
# MAGIC         DISTINCT f.customer_id
# MAGIC     ) AS unique_customers,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- VENTAS BRUTAS
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     ROUND(
# MAGIC         SUM(
# MAGIC             f.gross_amount
# MAGIC         ),
# MAGIC         2
# MAGIC     ) AS gross_sales,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- DESCUENTOS
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     ROUND(
# MAGIC         SUM(
# MAGIC             f.discount_amount
# MAGIC         ),
# MAGIC         2
# MAGIC     ) AS total_discount,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- VENTAS NETAS
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     ROUND(
# MAGIC         SUM(
# MAGIC             f.net_amount
# MAGIC         ),
# MAGIC         2
# MAGIC     ) AS net_sales,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- TICKET MEDIO
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     ROUND(
# MAGIC
# MAGIC         SUM(
# MAGIC             f.net_amount
# MAGIC         )
# MAGIC
# MAGIC         /
# MAGIC
# MAGIC         NULLIF(
# MAGIC             COUNT(
# MAGIC                 DISTINCT f.sale_id
# MAGIC             ),
# MAGIC             0
# MAGIC         ),
# MAGIC
# MAGIC         2
# MAGIC
# MAGIC     ) AS average_ticket
# MAGIC
# MAGIC
# MAGIC -- ============================================================
# MAGIC -- FACT_SALES
# MAGIC -- ============================================================
# MAGIC
# MAGIC FROM
# MAGIC     retail_analytics.`3_gold`.fact_sales f
# MAGIC
# MAGIC
# MAGIC -- ============================================================
# MAGIC -- DIM_DATE
# MAGIC -- ============================================================
# MAGIC
# MAGIC INNER JOIN
# MAGIC     retail_analytics.`3_gold`.dim_date d
# MAGIC
# MAGIC     ON f.date_key = d.date_key
# MAGIC
# MAGIC
# MAGIC -- ============================================================
# MAGIC -- GRANULARIDAD
# MAGIC -- ============================================================
# MAGIC --
# MAGIC -- Una fila por año + mes.
# MAGIC --
# MAGIC -- ============================================================
# MAGIC
# MAGIC GROUP BY
# MAGIC
# MAGIC     d.year,
# MAGIC
# MAGIC     d.month,
# MAGIC
# MAGIC     d.month_name,
# MAGIC
# MAGIC     d.year_month;

# COMMAND ----------

# DBTITLE 1,02.01 VALIDACIÓN DE VW_SALES_MONTHLY
# MAGIC %sql
# MAGIC -- ============================================================
# MAGIC -- 02.01 VALIDACIÓN DE VW_SALES_MONTHLY
# MAGIC -- ============================================================
# MAGIC
# MAGIC SELECT *
# MAGIC FROM retail_analytics.`3_gold`.vw_sales_monthly
# MAGIC ORDER BY year, month;

# COMMAND ----------

# DBTITLE 1,02.02 EVOLUCIÓN Y CRECIMIENTO MENSUAL DE VENTAS
# MAGIC %sql
# MAGIC
# MAGIC -- ============================================================
# MAGIC -- 02.02 EVOLUCIÓN Y CRECIMIENTO MENSUAL DE VENTAS
# MAGIC -- ============================================================
# MAGIC --
# MAGIC -- OBJETIVO
# MAGIC -- ------------------------------------------------------------
# MAGIC --
# MAGIC -- Comparar las ventas netas de cada mes con el mes anterior.
# MAGIC --
# MAGIC -- Calcularemos:
# MAGIC --
# MAGIC -- - ventas del mes actual
# MAGIC -- - ventas del mes anterior
# MAGIC -- - diferencia absoluta
# MAGIC -- - crecimiento porcentual MoM
# MAGIC --
# MAGIC -- IMPORTANTE
# MAGIC -- ------------------------------------------------------------
# MAGIC --
# MAGIC -- Utilizamos una CTE para calcular una sola vez:
# MAGIC --
# MAGIC --      LAG(net_sales)
# MAGIC --
# MAGIC -- evitando repetir la misma Window Function varias veces.
# MAGIC --
# MAGIC -- ============================================================
# MAGIC
# MAGIC
# MAGIC WITH monthly_sales AS (
# MAGIC
# MAGIC     SELECT
# MAGIC
# MAGIC         year,
# MAGIC
# MAGIC         month,
# MAGIC
# MAGIC         month_name,
# MAGIC
# MAGIC         year_month,
# MAGIC
# MAGIC         net_sales,
# MAGIC
# MAGIC
# MAGIC         -- ====================================================
# MAGIC         -- VENTAS DEL MES ANTERIOR
# MAGIC         -- ====================================================
# MAGIC
# MAGIC         LAG(
# MAGIC             net_sales
# MAGIC         ) OVER (
# MAGIC
# MAGIC             ORDER BY
# MAGIC                 year,
# MAGIC                 month
# MAGIC
# MAGIC         ) AS previous_month_sales
# MAGIC
# MAGIC
# MAGIC     FROM
# MAGIC         retail_analytics.`3_gold`.vw_sales_monthly
# MAGIC
# MAGIC )
# MAGIC
# MAGIC
# MAGIC SELECT
# MAGIC
# MAGIC     year,
# MAGIC
# MAGIC     month,
# MAGIC
# MAGIC     month_name,
# MAGIC
# MAGIC     year_month,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- VENTAS DEL MES ACTUAL
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     net_sales,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- VENTAS DEL MES ANTERIOR
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     previous_month_sales,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- VARIACIÓN ABSOLUTA
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     ROUND(
# MAGIC
# MAGIC         net_sales
# MAGIC         -
# MAGIC         previous_month_sales,
# MAGIC
# MAGIC         2
# MAGIC
# MAGIC     ) AS sales_variation,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- CRECIMIENTO MENSUAL MoM (%)
# MAGIC     -- ========================================================
# MAGIC     --
# MAGIC     -- Fórmula:
# MAGIC     --
# MAGIC     -- (ventas actuales - ventas anteriores)
# MAGIC     -- -------------------------------------- × 100
# MAGIC     --            ventas anteriores
# MAGIC     --
# MAGIC     -- NULLIF evita división entre cero.
# MAGIC     --
# MAGIC     -- Para el primer mes:
# MAGIC     --
# MAGIC     -- previous_month_sales = NULL
# MAGIC     --
# MAGIC     -- y por tanto mom_growth_pct también será NULL.
# MAGIC     --
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     ROUND(
# MAGIC
# MAGIC         (
# MAGIC             net_sales
# MAGIC             -
# MAGIC             previous_month_sales
# MAGIC         )
# MAGIC
# MAGIC         /
# MAGIC
# MAGIC         NULLIF(
# MAGIC             previous_month_sales,
# MAGIC             0
# MAGIC         )
# MAGIC
# MAGIC         * 100,
# MAGIC
# MAGIC         2
# MAGIC
# MAGIC     ) AS mom_growth_pct
# MAGIC
# MAGIC
# MAGIC FROM
# MAGIC     monthly_sales
# MAGIC
# MAGIC
# MAGIC ORDER BY
# MAGIC
# MAGIC     year,
# MAGIC
# MAGIC     month;

# COMMAND ----------

# DBTITLE 1,03. VIEW DE RENDIMIENTO DE PRODUCTOS
# MAGIC %sql
# MAGIC
# MAGIC -- ============================================================
# MAGIC -- 03. VIEW DE RENDIMIENTO DE PRODUCTOS
# MAGIC -- ============================================================
# MAGIC --
# MAGIC -- OBJETIVO
# MAGIC -- ------------------------------------------------------------
# MAGIC --
# MAGIC -- Analizar el rendimiento comercial de cada producto.
# MAGIC --
# MAGIC -- Calcularemos:
# MAGIC --
# MAGIC -- - número de tickets
# MAGIC -- - unidades vendidas
# MAGIC -- - ventas brutas
# MAGIC -- - descuentos
# MAGIC -- - ventas netas
# MAGIC -- - ranking global por ventas netas
# MAGIC --
# MAGIC -- Esta vista servirá posteriormente para:
# MAGIC --
# MAGIC -- - Top productos
# MAGIC -- - análisis por categoría
# MAGIC -- - análisis por marca
# MAGIC -- - visuales del dashboard
# MAGIC --
# MAGIC -- IMPORTANTE
# MAGIC -- ------------------------------------------------------------
# MAGIC --
# MAGIC -- Utilizamos nombres completamente cualificados para evitar
# MAGIC -- depender de USE CATALOG / USE SCHEMA.
# MAGIC --
# MAGIC -- La granularidad de esta View será:
# MAGIC --
# MAGIC --      1 fila = 1 producto
# MAGIC --
# MAGIC -- ============================================================
# MAGIC
# MAGIC
# MAGIC CREATE OR REPLACE VIEW
# MAGIC     retail_analytics.`3_gold`.vw_product_performance
# MAGIC AS
# MAGIC
# MAGIC
# MAGIC SELECT
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- ATRIBUTOS DEL PRODUCTO
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     p.product_id,
# MAGIC
# MAGIC     p.product_name,
# MAGIC
# MAGIC     p.category,
# MAGIC
# MAGIC     p.subcategory,
# MAGIC
# MAGIC     p.brand,
# MAGIC
# MAGIC     p.unit_cost,
# MAGIC
# MAGIC     p.base_price,
# MAGIC
# MAGIC     p.active,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- NÚMERO DE TICKETS
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     COUNT(
# MAGIC         DISTINCT f.sale_id
# MAGIC     ) AS total_tickets,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- UNIDADES VENDIDAS
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     SUM(
# MAGIC         f.quantity
# MAGIC     ) AS total_units,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- VENTAS BRUTAS
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     ROUND(
# MAGIC         SUM(
# MAGIC             f.gross_amount
# MAGIC         ),
# MAGIC         2
# MAGIC     ) AS gross_sales,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- DESCUENTOS
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     ROUND(
# MAGIC         SUM(
# MAGIC             f.discount_amount
# MAGIC         ),
# MAGIC         2
# MAGIC     ) AS total_discount,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- VENTAS NETAS
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     ROUND(
# MAGIC         SUM(
# MAGIC             f.net_amount
# MAGIC         ),
# MAGIC         2
# MAGIC     ) AS net_sales,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- RANKING GLOBAL POR VENTAS NETAS
# MAGIC     -- ========================================================
# MAGIC     --
# MAGIC     -- RANK() ordena los productos según ventas netas.
# MAGIC     --
# MAGIC     -- Si dos productos tienen exactamente el mismo importe,
# MAGIC     -- compartirán la misma posición.
# MAGIC     --
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     RANK() OVER (
# MAGIC
# MAGIC         ORDER BY
# MAGIC             SUM(f.net_amount) DESC
# MAGIC
# MAGIC     ) AS sales_rank
# MAGIC
# MAGIC
# MAGIC -- ============================================================
# MAGIC -- FACT_SALES
# MAGIC -- ============================================================
# MAGIC
# MAGIC FROM
# MAGIC     retail_analytics.`3_gold`.fact_sales f
# MAGIC
# MAGIC
# MAGIC -- ============================================================
# MAGIC -- DIM_PRODUCT
# MAGIC -- ============================================================
# MAGIC
# MAGIC INNER JOIN
# MAGIC     retail_analytics.`3_gold`.dim_product p
# MAGIC
# MAGIC     ON f.product_id = p.product_id
# MAGIC
# MAGIC
# MAGIC -- ============================================================
# MAGIC -- GRANULARIDAD
# MAGIC -- ============================================================
# MAGIC --
# MAGIC -- Una fila por producto.
# MAGIC --
# MAGIC -- ============================================================
# MAGIC
# MAGIC GROUP BY
# MAGIC
# MAGIC     p.product_id,
# MAGIC
# MAGIC     p.product_name,
# MAGIC
# MAGIC     p.category,
# MAGIC
# MAGIC     p.subcategory,
# MAGIC
# MAGIC     p.brand,
# MAGIC
# MAGIC     p.unit_cost,
# MAGIC
# MAGIC     p.base_price,
# MAGIC
# MAGIC     p.active;

# COMMAND ----------

# DBTITLE 1,03.01 VALIDACIÓN DE VW_PRODUCT_PERFORMANCE
# MAGIC %sql
# MAGIC
# MAGIC -- ============================================================
# MAGIC -- 03.01 VALIDACIÓN DE VW_PRODUCT_PERFORMANCE
# MAGIC -- ============================================================
# MAGIC
# MAGIC SELECT *
# MAGIC FROM retail_analytics.`3_gold`.vw_product_performance
# MAGIC ORDER BY sales_rank
# MAGIC LIMIT 20;

# COMMAND ----------

# DBTITLE 1,03.02 RANKING DE PRODUCTOS DENTRO DE CADA CATEGORÍA
# MAGIC %sql
# MAGIC
# MAGIC -- ============================================================
# MAGIC -- 03.02 RANKING DE PRODUCTOS DENTRO DE CADA CATEGORÍA
# MAGIC -- ============================================================
# MAGIC --
# MAGIC -- OBJETIVO
# MAGIC -- ------------------------------------------------------------
# MAGIC --
# MAGIC -- Identificar qué productos tienen mejor rendimiento dentro
# MAGIC -- de su propia categoría.
# MAGIC --
# MAGIC -- A diferencia del ranking global de VW_PRODUCT_PERFORMANCE,
# MAGIC -- aquí utilizamos:
# MAGIC --
# MAGIC --      PARTITION BY category
# MAGIC --
# MAGIC -- para reiniciar el ranking en cada categoría.
# MAGIC --
# MAGIC -- Esto permite responder preguntas como:
# MAGIC --
# MAGIC -- - ¿Cuál es el producto nº1 de cada categoría?
# MAGIC -- - ¿Cuáles son los Top 5 productos de cada categoría?
# MAGIC -- - ¿Qué productos concentran más ventas dentro de su categoría?
# MAGIC --
# MAGIC -- ============================================================
# MAGIC
# MAGIC
# MAGIC SELECT
# MAGIC
# MAGIC     category,
# MAGIC
# MAGIC     product_id,
# MAGIC
# MAGIC     product_name,
# MAGIC
# MAGIC     brand,
# MAGIC
# MAGIC     total_units,
# MAGIC
# MAGIC     net_sales,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- RANKING DENTRO DE CADA CATEGORÍA
# MAGIC     -- ========================================================
# MAGIC     --
# MAGIC     -- El ranking empieza de nuevo para cada category.
# MAGIC     --
# MAGIC     -- Si dos productos tienen exactamente las mismas ventas,
# MAGIC     -- compartirán posición.
# MAGIC     --
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     RANK() OVER (
# MAGIC
# MAGIC         PARTITION BY
# MAGIC             category
# MAGIC
# MAGIC         ORDER BY
# MAGIC             net_sales DESC
# MAGIC
# MAGIC     ) AS category_sales_rank
# MAGIC
# MAGIC
# MAGIC FROM
# MAGIC     retail_analytics.`3_gold`.vw_product_performance
# MAGIC
# MAGIC
# MAGIC ORDER BY
# MAGIC
# MAGIC     category,
# MAGIC
# MAGIC     category_sales_rank,
# MAGIC
# MAGIC     product_id;

# COMMAND ----------

# DBTITLE 1,04. VIEW DE RENDIMIENTO DE TIENDAS
# MAGIC %sql
# MAGIC
# MAGIC -- ============================================================
# MAGIC -- 04. VIEW DE RENDIMIENTO DE TIENDAS
# MAGIC -- ============================================================
# MAGIC --
# MAGIC -- OBJETIVO
# MAGIC -- ------------------------------------------------------------
# MAGIC --
# MAGIC -- Analizar el rendimiento comercial de cada tienda.
# MAGIC --
# MAGIC -- Calcularemos:
# MAGIC --
# MAGIC -- - número de tickets
# MAGIC -- - unidades vendidas
# MAGIC -- - ventas brutas
# MAGIC -- - descuentos
# MAGIC -- - ventas netas
# MAGIC -- - ticket medio
# MAGIC -- - ranking global por ventas netas
# MAGIC --
# MAGIC -- La granularidad será:
# MAGIC --
# MAGIC --      1 fila = 1 tienda
# MAGIC --
# MAGIC -- ============================================================
# MAGIC
# MAGIC
# MAGIC CREATE OR REPLACE VIEW
# MAGIC     retail_analytics.`3_gold`.vw_store_performance
# MAGIC AS
# MAGIC
# MAGIC
# MAGIC SELECT
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- ATRIBUTOS DE LA TIENDA
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     s.store_id,
# MAGIC
# MAGIC     s.store_name,
# MAGIC
# MAGIC     s.city,
# MAGIC
# MAGIC     s.region,
# MAGIC
# MAGIC     s.store_type,
# MAGIC
# MAGIC     s.opening_date,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- NÚMERO DE TICKETS
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     COUNT(
# MAGIC         DISTINCT f.sale_id
# MAGIC     ) AS total_tickets,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- UNIDADES VENDIDAS
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     SUM(
# MAGIC         f.quantity
# MAGIC     ) AS total_units,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- VENTAS BRUTAS
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     ROUND(
# MAGIC         SUM(
# MAGIC             f.gross_amount
# MAGIC         ),
# MAGIC         2
# MAGIC     ) AS gross_sales,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- DESCUENTOS
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     ROUND(
# MAGIC         SUM(
# MAGIC             f.discount_amount
# MAGIC         ),
# MAGIC         2
# MAGIC     ) AS total_discount,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- VENTAS NETAS
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     ROUND(
# MAGIC         SUM(
# MAGIC             f.net_amount
# MAGIC         ),
# MAGIC         2
# MAGIC     ) AS net_sales,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- TICKET MEDIO
# MAGIC     -- ========================================================
# MAGIC     --
# MAGIC     -- Ventas netas / número de tickets.
# MAGIC     --
# MAGIC     -- NULLIF protege frente a una posible división entre cero.
# MAGIC     --
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     ROUND(
# MAGIC
# MAGIC         SUM(
# MAGIC             f.net_amount
# MAGIC         )
# MAGIC
# MAGIC         /
# MAGIC
# MAGIC         NULLIF(
# MAGIC
# MAGIC             COUNT(
# MAGIC                 DISTINCT f.sale_id
# MAGIC             ),
# MAGIC
# MAGIC             0
# MAGIC
# MAGIC         ),
# MAGIC
# MAGIC         2
# MAGIC
# MAGIC     ) AS average_ticket,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- RANKING GLOBAL DE TIENDAS
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     RANK() OVER (
# MAGIC
# MAGIC         ORDER BY
# MAGIC             SUM(f.net_amount) DESC
# MAGIC
# MAGIC     ) AS sales_rank
# MAGIC
# MAGIC
# MAGIC -- ============================================================
# MAGIC -- FACT_SALES
# MAGIC -- ============================================================
# MAGIC
# MAGIC FROM
# MAGIC     retail_analytics.`3_gold`.fact_sales f
# MAGIC
# MAGIC
# MAGIC -- ============================================================
# MAGIC -- DIM_STORE
# MAGIC -- ============================================================
# MAGIC
# MAGIC INNER JOIN
# MAGIC     retail_analytics.`3_gold`.dim_store s
# MAGIC
# MAGIC     ON f.store_id = s.store_id
# MAGIC
# MAGIC
# MAGIC -- ============================================================
# MAGIC -- GRANULARIDAD
# MAGIC -- ============================================================
# MAGIC --
# MAGIC -- Una fila por tienda.
# MAGIC --
# MAGIC -- ============================================================
# MAGIC
# MAGIC GROUP BY
# MAGIC
# MAGIC     s.store_id,
# MAGIC
# MAGIC     s.store_name,
# MAGIC
# MAGIC     s.city,
# MAGIC
# MAGIC     s.region,
# MAGIC
# MAGIC     s.store_type,
# MAGIC
# MAGIC     s.opening_date;

# COMMAND ----------

# DBTITLE 1,04.01 VALIDACIÓN DE VW_STORE_PERFORMANCE
# MAGIC %sql
# MAGIC
# MAGIC -- ============================================================
# MAGIC -- 04.01 VALIDACIÓN DE VW_STORE_PERFORMANCE
# MAGIC -- ============================================================
# MAGIC
# MAGIC SELECT *
# MAGIC FROM retail_analytics.`3_gold`.vw_store_performance
# MAGIC ORDER BY sales_rank;

# COMMAND ----------

# DBTITLE 1,04.02 VENTAS POR REGIÓN Y PESO SOBRE EL TOTAL
# MAGIC %sql
# MAGIC
# MAGIC -- ============================================================
# MAGIC -- 04.02 VENTAS POR REGIÓN Y PESO SOBRE EL TOTAL
# MAGIC -- ============================================================
# MAGIC --
# MAGIC -- OBJETIVO
# MAGIC -- ------------------------------------------------------------
# MAGIC --
# MAGIC -- Analizar cuánto aporta cada región al total de ventas.
# MAGIC --
# MAGIC -- Calcularemos:
# MAGIC --
# MAGIC -- - ventas netas por región
# MAGIC -- - número de tickets
# MAGIC -- - unidades vendidas
# MAGIC -- - ticket medio
# MAGIC -- - porcentaje de ventas de cada región sobre el total
# MAGIC -- - ranking de regiones por ventas netas
# MAGIC --
# MAGIC -- ============================================================
# MAGIC
# MAGIC
# MAGIC WITH regional_sales AS (
# MAGIC
# MAGIC     SELECT
# MAGIC
# MAGIC         -- ====================================================
# MAGIC         -- REGIÓN
# MAGIC         -- ====================================================
# MAGIC
# MAGIC         s.region,
# MAGIC
# MAGIC
# MAGIC         -- ====================================================
# MAGIC         -- TICKETS
# MAGIC         -- ====================================================
# MAGIC
# MAGIC         COUNT(
# MAGIC             DISTINCT f.sale_id
# MAGIC         ) AS total_tickets,
# MAGIC
# MAGIC
# MAGIC         -- ====================================================
# MAGIC         -- UNIDADES VENDIDAS
# MAGIC         -- ====================================================
# MAGIC
# MAGIC         SUM(
# MAGIC             f.quantity
# MAGIC         ) AS total_units,
# MAGIC
# MAGIC
# MAGIC         -- ====================================================
# MAGIC         -- VENTAS NETAS
# MAGIC         -- ====================================================
# MAGIC
# MAGIC         ROUND(
# MAGIC             SUM(
# MAGIC                 f.net_amount
# MAGIC             ),
# MAGIC             2
# MAGIC         ) AS net_sales,
# MAGIC
# MAGIC
# MAGIC         -- ====================================================
# MAGIC         -- TICKET MEDIO
# MAGIC         -- ====================================================
# MAGIC
# MAGIC         ROUND(
# MAGIC
# MAGIC             SUM(
# MAGIC                 f.net_amount
# MAGIC             )
# MAGIC
# MAGIC             /
# MAGIC
# MAGIC             NULLIF(
# MAGIC
# MAGIC                 COUNT(
# MAGIC                     DISTINCT f.sale_id
# MAGIC                 ),
# MAGIC
# MAGIC                 0
# MAGIC
# MAGIC             ),
# MAGIC
# MAGIC             2
# MAGIC
# MAGIC         ) AS average_ticket
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- FACT_SALES
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     FROM
# MAGIC         retail_analytics.`3_gold`.fact_sales f
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- DIM_STORE
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     INNER JOIN
# MAGIC         retail_analytics.`3_gold`.dim_store s
# MAGIC
# MAGIC         ON f.store_id = s.store_id
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- GRANULARIDAD
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     GROUP BY
# MAGIC         s.region
# MAGIC )
# MAGIC
# MAGIC
# MAGIC SELECT
# MAGIC
# MAGIC     region,
# MAGIC
# MAGIC     total_tickets,
# MAGIC
# MAGIC     total_units,
# MAGIC
# MAGIC     net_sales,
# MAGIC
# MAGIC     average_ticket,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- PESO DE LA REGIÓN SOBRE LAS VENTAS TOTALES
# MAGIC     -- ========================================================
# MAGIC     --
# MAGIC     -- SUM(net_sales) OVER () calcula el total global sin
# MAGIC     -- perder la granularidad por región.
# MAGIC     --
# MAGIC     -- NULLIF evita una posible división entre cero.
# MAGIC     --
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     ROUND(
# MAGIC
# MAGIC         net_sales
# MAGIC
# MAGIC         /
# MAGIC
# MAGIC         NULLIF(
# MAGIC             SUM(net_sales) OVER (),
# MAGIC             0
# MAGIC         )
# MAGIC
# MAGIC         * 100,
# MAGIC
# MAGIC         2
# MAGIC
# MAGIC     ) AS sales_share_pct,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- RANKING DE REGIONES
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     RANK() OVER (
# MAGIC
# MAGIC         ORDER BY
# MAGIC             net_sales DESC
# MAGIC
# MAGIC     ) AS region_sales_rank
# MAGIC
# MAGIC
# MAGIC FROM
# MAGIC     regional_sales
# MAGIC
# MAGIC
# MAGIC ORDER BY
# MAGIC
# MAGIC     region_sales_rank,
# MAGIC
# MAGIC     region;

# COMMAND ----------

# DBTITLE 1,05. VIEW DE ANÁLISIS DE CLIENTES
# MAGIC %sql
# MAGIC
# MAGIC -- ============================================================
# MAGIC -- 05. VIEW DE ANÁLISIS DE CLIENTES
# MAGIC -- ============================================================
# MAGIC --
# MAGIC -- OBJETIVO
# MAGIC -- ------------------------------------------------------------
# MAGIC --
# MAGIC -- Analizar el comportamiento comercial de cada cliente
# MAGIC -- identificado.
# MAGIC --
# MAGIC -- Calcularemos:
# MAGIC --
# MAGIC -- - número de tickets
# MAGIC -- - unidades compradas
# MAGIC -- - ventas netas
# MAGIC -- - ticket medio
# MAGIC -- - gasto medio por línea
# MAGIC -- - ranking global de clientes
# MAGIC --
# MAGIC -- IMPORTANTE
# MAGIC -- ------------------------------------------------------------
# MAGIC --
# MAGIC -- Las ventas con customer_id = NULL representan compras
# MAGIC -- anónimas.
# MAGIC --
# MAGIC -- Estas ventas quedan deliberadamente fuera de esta vista
# MAGIC -- mediante el INNER JOIN con DIM_CUSTOMER.
# MAGIC --
# MAGIC -- La granularidad será:
# MAGIC --
# MAGIC --      1 fila = 1 cliente identificado
# MAGIC --
# MAGIC -- ============================================================
# MAGIC
# MAGIC
# MAGIC CREATE OR REPLACE VIEW
# MAGIC     retail_analytics.`3_gold`.vw_customer_analysis
# MAGIC AS
# MAGIC
# MAGIC
# MAGIC SELECT
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- ATRIBUTOS DEL CLIENTE
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     c.customer_id,
# MAGIC
# MAGIC     c.customer_name,
# MAGIC
# MAGIC     c.city,
# MAGIC
# MAGIC     c.region,
# MAGIC
# MAGIC     c.registration_date,
# MAGIC
# MAGIC     c.customer_segment,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- NÚMERO DE TICKETS
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     COUNT(
# MAGIC         DISTINCT f.sale_id
# MAGIC     ) AS total_tickets,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- UNIDADES COMPRADAS
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     SUM(
# MAGIC         f.quantity
# MAGIC     ) AS total_units,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- VENTAS NETAS
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     ROUND(
# MAGIC         SUM(
# MAGIC             f.net_amount
# MAGIC         ),
# MAGIC         2
# MAGIC     ) AS net_sales,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- TICKET MEDIO
# MAGIC     -- ========================================================
# MAGIC     --
# MAGIC     -- Ventas netas / número de tickets.
# MAGIC     --
# MAGIC     -- NULLIF evita una posible división entre cero.
# MAGIC     --
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     ROUND(
# MAGIC
# MAGIC         SUM(
# MAGIC             f.net_amount
# MAGIC         )
# MAGIC
# MAGIC         /
# MAGIC
# MAGIC         NULLIF(
# MAGIC
# MAGIC             COUNT(
# MAGIC                 DISTINCT f.sale_id
# MAGIC             ),
# MAGIC
# MAGIC             0
# MAGIC
# MAGIC         ),
# MAGIC
# MAGIC         2
# MAGIC
# MAGIC     ) AS average_ticket,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- GASTO MEDIO POR LÍNEA
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     ROUND(
# MAGIC
# MAGIC         AVG(
# MAGIC             f.net_amount
# MAGIC         ),
# MAGIC
# MAGIC         2
# MAGIC
# MAGIC     ) AS average_line_amount,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- RANKING GLOBAL DE CLIENTES
# MAGIC     -- ========================================================
# MAGIC     --
# MAGIC     -- Los clientes se ordenan según sus ventas netas
# MAGIC     -- acumuladas.
# MAGIC     --
# MAGIC     -- RANK permite que clientes con exactamente las mismas
# MAGIC     -- ventas compartan posición.
# MAGIC     --
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     RANK() OVER (
# MAGIC
# MAGIC         ORDER BY
# MAGIC             SUM(f.net_amount) DESC
# MAGIC
# MAGIC     ) AS customer_sales_rank
# MAGIC
# MAGIC
# MAGIC -- ============================================================
# MAGIC -- FACT_SALES
# MAGIC -- ============================================================
# MAGIC
# MAGIC FROM
# MAGIC     retail_analytics.`3_gold`.fact_sales f
# MAGIC
# MAGIC
# MAGIC -- ============================================================
# MAGIC -- DIM_CUSTOMER
# MAGIC -- ============================================================
# MAGIC --
# MAGIC -- INNER JOIN excluye automáticamente las ventas anónimas
# MAGIC -- porque customer_id = NULL no puede hacer match.
# MAGIC --
# MAGIC -- ============================================================
# MAGIC
# MAGIC INNER JOIN
# MAGIC     retail_analytics.`3_gold`.dim_customer c
# MAGIC
# MAGIC     ON f.customer_id = c.customer_id
# MAGIC
# MAGIC
# MAGIC -- ============================================================
# MAGIC -- GRANULARIDAD
# MAGIC -- ============================================================
# MAGIC --
# MAGIC -- Una fila por cliente identificado.
# MAGIC --
# MAGIC -- ============================================================
# MAGIC
# MAGIC GROUP BY
# MAGIC
# MAGIC     c.customer_id,
# MAGIC
# MAGIC     c.customer_name,
# MAGIC
# MAGIC     c.city,
# MAGIC
# MAGIC     c.region,
# MAGIC
# MAGIC     c.registration_date,
# MAGIC
# MAGIC     c.customer_segment;

# COMMAND ----------

# DBTITLE 1,05.01 VALIDACIÓN DE VW_CUSTOMER_ANALYSIS
# MAGIC %sql
# MAGIC
# MAGIC -- ============================================================
# MAGIC -- 05.01 VALIDACIÓN DE VW_CUSTOMER_ANALYSIS
# MAGIC -- ============================================================
# MAGIC
# MAGIC SELECT *
# MAGIC FROM retail_analytics.`3_gold`.vw_customer_analysis
# MAGIC ORDER BY customer_sales_rank
# MAGIC LIMIT 20;

# COMMAND ----------

# DBTITLE 1,05.02 ANÁLISIS POR SEGMENTO DE CLIENTE
# MAGIC %sql
# MAGIC
# MAGIC -- ============================================================
# MAGIC -- 05.02 ANÁLISIS POR SEGMENTO DE CLIENTE
# MAGIC -- ============================================================
# MAGIC --
# MAGIC -- OBJETIVO
# MAGIC -- ------------------------------------------------------------
# MAGIC --
# MAGIC -- Comparar el comportamiento comercial de los distintos
# MAGIC -- segmentos de clientes.
# MAGIC --
# MAGIC -- Analizaremos:
# MAGIC --
# MAGIC -- - clientes activos con compras
# MAGIC -- - tickets
# MAGIC -- - unidades compradas
# MAGIC -- - ventas netas
# MAGIC -- - venta media por cliente activo
# MAGIC -- - ticket medio
# MAGIC -- - peso sobre las ventas de clientes identificados
# MAGIC --
# MAGIC -- IMPORTANTE
# MAGIC -- ------------------------------------------------------------
# MAGIC --
# MAGIC -- active_customers representa únicamente clientes que tienen
# MAGIC -- al menos una compra en FACT_SALES.
# MAGIC --
# MAGIC -- No representa el total de clientes registrados en
# MAGIC -- DIM_CUSTOMER.
# MAGIC --
# MAGIC -- Esto es coherente con el objetivo comercial de esta query:
# MAGIC -- analizar comportamiento real de compra por segmento.
# MAGIC --
# MAGIC -- ============================================================
# MAGIC
# MAGIC
# MAGIC WITH segment_analysis AS (
# MAGIC
# MAGIC     SELECT
# MAGIC
# MAGIC         -- ====================================================
# MAGIC         -- SEGMENTO
# MAGIC         -- ====================================================
# MAGIC
# MAGIC         c.customer_segment,
# MAGIC
# MAGIC
# MAGIC         -- ====================================================
# MAGIC         -- CLIENTES ACTIVOS
# MAGIC         -- ====================================================
# MAGIC         --
# MAGIC         -- Clientes identificados con al menos una compra.
# MAGIC         --
# MAGIC         -- ====================================================
# MAGIC
# MAGIC         COUNT(
# MAGIC             DISTINCT c.customer_id
# MAGIC         ) AS active_customers,
# MAGIC
# MAGIC
# MAGIC         -- ====================================================
# MAGIC         -- TICKETS
# MAGIC         -- ====================================================
# MAGIC
# MAGIC         COUNT(
# MAGIC             DISTINCT f.sale_id
# MAGIC         ) AS total_tickets,
# MAGIC
# MAGIC
# MAGIC         -- ====================================================
# MAGIC         -- UNIDADES
# MAGIC         -- ====================================================
# MAGIC
# MAGIC         SUM(
# MAGIC             f.quantity
# MAGIC         ) AS total_units,
# MAGIC
# MAGIC
# MAGIC         -- ====================================================
# MAGIC         -- VENTAS NETAS
# MAGIC         -- ====================================================
# MAGIC
# MAGIC         ROUND(
# MAGIC             SUM(
# MAGIC                 f.net_amount
# MAGIC             ),
# MAGIC             2
# MAGIC         ) AS net_sales
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- FACT_SALES
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     FROM
# MAGIC         retail_analytics.`3_gold`.fact_sales f
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- DIM_CUSTOMER
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     INNER JOIN
# MAGIC         retail_analytics.`3_gold`.dim_customer c
# MAGIC
# MAGIC         ON f.customer_id = c.customer_id
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- GRANULARIDAD
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     GROUP BY
# MAGIC         c.customer_segment
# MAGIC )
# MAGIC
# MAGIC
# MAGIC SELECT
# MAGIC
# MAGIC     customer_segment,
# MAGIC
# MAGIC     active_customers,
# MAGIC
# MAGIC     total_tickets,
# MAGIC
# MAGIC     total_units,
# MAGIC
# MAGIC     net_sales,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- VENTA MEDIA POR CLIENTE ACTIVO
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     ROUND(
# MAGIC
# MAGIC         net_sales
# MAGIC
# MAGIC         /
# MAGIC
# MAGIC         NULLIF(
# MAGIC             active_customers,
# MAGIC             0
# MAGIC         ),
# MAGIC
# MAGIC         2
# MAGIC
# MAGIC     ) AS sales_per_active_customer,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- TICKET MEDIO
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     ROUND(
# MAGIC
# MAGIC         net_sales
# MAGIC
# MAGIC         /
# MAGIC
# MAGIC         NULLIF(
# MAGIC             total_tickets,
# MAGIC             0
# MAGIC         ),
# MAGIC
# MAGIC         2
# MAGIC
# MAGIC     ) AS average_ticket,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- PESO SOBRE LAS VENTAS DE CLIENTES IDENTIFICADOS
# MAGIC     -- ========================================================
# MAGIC     --
# MAGIC     -- La suma de net_sales aquí solo considera ventas
# MAGIC     -- asociadas a clientes identificados.
# MAGIC     --
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     ROUND(
# MAGIC
# MAGIC         net_sales
# MAGIC
# MAGIC         /
# MAGIC
# MAGIC         NULLIF(
# MAGIC             SUM(net_sales) OVER (),
# MAGIC             0
# MAGIC         )
# MAGIC
# MAGIC         * 100,
# MAGIC
# MAGIC         2
# MAGIC
# MAGIC     ) AS sales_share_pct
# MAGIC
# MAGIC
# MAGIC FROM
# MAGIC     segment_analysis
# MAGIC
# MAGIC
# MAGIC ORDER BY
# MAGIC
# MAGIC     net_sales DESC;

# COMMAND ----------

# DBTITLE 1,06. VIEW DE RENDIMIENTO POR CANAL DE VENTA
# MAGIC %sql
# MAGIC
# MAGIC -- ============================================================
# MAGIC -- 06. VIEW DE RENDIMIENTO POR CANAL DE VENTA
# MAGIC -- ============================================================
# MAGIC --
# MAGIC -- OBJETIVO
# MAGIC -- ------------------------------------------------------------
# MAGIC --
# MAGIC -- Analizar el rendimiento comercial de cada canal de venta.
# MAGIC --
# MAGIC -- Calcularemos:
# MAGIC --
# MAGIC -- - número de tickets
# MAGIC -- - número de líneas
# MAGIC -- - unidades vendidas
# MAGIC -- - ventas brutas
# MAGIC -- - descuentos
# MAGIC -- - ventas netas
# MAGIC -- - ticket medio
# MAGIC -- - porcentaje efectivo de descuento
# MAGIC -- - peso del canal sobre las ventas totales
# MAGIC -- - ranking de canales
# MAGIC --
# MAGIC -- GRANULARIDAD:
# MAGIC --
# MAGIC --      1 fila = 1 canal de venta
# MAGIC --
# MAGIC -- ============================================================
# MAGIC
# MAGIC
# MAGIC CREATE OR REPLACE VIEW
# MAGIC     retail_analytics.`3_gold`.vw_channel_performance
# MAGIC AS
# MAGIC
# MAGIC
# MAGIC WITH channel_sales AS (
# MAGIC
# MAGIC     SELECT
# MAGIC
# MAGIC         -- ====================================================
# MAGIC         -- CANAL DE VENTA
# MAGIC         -- ====================================================
# MAGIC
# MAGIC         f.sales_channel,
# MAGIC
# MAGIC
# MAGIC         -- ====================================================
# MAGIC         -- TICKETS
# MAGIC         -- ====================================================
# MAGIC
# MAGIC         COUNT(
# MAGIC             DISTINCT f.sale_id
# MAGIC         ) AS total_tickets,
# MAGIC
# MAGIC
# MAGIC         -- ====================================================
# MAGIC         -- LÍNEAS DE VENTA
# MAGIC         -- ====================================================
# MAGIC
# MAGIC         COUNT(
# MAGIC             f.line_id
# MAGIC         ) AS total_lines,
# MAGIC
# MAGIC
# MAGIC         -- ====================================================
# MAGIC         -- UNIDADES VENDIDAS
# MAGIC         -- ====================================================
# MAGIC
# MAGIC         SUM(
# MAGIC             f.quantity
# MAGIC         ) AS total_units,
# MAGIC
# MAGIC
# MAGIC         -- ====================================================
# MAGIC         -- VENTAS BRUTAS
# MAGIC         -- ====================================================
# MAGIC
# MAGIC         ROUND(
# MAGIC             SUM(
# MAGIC                 f.gross_amount
# MAGIC             ),
# MAGIC             2
# MAGIC         ) AS gross_sales,
# MAGIC
# MAGIC
# MAGIC         -- ====================================================
# MAGIC         -- DESCUENTOS
# MAGIC         -- ====================================================
# MAGIC
# MAGIC         ROUND(
# MAGIC             SUM(
# MAGIC                 f.discount_amount
# MAGIC             ),
# MAGIC             2
# MAGIC         ) AS total_discount,
# MAGIC
# MAGIC
# MAGIC         -- ====================================================
# MAGIC         -- VENTAS NETAS
# MAGIC         -- ====================================================
# MAGIC
# MAGIC         ROUND(
# MAGIC             SUM(
# MAGIC                 f.net_amount
# MAGIC             ),
# MAGIC             2
# MAGIC         ) AS net_sales
# MAGIC
# MAGIC
# MAGIC     FROM
# MAGIC         retail_analytics.`3_gold`.fact_sales f
# MAGIC
# MAGIC
# MAGIC     GROUP BY
# MAGIC         f.sales_channel
# MAGIC )
# MAGIC
# MAGIC
# MAGIC SELECT
# MAGIC
# MAGIC     sales_channel,
# MAGIC
# MAGIC     total_tickets,
# MAGIC
# MAGIC     total_lines,
# MAGIC
# MAGIC     total_units,
# MAGIC
# MAGIC     gross_sales,
# MAGIC
# MAGIC     total_discount,
# MAGIC
# MAGIC     net_sales,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- TICKET MEDIO
# MAGIC     -- ========================================================
# MAGIC     --
# MAGIC     -- Ventas netas / tickets.
# MAGIC     --
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     ROUND(
# MAGIC
# MAGIC         net_sales
# MAGIC
# MAGIC         /
# MAGIC
# MAGIC         NULLIF(
# MAGIC             total_tickets,
# MAGIC             0
# MAGIC         ),
# MAGIC
# MAGIC         2
# MAGIC
# MAGIC     ) AS average_ticket,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- PORCENTAJE EFECTIVO DE DESCUENTO
# MAGIC     -- ========================================================
# MAGIC     --
# MAGIC     -- No calculamos AVG(discount_pct), porque eso daría el
# MAGIC     -- mismo peso a todas las líneas independientemente de
# MAGIC     -- su importe.
# MAGIC     --
# MAGIC     -- Aquí calculamos:
# MAGIC     --
# MAGIC     --       descuento total
# MAGIC     --      ----------------- × 100
# MAGIC     --        venta bruta
# MAGIC     --
# MAGIC     -- De esta forma obtenemos el porcentaje real de ventas
# MAGIC     -- brutas que se ha destinado a descuentos.
# MAGIC     --
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     ROUND(
# MAGIC
# MAGIC         total_discount
# MAGIC
# MAGIC         /
# MAGIC
# MAGIC         NULLIF(
# MAGIC             gross_sales,
# MAGIC             0
# MAGIC         )
# MAGIC
# MAGIC         * 100,
# MAGIC
# MAGIC         2
# MAGIC
# MAGIC     ) AS effective_discount_pct,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- PESO DEL CANAL SOBRE LAS VENTAS TOTALES
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     ROUND(
# MAGIC
# MAGIC         net_sales
# MAGIC
# MAGIC         /
# MAGIC
# MAGIC         NULLIF(
# MAGIC             SUM(net_sales) OVER (),
# MAGIC             0
# MAGIC         )
# MAGIC
# MAGIC         * 100,
# MAGIC
# MAGIC         2
# MAGIC
# MAGIC     ) AS sales_share_pct,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- RANKING DE CANALES
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     RANK() OVER (
# MAGIC
# MAGIC         ORDER BY
# MAGIC             net_sales DESC
# MAGIC
# MAGIC     ) AS channel_sales_rank
# MAGIC
# MAGIC
# MAGIC FROM
# MAGIC     channel_sales;

# COMMAND ----------

# DBTITLE 1,06.01 VALIDACIÓN DE VW_CHANNEL_PERFORMANCE
# MAGIC %sql
# MAGIC
# MAGIC -- ============================================================
# MAGIC -- 06.01 VALIDACIÓN DE VW_CHANNEL_PERFORMANCE
# MAGIC -- ============================================================
# MAGIC
# MAGIC SELECT *
# MAGIC FROM retail_analytics.`3_gold`.vw_channel_performance
# MAGIC ORDER BY channel_sales_rank;

# COMMAND ----------

# DBTITLE 1,06.02 ANÁLISIS POR MÉTODO DE PAGO
# MAGIC %sql
# MAGIC -- ============================================================
# MAGIC -- 06.02 ANÁLISIS POR MÉTODO DE PAGO
# MAGIC -- ============================================================
# MAGIC --
# MAGIC -- OBJETIVO
# MAGIC -- ------------------------------------------------------------
# MAGIC --
# MAGIC -- Analizar el comportamiento de las ventas según el método
# MAGIC -- de pago utilizado.
# MAGIC --
# MAGIC -- Calcularemos:
# MAGIC --
# MAGIC -- - número de tickets
# MAGIC -- - número de líneas
# MAGIC -- - unidades vendidas
# MAGIC -- - ventas brutas
# MAGIC -- - descuentos
# MAGIC -- - ventas netas
# MAGIC -- - ticket medio
# MAGIC -- - porcentaje efectivo de descuento
# MAGIC -- - peso sobre las ventas totales
# MAGIC --
# MAGIC -- ============================================================
# MAGIC
# MAGIC
# MAGIC WITH payment_analysis AS (
# MAGIC
# MAGIC     SELECT
# MAGIC
# MAGIC         -- ====================================================
# MAGIC         -- MÉTODO DE PAGO
# MAGIC         -- ====================================================
# MAGIC
# MAGIC         f.payment_method,
# MAGIC
# MAGIC
# MAGIC         -- ====================================================
# MAGIC         -- TICKETS
# MAGIC         -- ====================================================
# MAGIC
# MAGIC         COUNT(
# MAGIC             DISTINCT f.sale_id
# MAGIC         ) AS total_tickets,
# MAGIC
# MAGIC
# MAGIC         -- ====================================================
# MAGIC         -- LÍNEAS
# MAGIC         -- ====================================================
# MAGIC
# MAGIC         COUNT(
# MAGIC             f.line_id
# MAGIC         ) AS total_lines,
# MAGIC
# MAGIC
# MAGIC         -- ====================================================
# MAGIC         -- UNIDADES
# MAGIC         -- ====================================================
# MAGIC
# MAGIC         SUM(
# MAGIC             f.quantity
# MAGIC         ) AS total_units,
# MAGIC
# MAGIC
# MAGIC         -- ====================================================
# MAGIC         -- VENTAS BRUTAS
# MAGIC         -- ====================================================
# MAGIC
# MAGIC         ROUND(
# MAGIC             SUM(
# MAGIC                 f.gross_amount
# MAGIC             ),
# MAGIC             2
# MAGIC         ) AS gross_sales,
# MAGIC
# MAGIC
# MAGIC         -- ====================================================
# MAGIC         -- DESCUENTOS
# MAGIC         -- ====================================================
# MAGIC
# MAGIC         ROUND(
# MAGIC             SUM(
# MAGIC                 f.discount_amount
# MAGIC             ),
# MAGIC             2
# MAGIC         ) AS total_discount,
# MAGIC
# MAGIC
# MAGIC         -- ====================================================
# MAGIC         -- VENTAS NETAS
# MAGIC         -- ====================================================
# MAGIC
# MAGIC         ROUND(
# MAGIC             SUM(
# MAGIC                 f.net_amount
# MAGIC             ),
# MAGIC             2
# MAGIC         ) AS net_sales
# MAGIC
# MAGIC
# MAGIC     FROM
# MAGIC         retail_analytics.`3_gold`.fact_sales f
# MAGIC
# MAGIC
# MAGIC     GROUP BY
# MAGIC         f.payment_method
# MAGIC )
# MAGIC
# MAGIC
# MAGIC SELECT
# MAGIC
# MAGIC     payment_method,
# MAGIC
# MAGIC     total_tickets,
# MAGIC
# MAGIC     total_lines,
# MAGIC
# MAGIC     total_units,
# MAGIC
# MAGIC     gross_sales,
# MAGIC
# MAGIC     total_discount,
# MAGIC
# MAGIC     net_sales,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- TICKET MEDIO
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     ROUND(
# MAGIC
# MAGIC         net_sales
# MAGIC
# MAGIC         /
# MAGIC
# MAGIC         NULLIF(
# MAGIC             total_tickets,
# MAGIC             0
# MAGIC         ),
# MAGIC
# MAGIC         2
# MAGIC
# MAGIC     ) AS average_ticket,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- PORCENTAJE EFECTIVO DE DESCUENTO
# MAGIC     -- ========================================================
# MAGIC     --
# MAGIC     -- Fórmula:
# MAGIC     --
# MAGIC     --      descuento total
# MAGIC     --     ----------------- × 100
# MAGIC     --       ventas brutas
# MAGIC     --
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     ROUND(
# MAGIC
# MAGIC         total_discount
# MAGIC
# MAGIC         /
# MAGIC
# MAGIC         NULLIF(
# MAGIC             gross_sales,
# MAGIC             0
# MAGIC         )
# MAGIC
# MAGIC         * 100,
# MAGIC
# MAGIC         2
# MAGIC
# MAGIC     ) AS effective_discount_pct,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- PESO SOBRE LAS VENTAS TOTALES
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     ROUND(
# MAGIC
# MAGIC         net_sales
# MAGIC
# MAGIC         /
# MAGIC
# MAGIC         NULLIF(
# MAGIC             SUM(net_sales) OVER (),
# MAGIC             0
# MAGIC         )
# MAGIC
# MAGIC         * 100,
# MAGIC
# MAGIC         2
# MAGIC
# MAGIC     ) AS sales_share_pct
# MAGIC
# MAGIC
# MAGIC FROM
# MAGIC     payment_analysis
# MAGIC
# MAGIC
# MAGIC ORDER BY
# MAGIC     net_sales DESC;

# COMMAND ----------

# DBTITLE 1,06.03 ANÁLISIS CANAL × MÉTODO DE PAGO
# MAGIC %sql
# MAGIC
# MAGIC -- ============================================================
# MAGIC -- 06.03 ANÁLISIS CRUZADO CANAL × MÉTODO DE PAGO
# MAGIC -- ============================================================
# MAGIC --
# MAGIC -- OBJETIVO
# MAGIC -- ------------------------------------------------------------
# MAGIC --
# MAGIC -- Analizar el comportamiento de cada método de pago dentro
# MAGIC -- de cada canal de venta.
# MAGIC --
# MAGIC -- Calcularemos:
# MAGIC --
# MAGIC -- - número de tickets
# MAGIC -- - unidades vendidas
# MAGIC -- - ventas brutas
# MAGIC -- - descuentos
# MAGIC -- - ventas netas
# MAGIC -- - ticket medio
# MAGIC -- - porcentaje efectivo de descuento
# MAGIC -- - peso del método de pago dentro de su canal
# MAGIC --
# MAGIC -- GRANULARIDAD:
# MAGIC --
# MAGIC --      1 fila = 1 combinación canal × método de pago
# MAGIC --
# MAGIC -- ============================================================
# MAGIC
# MAGIC
# MAGIC WITH channel_payment AS (
# MAGIC
# MAGIC     SELECT
# MAGIC
# MAGIC         -- ====================================================
# MAGIC         -- DIMENSIONES
# MAGIC         -- ====================================================
# MAGIC
# MAGIC         f.sales_channel,
# MAGIC
# MAGIC         f.payment_method,
# MAGIC
# MAGIC
# MAGIC         -- ====================================================
# MAGIC         -- TICKETS
# MAGIC         -- ====================================================
# MAGIC
# MAGIC         COUNT(
# MAGIC             DISTINCT f.sale_id
# MAGIC         ) AS total_tickets,
# MAGIC
# MAGIC
# MAGIC         -- ====================================================
# MAGIC         -- UNIDADES
# MAGIC         -- ====================================================
# MAGIC
# MAGIC         SUM(
# MAGIC             f.quantity
# MAGIC         ) AS total_units,
# MAGIC
# MAGIC
# MAGIC         -- ====================================================
# MAGIC         -- VENTAS BRUTAS
# MAGIC         -- ====================================================
# MAGIC
# MAGIC         ROUND(
# MAGIC             SUM(f.gross_amount),
# MAGIC             2
# MAGIC         ) AS gross_sales,
# MAGIC
# MAGIC
# MAGIC         -- ====================================================
# MAGIC         -- DESCUENTOS
# MAGIC         -- ====================================================
# MAGIC
# MAGIC         ROUND(
# MAGIC             SUM(f.discount_amount),
# MAGIC             2
# MAGIC         ) AS total_discount,
# MAGIC
# MAGIC
# MAGIC         -- ====================================================
# MAGIC         -- VENTAS NETAS
# MAGIC         -- ====================================================
# MAGIC
# MAGIC         ROUND(
# MAGIC             SUM(f.net_amount),
# MAGIC             2
# MAGIC         ) AS net_sales
# MAGIC
# MAGIC
# MAGIC     FROM
# MAGIC         retail_analytics.`3_gold`.fact_sales f
# MAGIC
# MAGIC
# MAGIC     GROUP BY
# MAGIC
# MAGIC         f.sales_channel,
# MAGIC
# MAGIC         f.payment_method
# MAGIC )
# MAGIC
# MAGIC
# MAGIC SELECT
# MAGIC
# MAGIC     sales_channel,
# MAGIC
# MAGIC     payment_method,
# MAGIC
# MAGIC     total_tickets,
# MAGIC
# MAGIC     total_units,
# MAGIC
# MAGIC     gross_sales,
# MAGIC
# MAGIC     total_discount,
# MAGIC
# MAGIC     net_sales,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- TICKET MEDIO
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     ROUND(
# MAGIC
# MAGIC         net_sales
# MAGIC
# MAGIC         /
# MAGIC
# MAGIC         NULLIF(
# MAGIC             total_tickets,
# MAGIC             0
# MAGIC         ),
# MAGIC
# MAGIC         2
# MAGIC
# MAGIC     ) AS average_ticket,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- DESCUENTO EFECTIVO
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     ROUND(
# MAGIC
# MAGIC         total_discount
# MAGIC
# MAGIC         /
# MAGIC
# MAGIC         NULLIF(
# MAGIC             gross_sales,
# MAGIC             0
# MAGIC         )
# MAGIC
# MAGIC         * 100,
# MAGIC
# MAGIC         2
# MAGIC
# MAGIC     ) AS effective_discount_pct,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- PESO DENTRO DEL CANAL
# MAGIC     -- ========================================================
# MAGIC     --
# MAGIC     -- Aquí NO calculamos el peso sobre las ventas globales.
# MAGIC     --
# MAGIC     -- PARTITION BY sales_channel hace que el denominador sea
# MAGIC     -- únicamente el total de ventas de ese canal.
# MAGIC     --
# MAGIC     -- Ejemplo:
# MAGIC     --
# MAGIC     -- ONLINE
# MAGIC     --   Card       60 %
# MAGIC     --   PayPal     25 %
# MAGIC     --   Transfer   15 %
# MAGIC     --
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     ROUND(
# MAGIC
# MAGIC         net_sales
# MAGIC
# MAGIC         /
# MAGIC
# MAGIC         NULLIF(
# MAGIC
# MAGIC             SUM(net_sales) OVER (
# MAGIC                 PARTITION BY sales_channel
# MAGIC             ),
# MAGIC
# MAGIC             0
# MAGIC
# MAGIC         )
# MAGIC
# MAGIC         * 100,
# MAGIC
# MAGIC         2
# MAGIC
# MAGIC     ) AS payment_share_within_channel_pct,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- RANKING DEL MÉTODO DE PAGO DENTRO DEL CANAL
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     RANK() OVER (
# MAGIC
# MAGIC         PARTITION BY
# MAGIC             sales_channel
# MAGIC
# MAGIC         ORDER BY
# MAGIC             net_sales DESC
# MAGIC
# MAGIC     ) AS payment_rank_within_channel
# MAGIC
# MAGIC
# MAGIC FROM
# MAGIC     channel_payment
# MAGIC
# MAGIC
# MAGIC ORDER BY
# MAGIC
# MAGIC     sales_channel,
# MAGIC
# MAGIC     payment_rank_within_channel,
# MAGIC
# MAGIC     payment_method;

# COMMAND ----------

# DBTITLE 1,07. VIEW DE KPIs EJECUTIVOS
# MAGIC %sql
# MAGIC
# MAGIC -- ============================================================
# MAGIC -- 07. VIEW DE KPIs EJECUTIVOS
# MAGIC -- ============================================================
# MAGIC --
# MAGIC -- OBJETIVO
# MAGIC -- ------------------------------------------------------------
# MAGIC --
# MAGIC -- Crear una vista ejecutiva con los principales indicadores
# MAGIC -- globales del negocio.
# MAGIC --
# MAGIC -- Esta vista devuelve UNA ÚNICA FILA y está diseñada para
# MAGIC -- alimentar directamente las KPI Cards del dashboard.
# MAGIC --
# MAGIC -- KPIs:
# MAGIC --
# MAGIC -- - Ventas netas
# MAGIC -- - Ventas brutas
# MAGIC -- - Descuentos
# MAGIC -- - Tickets
# MAGIC -- - Unidades vendidas
# MAGIC -- - Ticket medio
# MAGIC -- - Clientes identificados
# MAGIC -- - Ventas anónimas (%)
# MAGIC -- - Descuento efectivo (%)
# MAGIC --
# MAGIC -- ============================================================
# MAGIC
# MAGIC
# MAGIC CREATE OR REPLACE VIEW
# MAGIC     retail_analytics.`3_gold`.vw_executive_kpis
# MAGIC AS
# MAGIC
# MAGIC
# MAGIC SELECT
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- VENTAS NETAS
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     ROUND(
# MAGIC         SUM(f.net_amount),
# MAGIC         2
# MAGIC     ) AS net_sales,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- VENTAS BRUTAS
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     ROUND(
# MAGIC         SUM(f.gross_amount),
# MAGIC         2
# MAGIC     ) AS gross_sales,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- DESCUENTOS TOTALES
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     ROUND(
# MAGIC         SUM(f.discount_amount),
# MAGIC         2
# MAGIC     ) AS total_discounts,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- TOTAL DE TICKETS
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     COUNT(
# MAGIC         DISTINCT f.sale_id
# MAGIC     ) AS total_tickets,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- TOTAL DE UNIDADES VENDIDAS
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     SUM(
# MAGIC         f.quantity
# MAGIC     ) AS total_units_sold,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- TICKET MEDIO
# MAGIC     -- ========================================================
# MAGIC     --
# MAGIC     -- Ventas netas / número de tickets.
# MAGIC     --
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     ROUND(
# MAGIC
# MAGIC         SUM(f.net_amount)
# MAGIC
# MAGIC         /
# MAGIC
# MAGIC         NULLIF(
# MAGIC             COUNT(DISTINCT f.sale_id),
# MAGIC             0
# MAGIC         ),
# MAGIC
# MAGIC         2
# MAGIC
# MAGIC     ) AS average_ticket,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- CLIENTES IDENTIFICADOS
# MAGIC     -- ========================================================
# MAGIC     --
# MAGIC     -- COUNT DISTINCT ignora automáticamente los NULL.
# MAGIC     --
# MAGIC     -- Por tanto, únicamente contamos clientes identificados.
# MAGIC     --
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     COUNT(
# MAGIC         DISTINCT f.customer_id
# MAGIC     ) AS identified_customers,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- PORCENTAJE DE VENTAS ANÓNIMAS
# MAGIC     -- ========================================================
# MAGIC     --
# MAGIC     -- Medimos el porcentaje SOBRE EL IMPORTE DE VENTAS.
# MAGIC     --
# MAGIC     -- No el porcentaje de líneas ni de tickets.
# MAGIC     --
# MAGIC     -- Esto responde a:
# MAGIC     --
# MAGIC     -- "¿Qué porcentaje de nuestros ingresos procede de
# MAGIC     --  compras de clientes no identificados?"
# MAGIC     --
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     ROUND(
# MAGIC
# MAGIC         SUM(
# MAGIC             CASE
# MAGIC
# MAGIC                 WHEN f.customer_id IS NULL
# MAGIC                     THEN f.net_amount
# MAGIC
# MAGIC                 ELSE 0
# MAGIC
# MAGIC             END
# MAGIC         )
# MAGIC
# MAGIC         /
# MAGIC
# MAGIC         NULLIF(
# MAGIC             SUM(f.net_amount),
# MAGIC             0
# MAGIC         )
# MAGIC
# MAGIC         * 100,
# MAGIC
# MAGIC         2
# MAGIC
# MAGIC     ) AS anonymous_sales_pct,
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- PORCENTAJE EFECTIVO DE DESCUENTO
# MAGIC     -- ========================================================
# MAGIC     --
# MAGIC     -- Mantenemos exactamente la misma definición utilizada
# MAGIC     -- en los análisis de canal y método de pago:
# MAGIC     --
# MAGIC     --       descuentos
# MAGIC     --      ------------ × 100
# MAGIC     --      venta bruta
# MAGIC     --
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     ROUND(
# MAGIC
# MAGIC         SUM(f.discount_amount)
# MAGIC
# MAGIC         /
# MAGIC
# MAGIC         NULLIF(
# MAGIC             SUM(f.gross_amount),
# MAGIC             0
# MAGIC         )
# MAGIC
# MAGIC         * 100,
# MAGIC
# MAGIC         2
# MAGIC
# MAGIC     ) AS effective_discount_pct
# MAGIC
# MAGIC
# MAGIC FROM
# MAGIC     retail_analytics.`3_gold`.fact_sales f;

# COMMAND ----------

# DBTITLE 1,07.01 VALIDACIÓN DE VW_EXECUTIVE_KPIS
# MAGIC %sql
# MAGIC
# MAGIC -- ============================================================
# MAGIC -- 07.01 VALIDACIÓN DE VW_EXECUTIVE_KPIS
# MAGIC -- ============================================================
# MAGIC
# MAGIC SELECT *
# MAGIC FROM retail_analytics.`3_gold`.vw_executive_kpis;

# COMMAND ----------

# DBTITLE 1,08. VALIDACIÓN FINAL DE LA CAPA SQL ANALYTICS
# MAGIC %sql
# MAGIC -- ============================================================
# MAGIC -- 08. VALIDACIÓN FINAL DE LA CAPA SQL ANALYTICS
# MAGIC -- ============================================================
# MAGIC --
# MAGIC -- OBJETIVO
# MAGIC -- ------------------------------------------------------------
# MAGIC --
# MAGIC -- Validar de forma automática que las principales Views de
# MAGIC -- consumo SQL:
# MAGIC --
# MAGIC -- - existen
# MAGIC -- - contienen datos
# MAGIC -- - mantienen la granularidad esperada
# MAGIC -- - no presentan duplicados en sus claves principales
# MAGIC --
# MAGIC -- Si alguna validación crítica falla, el notebook devolverá
# MAGIC -- ERROR para que el Job no finalice como SUCCESS.
# MAGIC --
# MAGIC -- ============================================================
# MAGIC
# MAGIC
# MAGIC WITH validation_results AS (
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- VW_SALES_DAILY
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     SELECT
# MAGIC
# MAGIC         'vw_sales_daily' AS object_name,
# MAGIC
# MAGIC         COUNT(*) AS row_count,
# MAGIC
# MAGIC         COUNT(DISTINCT date_key) AS distinct_key_count,
# MAGIC
# MAGIC         CASE
# MAGIC             WHEN COUNT(*) = 0 THEN 'ERROR - EMPTY'
# MAGIC             WHEN COUNT(*) <> COUNT(DISTINCT date_key)
# MAGIC                 THEN 'ERROR - DUPLICATE KEY'
# MAGIC             ELSE 'OK'
# MAGIC         END AS status
# MAGIC
# MAGIC     FROM
# MAGIC         retail_analytics.`3_gold`.vw_sales_daily
# MAGIC
# MAGIC
# MAGIC     UNION ALL
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- VW_SALES_MONTHLY
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     SELECT
# MAGIC
# MAGIC         'vw_sales_monthly' AS object_name,
# MAGIC
# MAGIC         COUNT(*) AS row_count,
# MAGIC
# MAGIC         COUNT(DISTINCT year_month) AS distinct_key_count,
# MAGIC
# MAGIC         CASE
# MAGIC             WHEN COUNT(*) = 0 THEN 'ERROR - EMPTY'
# MAGIC             WHEN COUNT(*) <> COUNT(DISTINCT year_month)
# MAGIC                 THEN 'ERROR - DUPLICATE KEY'
# MAGIC             ELSE 'OK'
# MAGIC         END AS status
# MAGIC
# MAGIC     FROM
# MAGIC         retail_analytics.`3_gold`.vw_sales_monthly
# MAGIC
# MAGIC
# MAGIC     UNION ALL
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- VW_PRODUCT_PERFORMANCE
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     SELECT
# MAGIC
# MAGIC         'vw_product_performance' AS object_name,
# MAGIC
# MAGIC         COUNT(*) AS row_count,
# MAGIC
# MAGIC         COUNT(DISTINCT product_id) AS distinct_key_count,
# MAGIC
# MAGIC         CASE
# MAGIC             WHEN COUNT(*) = 0 THEN 'ERROR - EMPTY'
# MAGIC             WHEN COUNT(*) <> COUNT(DISTINCT product_id)
# MAGIC                 THEN 'ERROR - DUPLICATE KEY'
# MAGIC             ELSE 'OK'
# MAGIC         END AS status
# MAGIC
# MAGIC     FROM
# MAGIC         retail_analytics.`3_gold`.vw_product_performance
# MAGIC
# MAGIC
# MAGIC     UNION ALL
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- VW_STORE_PERFORMANCE
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     SELECT
# MAGIC
# MAGIC         'vw_store_performance' AS object_name,
# MAGIC
# MAGIC         COUNT(*) AS row_count,
# MAGIC
# MAGIC         COUNT(DISTINCT store_id) AS distinct_key_count,
# MAGIC
# MAGIC         CASE
# MAGIC             WHEN COUNT(*) = 0 THEN 'ERROR - EMPTY'
# MAGIC             WHEN COUNT(*) <> COUNT(DISTINCT store_id)
# MAGIC                 THEN 'ERROR - DUPLICATE KEY'
# MAGIC             ELSE 'OK'
# MAGIC         END AS status
# MAGIC
# MAGIC     FROM
# MAGIC         retail_analytics.`3_gold`.vw_store_performance
# MAGIC
# MAGIC
# MAGIC     UNION ALL
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- VW_CUSTOMER_ANALYSIS
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     SELECT
# MAGIC
# MAGIC         'vw_customer_analysis' AS object_name,
# MAGIC
# MAGIC         COUNT(*) AS row_count,
# MAGIC
# MAGIC         COUNT(DISTINCT customer_id) AS distinct_key_count,
# MAGIC
# MAGIC         CASE
# MAGIC             WHEN COUNT(*) = 0 THEN 'ERROR - EMPTY'
# MAGIC             WHEN COUNT(*) <> COUNT(DISTINCT customer_id)
# MAGIC                 THEN 'ERROR - DUPLICATE KEY'
# MAGIC             ELSE 'OK'
# MAGIC         END AS status
# MAGIC
# MAGIC     FROM
# MAGIC         retail_analytics.`3_gold`.vw_customer_analysis
# MAGIC
# MAGIC
# MAGIC     UNION ALL
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- VW_CHANNEL_PERFORMANCE
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     SELECT
# MAGIC
# MAGIC         'vw_channel_performance' AS object_name,
# MAGIC
# MAGIC         COUNT(*) AS row_count,
# MAGIC
# MAGIC         COUNT(DISTINCT sales_channel) AS distinct_key_count,
# MAGIC
# MAGIC         CASE
# MAGIC             WHEN COUNT(*) = 0 THEN 'ERROR - EMPTY'
# MAGIC             WHEN COUNT(*) <> COUNT(DISTINCT sales_channel)
# MAGIC                 THEN 'ERROR - DUPLICATE KEY'
# MAGIC             ELSE 'OK'
# MAGIC         END AS status
# MAGIC
# MAGIC     FROM
# MAGIC         retail_analytics.`3_gold`.vw_channel_performance
# MAGIC
# MAGIC
# MAGIC     UNION ALL
# MAGIC
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- VW_EXECUTIVE_KPIS
# MAGIC     -- ========================================================
# MAGIC     --
# MAGIC     -- Esta View debe devolver exactamente UNA fila.
# MAGIC     --
# MAGIC     -- ========================================================
# MAGIC
# MAGIC     SELECT
# MAGIC
# MAGIC         'vw_executive_kpis' AS object_name,
# MAGIC
# MAGIC         COUNT(*) AS row_count,
# MAGIC
# MAGIC         COUNT(*) AS distinct_key_count,
# MAGIC
# MAGIC         CASE
# MAGIC             WHEN COUNT(*) <> 1
# MAGIC                 THEN 'ERROR - EXPECTED 1 ROW'
# MAGIC             ELSE 'OK'
# MAGIC         END AS status
# MAGIC
# MAGIC     FROM
# MAGIC         retail_analytics.`3_gold`.vw_executive_kpis
# MAGIC
# MAGIC )
# MAGIC
# MAGIC SELECT
# MAGIC     object_name,
# MAGIC     row_count,
# MAGIC     distinct_key_count,
# MAGIC     status
# MAGIC
# MAGIC FROM
# MAGIC     validation_results
# MAGIC
# MAGIC ORDER BY
# MAGIC     object_name;