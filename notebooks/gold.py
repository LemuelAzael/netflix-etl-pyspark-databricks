# Databricks notebook source
#Librerías
from pyspark.sql.functions import col, lower, row_number
from pyspark.sql.window import Window

# COMMAND ----------

# Leer desde Silver
df_silver = spark.read.format("delta").load("/Volumes/workspace/workshop_proyecto/silver/")

# COMMAND ----------

# MAGIC %md
# MAGIC # Capa Gold 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Top 10 películas por puntuación IMDB

# COMMAND ----------

df_top10 = (
    df_silver
    .orderBy(col("imdb_score").desc(), col("imdb_votes").desc())
    .limit(10)
)

# COMMAND ----------

df_top10.show(vertical=True)

# COMMAND ----------

df_top10.write.format("delta").mode("overwrite").save("/Volumes/workspace/workshop_proyecto/gold/df_top10")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Tabla de distribución por age_certification

# COMMAND ----------

df_by_cert = (
    df_silver.groupBy("age_certification")
    .count()
    .withColumnRenamed("count", "num_peliculas")
)
df_by_cert.show()

# COMMAND ----------

df_by_cert.write.format("delta").mode("overwrite").save("/Volumes/workspace/workshop_proyecto/gold/df_by_cert")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Tendencia de lanzamientos por año

# COMMAND ----------

df_by_year = (
    df_silver.groupBy("release_year")
    .count()
    .withColumnRenamed("count", "peliculas_estrenadas")
    .orderBy("release_year")
)

df_by_year.show()

# COMMAND ----------

df_by_year.write.format("delta").mode("overwrite").save("/Volumes/workspace/workshop_proyecto/gold/df_by_year")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Duración promedio por tipo

# COMMAND ----------

df_by_type = (
    df_silver.groupBy("type")
    .agg({"runtime": "avg"})
    .withColumnRenamed("avg(runtime)", "duracion_promedio")
)

df_by_type.show()

# COMMAND ----------

df_by_type.write.format("delta").mode("overwrite").save("/Volumes/workspace/workshop_proyecto/gold/df_by_type")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Agregados por década

# COMMAND ----------

from pyspark.sql.functions import floor

df_by_decade = (
    df_silver
    .withColumn("decade", (col("release_year") / 10).cast("int") * 10)
    .groupBy("decade")
    .agg(
        {"id": "count", "imdb_score": "avg", "imdb_votes": "sum"}
    )
    .withColumnRenamed("count(id)", "num_peliculas")
    .withColumnRenamed("avg(imdb_score)", "promedio_score")
    .withColumnRenamed("sum(imdb_votes)", "total_votos")
)


# COMMAND ----------

df_by_decade.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data enrichment

# COMMAND ----------

# MAGIC %md
# MAGIC ### Clasificación por calidad del score (Excelente, Bueno, Malo)

# COMMAND ----------

from pyspark.sql.functions import when

df_score_category = df_silver.withColumn(
    "score_category",
    when(col("imdb_score") >= 9.0, "Excelente")
    .when((col("imdb_score") >= 6.0) & (col("imdb_score") < 9.0), "Bueno")
    .when(col("imdb_score") < 6.0, "Malo")
    .otherwise("Desconocido")
)

# COMMAND ----------

df_score_category.show(vertical=True)

# COMMAND ----------

df_score_category.write.format("delta").mode("overwrite").save("/Volumes/workspace/workshop_proyecto/gold/df_score_category")

# COMMAND ----------

# MAGIC %md
# MAGIC ### A partir de `score_category`

# COMMAND ----------

# MAGIC %md
# MAGIC **1. Distribución de películas por categoría de score**

# COMMAND ----------

df_score_dist = df_score_category.groupBy("score_category").count()

# COMMAND ----------

df_score_dist.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **2. Promedio de duración y votos por score_category**

# COMMAND ----------

df_score_stats = (
    df_score_category.groupBy("score_category")
    .agg(
        {"runtime": "avg", "imdb_votes": "avg", "*": "count"}
    )
    .withColumnRenamed("avg(runtime)", "duracion_promedio")
    .withColumnRenamed("avg(imdb_votes)", "votos_promedio")
    .withColumnRenamed("count(1)", "total_peliculas")
)

# COMMAND ----------

df_score_stats.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### A partir de la columna `is_classic`

# COMMAND ----------

# MAGIC %md
# MAGIC **Comparativa entre películas clásicas y modernas**

# COMMAND ----------

df_classic_comparison = (
    df_silver.groupBy("is_classic")
    .agg(
        {"imdb_score": "avg", "runtime": "avg", "*": "count"}
    )
    .withColumnRenamed("avg(imdb_score)", "score_promedio")
    .withColumnRenamed("avg(runtime)", "duracion_promedio")
    .withColumnRenamed("count(1)", "total_peliculas")
)

# COMMAND ----------

df_classic_comparison.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **Top 10 películas clásicas mejor valoradas**

# COMMAND ----------

df_top_classics = (
    df_silver.filter(col("is_classic") == True)
    .orderBy(col("imdb_score").desc(), col("imdb_votes").desc())
    .limit(10)
)

# COMMAND ----------

df_top_classics.show(vertical=True)

# COMMAND ----------

df_top_classics.write.format("delta").mode("overwrite").save("/Volumes/workspace/workshop_proyecto/gold/df_top_classics")

# COMMAND ----------

# MAGIC %md
# MAGIC ### A partir de las columna `runtime_hours` y `runtime_minutes`

# COMMAND ----------

# MAGIC %md
# MAGIC **Clasificación por duración: Cortas, Medias, Largas**

# COMMAND ----------

df_gold = df_silver.withColumn(
    "duracion_categoria",
    when(col("runtime") < 60, "Corta")
    .when((col("runtime") >= 60) & (col("runtime") <= 120), "Media")
    .otherwise("Larga")
)

df_runtime_dist = df_gold.groupBy("duracion_categoria").count()


# COMMAND ----------

df_runtime_dist.show()

# COMMAND ----------

df_runtime_dist.write.format("delta").mode("overwrite").save("/Volumes/workspace/workshop_proyecto/gold/df_runtime_dist")

# COMMAND ----------

# MAGIC %md
# MAGIC **Score promedio por tipo de duración**

# COMMAND ----------

df_runtime_score = (
    df_gold.groupBy("duracion_categoria")
    .agg(
        {"imdb_score": "avg", "*": "count"}
    )
    .withColumnRenamed("avg(imdb_score)", "score_promedio")
    .withColumnRenamed("count(1)", "num_peliculas")
)

# COMMAND ----------

df_runtime_score.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Clasificación temática por palabras clave

# COMMAND ----------

# MAGIC %md
# MAGIC A partir de `description`

# COMMAND ----------

df_thematic_key = df_silver.withColumn(
    "tema_principal",
    when(lower(col("description")).rlike("war|soldier|army|veteran|battle"), "Guerra")
    .when(lower(col("description")).rlike("love|romance|relationship|affair"), "Amor")
    .when(lower(col("description")).rlike("crime|murder|killer|police|detective"), "Crimen")
    .when(lower(col("description")).rlike("comedy|funny|humor|laugh|hilarious"), "Comedia")
    .when(lower(col("description")).rlike("horror|ghost|scary|haunted"), "Terror")
    .when(lower(col("description")).rlike("magic|dragon|kingdom|fantasy|sword"), "Fantasía")
    .when(lower(col("description")).rlike("alien|space|robot|future|sci-fi"), "Ciencia ficción")
    .when(lower(col("description")).rlike("family|struggle|poverty|society|life"), "Drama social")
    .when(lower(col("description")).rlike("adventure|journey|quest|explore"), "Aventura")
    .when(lower(col("description")).rlike("fight|explosion|hero|gun"), "Acción")
    .otherwise("Otro")
)

# COMMAND ----------

df_thematic_key.select("tema_principal").show()

# COMMAND ----------

df_thematic_key.write.format("delta").mode("overwrite").save("/Volumes/workspace/workshop_proyecto/gold/df_thematic_key")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Etiquetado por popularidad

# COMMAND ----------

from pyspark.sql.functions import when

df_gold_popular = df_silver.withColumn(
    "popularidad",
    when(col("imdb_votes") > 500000, "Muy Popular")
    .when(col("imdb_votes") > 100000, "Popular")
    .otherwise("Poco Conocida")
)

# COMMAND ----------

df_gold_popular.show(vertical=True)

# COMMAND ----------

df_gold_popular.write.format("delta").mode("overwrite").save("/Volumes/workspace/workshop_proyecto/gold/df_gold_popular")

# COMMAND ----------

