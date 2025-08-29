# Databricks notebook source
# Leer desde Bronze
df_bronze = spark.read.format("delta").load("/Volumes/workspace/workshop_proyecto/bronze/")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inspección inicial (Exploratory Data Check)

# COMMAND ----------

df_bronze.show(5)

# COMMAND ----------

df_bronze.printSchema()

# COMMAND ----------

df_bronze.count()

# COMMAND ----------

# Conteo de nulos por columna
from pyspark.sql.functions import col, sum, when

null_counts = df_bronze.select([
    sum(when(col(c).isNull(), 1).otherwise(0)).alias(c)
    for c in df_bronze.columns
])

null_counts.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Observación:
# MAGIC 1. Columna con muy alta cantidad de nulos (2275): `age_certification`
# MAGIC 2. Columnas con nulos moderados (100–140): `runtime`, `imdb_votes`, `imdb_score`, `imdb_id`
# MAGIC 3. Columnas clave: `id`, `title`, `release_year`
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC Al realizar el conteo de valores nulos por columna, no es suficiente tan solo con ver cuantos nulos hay, es necesario llevar a cabo una **detección de valores atípicos** o **malformateados**, especialmente en columnas numéricas.
# MAGIC Esto porque aunque una celda no esté vacía, puede contener datos **no lógicos** o **mal estructurados** para el tipo de dato esperado y si se realiza un casteo posterior esto puede traer problemas.
# MAGIC
# MAGIC Por ejemplo, al intentar convertir una columna como `imdb_score` a tipo `float`, la operación puede fallar en tiempo de ejecución (al usar `.show()` o `.count()`) si algún valor contiene texto no numérico, como `"N/A"`, `"null"` (como texto, no como valor nulo real), o frases como `"which stood for..."`.
# MAGIC Este tipo de errores ocurre porque `cast()` espera que **todos los valores sean válidos para el tipo objetivo**, y no tolera valores corruptos.
# MAGIC
# MAGIC Para evitar que el flujo se rompa, es recomendable:
# MAGIC
# MAGIC + **Limpiar el formato** de los datos antes del cast (por ejemplo, eliminar comas en números).
# MAGIC
# MAGIC + Usar `try_cast()` en lugar de `cast()`, ya que `try_cast` devuelve null en lugar de lanzar error cuando no puede convertir un valor.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Valores atípicos en las columnas numéricas

# COMMAND ----------

# MAGIC %md
# MAGIC Ver cuáles valores no se pueden convertir   
# MAGIC Hacer esto antes del cast para auditar qué valores fallan al castear.
# MAGIC

# COMMAND ----------

from pyspark.sql.functions import expr, col
from pyspark.sql.functions import length
from pyspark.sql.functions import count, when

# COMMAND ----------

# MAGIC %md
# MAGIC **Para la columna:** `imdb_votes`

# COMMAND ----------

# Crear columna auxiliar con try_cast
df_check = df_bronze.withColumn("imdb_votes_int", expr("try_cast(imdb_votes as int)"))


# COMMAND ----------

# Filtrar los valores no convertibles y ver su longitud para detectar errores ocultos
df_check.filter(col("imdb_votes_int").isNull() & col("imdb_votes").isNotNull()) \
        .select("imdb_votes", length("imdb_votes").alias("num_caracteres")) \
        .distinct() \
        .show(100, False)


# COMMAND ----------

# MAGIC %md
# MAGIC Viendo que los valores como `"21635.0"` tienen decimales innecesarios (como el `.0`) y están siendo tratados como no convertibles a `int`, esto es completamente esperable.
# MAGIC
# MAGIC En PySpark, cuando haces:
# MAGIC `expr("try_cast(imdb_votes as int)")`
# MAGIC espera que el valor sea estrictamente un número entero como string (`"21635"`), no un float (`"21635.0"`). Por eso los valores con `.0` fallan en el `cast` a entero.
# MAGIC
# MAGIC

# COMMAND ----------

# Contar valores válidos vs inválidos
df_check.select(
    count(when(col("imdb_votes_int").isNull() & col("imdb_votes").isNotNull(), True)).alias("no_convertibles"),
    count(when(col("imdb_votes_int").isNotNull(), True)).alias("convertibles")
).show()


# COMMAND ----------

# MAGIC %md
# MAGIC Vale la pena solucionar este detalle viendo la cantidad de no convertibles. 

# COMMAND ----------

# MAGIC %md
# MAGIC **Para la columna:** `imdb_score`

# COMMAND ----------

# Crear columna auxiliar con try_cast
df_check = df_bronze.withColumn("imdb_score_int", expr("try_cast(imdb_score as float)"))


# COMMAND ----------

# Filtrar los valores no convertibles y ver su longitud para detectar errores ocultos
df_check.filter(col("imdb_score_int").isNull() & col("imdb_score").isNotNull()) \
        .select("imdb_score", length("imdb_score").alias("num_caracteres")) \
        .distinct() \
        .show(100, False, vertical=True)


# COMMAND ----------

# MAGIC %md
# MAGIC Se puede observar como la mayoría de valores no convertibles son de valores de otras columnas como `description`, `imdb_id` y uno que otro de `age_certification`.

# COMMAND ----------

# Contar valores válidos vs inválidos
df_check.select(
    count(when(col("imdb_score_int").isNull() & col("imdb_score").isNotNull(), True)).alias("no_convertibles"),
    count(when(col("imdb_score_int").isNotNull(), True)).alias("convertibles")
).show()

# COMMAND ----------

# MAGIC %md
# MAGIC No son tantos no convertibles en comparación de la anterior columna. 

# COMMAND ----------

# MAGIC %md
# MAGIC **Para la columna:** `runtime`

# COMMAND ----------

# Crear columna auxiliar con try_cast
df_check = df_bronze.withColumn("runtime_int", expr("try_cast(runtime as int)"))

# COMMAND ----------

# Filtrar los valores no convertibles y ver su longitud para detectar errores ocultos
df_check.filter(col("runtime_int").isNull() & col("runtime").isNotNull()) \
        .select("runtime", length("runtime").alias("num_caracteres")) \
        .distinct() \
        .show(100, False, vertical=True)

# COMMAND ----------

# MAGIC %md
# MAGIC Se puede observar como la mayoría de valores no convertibles son valores de la columna `description`.

# COMMAND ----------

# Contar valores válidos vs inválidos
df_check.select(
    count(when(col("runtime_int").isNull() & col("runtime").isNotNull(), True)).alias("no_convertibles"),
    count(when(col("runtime_int").isNotNull(), True)).alias("convertibles")
).show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Limpieza de datos (Data Cleaning)

# COMMAND ----------

# MAGIC %md
# MAGIC Estas operaciones corrigen, rellenan, eliminan o normalizan datos corruptos, vacíos o mal formateados:

# COMMAND ----------

# MAGIC %md
# MAGIC ### Para la columna `imdb_votes`

# COMMAND ----------

# MAGIC %md
# MAGIC Solución decimales innecesarios (`.0`)

# COMMAND ----------

#Cast encadenado
df_clean = df_bronze.withColumn(
    "imdb_votes_clean",
    expr("try_cast(try_cast(imdb_votes as float) as int)")
)


# COMMAND ----------

# Ver cuáles no fueron convertibles:
df_clean.filter(
    col("imdb_votes").isNotNull() & col("imdb_votes_clean").isNull()
).select("imdb_votes").distinct().show(100, False)


# COMMAND ----------

# Ver cuántos no fueron convertibles:
from pyspark.sql.functions import col

df_clean.filter(
    col("imdb_votes").isNotNull() & col("imdb_votes_clean").isNull()
).count()


# COMMAND ----------

# Reemplazar la original
df_clean = df_clean.drop("imdb_votes").withColumnRenamed("imdb_votes_clean", "imdb_votes")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Para la columna `age_certification`

# COMMAND ----------

# MAGIC %md
# MAGIC Teniendo la mayor cantidad de nulos (2275)

# COMMAND ----------

# Rellenar valores nulos en age_certification con "NR" (Not Rated)
df_clean = df_clean.fillna({"age_certification": "NR"})

# COMMAND ----------

# MAGIC %md
# MAGIC Además los valores válidos de age_certification tienen hasta 5 caracteres filtrar los registros que exceden ese tamaño para limpiarlo

# COMMAND ----------

from pyspark.sql.functions import col, length, upper

# Regex para detectar números como string (entero o decimal)
regex_number = r"^\d+(\.\d+)?$"

# Filtrar:
# - longitud máxima 5
# - no sea un número
# - no contenga minúsculas (ya que lo convertimos a mayúsculas, esto elimina las que no se pudieron convertir)
df_clean = df_clean.filter(
    (length(col("age_certification")) <= 5) &
    (~col("age_certification").rlike(regex_number)) &
    (col("age_certification") == upper(col("age_certification")))  # solo valores enteramente en mayúsculas
)

# COMMAND ----------

df_clean.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Para columna `type`

# COMMAND ----------

# MAGIC %md
# MAGIC Limpiar la columna type para quedarte solo con los registros que tienen valores válidos como "MOVIE" o "SHOW"

# COMMAND ----------

df_clean.select("type") \
    .filter(~col("type").isin("MOVIE", "SHOW")) \
    .distinct() \
    .show(100, False)


# COMMAND ----------

df_clean = df_clean.filter(col("type").isin("MOVIE", "SHOW"))

# COMMAND ----------

df_clean.count()

# COMMAND ----------

# MAGIC %md
# MAGIC **Para las columnas** `id`, `title`, `release_year`

# COMMAND ----------

# Eliminar filas con nulos en columnas clave
df_clean = df_clean.dropna(subset=["id", "title", "release_year"])

# COMMAND ----------

# Conteo de nulos por columna
from pyspark.sql.functions import col, sum, when

null_counts = df_clean.select([
    sum(when(col(c).isNull(), 1).otherwise(0)).alias(c)
    for c in df_clean.columns
])

null_counts.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Transformación

# COMMAND ----------

# MAGIC %md
# MAGIC **Convertir columnas al tipo de dato adecuado con `try_cast` (estructura)**

# COMMAND ----------

from pyspark.sql.functions import expr

df_silver = (
    df_clean
    .withColumn("release_year", expr("try_cast(release_year as int)"))
    .withColumn("runtime", expr("try_cast(runtime as int)"))
    .withColumn("imdb_score", expr("try_cast(imdb_score as float)"))
    .withColumn("imdb_votes", expr("try_cast(imdb_votes as int)"))
)


# COMMAND ----------

# revisar el esquema para comprobar
df_silver.printSchema()

# COMMAND ----------

# Conteo de nulos por columna
from pyspark.sql.functions import col, sum, when

null_counts = df_silver.select([
    sum(when(col(c).isNull(), 1).otherwise(0)).alias(c)
    for c in df_silver.columns
])

null_counts.show()

# COMMAND ----------

df_silver.count()

# COMMAND ----------

# MAGIC %md
# MAGIC **Limpiar y normalizar texto (título y descripción)**
# MAGIC

# COMMAND ----------

from pyspark.sql.functions import lower, trim

df_silver = (
    df_silver
    .withColumn("title", trim(lower(col("title"))))
    .withColumn("description", trim(lower(col("description"))))
)

# COMMAND ----------

# MAGIC %md
# MAGIC **Calcular duración en formato "horas:minutos"**
# MAGIC

# COMMAND ----------

from pyspark.sql.functions import floor

df_silver = df_silver.withColumn("runtime_hours", floor(col("runtime") / 60))
df_silver = df_silver.withColumn("runtime_minutes", col("runtime") % 60)


# COMMAND ----------

# MAGIC %md
# MAGIC **Agregar campo booleano: ¿Es clásica? (antes del año 2000)**

# COMMAND ----------

df_silver = df_silver.withColumn(
    "is_classic",
    when(col("release_year") < 2000, True).otherwise(False)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Hacía la capa Silver

# COMMAND ----------

#Carpeta silver 
silver_path = '/Volumes/workspace/workshop_proyecto/silver/'

# COMMAND ----------

# Guardar en formato Delta
df_silver.write.format("delta").mode("overwrite").save(silver_path)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC