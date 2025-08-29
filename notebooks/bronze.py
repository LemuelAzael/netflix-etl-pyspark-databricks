# Databricks notebook source
# Leer CSV
df_bronze = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "false")
    .csv("/Volumes/workspace/workshop_proyecto/raw/netflix-tvshows-movies-raw.csv")
)

# COMMAND ----------

df_bronze.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Hacía la capa bronce

# COMMAND ----------

#Carpeta bronze 
bronze_path = '/Volumes/workspace/workshop_proyecto/bronze/'

# COMMAND ----------

# Guardar en formato Delta
df_bronze.write.format("delta").mode("overwrite").save(bronze_path)