# Netflix Dataset ETL with Databricks using Medallion Architecture

## Descripción del proyecto

Este proyecto implementa un pipeline ETL completo sobre un dataset de Netflix, utilizando la **arquitectura Medallion** en **Databricks**. El objetivo es transformar datos crudos en información refinada lista para análisis y visualización, pasando por capas de ingesta, limpieza y enriquecimiento.

El proyecto permite explorar preguntas como:

* ¿Cuáles son los contenidos más populares por año, tipo o puntuación?
* ¿Cómo se clasifican las películas y series según duración o calidad del score?
* ¿Qué patrones de popularidad o temática se pueden identificar?

## Fuente de datos

* **Origen:** Kaggle
* **Archivo original:** `netflix-tvshows-movies.csv`
* **Contenido:** Información sobre películas y series (título, tipo, año, duración, puntuación IMDb, etc.)
* **Formato:** CSV plano, sin transformación previa
* **Carga inicial:** Directamente al volumen `raw` del proyecto en Databricks

## Infraestructura y almacenamiento

* **Plataforma:** Databricks Free Edition
* **Catálogo:** Esquema propio creado para el proyecto
* **Volúmenes:** Cada uno representa una capa de la arquitectura Medallion

  * `/Volumes/workspace/workshop_proyecto/raw/` – Datos originales
  * `/Volumes/workspace/workshop_proyecto/bronze/` – Bronze (ingesta mínima)
  * `/Volumes/workspace/workshop_proyecto/silver/` – Silver (limpieza y estandarización)
  * `/Volumes/workspace/workshop_proyecto/gold/` – Gold (enriquecimiento y modelado)

## Procesamiento (ETL)

El pipeline se implementa mediante notebooks independientes para cada capa:

* **Bronze – Ingesta cruda**

  * **Notebook:** `bronze`
  * Lee los datos desde `raw/` y los guarda en formato Delta con tipado mínimo.

* **Silver – Limpieza y estandarización**

  * **Notebook:** `silver`
  * Validación de tipos de datos, limpieza de registros inválidos o duplicados, tratamiento de nulos y transformación de columnas.
  * Resultado almacenado en `/silver/`.

* **Gold – Enriquecimiento y modelado**

  * **Notebook:** `gold`
  * Genera tablas derivadas para análisis: clasificación por score, duración, top 10 por año, tipo y popularidad, etiquetado por temática.
  * Resultado listo para análisis y visualización.

## Orquestación y automatización

* Se configuró un **Job en Databricks** para automatizar la ejecución del pipeline completo.
* Secuencia de tareas:

  1. Ejecutar `etl_bronze`
  2. Ejecutar `etl_silver`
  3. Ejecutar `etl_gold`
* Se puede activar mediante trigger manual o programado, garantizando que los notebooks se ejecuten en orden.

## Tecnologías utilizadas

* Databricks (Free Edition)
* Apache Spark / PySpark
* Delta Lake
* Python (notebooks)

## Resultado esperado

* Datasets refinados y enriquecidos en la capa Gold.
* Información lista para análisis exploratorio, visualización y consultas analíticas.
* Ejemplos de análisis: top 10 de contenidos, clasificación por duración y puntuación, segmentación temática.

## 👤 Autor
Lemuel Azael Carrillo Barrera
Proyecto para portafolio de ingeniería de datos.

## 📚 Desarrollo completo del proyecto

Revisa todo el proceso paso a paso: desde la configuración inicial hasta la ejecución automatizada en AWS.

🔎 [Ver desarrollo completo en Notion](https://www.notion.so/ETL-on-Netflix-Dataset-using-Databricks-and-Medallion-Architecture-2569ec6ab852803c9fecd5297fd2ecb9?source=copy_link)

## 🌐 Portafolio completo

Explora más proyectos y recursos en mis portafolios personales:

🔗 [GitHub – Portafolio](https://github.com/LemuelAzael/LemuelAzael)
🔗 [Notion – Portafolio](https://www.notion.so/Portafolio-Lemuel-Carrillo-2179ec6ab8528029ba54f3bf3363f993?source=copy_link)