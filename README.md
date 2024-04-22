# 🚗 Descripción del Proyecto 🚗

Este proyecto se centra en el procesamiento y análisis de datos relacionados con despachos de artículos y su maestro, con el objetivo de preparar los datos para su uso en modelos de machine learning. Incluye scripts para la preparación de datos, entrenamiento de modelos y generación de predicciones.

## 📁 Archivos del Proyecto 📁

### 1. `data_preparation.py` 🛠️

Este script se encarga de preparar los datos relacionados con los despachos de artículos y su maestro. Contiene funciones para consultas SQL, limpieza y transformación de datos en DataFrames de Pandas, y combinación de datos de diferentes fuentes.

### 2. `data_processing.py` 🔄

El script `data_processing.py` se encarga de preparar los datos para su uso en modelos de machine learning. Contiene funciones para filtrar datos, dividirlos en conjuntos de entrenamiento y prueba, y crear pipelines de procesamiento de datos para su posterior utilización en el entrenamiento de modelos.

### 3. `model.py` 🤖

Este script se centra en el entrenamiento de modelos de machine learning utilizando diferentes clasificadores. Se encarga de cargar los datos preparados, entrenar los modelos y generar predicciones para un conjunto específico de datos, con el fin de facilitar la toma de decisiones basadas en el análisis de los resultados obtenidos.

## 🚀 Ejecución del Proyecto 🚀

Para ejecutar el proyecto, sigue estos pasos:

1. Asegúrate de tener instaladas todas las dependencias necesarias. Puedes instalarlas utilizando el siguiente comando:
  pip install -r requirements.txt

2. Ejecuta los scripts en el siguiente orden:

   - `data_preparation.py`
   - `data_processing.py`
   - `model.py`

Asegúrate de ajustar los parámetros necesarios en el archivo `model.py` antes de ejecutarlo.

## 📦 Dependencias 📦

El proyecto utiliza las siguientes dependencias:

- `scikit-learn`: Para el preprocesamiento de datos y entrenamiento de modelos de machine learning.
- `pandas`: Para la manipulación y análisis de datos.
- `numpy`: Para operaciones numéricas.
- `sqlalchemy`: Para la conexión y consultas a bases de datos SQL.
- `openpyxl`: Para leer archivos Excel.
- `matplotlib`: Para visualización de datos.

Asegúrate de tener estas dependencias instaladas antes de ejecutar el proyecto.

## ✍️ Autor ✍️

Este proyecto fue desarrollado por [David Gonzalez](https://github.com/DeiviGT1).

