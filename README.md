# Proyecto MLOps — Pipeline Covertype

Este proyecto implementa un pipeline completo de MLOps para el entrenamiento, almacenamiento y consumo de modelos de Machine Learning utilizando el dataset **Covertype**.

La arquitectura está basada en microservicios desplegados con **Docker Compose**, e integra orquestación con Airflow, almacenamiento de modelos en MinIO, entrenamiento desde Jupyter y un API de inferencia.

---

## 🧠 Flujo General del Sistema

### ⏱️ Airflow — Orquestación del pipeline

* **Puerto:** `8080`
* **Credenciales:** `airflow / airflow`

Airflow ejecuta un DAG programado **cada 5 minutos**.

El flujo es el siguiente:

1. Consulta si existen **nuevos datos disponibles para entrenamiento**.

2. Si encuentra datos nuevos:

   * Los inserta en la base de datos **MySQL** en la tabla:

     ```
     covertype_raw
     ```

     Esta tabla contiene los datos **en estado crudo (raw)**.

3. Posteriormente, toma los datos desde `covertype_raw`, aplica el **preprocesamiento** y los inserta en:

   ```
   covertype_processed
   ```

Este enfoque permite mantener:

* trazabilidad de datos crudos
* separación clara entre ingestión y transformación
* atomicidad académica del pipeline

---

### 🪣 MinIO — Object Storage de Modelos

* **Puerto consola:** `9001`
* **Credenciales:** `admin / supersecret`

Antes de entrenar un modelo, es necesario **crear el bucket** donde se almacenarán los artefactos.

Los modelos entrenados (por ejemplo `.pkl`, métricas, resultados) se almacenan en este bucket.

---

### 📓 Jupyter — Entrenamiento de Modelos

* **Puerto:** `8888`
* **Token:** `minio123`

Desde Jupyter se puede:

* Entrenar modelos utilizando los datos procesados
* Crear el bucket en MinIO (si aún no existe)
* Subir modelos entrenados al bucket
* Evaluar métricas y comparar algoritmos

Este servicio funciona como entorno de experimentación y desarrollo.

---

### 🚀 API de Inferencia

* **Puerto:** `8001`

Permite:

* Probar el modelo enviando parámetros de entrada
* Configurar el **bucket desde donde se cargarán los modelos**
* Obtener predicciones en tiempo real

---

## 🗄️ Base de Datos MySQL

El sistema utiliza dos tablas principales:

### `covertype_raw`

Contiene los datos tal como llegan desde la API:

* Sin transformación
* Con UUID para control de procesamiento
* Sirve como staging layer

### `covertype_processed`

Contiene los datos:

* Limpios
* Con variables categóricas codificadas (OneHotEncoder)
* Listos para entrenamiento

---

## 🔁 Frecuencia del Pipeline

El pipeline está configurado para ejecutarse:

```
Cada 5 minutos
```

Esto permite:

* Simular ingestión continua de datos
* Entrenamiento batch académico
* Validar arquitectura MLOps realista

---

## 📦 Servicios del Proyecto

| Servicio      | Puerto | Descripción                     |
| ------------- | ------ | ------------------------------- |
| Airflow       | 8080   | Orquestación del pipeline       |
| MinIO         | 9001   | Almacenamiento de modelos       |
| Jupyter       | 8888   | Entrenamiento y experimentación |
| API Inference | 8001   | Servicio de predicción          |
| MySQL         | 3306   | Persistencia de datos           |

---

## ✅ Consideraciones

* El bucket debe existir antes de subir modelos
* Airflow solo procesa datos **nuevos**
* Los modelos pueden versionarse en MinIO
* El sistema está diseñado con enfoque académico pero siguiendo prácticas reales de MLOps
* La api de inferencia solo funciona, cuando ya hay modelos subidos en el bucket.

---

## 👨‍💻 Autores

Proyecto desarrollado como parte del curso de **MLOps**.
