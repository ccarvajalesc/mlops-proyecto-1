"""
CoverType Prediction API
----------------------

Servicio REST construido con FastAPI para predecir el tipo de cobertura
forestal (Cover_Type) a partir de características geográficas.

Características:
- Permite ejecutar múltiples modelos en una sola petición
- Modelos cargados en memoria al iniciar la aplicación
- Validación automática de parámetros mediante Query + Enum
- Preparado para despliegue en Docker

Autor: Taller MLOps
Versión: 1.0.0
"""

from fastapi import FastAPI, Query
import pandas as pd
from typing import List, Annotated
from predict import predict_new_data, load_model
from enum import Enum
import logging


# ------------------------------------------------------------------------------
# Inicialización de la aplicación
# ------------------------------------------------------------------------------

app = FastAPI(
    title="MLOPS CoverType Prediction API",
    description="Servicio para clasificar cobertura forestal",
    version="1.0.0",
)


logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Carga de modelos
# ------------------------------------------------------------------------------

tree_model, tree_encoders, tree_scaler = load_model("models/decision_tree.pkl")
knn_model, knn_encoders, knn_scaler = load_model("models/knn.pkl")
svm_model, svm_encoders, svm_scaler = load_model("models/svm.pkl")

models_dict = {
    "TREE": {"model": tree_model, "encoders": tree_encoders, "scaler": tree_scaler},
    "KNN": {"model": knn_model, "encoders": knn_encoders, "scaler": knn_scaler},
    "SVM": {"model": svm_model, "encoders": svm_encoders, "scaler": svm_scaler},
}


# ------------------------------------------------------------------------------
# Enumeraciones
# ------------------------------------------------------------------------------

class model_class(str, Enum):
    """Modelos disponibles en la API."""
    TREE = "TREE"
    KNN = "KNN"
    SVM = "SVM"


# ------------------------------------------------------------------------------
# Endpoint de predicción
# ------------------------------------------------------------------------------

@app.post(
    "/predict",
    summary="Predecir tipo de cobertura forestal",
    description="Predice Cover_Type utilizando uno o varios modelos de ML"
)

async def predict(
    models: Annotated[
        List[model_class],
        Query(..., description="Modelos a utilizar: TREE, KNN, SVM")
    ],

    Elevation: float = Query(...),
    Aspect: float = Query(...),
    Slope: float = Query(...),
    Horizontal_Distance_To_Hydrology: float = Query(...),
    Vertical_Distance_To_Hydrology: float = Query(...),
    Horizontal_Distance_To_Roadways: float = Query(...),
    Hillshade_9am: float = Query(...),
    Hillshade_Noon: float = Query(...),
    Hillshade_3pm: float = Query(...),
    Horizontal_Distance_To_Fire_Points: float = Query(...),

    Wilderness_Area: str = Query(...),
    Soil_Type: str = Query(...)
):
    """
    Realiza la predicción de tipo de cobertura forestal.

    Permite ejecutar múltiples modelos sobre el mismo registro.

    Returns
    -------
    dict

    Ejemplo:

    {
        "TREE": [2],
        "SVM": [2]
    }
    """

    df = pd.DataFrame([{
        "Elevation": Elevation,
        "Aspect": Aspect,
        "Slope": Slope,
        "Horizontal_Distance_To_Hydrology": Horizontal_Distance_To_Hydrology,
        "Vertical_Distance_To_Hydrology": Vertical_Distance_To_Hydrology,
        "Horizontal_Distance_To_Roadways": Horizontal_Distance_To_Roadways,
        "Hillshade_9am": Hillshade_9am,
        "Hillshade_Noon": Hillshade_Noon,
        "Hillshade_3pm": Hillshade_3pm,
        "Horizontal_Distance_To_Fire_Points": Horizontal_Distance_To_Fire_Points,
        "Wilderness_Area": Wilderness_Area,
        "Soil_Type": Soil_Type
    }])

    logger.info(f"Input recibido: {df.to_json()}")

    response = {}

    for m in models:

        model_name = m.value

        prediction = predict_new_data(
            df,
            models_dict[model_name]["model"],
            models_dict[model_name]["encoders"],
            models_dict[model_name]["scaler"]
        )

        response[model_name] = prediction.tolist()

    logger.info(f"Response enviado: {response}")

    return response