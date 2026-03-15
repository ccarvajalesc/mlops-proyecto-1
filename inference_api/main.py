from fastapi import FastAPI, Query
from typing import List, Annotated
import pandas as pd
from enum import Enum

from predict import load_model_from_minio, predict_new_data


app = FastAPI(
    title="CoverType Prediction API",
    version="1.0"
)


# ------------------------------------------------------------------------------
# Cargar modelos al iniciar
# ------------------------------------------------------------------------------

tree_model, tree_scaler = load_model_from_minio("models/decision_tree.pkl")
knn_model, knn_scaler = load_model_from_minio("models/knn.pkl")
svm_model, svm_scaler = load_model_from_minio("models/svm.pkl")


models_dict = {
    "TREE": {"model": tree_model, "scaler": tree_scaler},
    "KNN": {"model": knn_model, "scaler": knn_scaler},
    "SVM": {"model": svm_model, "scaler": svm_scaler},
}


class model_class(str, Enum):
    TREE = "TREE"
    KNN = "KNN"
    SVM = "SVM"


@app.post("/predict")

async def predict(
    models: Annotated[List[model_class], Query(...)],

    Elevation: float,
    Aspect: float,
    Slope: float,
    Horizontal_Distance_To_Hydrology: float,
    Vertical_Distance_To_Hydrology: float,
    Horizontal_Distance_To_Roadways: float,
    Hillshade_9am: float,
    Hillshade_Noon: float,
    Hillshade_3pm: float,
    Horizontal_Distance_To_Fire_Points: float,

    Wilderness_Area: str,
    Soil_Type: str
):

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

    response = {}

    for m in models:

        model_name = m.value

        prediction = predict_new_data(
            df,
            models_dict[model_name]["model"],
            models_dict[model_name]["scaler"]
        )

        response[model_name] = prediction.tolist()

    return response