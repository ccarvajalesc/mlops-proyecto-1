import pandas as pd
import pickle


# ------------------------------------------------------------------------------
# Columnas esperadas en el modelo
# ------------------------------------------------------------------------------

EXPECTED_NUM_COLS = [
    "Elevation",
    "Aspect",
    "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points"
]
"""
Lista de columnas numéricas esperadas por el modelo.
"""


EXPECTED_CAT_COLS = [
    "Soil_Type",
    "Wilderness_Area"
]
"""
Columnas categóricas que requieren OneHotEncoding.
"""


ALL_EXPECTED_COLS = EXPECTED_NUM_COLS + EXPECTED_CAT_COLS
"""
Lista completa de columnas esperadas en los datos de entrada.
"""


# ------------------------------------------------------------------------------
# Cargar modelo
# ------------------------------------------------------------------------------

def load_model(filename):
    """
    Carga un modelo entrenado junto con sus componentes de preprocesamiento.

    Args:
        filename (str): Ruta al archivo pickle del modelo.

    Returns
    -------
    tuple
        model : modelo entrenado
        encoders : diccionario de encoders
        scaler : scaler utilizado durante entrenamiento (puede ser None)

    Notes
    -----
    El archivo debe contener un diccionario con estructura:

    {
        "model": model,
        "encoders": encoders,
        "scaler": scaler
    }
    """

    with open(filename, "rb") as f:
        payload = pickle.load(f)

    return payload["model"], payload.get("encoders"), payload.get("scaler")


# ------------------------------------------------------------------------------
# Predicción sobre nuevos datos
# ------------------------------------------------------------------------------

def predict_new_data(df_new, model, encoders, scaler=None):
    """
    Realiza predicciones para nuevos datos utilizando un modelo entrenado.

    Aplica exactamente el mismo pipeline de preprocesamiento utilizado
    durante el entrenamiento:

    - Limpieza de valores faltantes
    - OneHotEncoding para variables categóricas
    - Concatenación de variables numéricas y categóricas
    - Escalado opcional
    - Predicción final

    Parameters
    ----------
    df_new : pandas.DataFrame
        DataFrame con los nuevos datos para inferencia.

    model :
        Modelo entrenado (DecisionTree, SVM, KNN, etc.).

    encoders : dict
        Diccionario con el OneHotEncoder bajo la clave `"onehot"`.

    scaler : object, optional
        Scaler utilizado durante el entrenamiento (StandardScaler u otro).

    Returns
    -------
    numpy.ndarray
        Predicciones generadas por el modelo.
    """

    df_new = df_new.copy()

    num_cols = EXPECTED_NUM_COLS
    cat_cols = EXPECTED_CAT_COLS

    # --------------------------------------------------------------------------
    # Verificar columnas esperadas
    # --------------------------------------------------------------------------

    missing_cols = set(ALL_EXPECTED_COLS) - set(df_new.columns)

    if missing_cols:
        raise ValueError(f"Faltan columnas en la entrada: {missing_cols}")

    # --------------------------------------------------------------------------
    # Limpieza de datos
    # --------------------------------------------------------------------------

    df_new[num_cols] = df_new[num_cols].fillna(df_new[num_cols].median())

    for col in cat_cols:
        df_new[col] = df_new[col].fillna("Unknown")

    # --------------------------------------------------------------------------
    # Aplicar OneHotEncoder
    # --------------------------------------------------------------------------

    ohe = encoders["onehot"]

    X_cat = ohe.transform(df_new[cat_cols])

    cat_feature_names = ohe.get_feature_names_out(cat_cols)

    X_cat_df = pd.DataFrame(
        X_cat,
        columns=cat_feature_names,
        index=df_new.index
    )

    # --------------------------------------------------------------------------
    # Concatenar variables numéricas + categóricas
    # --------------------------------------------------------------------------

    X_final = pd.concat([df_new[num_cols], X_cat_df], axis=1)

    # --------------------------------------------------------------------------
    # Ordenar columnas igual que en entrenamiento
    # --------------------------------------------------------------------------

    ordered_cols = list(num_cols) + list(cat_feature_names)

    X_final = X_final[ordered_cols]

    # --------------------------------------------------------------------------
    # Escalado (si aplica)
    # --------------------------------------------------------------------------

    if scaler is not None:

        X_scaled = scaler.transform(X_final)

        X_final = pd.DataFrame(
            X_scaled,
            columns=X_final.columns,
            index=X_final.index
        )

    # --------------------------------------------------------------------------
    # Predicción
    # --------------------------------------------------------------------------

    predictions = model.predict(X_final)

    return predictions