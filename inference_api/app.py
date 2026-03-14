from fastapi import FastAPI, Query, Request,  Response
import pandas as pd
from typing import List, Annotated
from predict import predict_new_data, load_model
from enum import Enum
import logging
from logging.handlers import RotatingFileHandler
import os
import time 
from fastapi.responses import JSONResponse





# ------------------------------------------------------------------------------
# Inicialización de la aplicación
# ------------------------------------------------------------------------------

app = FastAPI(
    title="Taller 1 MLOPS: Penguin Prediction API",
    description="Servicio para clasificar pingüinos según características físicas",
    version="1.0.0",
)
