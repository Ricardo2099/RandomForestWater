from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from joblib import load
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
import httpx
from typing import Literal
from fastapi import HTTPException

carpeta = Path(__file__).resolve().parent
ruta_modelo = carpeta / "model" / "random_forest_agua.joblib"
paquete = load(ruta_modelo)
modelo = paquete["modelo"]
features = paquete["features"]

app = FastAPI(title="Predicción de potabilidad del agua")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class InputData(BaseModel):
    ph: float
    hardness: float
    tds: float
    chlorine: float
    sulfate: float
    conductivity: float
    organic_carbon: float
    trihalomethanes: float
    turbidity: float


class OutputData(BaseModel):
    prediccion: int
    probabilidad_potable: float


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/score", response_model=OutputData)
def score(data: InputData) -> OutputData:
    valores = data.model_dump()
    entrada = pd.DataFrame([[valores[nombre] for nombre in features]], columns=features)
    prediccion = int(modelo.predict(entrada)[0])
    probabilidad = float(modelo.predict_proba(entrada)[0, 1])
    return OutputData(
        prediccion=prediccion,
        probabilidad_potable=probabilidad,
    )


# Solicitud para reentrenar el modelo
class WorkflowInput(BaseModel):
    porcentaje: Literal[25, 50, 100]


@app.post("/workflow")
def ejecutar_workflow(data: WorkflowInput):

    # Obtener la URL del dataset seleccionado
    dataset_url = os.getenv(f"DATASET_URL_{data.porcentaje}")

    # Token de GitHub almacenado en Render
    github_token = os.getenv("GITHUB_TOKEN")

    if not dataset_url or not github_token:
        raise HTTPException(
            status_code=500,
            detail="Faltan variables de entorno"
        )

    # Repositorio de GitHub
    url_github = (
        "https://api.github.com/repos/"
        "Ricardo2099/RandomForestWater/dispatches"
    )

    # Datos que recibirá GitHub Actions
    payload = {
        "event_type": "ml_ci_cd",
        "client_payload": {
            "dataseturl": dataset_url,
            "sha": f"agua-{data.porcentaje}"
        }
    }

    # Enviar solicitud a GitHub
    try:
        respuesta = httpx.post(
            url_github,
            headers={
                "Authorization": f"Bearer {github_token}",
                "Accept": "application/vnd.github+json"
            },
            json=payload,
            timeout=20
        )
        respuesta.raise_for_status()

    except httpx.HTTPError:
        raise HTTPException(
            status_code=502,
            detail="No se pudo activar GitHub Actions"
        )

    return {
        "mensaje": "Workflow solicitado correctamente",
        "porcentaje": data.porcentaje,
        "sha": f"agua-{data.porcentaje}"
    }

