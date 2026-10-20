from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from joblib import load
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


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

