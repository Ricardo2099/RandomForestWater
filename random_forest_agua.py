from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# 1. Cargar el CSV ubicado junto a este programa.
carpeta = Path(__file__).resolve().parent
datos = pd.read_csv(carpeta / "Watera.csv")

# 2. Separar el objetivo de las características numéricas.
# potability: 0 = no potable, 1 = potable.
X = datos.drop(columns=["potability"]).select_dtypes(include="number")
y = datos["potability"]

# 3. Separar entrenamiento y prueba antes de imputar.
# La estratificación conserva la proporción de ambas clases.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)
print("Muestras totales:", len(datos))
print("Datos de entrenamiento:", len(X_train))
print("Datos de prueba:", len(X_test))

# 4. Crear un pipeline que conserva el preprocesamiento junto con el modelo.
# Cada árbol usa muestras y características aleatorias; el bosque combina
# sus resultados para obtener una predicción más estable.
modelo = Pipeline(
    steps=[
        ("imputador", SimpleImputer(strategy="mean")),
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=100,
                max_depth=2,
                random_state=0,
                n_jobs=1,
            ),
        ),
    ]
)
print("Entrenando modelo Random Forest...")
modelo.fit(X_train, y_train)

# 5. Guardar el pipeline completo para que la API aplique el mismo
# preprocesamiento que se utilizó durante el entrenamiento.
ruta_modelo = carpeta / "model" / "random_forest_agua.joblib"
ruta_modelo.parent.mkdir(exist_ok=True)
dump({"modelo": modelo, "features": X.columns.tolist()}, ruta_modelo)
print(f"Modelo guardado en: {ruta_modelo}")

# 6. Predecir la clase y la probabilidad de la clase potable.
predicciones = modelo.predict(X_test)
probabilidades = modelo.predict_proba(X_test)[:, 1]

# 7. Evaluar el modelo como en el ejemplo de clasificación.
print(f"\nAccuracy: {accuracy_score(y_test, predicciones):.4f}")
print("\nReporte de clasificación:")
print(
    classification_report(
        y_test,
        predicciones,
        labels=[0, 1],
        target_names=["No potable", "Potable"],
        zero_division=0,
    )
)

matriz = confusion_matrix(y_test, predicciones, labels=[0, 1])
print("Matriz de confusión (filas: real; columnas: predicción):")
print(
    pd.DataFrame(
        matriz,
        index=["Real 0: no potable", "Real 1: potable"],
        columns=["Predicción 0", "Predicción 1"],
    )
)

# 8. Mostrar algunos resultados concretos y su probabilidad de potabilidad.
ejemplos = pd.DataFrame(
    {
        "real": y_test.head(5).to_numpy(),
        "prediccion": predicciones[:5],
        "probabilidad_potable": probabilidades[:5].round(4),
    },
    index=y_test.head(5).index,
)
print("\nEjemplos de predicción:")
print(ejemplos)
