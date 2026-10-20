from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


# 1. Cargar el CSV ubicado junto a este programa.
carpeta = Path(__file__).resolve().parent
datos = pd.read_csv(carpeta / "Watera.csv")

# 2. X contiene las nueve características numéricas; y es el objetivo.
X = datos.drop(columns=["potability"]).select_dtypes(include="number")
y = datos["potability"]

# 3. Separar antes de imputar para no usar información de prueba.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)
print("Muestras totales:", len(datos))
print("Datos de entrenamiento:", len(X_train))
print("Datos de prueba:", len(X_test))

# 4. Calcular las medias solo con entrenamiento y aplicarlas a ambos conjuntos.
imputador = SimpleImputer(strategy="mean")
X_train_imputado = pd.DataFrame(
    imputador.fit_transform(X_train), columns=X.columns, index=X_train.index
)
X_test_imputado = pd.DataFrame(
    imputador.transform(X_test), columns=X.columns, index=X_test.index
)

# 5. Crear y entrenar el modelo XGBoost de regresión.
modelo = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=100,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=0,
    n_jobs=1,
)
modelo.fit(X_train_imputado, y_train)

# 6. Predecir y evaluar exclusivamente este modelo con MAE.
predicciones = modelo.predict(X_test_imputado)
mae = mean_absolute_error(y_test, predicciones)
print(f"MAE XGBoost: {mae:.4f}")
