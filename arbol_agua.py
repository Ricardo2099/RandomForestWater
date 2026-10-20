from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error


# 1. Cargar el CSV ubicado junto a este programa.
carpeta = Path(__file__).resolve().parent
datos = pd.read_csv(carpeta / "Watera.csv")

# 2. Mostrar la cantidad total de muestras.
print("Muestras totales:", len(datos))

# 3. X contiene las nueve características numéricas; y es el objetivo.
# Aunque potability contiene 0 y 1, se predice como valor numérico para
# aplicar MAE, tal como en el ejemplo de regresión de las presentaciones.
X = datos.drop(columns=["potability"]).select_dtypes(include="number")
y = datos["potability"]

# 4. Separar antes de imputar para no usar información de prueba.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)
print("Datos de entrenamiento:", len(X_train))
print("Datos de prueba:", len(X_test))

# 5. Rellenar faltantes con la media de cada columna de entrenamiento.
# En prueba se aplican esas mismas medias, sin volver a calcularlas.
imputador = SimpleImputer(strategy="mean")
X_train_imputado = pd.DataFrame(
    imputador.fit_transform(X_train), columns=X.columns, index=X_train.index
)
X_test_imputado = pd.DataFrame(
    imputador.transform(X_test), columns=X.columns, index=X_test.index
)

# 6. Crear el árbol de regresión. La profundidad se fija antes de evaluar.
modelo_arbol = DecisionTreeRegressor(max_depth=5, random_state=0)

# 7. Aprender a partir de las características y objetivo de entrenamiento.
modelo_arbol.fit(X_train_imputado, y_train)

# 8. Predecir el valor de potabilidad de las muestras de prueba.
predicciones_arbol = modelo_arbol.predict(X_test_imputado)

# 9. Evaluar el árbol con MAE. En este objetivo binario, un valor menor es mejor.
mae_arbol = mean_absolute_error(y_test, predicciones_arbol)
print(f"\nMAE Decision Tree: {mae_arbol:.4f}")
