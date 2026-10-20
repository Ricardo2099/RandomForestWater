# Modelos de predicción para potabilidad del agua

Ejemplo educativo con un árbol de regresión y XGBoost.

## Ejecutar en PowerShell

Desde la carpeta del proyecto:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe arbol_agua.py
.\.venv\Scripts\python.exe xgboost_agua.py
.\.venv\Scripts\python.exe random_forest_agua.py
```

Cada programa es independiente: `arbol_agua.py` entrena el árbol de decisión, `xgboost_agua.py` entrena XGBoost y `random_forest_agua.py` entrena Random Forest. Los dos primeros imprimen su propio MAE; Random Forest imprime métricas de clasificación y guarda el pipeline entrenado en `model/random_forest_agua.joblib`.

## Preparar la API de Random Forest

Primero entrena y guarda el modelo:

```powershell
.\.venv\Scripts\python.exe random_forest_agua.py
```

Después inicia la API:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app:app --reload
```

La documentación interactiva queda disponible en `http://127.0.0.1:8000/docs`. En otra terminal, con el servidor activo, ejecuta el cliente:

```powershell
.\.venv\Scripts\python.exe client.py
```

La API también tiene `GET /health` para comprobar que está disponible y `POST /score` para predecir potabilidad. El cliente usa una muestra real del CSV y devuelve la clase predicha y la probabilidad estimada de potabilidad.

## Crear imagen Docker para Random Forest

El `Dockerfile` contiene únicamente la API y el modelo guardado de Random Forest; no incluye los scripts de Decision Tree ni XGBoost. Primero genera el modelo:

```powershell
.\.venv\Scripts\python.exe random_forest_agua.py
```

Comprueba que Docker esté iniciado y construye la imagen. Sustituye `TU_USUARIO` por tu usuario de Docker Hub:

```powershell
docker build -t TU_USUARIO/random-forest-agua:latest .
```

Prueba localmente el contenedor:

```powershell
docker run --rm -p 8000:8000 -e PORT=8000 TU_USUARIO/random-forest-agua:latest
```

En otra terminal puedes comprobarlo:

```powershell
Invoke-RestMethod http://localhost:8000/health
.\.venv\Scripts\python.exe client.py
```

Inicia sesión, publica la imagen y verifica que esté disponible en Docker Hub:

```powershell
docker login
docker push TU_USUARIO/random-forest-agua:latest
docker pull TU_USUARIO/random-forest-agua:latest
```

En Render, crea un servicio de tipo **Existing Image** o **Private Service** según la configuración disponible, indica la imagen `TU_USUARIO/random-forest-agua:latest` y configura el puerto `8000`. El contenedor usa automáticamente la variable `PORT` de Render y escucha en `0.0.0.0`. Después podrás probar `/health`, `/docs` y `/score` con la URL pública.

## Relación con las presentaciones

Se revisaron `tecnologias_w3.pptx`, `tecnologias_w4.pptx` y `tecnologias_w5.pptx` de Downloads. W3 presenta `DecisionTreeRegressor`, separación de datos y MAE; W4 explica `.fit()`, `.predict()` y el efecto de la complejidad del árbol; W5 utiliza `SimpleImputer` y pipelines; W6 presenta Random Forest para clasificación. Aquí se fija `max_depth=5` en el árbol de regresión y `max_depth=2` con 100 árboles en Random Forest, sin búsqueda de parámetros. XGBoost y Random Forest están en programas separados.

W5, diapositivas 6 y 9, separa 80/20 y utiliza `SimpleImputer`: `fit_transform` en entrenamiento y `transform` en validación. Aquí se sigue ese ejemplo con `strategy="mean"` explícito. Cada faltante se sustituye por la media de su columna **calculada solo con entrenamiento**, para evitar fuga de información. Las columnas con faltantes son `ph`, `tds`, `sulfate` y `conductivity`; no se eliminan esas muestras.

La adaptación principal es que `potability` contiene 0 y 1. Para seguir el enfoque de MAE, `DecisionTreeRegressor` y `XGBRegressor` la predicen como valor numérico. Para seguir el ejemplo de W6, `RandomForestClassifier` la predice directamente como clase y se evalúa con accuracy, reporte de clasificación y matriz de confusión. El pipeline de Random Forest incluye la imputación y se guarda para que la API use exactamente el mismo preprocesamiento. Se preparan los archivos de API y cliente, pero el deployment se deja al usuario.

## Explicación para clase

- **X:** las nueve características numéricas reales: `ph`, `hardness`, `tds`, `chlorine`, `sulfate`, `conductivity`, `organic_carbon`, `trihalomethanes` y `turbidity`.
- **y:** `potability`, con 0 = no potable y 1 = potable. Se trata como objetivo numérico para calcular MAE.
- **Entrenamiento:** 80,000 filas seleccionadas aleatoriamente con `random_state=0`. Cada programa usa sus características y objetivos para entrenar su propio modelo. Random Forest usa estratificación para conservar la proporción de clases. También se usan para calcular las medias de imputación.
- **Prueba:** las otras 20,000 filas. Sus características se usan para predecir; sus etiquetas solo se usan para evaluar. No intervienen en el entrenamiento ni en el cálculo de medias.
- **`.fit()`:** aprende relaciones entre las características y el valor de potabilidad.
- **`.predict()`:** aplica lo aprendido a las características de prueba y devuelve un valor predicho por muestra.

## Interpretar los resultados

MAE (Mean Absolute Error) es el promedio de la diferencia absoluta entre el valor real y el predicho:
`MAE = promedio(|real - predicho|)`. Un valor menor indica predicciones más cercanas. Aunque el objetivo solo contiene 0 y 1, los regresores pueden producir valores decimales; el script muestra también una clase con umbral 0.5 para facilitar la interpretación.

Los scripts no generan imágenes PNG. La visualización del árbol no forma parte de este flujo porque no aparece como salida requerida en los ejemplos utilizados.

## Resultado de la ejecución verificada

Los resultados exactos de MAE se obtienen ejecutando los scripts del árbol y XGBoost. Random Forest se interpreta con sus métricas de clasificación: accuracy mide aciertos totales; precision, recall y F1 detallan el desempeño de cada clase; la matriz muestra reales frente a predicciones. El script también muestra la probabilidad estimada de que algunas muestras sean potables.
