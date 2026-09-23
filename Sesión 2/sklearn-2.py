# %% Importaciones y datos
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

CARPETA = Path(__file__).resolve().parent
datos = pd.read_csv(CARPETA / "boston.csv")
X = datos.drop(columns="MEDV")
y = datos["MEDV"].to_numpy()

# %% Modelos equivalentes a los de sklearn-1

# SelectKBest se ajusta dentro de cada fold para no utilizar el conjunto de
# prueba al elegir la variable. SVR incluye además el escalado necesario.

modelos = {
    "Lineal": make_pipeline(
        SelectKBest(score_func=f_regression, k=1), 
        LinearRegression()
    ),
    # Código incompleto
    "SVR": make_pipeline(
        SelectKBest(score_func=f_regression, k=2),
        StandardScaler(),
        SVR(...),
    ),
    # Código incompleto
    "Bosque": make_pipeline(
        SelectKBest(score_func=f_regression, k=2),
        RandomForestRegressor(...),
    ),
}

# %% Validación cruzada con las mismas particiones
particiones = KFold(n_splits=10, shuffle=True, random_state=42) #lo mezcla con shuffle
predicciones = {}
filas = []
for nombre, modelo in modelos.items():
    scores = cross_val_score(
        modelo, X, y, scoring="neg_mean_squared_error", cv=particiones

    )
    prediccion = cross_val_predict(modelo, X, y, cv=particiones)
    predicciones[nombre] = prediccion
    filas.append(
        {
            "modelo": nombre,
            "MSE: media de folds": -scores.mean(),
            "MSE: predicciones reunidas": mean_squared_error(y, prediccion),
        }
    )

resultados = pd.DataFrame(filas).set_index("modelo")
print(resultados.round(3))

# %% Predicciones de validación cruzada ordenadas por el precio observado

# Código incompleto
