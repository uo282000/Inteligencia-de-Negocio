# %% Importaciones y lectura
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd

CARPETA = Path(__file__).resolve().parent
df = pd.read_excel(CARPETA / "Churn_Modelling_sintetico_NANs.xlsx")

# %% Primer diagnóstico
print(df.head())
print(df.shape)
print(df.dtypes)
print(df.isna().sum())

# %% Ejercicio 8
# 1. Elimine las filas con valores perdidos.
df_limpio = df.dropna(axis=0, how="any")

# 2. Seleccione X con las columnas que vea razonables.
X = df_limpio.drop(columns=["CustomerId", "Surname", "EstimatedSalary"])
X["Gender"] = X["Gender"].map({"Female": 0, "Male": 1})
X["Geography"] = X["Geography"].map({"France": 0, "Spain": 1, "Germany": 2})

# 3. Seleccione y = EstimatedSalary.
y = df_limpio["EstimatedSalary"]

# 4. Compare regresión lineal, SVR y bosque aleatorio mediante validación
#    cruzada de 10 folds y error cuadrático medio.
modelos = {
    "Lineal": LinearRegression(),
    "SVR": make_pipeline(StandardScaler(), SVR(kernel="rbf", C=100)),
    "Bosque": RandomForestRegressor(n_estimators=100, random_state=42),
}

particiones = KFold(n_splits=10, shuffle=True, random_state=42)

print("\n--- Comparación de modelos (MSE en 10 folds) ---")
for nombre, modelo in modelos.items():
    # scoring="neg_mean_squared_error" devuelve el MSE en negativo
    puntuaciones = cross_val_score(
        modelo, selected_X, y, scoring="neg_mean_squared_error", cv=particiones
    )
    mse_medio = -puntuaciones.mean()
    print(f"{nombre}: {mse_medio:,.2f}")
# 5. Opcional: prediga Exited mediante tres clasificadores y compare aciertos.
# 6. Opcional: sustituya la eliminación por diferentes métodos de imputación.
# 7. Discuta qué variables influyen en cada predicción. 

# %%
