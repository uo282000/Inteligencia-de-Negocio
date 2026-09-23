# %% Importaciones y lectura
from pathlib import Path

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
# 2. Seleccione X con las columnas que vea razonables.
# 3. Seleccione y = EstimatedSalary.
# 4. Compare regresión lineal, SVR y bosque aleatorio mediante validación
#    cruzada de 10 folds y error cuadrático medio.
# 5. Opcional: prediga Exited mediante tres clasificadores y compare aciertos.
# 6. Opcional: sustituya la eliminación por diferentes métodos de imputación.
# 7. Discuta qué variables influyen en cada predicción. 
