# %% Imports
import matplotlib.pyplot as plt
import numpy as np
from scipy.datasets import ascent
from scipy.stats import norm

imagen = ascent().astype(float)

# Si no se consigue hacer el pip de pooch en el PC de clase
# import pickle
# from pathlib import Path
#
# CARPETA = Path(__file__).resolve().parent
# with (CARPETA / "ascent.dat").open("rb") as archivo:
#     imagen = np.array(pickle.load(archivo), dtype=float)

ruido = norm.rvs(loc=0, scale=16, size=imagen.shape, random_state=42)
imagen_ruido = np.clip(imagen + ruido, 0, 255)

ruido2 = norm.rvs(loc=0, scale=64, size=imagen.shape, random_state=42)
imagen_ruido2 = np.clip(imagen + ruido2, 0, 255)

# %% Celda 1
fig, ejes = plt.subplots(1, 3)
ejes[0].imshow(imagen, cmap="gray", vmin=0, vmax=255)
ejes[0].set_title("Original")
ejes[1].imshow(imagen_ruido, cmap="gray", vmin=0, vmax=255)
ejes[1].set_title("Ruido: desviación 16")
ejes[2].imshow(imagen_ruido2, cmap="gray", vmin=0, vmax=255)
ejes[2].set_title("Ruido: desviación 64")
for eje in ejes:
    eje.axis("off")
plt.tight_layout()
plt.show()

# %%
