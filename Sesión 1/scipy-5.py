import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import minkowski, chebyshev, euclidean, cityblock

rejilla = np.meshgrid(
    np.linspace(-1.1, 1.1, 512),
    np.linspace(-1.1, 1.1, 512),
    indexing="ij",
)
X, Y = rejilla

def en_bola_minkowski(x, y, p):
    return minkowski([x, y], [0, 0], p) <= 1

def bola_minkowski(p):
    return np.vectorize(en_bola_minkowski)(X, Y, p)

def en_bola_chebyshev(x, y):
    return chebyshev([x, y], [0, 0]) <= 1

def bola_chebyshev():
    return np.vectorize(en_bola_chebyshev)(X, Y)

def en_bola_euclidea(x, y):
    return euclidean([x, y], [0, 0]) <= 1

def bola_euclidea():
    return np.vectorize(en_bola_euclidea)(X, Y)

def en_bola_manhattan(x, y):
    return cityblock([x, y], [0, 0]) <= 1 # cityblock es el nombre técnico de Manhattan en SciPy

def bola_manhattan():
    return np.vectorize(en_bola_manhattan)(X, Y)

fig, ejes = plt.subplots(2, 3, figsize=(10, 6))

# Fila 1: Minkowski 1, 2 y 4
ejes[0, 0].imshow(bola_minkowski(1), extent=[-1.1, 1.1, -1.1, 1.1], origin="lower", cmap="Blues")
ejes[0, 0].set_title("Minkowski (p=1)")

ejes[0, 1].imshow(bola_minkowski(2), extent=[-1.1, 1.1, -1.1, 1.1], origin="lower", cmap="Blues")
ejes[0, 1].set_title("Minkowski (p=2)")

ejes[0, 2].imshow(bola_minkowski(4), extent=[-1.1, 1.1, -1.1, 1.1], origin="lower", cmap="Blues")
ejes[0, 2].set_title("Minkowski (p=4)")

# Fila 2: Manhattan, Euclídea y Chebyshev
ejes[1, 0].imshow(bola_manhattan(), extent=[-1.1, 1.1, -1.1, 1.1], origin="lower", cmap="Oranges")
ejes[1, 0].set_title("Manhattan")

ejes[1, 1].imshow(bola_euclidea(), extent=[-1.1, 1.1, -1.1, 1.1], origin="lower", cmap="Oranges")
ejes[1, 1].set_title("Euclídea")

ejes[1, 2].imshow(bola_chebyshev(), extent=[-1.1, 1.1, -1.1, 1.1], origin="lower", cmap="Oranges")
ejes[1, 2].set_title("Chebyshev")

# Ajustes de formato para ocultar ejes
for ax in ejes.flat:
    ax.set_aspect("equal")
    ax.axis("off")

plt.tight_layout()
plt.show()