from matplotlib import pyplot as plt
import numpy as np


def laplacien(f):
    '''
    calcule le laplacien d'une fonction 2d
    
    f: matrice représentant le champ scalaire
    return : la laplacien

    '''
    lap = np.zeros_like(f)

    # --- Centre : formule standard ---

    lap[1:-1, 1:-1] = (
        f[2:, 1:-1] + f[:-2, 1:-1] + f[1:-1, 2:] + f[1:-1, :-2] - 4 * f[1:-1, 1:-1]
    )

    # --- Bords : 3 voisins disponibles ---

    lap[0, 1:-1] = f[1, 1:-1] + f[0, 2:] + f[0, :-2] - 3 * f[0, 1:-1]
    lap[-1, 1:-1] = f[-2, 1:-1] + f[-1, 2:] + f[-1, :-2] - 3 * f[-1, 1:-1]
    lap[1:-1, 0] = f[1:-1, 1] + f[1:-1, 0] + f[2:, 0] - 3 * f[1:-1, 0]
    lap[1:-1, -1] = f[1:-1, -2] + f[2:, -1] + f[:-2, -1] - 3 * f[1:-1, -1]

    # --- Coins : 2 voisins disponibles ---
    lap[0, 0] = 0
    lap[0, -1] = 0
    lap[-1, 0] = 0
    lap[-1, -1] = 0

    return lap

f = np.array ([[1,2,3],[4,5,6],[7,8,9]])

print('laplacien avec la fonction=',laplacien(f))

dy_np, dx_np = np.gradient(f)
_, d2x = np.gradient(dx_np)
d2y, _ = np.gradient(dy_np)
lap_np = d2x + d2y
print("\nLaplacien via np.gradient x2 :\n", lap_np)


# au centre les deux méthodes fonnent le même resulats
# mais attention il faut faire attention sur les bords car en fonction de ce que l'on choisis c'est différents

print("\n--- Détection de contours ---")
image_path = 'td01_image.jpg'
try:
    image = plt.imread(image_path)
    if image.ndim == 3:
        image = image.mean(axis=2)  # RGB → niveaux de gris
    image = image.astype(float) / 255.0
except FileNotFoundError:
    raise FileNotFoundError(f"Image non trouvée : {image_path}")

dy, dx = np.gradient(image)  

norme_gradient = np.sqrt(dx**2 + dy**2)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title("Image originale")
plt.subplot(1, 2, 2)
plt.imshow(norme_gradient, cmap='gray')
plt.title("Norme du gradient")
plt.show()
