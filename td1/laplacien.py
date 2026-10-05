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
