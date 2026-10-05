from matplotlib import pyplot as plt
import numpy as np

def gradient2d(f):
    '''
    calcule le gradient d'une fonction 2d

    f: matrice représenatnt un champ scalaire 2d
    return: deux matrice représentant le gradient de f dans les direction x et y

    '''

    f = f.astype(float)
    df_dx = np.zeros_like(f)
    df_dy = np.zeros_like(f)

    # Axe x (colonnes)
    df_dx[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / 2.0
    df_dx[:, 0] = f[:, 1] - f[:, 0]
    df_dx[:, -1] = f[:, -1] - f[:, -2]

    # Axe y (lignes)
    df_dy[1:-1, :] = (f[2:, :] - f[:-2, :]) / 2.0
    df_dy[0, :] = f[1, :] - f[0, :]
    df_dy[-1, :] = f[-1, :] - f[-2, :]

    return df_dx, df_dy

f = np.array ([[1,2,3],[4,5,6],[7,8,9]])

print('gradient avec la fonction=',gradient2d(f))
print('verification=',np.gradient(f))

