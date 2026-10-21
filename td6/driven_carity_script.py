# -*- coding: utf-8 -*-
"""
Affichage d'un écoulement incompressible dans une cavité entrainée.

Phynum M1 2026, Université Paris Cité

"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as lg
import matplotlib.pyplot as plt


def Diffusion_step(u, v):
    # A compléter (Question 1)
    ustar = np.zeros((N_x, N_y))
    vstar = np.zeros((N_x, N_y))

    return [ustar, vstar]


def Laplacian2D(X):
    # Renvoie le Laplacien de X sur [1:-1,1:-1]
    d2X_dx2 = (X[:-2, 1:-1] - 2*X[1:-1, 1:-1] + X[2:, 1:-1]) / dx**2
    d2X_dy2 = (X[1:-1, :-2] - 2*X[1:-1, 1:-1] + X[1:-1, 2:]) / dy**2
    return d2X_dx2 + d2X_dy2


def Divergence(u, v):
    # Renvoie la divergence de (u,v) sur [1:-1,1:-1]
    div = np.zeros((N_x, N_y))
    # A compléter (Question 2)

    return div


def buildLaplacianN(L, N):
    """
    Laplacian matrix 1D for presure with Neumann BC

    """
    n = N - 2
    delta = L / (N - 1)

    # A compléter (Question 3)

    return LAP

def Gradient(p):
    gradp_x = np.zeros((N_x, N_y))
    gradp_y = np.zeros((N_x, N_y))
    # A compléter (Question 5)

    return gradp_x, gradp_y


def buildLaplacianN2D():
    A_x = buildLaplacianN(L_x, N_x)
    A_y = buildLaplacianN(L_y, N_y)
    LAP = sp.kron(A_x, sp.eye(N_y - 2, N_y - 2)) + sp.kron(
        sp.eye(N_x - 2, N_x - 2), A_y
    )
    # Ajout d'une condition de Dirichlet à un point unique (ici [1,2]) pour assurer l'unicité
    LAP0 = buildLaplacianCor2D()
    return LAP + LAP0


def buildLaplacianCor2D():
    n_x = N_x - 2
    n_y = N_y - 2
    B = [np.zeros(n_x * n_y)]
    dy = L_y / (N_y - 1)
    offset = np.array([1])
    B[0][1] = -1 / (dy**2)
    LAP0 = sp.dia_matrix((B, offset), shape=(n_x * n_y, n_x * n_y))
    return LAP0


def LUdecomposition(LAP):
    """
    Renvoie la décomposition LU de la matrice creuse LAP
    """
    return lg.splu(
        LAP.tocsc(),
    )


def ResoLap(LULap, f_X):
    """
    Résout le système linéaire

    LULap * p = f_X
    """
    # Applanit la matrice en vecteur
    f_vector = f_X.ravel()

    # Résout le système linéaire
    p = LULap.solve(f_vector)

    # Retransformes le vecteur en matrice
    return p.reshape(f_X.shape)


def VelocityGhostPoints(u, v):
    """
    Conditions de bord de Dirichlet

    """
    # A compléter (Question 6)
    ### Bas

    ### Haut

    ### Droite

    ### Gauche


def PressureGhostPoints(p):
    """
    Conditions de bord de Von Neumman

    """
    # A compléter (Question 7)
    ### Bas

    ### Haut

    ### Droite

    ### Gauche



L_x = 1.0
L_y = 1.0

N_x = int(30)
N_y = int(32)

dx = L_x / (N_x - 1)
dy = L_y / (N_y - 1)

x = np.linspace(dx / 2, L_x - dx / 2, N_x - 2)
y = np.linspace(dy / 2, L_y - dy / 2, N_y - 2)
[xx, yy] = np.meshgrid(x, y)

dt = 0.0001

t_array = np.arange(0.0, 0.3 + dt, dt)

U = 0.1  # Vitesse d'entrainement du fluide en haut

##### Vitesses initiales
u = np.zeros((N_x, N_y))
v = np.zeros((N_x, N_y))


##### Pression initiale
p = np.zeros((N_x, N_y))

divstar = np.zeros((N_x, N_y))

ustar = np.zeros((N_x, N_y))
vstar = np.zeros((N_x, N_y))


### Matrix construction for projection step
LAPN = buildLaplacianN2D()
LUPN = LUdecomposition(LAPN)


for t in t_array:
    ### Diffusion
    ustar, vstar = Diffusion_step(u, v)
    ### Mise à jour des points fantômes pour la vitesse
    VelocityGhostPoints(ustar, vstar)

    ### Calcul de la divergence de (u^*,v^*)
    divstar = Divergence(ustar, vstar)
    divstar = divstar - np.mean(divstar[1:-1, 1:-1])

    ### Résolution de l'équation de Poisson pour p sur Omega
    p[1:-1, 1:-1] = ResoLap(LUPN, divstar[1:-1, 1:-1] / dt)
    ### Mise à jour des points fantômes pour la pression
    PressureGhostPoints(p)
    ### Calcul du gradient de p
    gradp_x, gradp_y = Gradient(p)

    ### Projection de u^* et v^*
    u = ustar - dt * gradp_x
    v = vstar - dt * gradp_y
    ### Mise à jour des points fantômes pour la vitesse
    VelocityGhostPoints(u, v)

    if (t / dt) % 100 == 0:
        plotlabel = "t = %1.5f" % (t)
        plt.clf()
        plt.title(plotlabel)
        plt.quiver(xx, yy, np.transpose(u[1:-1, 1:-1]), np.transpose(v[1:-1, 1:-1]), 4)
        plt.axis("image")
        plt.draw()
        plt.pause(0.001)
