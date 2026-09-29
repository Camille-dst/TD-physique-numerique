# -*- coding: utf-8 -*-
"""
Résolution de l'équation de Poisson 1D par l'inversion de l'opérateur Laplacien

Phynum M1 2025, Université Paris Cité

"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as lg
import matplotlib.pyplot as plt


def f(L_x, X):
    return (np.pi / L_x) ** 2 * np.sin(np.pi * X / L_x)


def analytical(L_x, X):
    
    return -np.sin(np.pi * X / L_x)


def buildLaplacian(L_x, n_x):

    # dimension de la matrice
    m_x = n_x - 2
    # calcul du delta x
    dx = L_x / (n_x - 1)

    diag_x = [np.ones(m_x), -2*np.ones(m_x), np.ones(m_x)]
    offset = np.array([-1,0,1])
    LAP = sp.dia_matrix((diag_x,offset), shape = (m_x,m_x))/(dx**2)

    return LAP


def LUdecomposition(LAP):
    """
    Renvoie la décomposition LU de la matrice creuse LAP
    """
    return lg.splu(LAP.tocsc())   #splu retourne la decompositiob en LU


def ResoLap(LULap, f_X):
    """
    Résout le système linéaire

    LULap * U = f_X
    """
    # Résout le système linéaire
    return LULap.solve(f_X)   #L (partie basse de la matrice), U (partie haute de la matrice)


if __name__ == "__main__":
    # Domaine
    L_x = 1
    # Nombre de points
    n_x = 100

    X = np.linspace(0, L_x, num=n_x)

    print("Domaine : [0,{}]".format(L_x))
    print("n_x = {}".format(n_x))
    print("dx  = {:.3e}".format(L_x / (n_x - 1)))

    # Allocation du tableau solution
    U = np.empty((n_x,))

    
    # Conditions aux limites
    U[0] = U[-1] = 0


    # Construction du Laplacien
    LAP = buildLaplacian(L_x, n_x)



    #print("LAPoisson shape = {}".format(LAPoisson.shape))

    # Résolution à l'intérieur du domaine

    X_int = X[1:-1]
    f_X = f(L_x, X_int)

    # decomposition
    LULap = LUdecomposition(LAP)

    # Inversion du Laplacien

    U[1:-1] = ResoLap(LULap, f_X)

    # Tracé de la solution (TODO: à compléter)
    plt.figure(figsize=(10, 4))
    plt.plot(X,U)
    plt.title("Solution de l'équation de Poisson 1D")
    plt.xlabel("$x$")
    plt.ylabel("$u$")
    #plt.legend()
    plt.grid()
    plt.show()
    plt.savefig('Poisson_1D_sol.png')


    n_x_array = np.logspace(1, 6,num=10)
    epsilon = []
    for n_x in n_x_array:
        pass
        # TODO: A compléter
        
    plt.figure(figsize=(10, 4.5))

    plt.xlabel(r'$n_x$')
    plt.ylabel(r'$\epsilon(n_x)$')
    plt.legend()
    plt.grid()
    plt.tight_layout()   
    # plt.savefig('Poisson_1D_convergence.png')