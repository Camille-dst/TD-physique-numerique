
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

    entrée: 
    LuLap = décomposition de A 
    f_x = le vecteur f_X

    return = U

    LULap * U = f_X
    """
    # Résout le système linéaire
    return LULap.solve(f_X)   #L (partie basse de la matrice), U (partie haute de la matrice)

def compute_error(u_num, u_ana):
    return np.max(np.abs(u_num - u_ana))


def plot_solution(L_x=1.0, n_x=100):
    """Calcule et affiche la solution pour un n_x donné (Q5)"""
    X = np.linspace(0, L_x, num=n_x)
    U = np.zeros(n_x)

    LAP = buildLaplacian(L_x, n_x)
    LULap = LUdecomposition(LAP)

    f_X = f(L_x, X[1:-1])
    U[1:-1] = ResoLap(LULap, f_X)

    plt.figure(figsize=(8, 4))
    plt.plot(X, U, label="Solution numérique $u(x)$")
    plt.plot(X, analytical(L_x, X), "--", label="Solution analytique")
    plt.title("Solution de l'équation de Poisson 1D (Q5)")
    plt.xlabel("$x$")
    plt.ylabel("$u$")
    plt.legend()
    plt.grid(True)
    plt.savefig("Poisson_1D_sol.png")
    plt.show()


def plot_convergence(L_x=1.0):
    """Calcule et affiche le graphe de convergence log-log (Q7 & Q8)"""
    # Conversion explicite en entiers pour n_x
    n_x_array = np.logspace(1, 6, num=15, dtype=int)
    epsilon = []
    dx_array = []

    for n_x in n_x_array:  
        dx = L_x / (n_x - 1)
        x_grid = np.linspace(0, L_x, num=n_x)

        lap = buildLaplacian(L_x, n_x)
        lu = LUdecomposition(lap)

        u_sol = np.zeros(n_x)
        u_sol[1:-1] = ResoLap(lu, f(L_x, x_grid[1:-1]))

        u_exact = analytical(L_x, x_grid)
        err = compute_error(u_sol, u_exact)

        epsilon.append(err)
        dx_array.append(dx)

    plt.figure(figsize=(8, 5))
    plt.loglog(dx_array, epsilon, "o-", label=r"Erreur $\epsilon(\Delta x)$")
    plt.loglog(
        dx_array,
        [10 * d**2 for d in dx_array],
        "k--",
        label=r"Pente d'ordre 2 $\mathcal{O}(\Delta x^2)$",
    )
    plt.xlabel(r"Pas spatial $\Delta x$")
    plt.ylabel(r"Erreur $\epsilon$")
    plt.title("Étude de convergence (Q7 & Q8)")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig("Poisson_1D_convergence.png")
    plt.show()


if __name__ == "__main__":
    plot_solution(L_x=1.0, n_x=100)
    plot_convergence(L_x=1.0)