
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as lg
import matplotlib.pyplot as plt

from td04_Poisson1D_start import buildLaplacian

def f(L_x, X, L_y, Y):
    return ((np.pi / L_x)** 2 + (np.pi / L_y)** 2) * np.sin(np.pi * X / L_x)*np.sin(np.pi * Y / L_y)


def analytical(L_x, X, L_y, Y):
    
    return -np.sin(np.pi * X / L_x)* np.sin(np.pi * Y / L_y)


def buildLaplacian2D(L_x, n_x, L_y , n_y):

    Ax = buildLaplacian(L_x, n_x)  # Taille (m_x, m_x)
    Ay = buildLaplacian(L_y, n_y)  # Taille (m_y, m_y)

    Ix = sp.eye(n_x - 2)
    Iy = sp.eye(n_y - 2)

    # A = Iy x Ax + Ay x Ix
    LAP = sp.kron(Iy, Ax) + sp.kron(Ay, Ix)

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
    return LULap.solve(f_X.ravel())   #L (partie basse de la matrice), U (partie haute de la matrice)

def compute_error_2D(u_num, u_ana):
    return np.max(np.abs(u_num - u_ana))


def plot_solution(L_x=1.0, n_x=100, L_y=1.0, n_y = 100):
    """Calcule et affiche la solution pour un n_x donné (Q5)"""
    X = np.linspace(0, L_x, num=n_x)
    Y = np.linspace(0, L_y, num=n_y)
    X, Y = np.meshgrid(X, Y, indexing='ij')
    U = np.zeros((n_x, n_y))

    LAP = buildLaplacian2D(L_x, n_x, L_y, n_y)
    LULap = LUdecomposition(LAP)

    f_X = f(L_x, X[1:-1, 1:-1], L_y , Y[1:-1, 1:-1])
    U[1:-1, 1:-1] = ResoLap(LULap, f_X).reshape((n_x - 2,n_y - 2))

    plt.pcolormesh(X, Y, U, shading='auto', cmap='viridis')
    plt.colorbar(label='$u(x,y)$')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title('Solution de l\'équation de Poisson 2D (Q12)')
    plt.show()
    plt.savefig("Poisson_2D_sol.png")
    plt.show()


def plot_convergence_2D(L_x=1.0, L_y=1.0):
    # Tableau de taille de grille nx = ny allant de 10 à 100
    n_x_array = np.logspace(1, 2, num=10, dtype=int)
    epsilon_2D = []
    dx_array = []

    for nx in n_x_array:
        ny = nx  # Grille carrée (nx = ny)
        dx = L_x / (nx - 1)

        x = np.linspace(0, L_x, num=nx)
        y = np.linspace(0, L_y, num=ny)
        X, Y = np.meshgrid(x, y, indexing="ij")

        # 1. Matrice Laplacien 2D et résolution
        LAP2D = buildLaplacian2D(L_x, nx, L_y, ny)
        LULap2D = lg.splu(LAP2D.tocsc())

        f_int = f(L_x, X[1:-1, 1:-1], L_y, Y[1:-1, 1:-1])
        u_sol_1D = LULap2D.solve(f_int.ravel())

        # Reconstruction de la solution 2D
        U_num = np.zeros((nx, ny))
        U_num[1:-1, 1:-1] = u_sol_1D.reshape((nx - 2, ny - 2))

        # 2. Solution analytique exacte
        U_ana = analytical(L_x, X, L_y, Y)

        # 3. Calcul de l'erreur
        err = compute_error_2D(U_num, U_ana)

        epsilon_2D.append(err)
        dx_array.append(dx)

    # Tracé log-log de l'erreur en fonction de dx (Q13)
    plt.figure(figsize=(8, 5))
    plt.loglog(dx_array, epsilon_2D, "o-", label=r"Erreur $\epsilon_{2D}(\Delta x)$")

    # Droite de référence de pente 2 (Ordre 2)
    plt.loglog(
        dx_array,
        [5 * d**2 for d in dx_array],
        "k--",
        label=r"Ordre 2 $\mathcal{O}(\Delta x^2)$",
    )

    plt.xlabel(r"Pas spatial $\Delta x$")
    plt.ylabel(r"Erreur $\epsilon$ (Norme $L_\infty$)")
    plt.title("Convergence de l'équation de Poisson 2D (Q13)")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig("Poisson_2D_convergence.png")
    plt.show()


if __name__ == "__main__":
    plot_solution(L_x=1.0, n_x=100, L_y=1.0, n_y = 100)
    plot_convergence_2D(L_x=1.0,  L_y=1.0)
   