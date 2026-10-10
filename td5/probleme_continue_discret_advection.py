import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt


def set_parameters(case="dispersion"):
    """
    Définit les paramètres pour chaque cas du sujet.
    Args:
        case (str): "dispersion" (Section 4), "eigenvalues" (Section 3),
                   "advection_pure" (Section 3.1),
                   "diffusion_pure" (Section 3.1).
    """
    L = 5.0  # Longueur du domaine

    if case == "dispersion":
        # Cas pour la Question 11 de la Section 4 : Étudier u(t=1,x) pour c=1 et ν petit
        c = 1.0
        deltat = 2.0e-5
        NP = 100
        nu = (
            c**2 * deltat / 2
        )  # Diffusion artificielle pour observer la dispersion numérique
    elif case == "eigenvalues":
        # Cas général pour l'analyse des valeurs propres (Section 3)
        c = 25.0
        deltat = 2.0e-3
        NP = 50
        nu = 1.0
    elif case == "advection_pure":
        # Question 7, Section 3.1 : Cas c ≠ 0, ν = 0 (advection pure)
        c = 1.0
        deltat = 2.0e-3
        NP = 50
        nu = 0.0
    elif case == "diffusion_pure":
        # Question 8, Section 3.1 : Cas c = 0, ν > 0 (diffusion pure)
        c = 0.0
        deltat = 2.0e-3
        NP = 50
        nu = 1.0
    else:
        raise ValueError(f"Cas inconnu: {case}")

    deltax = L / (NP - 1)
    N = NP - 1
    return L, c, nu, deltat, NP, deltax, N

def build_advection_diffusion_matrix(c, nu, deltax, N):
    """
    Construit la matrice A pour le schéma en différences finies centré.
    """
    beta = (c/(2*deltax) + nu/deltax**2)
    alpha = (-c/(2*deltax) + nu/deltax**2)
    gamma = nu/deltax**2

    diag_x = [np.ones(beta), -2*np.ones(alpha), np.ones(gamma)]
    offsets = np.array([-1,0,1])

    A = sp.dia_matrix((diag_x, offsets), shape=(N, N))

    # Conditions aux limites périodiques (coins de la matrice)
    A[0, -1] = alpha
    A[-1, 0] = beta

    return A


def compute_eigenvalues(A, c, nu, deltax, N, deltat):
    """
    Calcule les valeurs propres pour les problèmes continu et discret.
    Correspond aux Questions 1 et 2 de la Section 2 (problème discret analytique et continu).
    """
    # Valeurs propres numériques (via la matrice A)
    p_discrete_numerical, _ = np.linalg.eig(A.todense())

    k_discrete = 2 * np.pi * np.arange(N) / (N * deltax)
    p_discrete_analytic =  -1j * c * np.sin(k_discrete * deltax) / deltax + \
                      (2 * nu / deltax**2) * (np.cos(k_discrete * deltax) - 1)

    k_array = np.linspace(0, np.pi / deltax, 1000)
    p_continous =  -1j * c * k_array - nu * k_array**2

    return p_discrete_numerical, p_discrete_analytic, p_continous, k_array

def analyze_stability():
    """
    Analyse la stabilité pour les cas advection pure et diffusion pure.
    """
    print("\n=== Question 7, Section 3.1 : Advection pure (c ≠ 0, ν = 0) ===")
    L, c, nu, deltat, NP, deltax, N = set_parameters(case="advection_pure")
    A = build_advection_diffusion_matrix(c, nu, deltax, N)
    p_discrete_numerical, p_discrete_analytic, p_continous, k_array = (
        compute_eigenvalues(A, c, nu, deltax, N, deltat)
    )
    m_discrete_numerical = p_discrete_numerical * deltat
    m_discrete_analytic = p_discrete_analytic * deltat
    m_continous = p_continous * deltat
    plot_eigenvalues(
        m_discrete_numerical,
        m_discrete_analytic,
        m_continous,
        title="Question 7, Section 3.1 : Advection pure (INSTABLE)",
    )

    print("\n=== Question 8, Section 3.1 : Diffusion pure (c = 0, ν > 0) ===")
    L, c, nu, deltat, NP, deltax, N = set_parameters(case="diffusion_pure")
    A = build_advection_diffusion_matrix(c, nu, deltax, N)
    p_discrete_numerical, p_discrete_analytic, p_continous, k_array = (
        compute_eigenvalues(A, c, nu, deltax, N, deltat)
    )
    m_discrete_numerical = p_discrete_numerical * deltat
    m_discrete_analytic = p_discrete_analytic * deltat
    m_continous = p_continous * deltat
    plot_eigenvalues(
        m_discrete_numerical,
        m_discrete_analytic,
        m_continous,
        title="Question 8, Section 3.1 : Diffusion pure (STABLE si dt ≤ Δx²/(2ν))",
    )

def simulate(L, c, nu, deltat, deltax, N):
    """
    Simule l'équation d'advection-diffusion avec un schéma d'Euler explicite.
    """
    x = np.arange(0, L, deltax)
    x_f = (
        np.arange(-c, L - c, deltax) % L
    )  # Position finale analytique (décalage de -c)
    t = np.arange(0, 1 + deltat, deltat)
    U = np.zeros((len(t), N))
    U[0, 0 : N // 2] = 1.0  # Condition initiale : fonction porte

    A = build_advection_diffusion_matrix(c, nu, deltax, N).todense()


    for i in range(len(t) - 1):
        #U^{n+1} = U^n + dt * A * U^n
        U[i + 1] = U[i] + deltat * np.dot(A, U[i]).A1
        
        

    return x, x_f, t, U

def plot_solution(x, x_f, U):
    """Affiche la solution numérique et analytique (Question 1, Section 4)."""
    plt.figure(figsize=(10, 6))
    plt.plot(x, U[-1, :], label="Solution numérique")
    plt.plot(x_f, U[0, :], "o", markersize=4, label="Solution analytique")
    plt.xlabel("x")
    plt.ylabel("u(x, t)")
    plt.title("Question 11 : Solution à t=1 (dispersion numérique)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_eigenvalues(m_discrete_numerical, m_discrete_analytic, m_continous, title):
    """Affiche les valeurs propres dans le plan complexe."""
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    plt.scatter(
        np.real(m_continous),
        np.imag(m_continous),
        c="g",
        s=10,
        label="Problème continu",
    )
    plt.scatter(
        np.real(m_discrete_analytic),
        np.imag(m_discrete_analytic),
        c="r",
        s=10,
        label="Problème discret analytique",
    )
    plt.plot(
        np.real(m_discrete_numerical),
        np.imag(m_discrete_numerical),
        "+b",
        markersize=8,
        label="Problème discret numérique",
    )

    circle_stable = plt.Circle(
        (-1, 0), 1, alpha=0.2, color="gray", label="Zone de stabilité (Euler explicite)"
    )
    ax.add_patch(circle_stable)
    plt.axis("image")
    plt.xlabel(r"$\Re(\lambda \, dt)$")
    plt.ylabel(r"$\Im(\lambda \, dt)$")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# =============================================================================
# EXÉCUTION PRINCIPALE
# =============================================================================
if __name__ == "__main__":
    # Partie 2 : Problème discret (Section 2)
    # Question 3 (Section 2.1) : Relation de dispersion discrète (à compléter dans compute_eigenvalues)
    # Question 6 (Section 2.2) : Matrice A (à compléter dans build_advection_diffusion_matrix)

    # Partie 3 : Stabilité (Section 3)
    # Questions 7 et 8 (Section 3.1) : Advection pure et diffusion pure
    analyze_stability()

    # Partie 4 : Intégration numérique (Section 4)
    # Question 11 : Simulation de u(t=1,x) pour c=1 et ν petit
    print("\n=== Question 1, Section 4 : Simulation de u(t=1,x) ===")
    L, c, nu, deltat, NP, deltax, N = set_parameters(case="dispersion")
    x, x_f, t, U = simulate(L, c, nu, deltat, deltax, N)
    plot_solution(x, x_f, U)