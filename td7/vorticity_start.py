import numpy as np
from scipy.fftpack import fft2, ifft2


import matplotlib.pyplot as plt

plt.ion()


def phys_spec(UR, NX, NY, NXSp, NYSp):
    #  Transformation de physique a spectral

    # Transformée de Fourier 2D
    UU = fft2(UR)
    UU_centred = np.fft.fftshift(UU)

    # Filtrage des modes de haute fréquence
    Unew_centred = 1j * np.zeros([NXSp, NYSp])
    Unew_centred = UU_centred[
        int((NX - NXSp) / 2) : int((NX + NXSp) / 2),
        int((NY - NYSp) / 2) : int((NY + NYSp) / 2),
    ]
    Unew = np.fft.fftshift(Unew_centred)
    return Unew


def spec_phys(U, NX, NY, NXSp, NYSp):
    # Transformation de spectral à physique

    UU_centred = 1j * np.zeros((NX, NY))

    # Remplie avec les modes de basse fréquence
    U_centred = np.fft.fftshift(U)
    UU_centred[
        int((NX - NXSp) / 2) : int((NX + NXSp) / 2),
        int((NY - NYSp) / 2) : int((NY + NYSp) / 2),
    ] = U_centred

    UU = np.fft.fftshift(UU_centred)

    # Tranformée de FOurier inverse 2D
    Unew = np.real(ifft2(UU))

    return Unew


# Nombre de Reynolds
Re = 1.0e3

# Resolution dans l'espace réel
NX = 128
NY = 128

# Resolution dans l'espace de Fourier(avec le de-aliasing)
NXSp = 2 * np.int32(NX / 3)
NYSp = 2 * np.int32(NY / 3)

dt = 2e-2  # pas de temps
NT = 100000  # nombre d'iterations en temps
modulo = 50  # modulo pour l'affichage

# Generation de la grille
dx = (2 * np.pi) / NX  # Et non 1/(NX-1) car on a des conditions périodiques
dy = (2 * np.pi) / NY
t = 0.0

x = np.linspace(1, NX + 1, NX + 1)
y = np.linspace(1, NY + 1, NY + 1)
yy, xx = np.meshgrid(y, x)
XP = xx * dx
YP = yy * dy
X = XP[:-1, :-1]
Y = YP[:-1, :-1]

# Initialisation du vortex
w = np.zeros([NX, NY])

g = (np.exp(-((X - np.pi)**2 + (Y - 3*np.pi/4)**2) / 0.2) +
np.exp(-((X - np.pi)**2 + (Y - 5*np.pi/4)**2) / 0.2) -
0.5 * np.exp(-((X - 5*np.pi/4)**2 + (Y - 5*np.pi/4)**2) / 0.2)
)
c = -np.mean(g)
w = g + c


# 2D arrays de vecteur d'onde
# Dans la direction x
kx = 2 * np.pi * np.fft.fftfreq(NXSp, d=2 * np.pi / NXSp)

# Dans la direction y
ky = 2 * np.pi * np.fft.fftfreq(NYSp, d=2 * np.pi / NYSp)

KX, KY = np.meshgrid(kx, ky, indexing="ij")


# Laplacien dans l'espace de Fourier
k2 = KX**2 + KY**2
k2_fixed = k2.copy()

# Laplacien dans l'espace de Fourier "fixé" pour l'équation de Poisson
k2_fixed[0, 0] = 1

w_hat = phys_spec(w, NX, NY, NXSp, NYSp)


for i in range(NT):
    # Laplacien(psi) = -w
    # A compléter (Question 2)
    psi_hat = w_hat/k2_fixed
    psi_hat[0,0] = 0

    # Vitesse à partir de la fonction de Psi
    # A compléter (Question 3)
    u = spec_phys(1j*KY*psi_hat,NX,NY,NXSp,NYSp) #attention a penser à revenir dans l'espace reel
    v = spec_phys(-1j*KX*psi_hat,NX,NY,NXSp,NYSp)   
    # Gradients de vorticité
    # A compléter (Question 4)
    grad_wx = spec_phys(1j*KX*w_hat,NX,NY,NXSp,NYSp)
    grad_wy = spec_phys(1j*KY*w_hat,NX,NY,NXSp,NYSp)

    # Calcul du terme non-linéaire
    # A compléter (Questions 5 et 6)
    NL = u*grad_wx + v*grad_wy
    NL_hat = phys_spec(NL,NX,NY,NXSp,NYSp)

    # Crank-Nicholson
    # A compléter (Question 7)
    w_hat = (-dt*NL_hat + (1 - dt*k2/(2*Re))*w_hat) / (1 + dt*k2/(2*Re))
    if i % modulo == 0:
        # Affichage
        w = spec_phys(w_hat, NX, NY, NXSp, NYSp)
        plt.figure(1)
        plt.set_cmap("jet")
        plt.clf()
        t = i * dt
        plotlabel = f"t = {t:1.5f}"
        plt.title(plotlabel)

        vmin, vmax = w.min(), w.max()

        # Affichage du champs de vorticité
        mesh = plt.pcolormesh(XP, YP, w, shading="flat", vmin=vmin, vmax=vmax)

        # Affichage du champ de vitesse
        # plt.quiver(X, Y, np.transpose(u), np.transpose(v), 4)

        cbar = plt.colorbar(mesh)
        cbar.set_label("Vorticity ω")

        plt.axis("image")
        plt.draw()
        plt.pause(0.001)
