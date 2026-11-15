#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# Monte Carlo
#
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import scipy.integrate as si
from scipy.special import gamma
##
rng = np.random.default_rng(seed=42)
#
plt.close('all')

##-------------------
## th. central limite
##-------------------

law = st.chi2(df=5)
#law = st.uniform()

# Q.4
k = 5
L = st.chi2(df=k)
mu = L.mean()
sigma = L.std()
n_values = [1,  10, 100]   

N = int(1e3)

Zn = {}
for n in n_values:
    X = L.rvs(size=(N, n))                       # N échantillons de taille n
    Xn_barre = X.mean(axis=1)                    # N moyennes empiriques
    Zn[n] = np.sqrt(n) * (Xn_barre - mu) / sigma

x = np.linspace(-4, 4, 500)
loi_normale = st.norm(loc=0, scale=1)

fig, axes = plt.subplots(1, len(n_values), figsize=(15, 4), sharey=True)

#on met les 3 graphes sur la même figure
for ax, n in zip(axes, n_values):
    ax.hist(Zn[n], bins=30, density=True, alpha=0.5, color="gray",
            label=r"histogramme de $Z_n$")
    ax.plot(x, loi_normale.pdf(x), "r", lw=2, label=r"$\mathcal{N}(0,1)$")
    # loi_normale.pdf donne la densité de N(0,1) sur la grille x
    ax.set_title(f"n = {n}")
    ax.set_xlabel(r"$Z_n$")
    ax.legend()

axes[0].set_ylabel("densité de probabilité")
plt.tight_layout()
plt.savefig("theoreme_central_limite.png", dpi=300)
plt.show()



##-------------------------
## illustration limite chi2
##-------------------------

# sample size
N = int(1e4)
# chi2 degrees of freedom
k_arr = (5, 500)

fig,ax = plt.subplots(1,2, figsize=(10,5))
for i,k in enumerate(k_arr):
    ## sample of size N
    X = st.chi2(df=k).rvs(size=N)
    mu = k
    sigma = (2*k)**0.5
    dist_norm = st.norm(loc=mu, scale=sigma)
    x = np.linspace(X.min(),X.max(),num=100)

    ax[i].hist(X, density=True,alpha=0.6, edgecolor='k',bins='auto')
    ax[i].plot(x, dist_norm.pdf(x),label=r'$\mathcal{N}(k,2k)$', color='r')
    ax[i].plot(x, st.chi2(df=k).pdf(x), label=f'pdf', color='b')
    ax[i].set_xlabel('x')
    ax[i].set_title('k={}'.format(k))
    ax[i].legend()

ax[0].set_ylabel('Density')
plt.savefig('fig09limite_chi2.png')
plt.show()

##---------------
## Calcul de Pi
##---------------
# sample size
N = int(1e4)
## 
X = rng.uniform(low=0.0, high=1.0, size=N)
Y = rng.uniform(low=0.0, high=1.0, size=N)

ok = (X**2 + Y**2) <= 1.0

plt.figure()
plt.plot(X[ok],Y[ok],marker='.',alpha=0.5, color='g',ls='none')
plt.plot(X[~ok],Y[~ok],marker='.',alpha=0.5,    color='r',ls='none')
plt.xlim(0,1)
plt.ylim(0,1)
plt.xlabel('X')
plt.ylabel('Y')
plt.gca().set_aspect('equal')
plt.title('Estimation de $\pi$ par Monte Carlo')
plt.savefig('valeur_aleaoire_cercle.png')
plt.show()


def estimate_pi(N):
    X,Y = rng.uniform(low=0.0, high=1.0, size=(2,N))
    ok = (X**2 + Y**2) <= 1.0
    n_ok = np.sum(ok)
    pi = 4 * n_ok / N
    return pi


N_arr = np.logspace(1, 6, 20).astype(int)
R = 100

pi_arr = np.zeros_like(N_arr, dtype=float)
error_arr = np.zeros_like(N_arr, dtype=float)

for i,n in enumerate(N_arr):
    pi_m = [estimate_pi(n) for k in range(R)]
    pi_arr[i] = pi_m[0] # one realisation
    error_arr[i] = np.std(pi_m) # standard deviation
    
plt.figure()
plt.plot(N_arr,pi_arr, ls='',color='k', marker='x')
plt.axhline(np.pi,label='$\pi$')
plt.xscale('log')
plt.xlabel('$N$')
plt.legend()
plt.ylabel('Estimate')
plt.savefig('fig09pi_estimation.png')
plt.show()

plt.figure()
plt.plot(N_arr, error_arr, ls='', color='k', marker='x')
plt.plot(N_arr, 1/N_arr**0.5, 'k:', label=r'$1/\sqrt{N}$')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.xlabel('$N$')
plt.ylabel('Standard deviation')
plt.savefig('fig09pi_error.png')
plt.show()



##-----------------------------
## deterministe vs stochastique
##-----------------------------    

def f(x):
    return 4*(1 - x**2)**0.5

def method_rectangle_left(f, N, a=0, b=1):
    h = (b - a) / N
    x = a + h*np.arange(N)  
    return h * np.sum(f(x))

def method_mc(f,N):
    U = rng.uniform(size=N)
    out = 1/N * np.sum(f(U))
    return out
    

err_rect = np.zeros_like(N_arr, dtype=float)
err_mc = np.zeros_like(N_arr, dtype=float)

for i,n in enumerate(N_arr):
    pass
    err_rect[i] = np.abs(method_rectangle_left(f,n) - np.pi)
    err_mc[i] = np.sqrt(np.mean([(method_mc(f,n) - np.pi)**2 for k in range(R)]))
plt.figure()
plt.loglog(N_arr, err_rect, 'x', color='b',label='Rectangle')
plt.loglog(N_arr, err_mc, 'x', color='k',label='MC')
plt.legend()
plt.xlabel('$N$')
plt.ylabel('Error')
plt.title('Deterministic vs Stochastic')
plt.show()



##-----------------------------
## Grande dimension
##-----------------------------
def volume_exact(D):
    return np.pi**(D/2) / gamma(D/2 + 1)

def estimate_volume(N,D):
    X = rng.uniform(low=-1.0, high=1.0, size=(N,D)) 
    ok = np.sum(X**2, axis=1) <= 1
    n_ok = np.sum(ok)
    V= 2**D * n_ok/N
    return V


def estimate_volume_grid(N,D): 
    M = round(N**(1./D))
    x = np.linspace(-1, 1, M)
    axes = [x] * D
    grids = np.meshgrid(*axes, indexing='ij',sparse=True)
    R2 = sum(g**2 for g in grids)
    ok = R2 <= 1    
    return 2**D/M**D * np.sum(ok) 

D = 5

M_arr = np.arange(5,20,2)
N_arr = M_arr**D
print(N_arr)
errors_grid = np.zeros_like(N_arr, dtype=float)
errors_mc = np.zeros_like(N_arr, dtype=float)

for i,n in enumerate(N_arr):
    errors_mc[i] = np.abs(estimate_volume(n,D) - volume_exact(D))  
    errors_grid[i] = np.abs(estimate_volume_grid(n,D) - volume_exact(D))  
    
idfix = -1
plt.figure()
plt.loglog(N_arr, errors_mc, 'x',color='k', label='MC')
plt.loglog(N_arr, errors_mc[idfix] * (N_arr/N_arr[idfix])**-0.5, ls=':', color='k',label=r'$\propto N^{-1/2}$')
plt.loglog(N_arr, errors_grid, 'x', color='b',label='Grid')
plt.loglog(N_arr, errors_grid[idfix] * (N_arr/N_arr[idfix])**(-1./D), ls='-.', color='b',label=r'$\propto N^{-1/D}$')
plt.legend()
plt.xlabel('$N$ ')
plt.ylabel('Error')
plt.title('$D={}$'.format(D))
plt.savefig('fig09volume.png')
plt.show()

        