import matplotlib.pyplot as plt
import numpy as np
from oscillateur_van_pol import analytical
from runge_kutta import rk2
from adams_moulton import am2  
from oscillateur_van_pol import f


# time parameters
t_start = 0
t_end = 500
N = 1000
#mu = 1e-12
mu_array = np.logspace(-12,-1,num=20)
    
# time array
times = np.linspace(t_start,t_end,num = N)

tol = 1e-10
max_iter = 10

X0 = np.array([0,1])


err_rk2 = []
err_am2 = []

for m in mu_array:

    f_m = lambda t, X: f(t, X, mu=m)

    sol_rk2 = rk2(f_m, X0, times)
    sol_am2 = am2(f_m, X0, times, tol, max_iter)

    
    xs_ana = analytical(times, mu=m)[:, 1]  # composante position x(t)

    
    e_rk2 = np.max(np.abs(sol_rk2[:, 1] - xs_ana))
    e_am2 = np.max(np.abs(sol_am2[:, 1] - xs_ana))

    err_rk2.append(e_rk2)
    err_am2.append(e_am2)

plt.figure(figsize=(8, 5))
plt.loglog(mu_array, err_rk2, 'o-', label=r'$\epsilon_{RK2}(\mu)$')
plt.loglog(mu_array, err_am2, 's--', label=r'$\epsilon_{AM2}(\mu)$')
plt.xlabel(r'$\mu$')
plt.ylabel(r'Erreur $\epsilon$')
plt.title(r'Erreur en fonction de $\mu$ (Q10)')
plt.savefig('comparaison 2 méthodes avec mu liste.png')
plt.legend()
plt.show()