from matplotlib import pyplot as plt
import numpy as np
from oscillateur_van_pol import f


def rk2(f, X0, t):
    '''
    renvoie la solution approché de X(t) 
    de l'équation 6 avec la méthode Runge Kutta

    k1 = f(tn,Xn)
    k2 = f(tn+1,Xn + dtk1)
    Xn+1 = Xn + dt/2 (k1 + k2)

    dt = t[1] - t[0]

    '''
    dt = t[1] - t[0]
    X = np.zeros((len(t), len(X0)))
    X[0] =X0
    for n in range(len(t) - 1): 
        k1 = f(t[n],X[n])
        k2 = f(t[n+1], X[n] + dt * k1)

        X[n+1] = X[n] + (dt/2)* (k1 + k2)

    return X

# time parameters
t_start = 0
t_end = 100
N = 1000
mu = 1e-12
# time array
times = np.linspace(t_start,t_end,num = N)

X0 = np.array([0,1])
X_rk2 = rk2(f, X0, times)

x_rk2 = X_rk2[:, 1]

plt.plot(times,x_rk2)
plt.xlabel('temps')
plt.ylabel('x_rk2')
plt.savefig('runge kutta.png')
plt.show
