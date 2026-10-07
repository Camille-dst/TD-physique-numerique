from matplotlib import pyplot as plt
import numpy as np
from oscillateur_van_pol import f
from runge_kutta import rk2


def am2(f,X0,t,tol,max_iter):
    '''
    renvoie la solution approché de X(t) 
        de l'équation 6 avec la méthode adams moulton 

    Paramètres : 
    tol  = precicion
    max_iter = nombre de fois max de la boucle
    '''

    X = np.zeros((len(t), len(X0)))
    X[0] =X0
    dt = t[1] - t[0]
    N = len(t)

    for n in range(N - 1):
        X_pred = X[n].copy()

        for _ in range(max_iter): 

            X_new = X[n] + (dt/2)* (f(t[n], X[n]) + f(t[n + 1], X_pred))

            if np.linalg.norm(X_new - X_pred) < tol :
                break
            else : 
                X_pred = X_new.copy()
        X[n+1] = X_new

    return X

def energie_mecanique(x,x_point):
    '''
    calcul de l'energie mécanique afin de comparer les deux solutions

    Pour am2, on doit avoir une energie mecanique constante
    Pour runge kutta , on doit avoir un energie mecanique croissante 
    '''
    return (1/2)* (x**2 + x_point**2)


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


x_rk2 = rk2(f, X0, times)[:, 1]
dx_rk2 = rk2(f, X0, times)[:, 0]
x_am2 = am2(f,X0,times,tol,max_iter)[:,1]
dx_am2 = am2(f,X0,times,tol,max_iter)[:,0]

plt.plot(times, energie_mecanique(x_rk2,dx_rk2),label = 'méthode runge-kutta' )
plt.plot(times, energie_mecanique(x_am2,dx_am2), label ='méthode adams moulton')
plt.xlabel('times')
plt.ylabel('energie_mecanique')
plt.legend()
plt.savefig('comparaison 2 méthodes.png')
plt.show()

    