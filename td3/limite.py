import matplotlib.pyplot as plt
import numpy as np
from oscillateur_van_pol import analytical
from adams_moulton import am2  
from oscillateur_van_pol import f
from adams_moulton import energie_mecanique


'''
on regarde la résolutino avec la méthode de adams_moulton 
attention on a pris comme condition initiale
X0 = [[0,1/2]]

'''


mu =0.1
#X0 = np.array([0,1/2])

X0 = []
for i in range(30):
    x0 = np.random.uniform(-3, 3)    
    dx0 = np.random.uniform(-4, 4)    
    X0.append(np.array([dx0, x0]))

# time parameters
t_start = 0
t_end = 100
N = 1000
# time array
times = np.linspace(t_start,t_end,num = N)

tol = 1e-10
max_iter = 10

#x = am2(f,X0,times,tol,max_iter)[:, 1]
#dx = am2(f,X0,times,tol,max_iter)[:, 0]

X = np.zeros((len(times), len(X0)))
lambda t, X: f(t, X, mu=0.1)

plt.figure(figsize=(7, 7))

for i in range(30):

    x_inf = am2(f,X0[i],times,tol,max_iter)[-1, 1] #-1 comme ca on prend le point à t_inf 
    dx_inf = am2(f,X0[i],times,tol,max_iter)[-1, 0]
    

    plt.scatter(x_inf, dx_inf, color='blue')

#plt.plot(times, x , label = 'x')
#plt.plot(times,energie_mecanique(x,dx),label ='energie mécanique')
plt.xlabel(r'x_inf')
plt.ylabel(r'dx_inf(t)')
plt.title(r'graphe de x_inf)')
plt.savefig('graphe de x_inf dans l espace des phases.png')
plt.legend()
plt.show()