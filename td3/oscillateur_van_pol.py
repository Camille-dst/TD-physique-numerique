from matplotlib import pyplot as plt
import numpy as np


F = 0

def f(t,X,mu,F):
    '''
    écriture de l'oscillateur de van de pol
    avec la forme

    dX/dt = AX + B avec X  = [x',x]

    avec A = [[mu(1-x), -1],[1,0]] et B = [F(t),0]

    '''
    A = np.array([[mu*(1-X)**2,-1],[1,0]])
    B = [[F,0]]

    return np.dot(A,X) + B

def analytical (t, mu):
    '''
    on regarde à l'odre 2 le développement de mu
    '''
    xs = np.cos(t) -mu *((1/4)*np.sin(t) - (1/12)*np.sin(3*t))
    xs_point = - np.sin(t) 

    X = np.array([[xs,xs_point]])

    return X

# time parameters
t_start = 0
t_end = 100
N = 1000
# physical parameter
mu = 1e-12
# time array
times = np.linspace(t_start,t_end,num = N)


plt.plot(times,analytical(times,mu)[0][0])
plt.xlabel('temps')
plt.ylabel('xs')
plt.savefig('xs en fonction du temps.png')
plt.show
