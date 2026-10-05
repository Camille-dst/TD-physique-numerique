#EXO1 problème de base

from timeit import default_timer as timer
from matplotlib import pyplot as plt
import numpy as np



def sum_base1(N):
    start = timer()
    '''
    calcul la somme des entiers de 1 à N

    :param N: entier positif
    :return: somme des entiers de 1 à N
    '''

    somme=0
    for i in range(1,N+1):
        somme += 1/i**2
    end = timer()
    time_spent = end - start
    return somme , time_spent


def sum_base_numpy(N):
    start = timer()
    N= np.arange(1,N+1)
    somme = np.sum(1/N**2)
    end = timer()
    time_spent = end - start
    return somme, time_spent

      



N_values = np.logspace(1,8, num=8, base=10, dtype=int)

temps_exec = []
for n in N_values:
    _, t = sum_base1(n)
    temps_exec.append(t)

temps_exec2 = []
for n in N_values:
    _, t = sum_base_numpy(n)
    temps_exec2.append(t)


plt.plot(N_values,temps_exec)
plt.plot(N_values, temps_exec2)
plt.xlabel('N')
plt.ylabel('Temps d\'exécution (s)')
plt.savefig('comparaison_temps_execution.png')
plt.show()


