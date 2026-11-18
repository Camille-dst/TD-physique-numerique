#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# Generation de lois
#
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import scipy.integrate as si
#
plt.close('all')
#### 
### numpy random generator
rng = np.random.default_rng(seed=54)

#######
## Q.3
#######
def Exponential(N,l):
    U = rng.uniform(low=0.0, high=1.0, size=N)
    pass


N = int(10_000)
l=10

#sample_exp = Exponential(N,l)
## check

plt.figure()
#plt.hist(sample_exp, density=True,bins="rice")

plt.xlabel('x')
plt.ylabel('Density')

#######
## Q.4
#######
print(10*'-')
print('Check ecart')
print(10*'-')


###############
#Q9 Box Muller  

def BoxMuller(N):
    pass

#X, Y = BoxMuller(N)


####################
# rejection sampling
####################


######
# Q.11
######
def rejection_sampling(p, xmin, xmax, pmax, N=1000):
    """ Rejection sampling algorithm
    """
    samples = []
    pass

######
# Q.12
######
def pdf_triangle(x, a=0, b=1, c=0.5):
    """ Probability density function of a triangle distribution
    """
    if x<c:
        f = 2*(x - a)/((b - a)*(c - a))
    else:
        f = 2*(b - x)/((b - a)*(b - c))
    return f



#########
# Q. 15
#########
def inverse_G(u):
    pass

def methode_rejet(N=10000):
    pass

########
# Q. 16
#########
#samples_rejet = methode_rejet()


plt.figure()
#plt.hist(samples_rejet, bins='rice', density=True, label='Rejection Sampling',histtype='stepfilled')

