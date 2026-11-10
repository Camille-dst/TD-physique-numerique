
# Proba & Python
# 
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
plt.close('all')
##
rng = np.random.default_rng(seed=42)

##--------------
## GCL      
##--------------

class Generator:
    def __init__(self,xo=1,m=1e12-11,a=427419669081):

        self.x = xo
        self.a = int(a)
        self.m = int(m)

    def uniform(self):
        '''
        GCL
        '''
        self.x = (self.a *self.x)%self.m
        return (self.x/self.m)

#Instance de la classe
gcl = Generator()
data = [gcl.uniform() for i in range(10)]
plt.figure()
plt.hist(data,density=True)
plt.savefig('GCL.png')
plt.title('GCL')

#deuxieme instance de classe 
gcl2 = Generator()
data2 = [gcl2.uniform() for i in range(10)]
plt.figure()
plt.hist(data2,density=True)
plt.savefig('GCL2.png')
plt.title('GCL2')

# troisème instance de classe
gcl3 = Generator(xo=10)
data3 = [gcl3.uniform() for i in range(10)]
plt.figure()
plt.hist(data3,density=True)
plt.savefig('GCL3.png')
plt.title('GCL3')

m=10000
glc_unif = Generator(m=m)
data_unif =[glc_unif.uniform() for i in range(m)]
data = [gcl.uniform() for i in range(m)]

plt.figure()
plt.plot(data[:-1], data[1:], '+', label="default")
plt.plot(data_unif[:-1], data_unif[1:], 'o',label ='m=10000')
plt.xlabel('$s_n$')
plt.ylabel('$s_{n+1}$')
plt.legend()
plt.xlim(0,1)
plt.ylim(0,1)
plt.savefig('GCL_compare(m=10000).png')

##-----------------------------------
## lois classiques
##-----------------------------------

# samole size
n = int(1e4)

## loi uniforme 
#def de la loi U(-1,1)

U = st.uniform(loc=-1, scale=2)

## loi exponentielle E(2)

E = st.expon(scale=1/2)

## loi normale N(0,1)
#loc =moyenne et scale = ecart type
N = st.norm(loc = 0, scale = 1) #attention dans la formule de la loi c'est ecart type au carré

U_pdf = U.pdf(n)
U_cdf = U.cdf(n)

E_pdf = E.pdf(n)
E_cdf = E.cdf(n)


N_pdf = N.pdf(n)
N_cdf = N.cdf(n)

lois = {"U(-1,1)": U, "E(2)": E, "N(0,1)": N}

# grille de valeurs de x
x = np.linspace(-4, 4, 1000)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, (nom, loi) in zip(axes, lois.items()):
    echantillon = loi.rvs(size=n). #histogramme
    ax.hist(echantillon, bins=50, density=True, alpha=0.4,
            color="gray", label="histogramme")

    # pdf et cdf
    ax.plot(x, loi.pdf(x), label="PDF")
    ax.plot(x, loi.cdf(x), label="CDF")

    ax.set_title(nom)
    ax.set_xlabel("x")
    ax.legend()

plt.tight_layout()
plt.savefig("pdf_cdf_hist.png", dpi=300)
plt.show()

##------------------------------
## chi2
##------------------------------



k = 5      
Ns = np.logspace(1,6,num=50).astype(int)


plt.show()
