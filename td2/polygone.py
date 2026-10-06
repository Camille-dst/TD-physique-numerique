import numpy as np


class Polygone:
    '''
    l: tuple de la longeur des cotés du polygone
    '''
    def __init__(self, cote):
        self.nb_cote = len(cote) # attribut d'instance
        self.perimetre = sum(cote) # attribut d'instance
        self.aire = 0.0 # attribut d'instance

    def __str__(self):
        return (f"{self.__class__.__name__} ({self.nb_cote} côtés) : "
            f"périmètre = {self.perimetre}, aire = {self.aire:.2f}")

    def __eq__(self, autre):
        if isinstance(autre, Polygone):
            return self.nb_cote == autre.nb_cote and self.perimetre == autre.perimetre and self.aire == autre.aire
        return False
    

    def calcul_aire(self):
        pass

class Rectangle(Polygone):

    def __init__(self, longeur, largeur):
        super().__init__([longeur, largeur, longeur, largeur])
        self.longeur, self.largeur = longeur, largeur
        self.calcul_aire()

    def calcul_aire(self):
        self.aire = self.longeur * self.largeur

class Carre(Rectangle):

    def __init__(self, cote):
        super().__init__(cote, cote)

class hexagone_regulier(Polygone):

    def __init__(self, cote):
         super().__init__([cote]*6)
         self.taille_cote = cote
         self.calcul_aire()

    def calcul_aire(self):
        self.aire = (3*np.sqrt(3)/2)* self.taille_cote**2

def tris_par_aire(polys):  #polys est une liste de polynome
    return sorted(polys, key=lambda p: p.aire)

ma_liste = [
    Rectangle(10, 5),
    Carre(4),
    hexagone_regulier(3),
]

print("--- Liste des formes ---")
for p in ma_liste:
    print(p)

print("\n--- Formes triées par aire croissante ---")
for p in tris_par_aire(ma_liste):
    print(p)