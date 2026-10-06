import numpy as np
import math


class rational_number :
    def __init__(self,numerateur, denominateur):
        self.numerateur= numerateur
        self.denominateur= denominateur
        if denominateur == 0:
            raise ValueError("Le dénominateur ne peut pas être nul.")
        self.reduce()

    def __add__(self,other):
        num = (
            self.numerateur * other.denominateur
            + other.numerateur * self.denominateur
        )
        den = self.denominateur * other.denominateur

        return rational_number(num, den)

    def __sub__(self,other):
        num = (
                self.numerateur * other.denominateur
                - other.numerateur * self.denominateur
            )
        den = self.denominateur * other.denominateur

        return rational_number(num, den)

    def __mul__(self,other):
        num = self.numerateur * other.numerateur
        den = self.denominateur * other.denominateur
        return rational_number(num, den)

    def __neg__(self):
        return rational_number(-self.numerateur, self.denominateur)


    def reduce(self):
        pgcd = math.gcd(self.numerateur, self.denominateur)
        self.numerateur //= pgcd
        self.denominateur //= pgcd

        if self.denominateur < 0 :
            self.numerateur, self.denominateur = - self.numerateur, - self.denominateur
        
    def __floordiv__(self, other):
        num = self.numerateur * other.denominateur
        den =self.denominateur * other.numerateur

        if den ==0 :
            raise ValueError("Le dénominateur ne peut pas être nul.")

        return rational_number(num / den)

    def __truediv__(self, other):
        return self // other

    def __str__(self):
        return f"{self.numerateur}/{self.denominateur}"


nombre1 = rational_number(1,2)
nombre2 = rational_number(1,0)

print(f"Nombre 1 : {nombre1}")
print(f"Nombre 2 : {nombre2}\n")

print(f"Addition : {nombre1} + {nombre2} = {nombre1 + nombre2}")
print(f"Soustraction : {nombre1} - {nombre2} = {nombre1 - nombre2}")
print(f"Multiplication : {nombre1} * {nombre2} = {nombre1 * nombre2}")
print(f"Opposé de n1 : -({nombre1}) = {-nombre1}")
