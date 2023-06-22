def gcd(m,n):
    while m%n!=0:
        oldm = m
        oldn = n
        m = oldn
        n = oldm%oldn
    return n
print(gcd(6,8))

class Fraction:
    def __init__(self, top, bottom):
        common = gcd(top, bottom)
        self.num = top / common
        self.den = bottom / common

    def __str__(self):
        return str(self.num) + "/" + str(self.den)


f1 = Fraction(6, 8)
print(f1)