from numba import njit
from pocs.Math.simple_func import delta 

@njit
def protein_exp(i, j, pmax, mmax, alpha, beta, gamma):
    
    m = i // (pmax + 1)
    p = i % (pmax + 1)
    
    mp = j // (pmax + 1)
    pp = j % (pmax + 1)
    
    Mij =   alpha * (      delta(m-1, mp)*delta(p, pp) - delta(m, mp)*delta(p, pp) ) 
    Mij +=  beta * gamma * m * (  delta(m, mp)*delta(p-1, pp) - delta(m, mp)*delta(p, pp) )
    Mij +=  gamma * (  (m+1)*delta(m+1, mp)*delta(p, pp) - m*delta(m,mp)*delta(p,pp) )    
    Mij +=  (p+1)*delta(m, mp)*delta(p+1, pp) - p*delta(m,mp)*delta(p,pp)


    return Mij 

