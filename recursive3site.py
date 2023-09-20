"""
Green's functions for the 3-site hopping, resulting from higher order exchange interaction terms.
"""

from functools import lru_cache

from numpy import *
from scipy.linalg import inv

from propagators.recursive import G0
from propagators.utils import *


def E3s(k,J):
    return 4*J*(sum(cos(k))**2-1)


def G3sk(k,omega,J):
    return 1/(omega-E3s(k,J))


@lru_cache(maxsize=None)
def G3s(omega,J,a):
    om = sqrt(omega+4*J)
    t = sqrt(J)
    if sum(abs(array(a)))%2==0:
        return G0(om,t,a)/om
    else:
        return 0.0+0.0j


@lru_cache(maxsize=None)
def Gm(sites,end,start,omega,J,cnst=True):
    '''Restricted Green's function for 3-site and exchange interaction.'''
    a = array(end)
    b = array(start)
    bos = array(sorted(sites))
    q = NNlist(bos)
    n = len(q)    #number of NN vectors
    d = len(start) #space dimension
    m = 2*n+2*d   #multiplier for J
    if cnst==True:
        v = neighbor(sites,2) # 1st+2nd neighbourhood of bosons
        va = array(sorted(v))
        vs = {r:i for (i,r) in enumerate(sorted(v))}
        
        V = zeros((len(v),len(v)),dtype=int)
        
        for s in sites:
            nn = sorted([tuple(i) for i in NNlist([s])])
            nnn = sorted([tuple(i) for i in NNlist(NNlist([s])) if tuple(i)!=s])
            for i in nn:
                for j in nn:
                    if i!=j: # subtract 3s hopping going over a boson
                        V[vs[i],vs[j]] += 1
                if i not in sites: # correct exchange for el-bos proximity
                    V[vs[i],vs[i]] += 2
            for i in nnn: # subtract 3s hopping ending/beginning at boson
                V[vs[s],vs[i]] += 1
                if i not in sites: # to avoid double counting for a pair of bosons
                    V[vs[i],vs[s]] += 1
        
        ged = array([[G3s(omega-m*J,J,tuple(x-y)) for y in va] for x in va])
        geb = array([[G3s(omega-m*J,J,tuple(x-b))] for x in va])
        gad = array([[G3s(omega-m*J,J,tuple(a-x)) for x in va]])
        gab = G3s(omega-m*J,J,tuple(a-b))
        
        res = gab -J*gad.dot(inv(identity(len(v))+J*V.dot(ged))).dot(V.dot(geb))
        return res[0,0]
    else:
        return G3s(omega-m*J,J,tuple(a-b))
