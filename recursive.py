"""
Recursive algorithms for calculating Green's functions on square lattices (works for 0D, 1D, 2D), and their versions for hard-core boson restricted models. 
"""

from functools import lru_cache
from collections import Counter

from numpy import *
from scipy.linalg import inv
#from scipy.special import hyp2f1
from mpmath import hyp2f1,ellipk,ellipe

from propagators.utils import *


def Eps(k,t,d=None):
    '''Free particle dispersion for holes (includes -)'''
    k = array(k)
    if d!=None:
        k = k[:d]
    return 2*t*sum(cos(k))

def G0k(k,omega,t,d=None):
    return 1/(omega-Eps(k,t,d))


@lru_cache(maxsize=None)
def G0(omega,t,a):
    '''2D Lattice Green's functions, as derived by Morita in
    J. Math. Phys. 12, 1744;
    J. Phys. Soc. Jpn. 30, 957'''
    a = -sort(-abs(array(a))) #convert to array and apply symmetries
    s = 2*t
    if abs(t)<1e-6 or len(a)==0:#0D solution for t=0
        if all(a==0):
            return 1/omega
        else:
            return 0j
    elif len(a)==2:
        m = (2*s/omega)**2
        (i,j) = a
        # Basic G0 functions
        if (i==0 and j==0):
            R = (1/omega)*hyp2f1(1/2,1/2,1,m)
        elif (i==1 and j==1):
            R = (1/omega)*m/8*hyp2f1(3/2,3/2,3,m)
            #(hyp2f1(3/2,1/2,2,m)-hyp2f1(1/2,1/2,1,m))
        elif (i==1 and j==0):
            R = (omega*G0(omega,t,(0,0))-1)/2/s
        # Recursive definitions
        elif (i>1 and j==i):
            #R = binom(2*j,j)/2**(4*j+1)/omega*m**j*hyp2f1(j+1/2,j+1/2,2*j+1,m)
            R = (4*(i-1)*(2/m-1)*G0(omega,t,(i-1,i-1))
                  -(2*i-3)*G0(omega,t,(i-2,i-2)))/(2*i-1)
        elif (i>1 and j==i-1):
            R = (omega*G0(omega,t,(j,j))-s*G0(omega,t,(j,j-1)))/s
        else:
            R = (2*omega*G0(omega,t,(i-1,j))/s
                    -(G0(omega,t,(i-2,j))+G0(omega,t,(i-1,j+1))
                        +G0(omega,t,(i-1,j-1))))
        return complex(R)
    elif len(a)==1:
        i = a[0]
        m = omega/s
        xi = -m + sqrt(m-1)*sqrt(m+1)
        return xi**i/(s*sqrt(m-1)*sqrt(m+1))
#### Legacy code for reference
        # if i==0:
        #     return -1j*sign(imag(sqrt((1+m)/(1-m))))/(s*sqrt(1-m**2))
        # if i==1:
        #     return (omega*G0(omega,t,(0,))-1)/s
        # if i>1:
        #     return 2*m*G0(omega,t,(i-1,))-G0(omega,t,(i-2,))
    else:
        raise NotImplementedError("Higher dimensions are not implemented here.")


@lru_cache(maxsize=None)
def Gu(z,t,U,end,start):
    """Free propagator corrected for core hole potential U. Needed for RIXS."""
    dim = len(start)
    centre = (0,)*dim
    z = z - U
    diff = tuple(array(end)-array(start))
    if U==0:
        return G0(z,t,diff)
    else:
        return G0(z,t,diff)-G0(z,t,end)*G0(z,t,start)/(1./U+G0(z,t,centre))


@lru_cache(maxsize=None)
def Gi(sites,end,start,omega,t,cnst=True):
    '''Restricted Green's function for kinetic interaction. Works only for the 2D model with orbitons.'''
    a = array(end)
    b = array(start)
    if cnst==True and len(sites)>0:
        B = array(sorted(sites))
        v = NNlist(B)
        w = coNNlist(B)
        n = len(v)    #number of NN vectors
        g0d = array([[G0(omega,t,tuple(x-y)) for y in v] for x in w])
        g0b = array([[G0(omega,t,tuple(x-b))] for x in w])
        gad = array([[G0(omega,t,tuple(a-x)) for x in v]])
        gab = G0(omega,t,tuple(a-b))
        res = gab-dot(gad,dot(inv(identity(n)+t*g0d),t*g0b))
        return res[0,0]
    else:
        return G0(omega,t,tuple(a-b))


@lru_cache(maxsize=None)
def Gj(sites,end,start,omega,t,J,cnst=True):
    '''Restricted Green's function for kinetic and exchange interaction. Works only for the 2D model with orbitons.'''
    a = array(end)
    b = array(start)
    B = array(sorted(sites))
    v = NNlist(B)
    w = coNNlist(B)
    n = len(v)    #number of NN vectors
    d = len(v[0]) #space dimension
    m = 2*n+2*d   #multiplier for J
    if cnst==True:
        ged = array([[G0(omega-m*J,t,tuple(x-y)) for y in v] for x in v])
        g0d = array([[G0(omega-m*J,t,tuple(x-y)) for y in v] for x in w])
        geb = array([[G0(omega-m*J,t,tuple(x-b))] for x in v])
        g0b = array([[G0(omega-m*J,t,tuple(x-b))] for x in w])
        gad = array([[G0(omega-m*J,t,tuple(a-x)) for x in v]])
        gab = G0(omega-m*J,t,tuple(a-b))
        res = gab-dot(gad,dot(inv(identity(n)+t*g0d+2*J*ged),t*g0b+2*J*geb))
        return res[0,0]
    else:
        return G0(omega-m*J,t,tuple(a-b))
