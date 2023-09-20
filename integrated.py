"""
Funcitions to numerically integrate the Green's functions. Very slow, only useful for comparison and testing of the recursive functions.
"""

from functools import lru_cache

from numpy import *
from scipy.integrate import nquad

from propagators.recursive import G0k


@lru_cache(maxsize=None)
def Go(omega,t,a):
    '''Brute force calculation of the real space Green's funtions by integrating the G0k function.'''
    a = abs(array(a))
    n = len(a)
    F = lambda k: exp(1j*dot(k,a))*G0k(k,omega,t)/(2*pi)**n
    Fr = lambda *k: real(F(k))
    Fi = lambda *k: imag(F(k))
    Ir = nquad(Fr,[[-pi,pi]]*n)
    Ii = nquad(Fi,[[-pi,pi]]*n)
    return Ir[0]+1j*Ii[0],Ir[1]+1j*Ii[1]


@lru_cache(maxsize=None)
def G3so(omega,J,a):
    '''Brute force calculation of the real space Green's funtions by integrating the G3sk function.'''
    a = abs(array(a))
    n = len(a)
    F = lambda k: exp(1j*dot(k,a))*G3sk(k,omega,J)/(2*pi)**n
    Fr = lambda *k: real(F(k))
    Fi = lambda *k: imag(F(k))
    Ir = nquad(Fr,[[-pi,pi]]*n)
    Ii = nquad(Fi,[[-pi,pi]]*n)
    return Ir[0]+1j*Ii[0],Ir[1]+1j*Ii[1]

