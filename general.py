"""
General algorithm for calculating lattice Green's functions on hypercubic lattices.
Not efficient beyond 3D, but theoretically possible to use.
Description can be found in:
M. Berciu and A. M. Cook, EPL 92, 40003 (2010)
"""

from itertools import combinations_with_replacement as combwr
from functools import lru_cache, reduce
from collections import Counter

from numpy import *
from scipy.linalg import inv

@lru_cache(maxsize=None)
def An(om,t,d,N):
    m = t/om
    idx = combwr(range(N,-1,-1),d)
    D = {i:[] for i in range(N+1)}
    C = Counter()
    R = dict()
    for i in idx:
        s = sum(i)
        C[s] +=1
        if (s <= N):
            D[s].append(i)
            R[i] = (s, C[s]-1)
    An = {N:zeros((len(D[N]),len(D[N-1])),dtype=complex)}
    lmbd = (sqrt(1-(2*d*m)**2)-1)/(2*d*m)
    for i,v in enumerate(array(D[N])):
        for u in abs(v-identity(d)):
            if sum(u)<N:
                p,j = R[tuple(sorted(u,reverse=True))]
                An[N][i,j] += lmbd
    for n in range(N-1,0,-1):
        alpha = zeros((len(D[n]),len(D[n-1])),dtype=complex)
        beta = zeros((len(D[n]),len(D[n+1])),dtype=complex)
        for i,v in enumerate(array(D[n])):
            for u in abs(v-identity(d)):
                p,j = R[tuple(sorted(u,reverse=True))]
                if p<n:
                    alpha[i,j] -= m
                else:
                    beta[i,j] -= m
            for w in v+identity(d):
                l = R[tuple(sorted(w,reverse=True))][1]
                beta[i,l] -= m
        An[n] = dot(inv(identity(len(D[n]))-dot(beta,An[n+1])),alpha)
    An[0] = 1/(om+2*d*t*An[1])
    return An, R


def Gn(om,t,a,N=100):
    a = tuple(sorted(abs(array(a)),reverse=True))
    A,R = An(om,t,len(a),N)
    s,i = R[a]
    G = reduce(dot,(A[j] for j in range(s,-1,-1)))
    return G[i][0]
