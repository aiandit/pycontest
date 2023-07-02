from math import sin, cos, sqrt
from pycontest import *
import numpy as np
import random

def eyeList(l):
    return [ [ l[i] if i==j else 0 for i in range(len(l)) ] for j in range(len(l)) ]

def getDiag(A):
    return [ A[i][i] for i in range(len(A)) ]

def sumAllElements(A):
    return sum(A)

def sumAllElements2(A):
    return sum([ sum(A[i]) for i in range(len(A)) ])

def eLaS(l):
    A = eyeList(l)
    return sumAllElements2(A)

def eLaSDiag(l):
    A = eyeList(l)
    return sum(getDiag(A))

def eAaS(l):
    A = np.diag(l)
    return np.sum(A)

def eAaSDiag(l):
    A = np.diag(l)
    return np.sum(np.diag(A))


def mkInput(N):
    Ns = int(sqrt(N))
    if Ns > 20e3:
        raise ValueError()
    return ([random.random() for i in range(Ns)],),{}

cfs = [eLaS, eLaSDiag, eAaS, eAaSDiag]
print(contest(*cfs, timeout=1e-1, input=mkInput))
