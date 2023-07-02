from math import sin, cos, sqrt
from pycontest import *

def getTest(bfun):

    def inner(l):
        return [ bfun(v) for v in l]

    return inner

contest(getTest(sin), getTest(cos), getTest(sqrt))
