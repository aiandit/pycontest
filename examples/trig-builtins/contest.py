from math import sin, cos, sqrt
from pycontest import *

def getTest(bfun):

    def inner(l):
        return [ bfun(v) for v in l]

    return inner

contest(dict(sin=getTest(sin), cos=getTest(cos), sqrt=getTest(sqrt)), name="Trig. test", outdir='out')
