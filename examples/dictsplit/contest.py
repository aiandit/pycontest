from math import sin, cos, sqrt
from pycontest import *
import numpy as np
import random

def unz1(d):
    names = [n for n in d if 'd_' + n in d]
    dd = {k: d['d_' + k] for k in names}
    d = {k: d[k] for k in names}
    return dd, d

def unz2(d):
    names = list(d.keys())
    dd = {k: d['d_' + k] for k in names[len(names)//2:]}
    d = {k: d[k] for k in names[len(names)//2:]}
    return dd, d

def unz3(d):
    names = list(d.keys())
    values = list(d.values())
    dd = dict(zip(names[0:len(names)//2], values[0:len(names)//2]))
    d = dict(zip(names[len(names)//2:], values[len(names)//2:]))
    return dd, d

def unz4(d):
    items = list(d.items())
    dd = dict(items[0:len(items)//2])
    d = dict(items[len(items)//2:])
    return dd, d

def joind(d):
    return { 'd_' + k: v for k, v in d.items() } | { k: v for k, v in d.items() }

def mkInput(N):
    Ns = int(N/2)

    dd = { f'd_k{i}': i for i in range(Ns) }
    d = { f'k{i}': i for i in range(Ns) }
    return [dd | d], {}

cfs = [unz1, unz2, unz3, unz4, joind]
print(contest(*cfs, timeout=1e-1, input=mkInput))
