import io
import sys
import time
import random
import math

from math import sin, asin, atan2, log, sqrt

import matplotlib.pyplot as plt
import numpy as np
from itertools import chain

from .value import Value
from .table import Table

class Timer:
    def __init__(self, name, restarget=None, print=True, tsrc=None):
        self.t0 = 0
        self.t1 = 0
        self.run = False
        self.name = name
        self.result = lambda t: restarget.report(dict(t=t, name=self.name))
        self.tsrc = time.time if tsrc is None else tsrc
        self.print = print

    def __enter__(self):
        self.run = True
        self.t0 = self.tsrc()

    def gettime(self):
        return Value((self.t1 if not self.run else self.tsrc()) - self.t0, 's')

    def __exit__(self, *args):
        self.t1 = self.tsrc()
        self.run = False
        self.result(self.gettime())
        if self.print:
            print(f'Timing {self.name}: {self.gettime()}')


class Vecprop:
    def __init__(self, l):
        v = [ e() for e in l ]
        unit = l[0].unit
        s = sum(v)
        self.s = Value(s, unit)
        self.n = len(v)
        avg = s / self.n
        self.mean = Value(avg, unit)
        self.std = Value(sqrt(sum([(x - avg)**2 for x in v])) / (self.n - 1), unit)


def getTestIteration(bfun):
    def inner(l):
        return [ bfun(v) for v in l]
    return inner


class Benchmark:

    ncool = 2
    nwarm = 5
    nrun = 10
    results = []
    name = 'test'
    func = None

    def __init__(self, print=True, **kw):
        self.print = print
        self.vectorize = kw.get('vectorize', '')

    def report(self, res):
        self.results += [res]

    def bench(self, func, *arg, **kw):
        self.func = func
        self.results = []
        N = self.nwarm + self.nrun + self.ncool
        res = [0] * N

        if self.vectorize:
            #print(f'Create vector iteratin function for {func.__name__}')
            func = getTestIteration(func)

        for i in range(N):
            with Timer(func.__qualname__, self, print=False) as t:
                res[i] = func(*arg, **kw)

        self.rwarm = self.results[0:self.nwarm]
        self.rrun = self.results[self.nwarm:self.nwarm+self.nrun]
        self.rcool = self.results[self.nwarm+self.nrun:]
#        print('bench results', self.results)
        self.vp = Vecprop([ v['t'] for v in self.results ])
        self.vpw = Vecprop([ v['t'] for v in self.rwarm ])
        self.vpr = Vecprop([ v['t'] for v in self.rrun ])
        self.vpc = Vecprop([ v['t'] for v in self.rcool ])
        if self.print:
            print(f'Benchmark {self.name} results: '
                  f'run=(mean={self.vpr.mean}, std={self.vpr.std}), '
                  f'warm=(mean={self.vpw.mean}, std={self.vpw.std}), '
                  f'cool=(mean={self.vpc.mean}, std={self.vpc.std})'
                  )

    def plot(self):
        # plot
        fig, ax = plt.subplots()
        N = self.nwarm + self.nrun + self.ncool
        x = [i+1 for i in range(N)]

        adisp, aunit = self.vpr.mean.getpair()
        plsc = adisp / self.vpr.mean()

        y = [ plsc * v['t']() for v in self.results ]
        h1 = ax.stem(x, y)
        ax.set(xlim=(1, N+1), xticks=np.arange(1, N+1))
        ax.legend([h1], ['Results'])
        #plt.yscale('log')
        plt.ylabel(f'Time ({aunit})')
        plt.xlabel('Run')
        plt.show()


def displArgs(*args, **kw):
    print('Positional args:')
    for i, a in enumerate(args):
        print(f'[{i}]: type={type(a).__name__}, len={len(a)}')
    if len(kw):
        print('Keyword args:')
        for i, (k, v) in enumerate(kw.items()):
            print(f'{k} = type={type(v).__name__}, len={len(v)}')


class MyIO:
    def __init__(self, echo=sys.stdout):
        self.echo = echo
        self.sink = io.StringIO()

    def flush(self):
        if self.echo:
            self.echo.flush()
        self.sink.flush()

    def write(self,  str):
        if self.echo:
            self.echo.write(str)
        self.sink.write(str)

    def getvalue(self):
        return self.sink.getvalue()

class Contest:

    def __init__(self):
        pass

    def report(self, res):
        self.results += [res]

    def inputList(self, N):
        return ([random.random() for i in range(N)],), {}

    def getTable(self):
        return self.myio.getvalue()

    def run(self, funcs, **kw_):
        detail = 4
        Ns = [int(math.floor(10 ** (i/detail))) for i in range(20*detail)]
        Ns = list(dict.fromkeys(Ns).keys())
        inputszs = kw_.get('Ns', Ns)
        timeout = kw_.get('timeout', 0.05)
        echo = kw_.get('echo', None)

        vectorize = kw_.get('vectorize', '')

        if vectorize == 'list':
            geninput = kw_.get('input', self.inputList)
        elif vectorize == 'array':
            geninput = kw_.get('input', self.inputArray)
        else:
            geninput = kw_.get('input', self.inputList)

        print(f'input sizes = {inputszs}')
        print(f'timeout = {Value(timeout,"s")}')

        self.funcs = funcs
        self.args = inputszs
        self.results = []


        ninputs = len(inputszs)

        self.results = [[] for i in range(ninputs)]

        self.myio = MyIO(echo=sys.stdout if echo else None)

        scales = [1] + list(chain(*((1e3, 1e6) for i in range(3))))
        tb = Table(
            fd=self.myio,
            headers=['N', 'R', 'R std', 'W', 'W std', 'C', 'C std'],
            namehead='Func', colPrec=5, colScale=scales, unit='s')
        print(tb)
        tb.writeHeader()

        self.bailout = [False] * len(funcs)

        vnan = math.nan

        for inp in range(ninputs):

            psize = int(inputszs[inp])
            try:
                args, kw = geninput(psize)
                print(f'Obtain inputs for problem size {psize}:')
                displArgs(*args, **kw)
            except BaseException as ex:
                print(ex)
                break

            for fi, f in enumerate(funcs):
                if self.bailout[fi]:
                    self.results[inp].append(None)
                    continue
                bm = Benchmark(print=False, vectorize=vectorize)
                bm.bench(f, *args, **kw)
                self.results[inp].append(bm)
                #print('Contest results', self.results)

                #print(tb)

#            for i in range(ninputs):
#                print(f'results for inp. {i}: {self.results[inp]}')

            [ tb.writeRow(
                (psize, t.vpr.mean(), t.vpr.std(), t.vpw.mean(), t.vpw.std(), t.vpc.mean(), t.vpc.std())
                if t is not None else (psize, vnan, vnan, vnan, vnan, vnan, vnan),
                funcs[k].__qualname__)
              for k, t in enumerate(self.results[inp]) ]

            self.bailout = [t is None or t.vpr.mean() > timeout for t in self.results[inp]]
            print('bail out', self.bailout)
            if all(self.bailout):
                break

        return self.getTable()

    def plot(self):

        rlist = [ f for f in self.results if len(f) ]

        print('plot', len(rlist))

        # plot
        fig, ax = plt.subplots()
        N = len(rlist)
        x = self.args[0:N]

        print('plot', rlist)
        print('plot lasr res', rlist[-1])
        adisp, aunit = rlist[0][0].vpr.mean.getpair()
        plsc = adisp / rlist[0][0].vpr.mean()

        ntests = len(self.funcs)
        handles = []
        fnames = [self.funcs[i].__qualname__ for i in range(ntests)]

        vnan = math.nan

        for i in range(ntests):

            y    = [ rlist[j][i].vpr.mean(plsc) if rlist[j][i] is not None else vnan for j in range(N) ]
            yerr = [ rlist[j][i].vpr.std(plsc) if rlist[j][i] is not None else vnan for j in range(N) ]

            print('x', x, 'mean', y, 'std', yerr)

            h1 = ax.errorbar(x, y, yerr, label=fnames[i])
            handles += [h1]

            #        ax.set(xlim=(1, N+1), xticks=np.arange(1, N+1))
            #        ax.legend([h1], ['Results'])
        print(fnames)
        plt.xscale('log')
        plt.yscale('log')
        plt.ylabel(f'Time ({aunit})')
        plt.xlabel('Input size N')
        ax.legend()
        plt.show()

def contest(*funcs, **kw):
    c = Contest()
    tbl = c.run(funcs, **kw)
    plot = kw.get('plot', True)
    if plot:
        c.plot()
    return tbl


def fl_sin(l):
    return [sin(v) for v in l]

def fl_asin(l):
    return [asin(v) for v in l]

def fl_sin2(l):
    return list(map(sin, l))

def d_unzip1(l):
    ds = tuple({k: v[i] for k, v in l.items()} for i in range(2))
    return ds

def d_unzip2(l):
    d1 = {k: v[0] for k, v in l.items()}
    d2 = {k: v[1] for k, v in l.items()}
    return d1, d2


def fl_find1(l):
    return [sin(v) for v in l]

def fl_find2(l):
    return [asin(v) for v in l]

def testbench():
    bm = Benchmark()
    x = [random.random() for i in range(int(1e5))]
    bm.bench(fl_sin, x)
    bm.bench(fl_asin, x)
    bm.bench(fl_sin2, x)
    bm.plot()

    x2 = {v: (v, 1e2*v) for v in x}
    bm.bench(d_unzip1, x2)
    bm.bench(d_unzip2, x2)
    bm.plot()
    #    x2s = {v: (v, 1e2*v) for v in x[0:5]}
    #    print(d_unzip1(x2s))
    #    print(d_unzip2(x2s))

def testcontest():
    def genInput(N):
        print('geninput', N)
        x = [random.random() for i in range(N)]
        args = [x]
        return args, {}
    c = Contest(genInput)
    c.run([fl_sin, fl_sin2, fl_asin], [1e2, 1e3, 1e4, 2e4])
    c.plot()

#testcontest()
