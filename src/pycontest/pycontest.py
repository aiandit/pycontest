import io
import sys
import os
import time
import random
import math
import gc

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

    def __repr__(self):
        return f'{self.mean} +/-{self.std}'

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
        print(f'[{i}]: type={type(a).__name__}')
    if len(kw):
        print('Keyword args:')
        for i, (k, v) in enumerate(kw.items()):
            print(f'{k} = type={type(v).__name__}')


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

    def __init__(self, **kw):
        self.name = kw.get('name', '')
        self.title = kw.get('title', self.name)
        self.show = kw.get('show', True)
        self.print = kw.get('print', True)
        self.outdir = kw.get('outdir', '.')
        self.timeout = kw.get('timeout', 1e-2)
        self.echo = kw.get('echo', False)
        self.verbose = kw.get('verbose', False)
        self.base = kw.get('base', 10)
        self.detail = kw.get('detail', 4)
        self.start = kw.get('start', 1)

    def report(self, res):
        self.results += [res]

    def inputSizes(self, **kw):
        Ns = [int(math.floor(self.base ** (i/self.detail))) for i in range(int(self.start*self.detail), int(100*self.detail))]
        Ns = list(dict.fromkeys(Ns).keys())
        return Ns

    def inputList(self, N):
        return ([random.random() for i in range(N)],), {}

    def getTable(self):
        return self.myio.getvalue()

    def run(self, funcs, **kw_):
        inputszs = self.inputSizes(**kw_)
        timeout = kw_.get('timeout', self.timeout)
        echo = kw_.get('echo', self.echo)
        verbose = kw_.get('verbose', self.verbose)

        vectorize = kw_.get('vectorize', '')

        if vectorize == 'list':
            geninput = kw_.get('input', self.inputList)
        elif vectorize == 'array':
            geninput = kw_.get('input', self.inputArray)
        else:
            geninput = kw_.get('input', self.inputList)

        reset = kw_.get('reset', lambda: 0)

        if verbose:
            print(f'input sizes = {inputszs[0]}, {inputszs[1]}, ..., {inputszs[-1]}')
            print(f'timeout = {Value(timeout,"s")}')

        if isinstance(funcs, dict):
            self.names = list(funcs.keys())
            self.funcs = list(funcs.values())
        else:
            self.names = [f.__name__ for f in funcs]
            self.funcs = funcs
        self.args = inputszs
        self.results = []


        ninputs = len(inputszs)

        self.results = [[] for i in range(ninputs)]

        self.myio = MyIO(echo=sys.stdout if echo else None)

        scales = [1] + list(chain(*((1e6, 1e6) for i in range(3))))
        colWidths = 6
        namewidth = max(*[len(f) for f in self.names])+2
        self.table = tb = Table(
            fd=self.myio,
            headers=['N', 'R', 'R std', 'W', 'W std', 'C', 'C std'],
            namehead='Func', namewidth=namewidth,
            colPrec=colWidths, colScale=scales, unit='s')
        if verbose > 1:
            print(tb)
        tb.writeHeader()

        self.bailout = [False] * len(self.funcs)

        vnan = math.nan

        for inp in range(ninputs):

            psize = int(inputszs[inp])
            try:
                print(f'Obtain inputs for problem size {psize}:')
                args, kw = geninput(psize)
                displArgs(*args, **kw)
            except ValueError as ex:
                print(ex)
                continue
            except BaseException as ex:
                print(f'Failed to run input() for N={psize}, stop')
                break


            for fi, f in enumerate(self.funcs):
                if self.bailout[fi]:
                    self.results[inp].append(None)
                    continue
                gc.collect()
                reset()

                bm = Benchmark(print=False, vectorize=vectorize)
                bm.bench(f, *args, **kw)
                self.results[inp].append(bm)
                #print('Contest results', self.results)

                # print(bm.vpr.mean(), 's')

#            for i in range(ninputs):
#                print(f'results for inp. {i}: {self.results[inp]}')

            [ tb.addRow(
                (psize, t.vpr.mean(), t.vpr.std(), t.vpw.mean(), t.vpw.std(), t.vpc.mean(), t.vpc.std())
                if t is not None else (psize, vnan, vnan, vnan, vnan, vnan, vnan),
                self.names[k])
              for k, t in enumerate(self.results[inp]) ]

            meants = [f'{t.vpr}' if t is not None else math.inf for t in self.results[inp]]
            self.bailout = [t is None or t.vpr.mean() > timeout for t in self.results[inp]]
            print(f'Results step i={inp}, N={psize}: ', meants)
            if all(self.bailout):
                break

        return self.getTable()

    def plot(self):

        self.plots = []
        rlist = [ f for f in self.results if len(f) ]

        # print('plot', len(rlist))

        # plot
        fig, ax = plt.subplots()
        N = len(rlist)
        x = self.args[0:N]

        # print('plot', rlist)
        # print('plot lasr res', rlist[-1])
        adisp, aunit = rlist[0][0].vpr.mean.getpair()
        plsc = adisp / rlist[0][0].vpr.mean()

        ntests = len(self.funcs)
        handles = []
        fnames = [self.names[i] for i in range(ntests)]

        vnan = math.nan

        def getSeries(i, rlist):
            xinnan = [ j for j in range(N) if rlist[j][i] is not None ]
            xnnan = [ x[j] for j in xinnan ]

            y    = [ rlist[j][i].vpr.mean(plsc) for j in xinnan ]
            yerr = [ rlist[j][i].vpr.std(plsc) for j in xinnan ]
            xplt = [ x[j] for j in xinnan ]

            return xplt, y, yerr


        for i in range(ntests):

            xplt, y, yerr = getSeries(i, rlist)
            # print('x', xplt, 'mean', y, 'std', yerr)

            h1 = ax.errorbar(xplt, y, yerr, label=fnames[i])
            handles += [h1]

            #        ax.set(xlim=(1, N+1), xticks=np.arange(1, N+1))
            #        ax.legend([h1], ['Results'])
        # print(fnames)
        plt.xscale('log')
        plt.yscale('log')
        plt.ylabel(f'Time ({aunit})')
        plt.xlabel('Input size N')
        ax.legend()
        plt.title(f'{self.title} - Runtimes')
        self.plots += [fig]


        # plot
        fig, ax = plt.subplots()

        reltSums = [0]*ntests
        reltSums2 = [0]*ntests

        for i in range(ntests):

            xplt, y, yerr = getSeries(i, rlist)

            reltSums[i] = Value(sum( [ y[i] / plsc / xplt[i] for i in range(len(xplt)) ] ) / len(xplt), 's')
            reltSums2[i] = Value(sum(y) / plsc / sum(xplt), 's')
            print('x', xplt, 'mean', y, 'std', yerr)
            print('xsum', sum(xplt), 'ysum', sum(y), 'rel', reltSums[i], 'rel2', reltSums2[i])
            # reltSums[i] = sum([ x[i] * y[i] for i in range(len(xplt)]) / sum(x)

            y    = [ y[j]    / xplt[j] for j in range(len(xplt)) ]
            yerr = [ yerr[j] / xplt[j] for j in range(len(xplt)) ]

            h1 = ax.errorbar(xplt, y, yerr, label=fnames[i])
            handles += [h1]

            #        ax.set(xlim=(1, N+1), xticks=np.arange(1, N+1))
            #        ax.legend([h1], ['Results'])

        print('reltimes', fnames, reltSums)
        plt.xscale('log')
        plt.yscale('log')
        plt.ylabel(f'Time ({aunit}) / N')
        plt.xlabel('Input size N')
        ax.legend()
        plt.title(f'{self.title} - Relative runtimes per input size N')
        self.plots += [fig]


        inds = list(range(len(reltSums)))
        inds = sorted(inds, key=lambda i: reltSums[i])

        rank = [ self.names[i] for i in inds ]
        rankTimes = [ reltSums[i] for i in inds ]
        print(f'Performance ranking: {list(zip(rank, rankTimes))}')

        winner = inds[0]
        namewin = self.names[winner]
        perfwin = reltSums[winner]
        print(f'And the winner is cand. {winner}, {namewin}, ({self.funcs[winner]}) with {reltSums[winner]} s/N')

        colWidths = 6
        namewidth = max(*[len(f) for f in self.names])+2
        colscale = 1e6
        self.ranktb = Table(headers=('Rel. T', 'Rel. T 2', 'Speedup'), colScale=(colscale, colscale, 1), colPrec=colWidths, namewidth=namewidth, unit=('s/N', 's/N', ''), namehead='Func')
        for i in range(len(inds)):
            self.ranktb.addRow((reltSums[inds[i]](), reltSums2[inds[i]](), reltSums[inds[i]]() / perfwin()), self.names[inds[i]])

        # plot
        fig, ax = plt.subplots()

        win_xplt, win_y, win_yerr = getSeries(winner, rlist)
        print('win x', xplt, 'win mean', y, 'win std', yerr)

        for i in range(ntests):

            xplt, y, yerr = getSeries(i, rlist)
            print('x', xplt, 'mean', y, 'std', yerr)

            y =    [ y[j]    / win_y[j] for j in range(len(xplt)) ]
            yerr = [ yerr[j] / win_y[j] for j in range(len(xplt)) ]

            reltSums[i] = sum([ v for v in y ])

            h1 = ax.errorbar(xplt, y, yerr, label=fnames[i])
            handles += [h1]

            #        ax.set(xlim=(1, N+1), xticks=np.arange(1, N+1))
            #        ax.legend([h1], ['Results'])

        print(fnames, reltSums)
        plt.xscale('log')
        plt.yscale('log')
        plt.ylabel(f'Time / Time[{namewin}]')
        plt.xlabel('Input size N')
        ax.legend()
        plt.title(f'{self.title} - Runtimes relative to best results')
        self.plots += [fig]

        names = ['plot-times', 'plot-reltimes-N', 'plot-reltimes-best']
        for i, p in enumerate(self.plots):
            ofname = os.path.join(self.outdir, f'plt-{self.name}-{names[i]}.png')
            print(f'Write PNG plot {i} to {ofname}')
            p.savefig(ofname)

        plt.draw()

    def printTables(self):
        print('Performance results')
        print(self.table.gets())
        print('Contest Rank')
        print(self.ranktb.gets())

    def writeTable(self, tb, ofname):
        bdir = self.outdir
        if not os.path.exists(bdir):
            os.mkdir(bdir)
        resTable = os.path.join(bdir, f'{self.name}-{ofname}.txt')
        if self.verbose:
            print(f'Write TXT result table to {resTable}')
        print(f'{tb.gets()}', file=open(resTable, 'w'))
        resTable = os.path.join(bdir, f'{self.name}-{ofname}.csv')
        if self.verbose:
            print(f'Write CSV result table to {resTable}')
        self.table.sep = ';'
        print(f'{tb.gets()}', file=open(resTable, 'w'))

    def writeResults(self):
        self.writeTable(self.table, 'benchmark')
        self.writeTable(self.ranktb, 'rank')

def contest(*funcs, **kw):
    c = Contest(**kw)
    if len(funcs) == 1:
        tbl = c.run(funcs[0], **kw)
    else:
        tbl = c.run(funcs, **kw)
    c.plot()
    #pyplot.pause(1e-2)
    if c.print:
        c.printTables()
    c.writeResults()
    if c.show:
        plt.show()
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
