import unittest
from math import sin, cos, sqrt, floor
import math

from pycontest import Value, Table, contest

import numpy as np


class TestValue(unittest.TestCase):

    def test_value(self):
        val = 10.12345
        v = Value(val, 's')
        print(f'v = {v.gets(1e-3)}')
        print(f'v = {v.gets("k")}')

        print(f'v = {v.gets(1e3)}')
        print(f'v = {v.gets("m")}')

    def test_value2(self):
        val = 10.12345
        v = Value(val, 's')

        print(f'v = {v}')

        self.assertEqual(v(), val)
        self.assertEqual(v(1e-3), 1e-3*val)
        self.assertEqual(v(1e6), 1e6*val)

        self.assertEqual(v.get(), val)
        self.assertEqual(v.get(1e-3), 1e-3*val)
        self.assertEqual(v.get(1e6), 1e6*val)

        self.assertEqual(v(), val)
        self.assertEqual(v('k'), 1e-3*val)
        self.assertEqual(v('µ'), 1e6*val)

        self.assertEqual(v.get(), val)
        self.assertEqual(v.get('k'), 1e-3*val)
        self.assertEqual(v.get('n'), 1e9*val)

    def test_value3(self):
        val = 10.12345e3
        v = Value(val, 's')

        self.assertEqual(v.getunit(), 'ks')
        self.assertEqual(v.getunit(1e-6), 'Ms')
        self.assertEqual(v.getunit(1e9), 'ns')

        self.assertEqual(v.scale(), 1e-3)

        self.assertEqual(v.scale('P'), 1e-15)
        self.assertEqual(v.scale('T'), 1e-12)
        self.assertEqual(v.scale('G'), 1e-9)
        self.assertEqual(v.scale('M'), 1e-6)
        self.assertEqual(v.scale('k'), 1e-3)

        self.assertEqual(v.scale('m'), 1e3)
        self.assertEqual(v.scale('µ'), 1e6)
        self.assertEqual(v.scale('n'), 1e9)


    def test_value4(self):
        val = 10.12345e-6
        v = Value(val, 's')

        print(f'v = {v}')
        self.assertEqual(v.getunit(), 'µs')
        self.assertEqual(v.gets(), '10.12 µs')

        self.assertEqual(v.gets(fmt='%.4f'), '10.1235 µs')


class TestTable(unittest.TestCase):

    def test_table(self):
        val = 10.12345
        v = Value(val, 's')

        tbl = Table(headers=['A', 'B', 'C', 'D'], namehead='Name')
        r0 = [1,2,3,4]
        data = []
        for i in range(10):
            r = [i*v for v in r0]
            data.append(r)
            tbl.addRow(r, f'i={i}')

        print(data)
        self.assertEqual(data, tbl.getData())
        print(tbl.gets())


    def test_table2(self):
        val = 10.12345
        v = Value(val, 's')

        tbl = Table(headers=['A', 'B', 'C', 'D'], namehead='Name')
        r0 = [1,2,3,4]
        data = []
        for i in range(10):
            r = [i**v for v in r0]
            data.append(r)
            tbl.addRow(r, f'i={i}')

        print(data)
        self.assertEqual(data, tbl.getData())
        print(tbl.gets())


    def test_table3(self):
        val = 10.12345
        v = Value(val, 's')

        tbl = Table(headers=['A', 'B', 'C', 'D'], namehead='Name', colScale=[1, 1, 1e-3, 1e-3])
        r0 = [1,2,3,4]
        data = []
        for i in range(10):
            r = [i**v for v in r0]
            r[i % 4] = math.nan
            data.append(r)
            tbl.addRow(r, f'i={i}')

        print(data)
        self.assertEqual(data, tbl.getData())
        print(tbl.gets())


def getElemFTestList(bfun):
    def inner(l):
        return [ bfun(v) for v in l]
    return inner

class TestContest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.verbose = 1

    def test_contest(self):

        cfs = [getElemFTestList(f) for f in [sin, cos, sqrt]]
        contest(*cfs, show=False, timeout=1e-3, verbose=self.verbose)


    def test_contest2(self):

        cfs = [sin, cos, sqrt]
        res = contest(*cfs, show=False, vectorize=True, timeout=1e-3, verbose=self.verbose)

        print('Contest results:')
        print(res)


    def test_contest3(self):

        def eyeList(l):
            return [ [ l[i] if i==j else 0 for i in range(len(l)) ] for j in range(len(l)) ]

        def sumAllElements(A):
            return sum(A)

        def sumAllElements2(A):
            return sum([ sum(A[i]) for i in range(len(A)) ])

        def getDiag(A):
            return [ A[i][i] for i in range(len(A)) ]

        def eyeListAndSum(l):
            A = eyeList(l)
            return sumAllElements2(A)

        def eyeListAndSumDiag(l):
            A = eyeList(l)
            return sum(getDiag(A))

        def eyeArrAndSum(l):
            A = np.diag(l)
            return np.sum(A)

        cfs = [eyeArrAndSum, eyeListAndSum, eyeListAndSumDiag]
        res = contest(*cfs, show=False, timeout=1e-3, verbose=self.verbose)

        print('Contest results:')
        print(res)


    def test_contest4(self):

        def fSumAll(l):
            return np.sum(l)

        def fSumDiag(l):
            return np.sum(np.diag(l))

        def input(N):
            Ns = int(math.ceil(math.sqrt(N)))
            l = np.random.rand(Ns, Ns)
            return (l,),{}

        A = np.diag([1,2,3])
        assert np.shape(A) == (3,3)
        assert np.shape(np.diag(A)) == (3,)

        cfs = [fSumAll, fSumDiag]
        res = contest(*cfs, show=False, timeout=1e-4, input=input,
                      base=2, detail=2, start=8,
                      name="sum_vs_sumdiag", title="sum() vs. diag(sum())", verbose=self.verbose)

        print('Contest results:')
        print(res)
