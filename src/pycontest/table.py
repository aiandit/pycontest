# author: Johannes Willkomm <j.willkomm@fionec.de>

import json
import sys
import io

from .value import Value

def flatten(data):
    if isinstance(data, list):
        if len(data) == 0:
            return ()
        else:
            return flatten(data[0]) + flatten(data[1:])
    elif isinstance(data, tuple):
        if len(data) == 0:
            return ()
        else:
            return flatten(data[0]) + flatten(data[1:])
    else:
        return (data,)

serFields = ['headers', 'colWidths', 'colPrecs', 'valfmts']

class Table:

    colWidths = [8, 8, 8]
    sep = ';'
    _nchar = 0

    def __init__(self, headers, fd=sys.stdout, namehead=None, colPrec=2, colScale=1, unit='s', sep=' ', format='var'):
        self.fd = fd
        self.headers = tuple(headers)
        self.namehead = namehead

        self.sep = sep
        self.format = format

        self.fmtletter = 'f' if format == 'fixed' else 'e' if format == 'eng' else 'g'
        self.fmtaddspace = 5 if format == 'fixed' else 7 if format == 'eng' else 6

        if type(colPrec) == type([]) or type(colPrec) == type(()):
            self.colPrecs = colPrec
        else:
            self.colPrecs = [colPrec] * len(headers)

        self.colWidths = [v + self.fmtaddspace for v in self.colPrecs]

        if isinstance(colScale, list) or isinstance(colScale, tuple):
            self.colScales = colScale
        else:
            self.colScales = [colScale] * len(headers)

        if isinstance(unit, list) or isinstance(unit, tuple):
            self.colUnits = units
        else:
            self.colUnits =  [unit] * len(headers)

        self.valfmts = ['%%.%d%s' % (p, self.fmtletter) for p in self.colPrecs]

        self.colUnits = [Value(0, u).getunit(s) for s, u in zip(self.colScales, self.colUnits)]
        self.uHeaders = tuple('%s (%s)' %(h, u) for h, u in zip(self.headers, self.colUnits))

        if namehead:
            self.colWidths = [8] + self.colWidths
            self.headers = tuple([namehead] + list(self.uHeaders))
        else:
            self.headers = self.uHeaders

        self.update()
        self.data = []
        self.names = []

    def setfd(self, fd):
        self.fd = fd

    def writeTable(self, fd):
        print('write!', self.data, self.names)
        self.setfd(fd)
        self.writeHeader()
        for i in range(len(self.data)):
            self.writeRow(self.data[i], self.names[i])

    def update(self):
        self.lnFormat = self.sep.join(['%% %ds' % w for w in self.colWidths])

    def addRow(self, row, name=None):
        self.data += [row]
        self.names += [name]

    def getData(self):
        return self.data

    def writeHeader(self):
        self.write('%s\n' % self.getHeader())
        self.fd.flush()

    def getHeader(self):
        return self.fmtLine(self.headers)

    def writeLine(self, vline):
        self.write('%s\n' % vline)

    def writeValues(self, values, name=None):
        self.addRow(values, name=name)
        vstrs = self.fmtValues(values)
        if name:
            vstrs = tuple([name] + list(vstrs))
        self.writeLine(self.fmtLine(vstrs))

    writeRow = writeValues

    def fmtLine(self, values):
        # print(self.lnFormat, values)
        return self.lnFormat % values

    def fmtValues(self, values):
        fmtvals = tuple((self.valfmts[i] % (f*self.colScales[i])) for (i,f) in enumerate(values))
        fmtvals = [ '' if f == 'nan' else f  for f in fmtvals]
        return fmtvals

    def write(self, res):
        self._nchar += len(res)
        self.fd.write(res)

    def gets(self):
        outf = io.StringIO()
        self.writeTable(outf)
        return outf.getvalue()

    @property
    def nchar(self):
        return self._nchar

    def __str__(self):
        s = ''
        s += 'sep: ' + self.sep + '\n'
        s += 'headers (%d): ' % (len(self.headers),) + str(self.headers) + '\n'
        s += 'units (%d): ' % (len(self.colUnits),) + str(self.colUnits) + '\n'
        s += 'colPrecs (%d): ' % (len(self.colPrecs),) + str(self.colPrecs) + '\n'
        s += 'colWidths (%d): ' % (len(self.colWidths),) + str(self.colWidths) + '\n'
        s += 'valfmts (%d): ' % (len(self.valfmts),) + str(self.valfmts) + '\n'
        s += 'lnFormat: ' + self.lnFormat + '\n'
        return s

    def __repr__(self):
        tdata = {v: getattr(self, v) for v in serFields}
        return json.dumps(tdata, indent=1)

    def readJSON(self, jsdata):
        tdata = json.loads(jsdata)
        for i in tdata:
            if hasattr(self, i):
                setattr(self, i, flatten(tdata[i]))
        self.update()

    def getV3Table(self, newfd):
        v3Table = Table(newfd, self.headers, self.colPrecs, self.sep, self.format)

        v3Table.headers = flatten([(h + ' (min)', h + ' (mean)', h + ' (max)') for h in self.headers])
        v3Table.colPrecs = flatten([(c,c,c) for c in self.colPrecs])
        v3Table.colWidths = flatten([(c,c,c) for c in self.colWidths])
        v3Table.valfmts = flatten([(c,c,c) for c in self.valfmts])

        v3Table.update()

        return v3Table


if __name__ == "__main__":
    import sys, random
    t = Table(sys.stdout, ('A', 'B', 'C', 'D'))
    print(t)
    t.writeHeader()
    v1 = [random.random() for i in range(4)]
    v2 = [1e3 * v for v in v1]
    t.writeRow(v1)
    t.writeRow(v2)

    t = Table(sys.stdout, ('A', 'B', 'C', 'D'), colPrec=6, namehead='Name', unit='g', colScale=[1e3, 1, 1e-3, 1])
    print(t)
    t.writeHeader()
    v1 = [random.random() for i in range(4)]
    v2 = [1e3 * v for v in v1]
    t.writeRow(v1, 'test1')
    t.writeRow(v2, 'test2')
