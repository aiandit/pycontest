from math import log10

def getpair(t, unit='s', scale=None):
    pref = ['P', 'T', 'G', 'M', 'k', '', 'm', 'µ', 'n', 'p', 'f']
    if scale is None:
        scale = 1
        index = 5
        while True:
            if (t*scale >= 1 and t*scale < 1e3) or index == 0 or index == len(pref)-1:
                break
            elif t*scale < 1:
                scale *= 1e3
                index += 1
            else:
                scale *= 1e-3
                index -= 1
    else:
        if isinstance(scale, str):
            index = pref.index(scale)
            scale = 1e3 ** (index - 5)
        else:
            index = 5 + int(log10(scale) / 3)
    return (t, unit, pref[index], scale)


class Value:

    def __init__(self, v, unit='s'):
        self.value = v
        self.unit = unit

    def scale(self, scale=None):
        (ts, u, p, sc) = getpair(self.value, self.unit, scale=scale)
        return sc

    def getunit(self, scale=None):
        (ts, u, p, sc) = getpair(self.value, self.unit, scale=scale)
        return f'{p}{u}'

    getu = getunit

    def getpair(self, scale=None):
        (ts, u, p, sc) = getpair(self.value, self.unit, scale=scale)
        return (ts*sc, f'{p}{u}')

    def get(self, scale=1):
        (v, u) = self.getpair(scale)
        return v

    def __call__(self, sc=1):
        return self.get(sc)

    def __repr__(self):
        return f'Value({repr(self.value)}, {repr(self.unit)})'

    def gets(self, scale=None, fmt=None):
        (v, u) = self.getpair(scale)
        if fmt == None:
            fmt = '%.2f'
        vs = fmt % (v,)
        return f'{vs} {u}'

    def __str__(self):
        return self.gets()
