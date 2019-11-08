from math import ceil, sqrt
from utils import FuncCounter


def naive_method(func, a, b, eps=0.01, debug=False):
    """Минимизация наивным методом"""
    f = FuncCounter(func)
    n = ceil((b - a) / eps)
    tests = []

    for i in range(n + 1):
        x = a + eps * i
        tests.append((f(x), x))

    minimum = min(tests)[1]
    if debug:
        print("naive_method func counter", f.counter)
    return minimum


def dichotomy_method(func, a, b, eps=0.01, delta_mult=0.5, debug=False):
    """Минимизация методом дихотомии"""
    f = FuncCounter(func)
    delta = eps * delta_mult

    ai = a
    bi = b

    while (bi - ai) > eps * 2:
        ci = (ai + bi - delta) / 2
        di = (ai + bi + delta) / 2

        if f(ci) <= f(di):
            ai = ai
            bi = di
        else:
            ai = ci
            bi = bi

    if debug:
        print("dichotomy_method func counter", f.counter)
    return (bi + ai) / 2


def gold_sech_method(func, a, b, eps=0.01, debug=False):
    """Минимизация методом золотого сечения"""
    f = FuncCounter(func)
    ai = a
    bi = b
    ci = ai + (bi - ai) * (3 - sqrt(5)) / 2
    di = ai + (bi - ai) * (sqrt(5) - 1) / 2

    while (bi - ai) > eps * 2:
        if f(ci) <= f(di):
            ai = ai
            bi = di
            di = ci
            ci = ai + (bi - ai) * (3 - sqrt(5)) / 2
        else:
            ai = ci
            bi = bi
            ci = di
            di = ai + (bi - ai) * (sqrt(5) - 1) / 2

    if debug:
        print("gold_sech_method func counter", f.counter)

    return (bi + ai) / 2
