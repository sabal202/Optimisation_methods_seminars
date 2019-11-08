from math import ceil, sqrt
from functions import all_funcs
from utils import FuncCounter


def simple_minimize(func, a, b, eps=0.01, debug=False):
    """Минимизация наивным методом"""
    f = FuncCounter(func)
    n = ceil((b - a) / eps)
    tests = []

    for i in range(n + 1):
        x = a + eps * i
        tests.append((f(x), x))

    minimum = min(tests)[1]
    if debug:
        print("simple_minimize counter", f.counter)
    return minimum


def dichotomy(func, a, b, eps=0.01, delta_mult=1 / 2, debug=False):
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
        print("dichotomy counter", f.counter)
    return (bi + ai) / 2


def gold_sech(func, a, b, eps=0.01, debug=False):
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
        print("gold_sech counter", f.counter)

    return (bi + ai) / 2


func = all_funcs[32]
a = func.a
b = func.b
eps = func.aps

print("Minimize f(x)=5*(x**2) - 8*(x**(5/4)) - 20*x")
print("simple_minimize min:", simple_minimize(func, a, b, eps, debug=True))
print("dichotomy min:", dichotomy(func, a, b, eps, 1 / 4, debug=True))
print("gold_sech min:", gold_sech(func, a, b, eps, debug=True))
