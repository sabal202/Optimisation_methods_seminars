from functools import wraps

import numpy as np

from utils import FuncCounter
from zero_order_methods import gold_sech_method


def deco_maker(ind):
    def deco(func, x0):
        @wraps(func)
        def wrapper(x):
            new_args = list(x0)
            new_args[ind] = x
            return func(new_args)

        return wrapper

    return deco


def norm(x):
    return sum([i * i for i in x]) ** 0.5


def per_coordinate_descent(func, x0, eps=0.01, debug=False):
    a = np.array(x0)
    b = np.array(x0)
    dim = len(b)
    count = 0
    minimize = FuncCounter(gold_sech_method, without_memoization=True)
    while True:
        for i in range(dim):
            phi = deco_maker(i)(func, a)
            b[i] = minimize(phi, a[i] - 5 * eps, a[i] + 5 * eps, eps)
        c = b - a
        if np.linalg.norm(c) <= eps:
            count += 1
        else:
            count = 0
        if count == 3:
            break
        a = b.copy()
    if debug:
        print("coordinate_descent minimizations counter", minimize.counter)
    return tuple(b)


def step_partition_descent(func, grad, x0, alpha=1, lr=0.5, d=0.5, eps=0.01, debug=False):
    x0 = np.array(x0)
    alph = alpha

    func = FuncCounter(func, without_memoization=True)
    grad = FuncCounter(grad, without_memoization=True)
    gr = np.array(grad(x0))

    x1 = x0 - alph * gr
    while True:
        temp = func(x0)
        while func(x1) - temp > -1 * d * alph * np.linalg.norm(gr) ** 2:
            alph *= lr
            x1 = x0 - alph * gr
        gr = np.array(grad(x1))
        if np.linalg.norm(gr) <= eps:
            break
        else:
            x0 = x1.copy()
            alph = alpha
            x1 = x0 - alph * gr
    if debug:
        print("step_partition_descent func counter", func.counter)
        print("step_partition_descent grad counter", grad.counter)
    return tuple(x1)
