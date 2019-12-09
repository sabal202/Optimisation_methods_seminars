import numpy as np

from utils import FuncCounter
from zero_order_methods import gold_sech_method


def constant_step_descent(grad, x0, alpha=0.02, eps=0.01, debug=False):
    x0 = np.array(x0)
    grad = FuncCounter(grad, without_memoization=True)
    gr = np.array(grad(x0))
    x1 = x0 - alpha * gr
    while True:
        gr = np.array(grad(x1))
        if np.linalg.norm(gr) <= eps:
            break
        else:
            x0 = x1.copy()
            print(x0, alpha, gr)
            x1 = x0 - alpha * gr
    if debug:
        print("constant_step_descent grad counter", grad.counter)
    return tuple(x1)


def harmonic_series(n):
    return 1 / n


def divergent_series_method(grad, x0, eps=0.01, div_series=None, debug=False):
    x0 = np.array(x0)
    grad = FuncCounter(grad, without_memoization=True)
    gr = np.array(grad(x0))
    k = 1
    if div_series is None:
        div_series = harmonic_series
    x1 = x0 - div_series(k) * gr
    while True:
        gr = np.array(grad(x1))
        if np.linalg.norm(gr) <= eps:
            break
        else:
            x0 = x1.copy()
            k += 1
            x1 = x0 - div_series(k) * gr
    if debug:
        print("divergent_series_method grad counter", grad.counter)
    return tuple(x1)


def fastest_gradient_descent(func, grad, x0, eps=0.01, max_iter=float('inf'), debug=False):
    x0 = np.array(x0)
    func = FuncCounter(func, without_memoization=True)
    grad = FuncCounter(grad, without_memoization=True)
    minimize = FuncCounter(gold_sech_method, without_memoization=True)

    gr = np.array(grad(x0))

    alpha = minimize(lambda a: func(x0 - a * gr), 0, 1, eps * 10)
    x1 = x0 - alpha * gr
    count = 1
    while count <= max_iter:
        gr = np.array(grad(x1))
        if np.linalg.norm(gr) <= eps:
            break
        else:
            x0 = x1.copy()
            alpha = minimize(lambda a: func(x0 - a * gr), 0, 1, eps * 10)
            x1 = x0 - alpha * gr
    if debug:
        print("step_partition_descent func counter", func.counter)
        print("step_partition_descent grad counter", grad.counter)
        print("step_partition_descent minimizations counter", minimize.counter)
    return tuple(x1)


def accelerated_gradient_descent(func, grad, x0, eps=0.01, debug=False):
    x0 = np.array(x0)
    dim = len(x0)
    func = FuncCounter(func, without_memoization=True)
    grad = FuncCounter(grad, without_memoization=True)
    FGD = FuncCounter(fastest_gradient_descent, without_memoization=True)

    y0 = np.array(FGD(func, grad, x0, eps, max_iter=dim))
    alpha = gold_sech_method(lambda a: func(x0 + a * (y0 - x0)), 0, 1, eps * 10)
    x1 = x0 + alpha * (y0 - x0)
    while True:
        gr = np.array(grad(x1))
        if np.linalg.norm(gr) <= eps:
            break
        else:
            x0 = x1.copy()
            y0 = np.array(FGD(func, grad, x0, eps, max_iter=dim))
            alpha = gold_sech_method(lambda a: func(x0 + a * (y0 - x0)), 0, 1, eps * 10)
            x1 = x0 + alpha * (y0 - x0)
    if debug:
        print("step_partition_descent func counter", func.counter)
        print("step_partition_descent grad counter", grad.counter)
        print("step_partition_descent FGD counter", FGD.counter)
    return tuple(x1)
