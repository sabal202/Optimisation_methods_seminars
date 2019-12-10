import numpy as np

from fast_descents import fastest_gradient_descent
from utils import FuncCounter
from zero_order_methods import gold_sech_method


def ravine_method(func, grad, x0, eps=0.02, debug=False):
    x0 = np.array(x0)

    func = FuncCounter(func, without_memoization=True)
    grad = FuncCounter(grad, without_memoization=True)
    FGD = FuncCounter(fastest_gradient_descent, without_memoization=True)
    count = 0
    while count < 3:
        _x = x0 + 10 * eps
        y0 = np.array(FGD(func, grad, x0, eps, max_iter=1))

        _y = y0 + 10 * eps
        alpha = gold_sech_method(lambda a: func(y0 + a * (_y - y0)), -10, 10, eps * 10)
        x1 = y0 + alpha * (_y - y0)
        c = x1 - x0
        if np.linalg.norm(c) <= eps:
            count += 1
        else:
            count = 0
        x0 = x1.copy()
    if debug:
        print("ravine_method func counter", func.counter)
        print("ravine_method grad counter", grad.counter)
        print("ravine_method FGD counter", FGD.counter)
    return tuple(x0)


def newton_method(grad, inv_hesse_mat, x0, eps=0.02, debug=False):
    x0 = np.array(x0)
    grad = FuncCounter(grad, without_memoization=True)
    inv_hesse_mat = FuncCounter(inv_hesse_mat, without_memoization=True)
    gr = np.array(grad(x0))
    while np.linalg.norm(gr) > eps:
        m = np.array(inv_hesse_mat(x0))
        x0 = x0 - np.dot(m, gr)
        gr = np.array(grad(x0))
    if debug:
        print("newton_method grad counter", grad.counter)
        print("newton_method inv_hesse_mat counter", inv_hesse_mat.counter)
    return tuple(x0)


def modified_newton_method(func, grad, inv_hesse_mat, x0, eps=0.02, debug=False):
    x0 = np.array(x0)
    func = FuncCounter(func, without_memoization=True)
    grad = FuncCounter(grad, without_memoization=True)
    minimize = FuncCounter(gold_sech_method, without_memoization=True)
    inv_hesse_mat = FuncCounter(inv_hesse_mat, without_memoization=True)

    gr = np.array(grad(x0))
    while np.linalg.norm(gr) > eps:
        m = np.array(inv_hesse_mat(x0))
        d = np.dot(m, gr)
        alpha = minimize(lambda a: func(x0 - a * d), -10, 10, eps * 10)
        x0 = x0 - alpha * d
        gr = np.array(grad(x0))
    if debug:
        print("modified_newton_method func counter", func.counter)
        print("modified_newton_method grad counter", grad.counter)
        print("modified_newton_method minimizations counter", minimize.counter)
        print("modified_newton_method inv_hesse_mat counter", inv_hesse_mat.counter)

    return tuple(x0)
