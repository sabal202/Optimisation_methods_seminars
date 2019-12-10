import numpy as np

from utils import FuncCounter
from zero_order_methods import gold_sech_method


def fletcher_reeves_method(func, grad, x0, eps=0.02, debug=False):
    x0 = np.array(x0)
    func = FuncCounter(func, without_memoization=True)
    grad = FuncCounter(grad, without_memoization=True)
    minimize = FuncCounter(gold_sech_method, without_memoization=True)
    dim = len(x0)

    gr0 = np.array(grad(x0))
    d = gr0
    alpha = minimize(lambda a: func(x0 - a * d), -10, 10, eps * 10)
    x1 = x0 - alpha * d

    gr1 = np.array(grad(x1))
    nr0 = np.linalg.norm(-1 * gr0)
    nr1 = np.linalg.norm(-1 * gr1)
    k = 0
    while nr1 > eps:
        if (k + 1) % dim == 0:
            beta = 0
        else:
            beta = nr1 ** 2 / nr0 ** 2
        d = -1 * gr1 + beta * d
        k += 1
        x0 = x1.copy()

        alpha = minimize(lambda a: func(x0 - a * d), -10, 10, eps * 10)
        x1 = x0 - alpha * d
        nr0 = nr1
        gr1 = np.array(grad(x1))
        nr1 = np.linalg.norm(-1 * gr1)

    if debug:
        print("fletcher_reeves_method func counter", func.counter)
        print("fletcher_reeves_method grad counter", grad.counter)
        print("fletcher_reeves_method minimizations counter", minimize.counter)
    return tuple(x1)
