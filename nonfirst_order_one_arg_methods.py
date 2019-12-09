import sympy
from sympy.parsing.sympy_parser import parse_expr

from utils import FuncCounter


def tangent_lines_method(func, a, b, eps=0.01, n=16, debug=False):
    """Метод касательных"""
    x = sympy.symbols('x')
    f = parse_expr(func.__doc__)
    diff = f.diff(x)
    func = FuncCounter(lambda x0: f.evalf(subs={x: x0}, n=n))
    diff_f = FuncCounter(lambda x0: diff.evalf(subs={x: x0}, n=n))
    diff_f_a = diff_f(a)
    diff_f_b = diff_f(b)
    ai = a
    bi = b
    f_a = func(ai)
    f_b = func(bi)
    ci = (f_a - f_b - ai * diff_f_a + bi * diff_f_b) / (diff_f_b - diff_f_a)
    diff_f_c = diff_f(ci)
    while abs(diff_f_c) > eps or bi - ai > eps * 2:
        if diff_f_c < 0:
            ai = ci
            diff_f_a = diff_f_c
            f_a = func(ai)
            ci = (f_a - f_b - ai * diff_f_a + bi * diff_f_b) / (diff_f_b - diff_f_a)
            diff_f_c = diff_f(ci)
        else:
            bi = ci
            diff_f_b = diff_f_c
            f_b = func(bi)
            ci = (f_a - f_b - ai * diff_f_a + bi * diff_f_b) / (diff_f_b - diff_f_a)
            diff_f_c = diff_f(ci)
    if debug:
        print("tangent_lines_method func counter", func.counter)
        print("tangent_lines_method diff counter", diff_f.counter)
    return ci


def newton_raphson_method(func, x0, eps=0.01, n=16, debug=False):
    """Метод Ньютона-Рафсона"""
    x = sympy.symbols('x')
    f = parse_expr(func.__doc__)
    diff = f.diff(x)
    func = FuncCounter(lambda _x: f.evalf(subs={x: _x}, n=n))
    diff_f = FuncCounter(lambda _x: diff.evalf(subs={x: _x}, n=n))
    diff2 = diff.diff(x)
    diff2_f = FuncCounter(lambda _x: diff2.evalf(subs={x: _x}, n=n))
    diff_f_xk = diff_f(x0)
    diff2_f_xk = diff2_f(x0)
    xk = x0 - diff_f_xk / diff2_f_xk
    while abs(xk - x0) > eps:
        x0 = xk
        diff_f_xk = diff_f(x0)
        diff2_f_xk = diff2_f(x0)
        xk = x0 - diff_f_xk / diff2_f_xk
    if debug:
        print("newton_raphson_method func counter", func.counter)
        print("newton_raphson_method diff counter", diff_f.counter)
        print("newton_raphson_method diff2 counter", diff_f.counter)

    return xk


def chord_method(func, x0, eps=0.01, n=16, debug=False):
    """Метод Хорд"""
    x = sympy.symbols('x')
    f = parse_expr(func.__doc__)
    diff = f.diff(x)
    func = FuncCounter(lambda _x: f.evalf(subs={x: _x}, n=n))
    diff_f = FuncCounter(lambda _x: diff.evalf(subs={x: _x}, n=n))
    diff_f_x0 = diff_f(x0)
    xk = x0 + 2.1 * eps
    diff_f_xk = diff_f(xk)
    xk1 = xk - diff_f_xk * (xk - x0) / (diff_f_xk - diff_f_x0)
    while abs(diff_f_xk) > eps:
        x0 = xk
        xk = xk1
        diff_f_x0 = diff_f_xk
        diff_f_xk = diff_f(xk)
        xk1 = xk - diff_f_xk * (xk - x0) / (diff_f_xk - diff_f_x0)
    if debug:
        print("chord_method func counter", func.counter)
        print("chord_method diff counter", diff_f.counter)
    return xk
