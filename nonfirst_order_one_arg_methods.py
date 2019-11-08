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
