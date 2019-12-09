import sympy
from sympy.parsing.sympy_parser import parse_expr

from fast_descents import accelerated_gradient_descent, constant_step_descent, divergent_series_method, \
    fastest_gradient_descent
from functions import all_funcs
from nonfirst_order_one_arg_methods import tangent_lines_method, newton_raphson_method, chord_method
from simple_descents import per_coordinate_descent, step_partition_descent
from zero_order_methods import dichotomy_method, gold_sech_method, naive_method

func = all_funcs[32]
a = func.a
b = func.b
eps = func.eps

print(f"Minimize f(x)={func.__doc__}")
print("=" * 20)
print("naive_method min:", naive_method(func, a, b, eps=eps, debug=True))
print("=" * 20)
print("dichotomy_method min:", dichotomy_method(func, a, b, eps=eps, delta_mult=0.25, debug=True))
print("=" * 20)
print("gold_sech_method min:", gold_sech_method(func, a, b, eps=eps, debug=True))
print("=" * 20)
print("tangent_lines_method min:", tangent_lines_method(func, a, b, eps=eps, debug=True))
print("=" * 20)
print("newton_raphson_method min:", newton_raphson_method(func, x0=a, eps=eps, debug=True))
print("=" * 20)
print("chord_method min:", chord_method(func, x0=a, eps=eps, debug=True))
print("=" * 20)

func_ = all_funcs[117]
x0 = func_.x0
x1, x2 = sympy.symbols('x1 x2')
f = parse_expr(func_.__doc__)
diff_x1 = f.diff(x1)
diff_x2 = f.diff(x2)
a1 = 0.1
a2 = 0.265
a3 = 0.5

func = lambda x: float(f.evalf(subs={x1: x[0], x2: x[1]}, n=16))
grad = lambda x: (float(diff_x1.evalf(subs={x1: x[0], x2: x[1]}, n=16)), float(diff_x2.evalf(subs={x1: x[0], x2: x[1]}, n=16)))

print(f"Minimize f(x)={func_.__doc__}")

print("=" * 20)
print("per_coordinate_descent min:", per_coordinate_descent(func, x0, eps=0.001, debug=True))
print("=" * 20)
print("step_partition_descent min:",
      step_partition_descent(func, grad, x0, alpha=1, lr=0.5, d=0.5, eps=0.01, debug=True))

print("=" * 20)
print("constant_step_descent with 0.1 min:", constant_step_descent(grad, x0, alpha=0.1, eps=0.01, debug=True))
print("=" * 20)
print("constant_step_descent with 0.265 min:", constant_step_descent(grad, x0, alpha=0.265, eps=0.01, debug=True))
print("=" * 20)
# print("constant_step_descent with 0.5 min:", constant_step_descent(grad, x0, alpha=0.5, eps=0.01, debug=True))
# print("=" * 20) # Расходится
print("divergent_series_method min:", divergent_series_method(grad, x0, eps=0.01, debug=True))
print("=" * 20)
print("fastest_gradient_descent min:", fastest_gradient_descent(func, grad, x0, eps=0.01, debug=True))
print("=" * 20)
print("accelerated_gradient_descent min:", accelerated_gradient_descent(func, grad, x0, eps=0.01, debug=True))
