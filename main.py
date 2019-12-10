import numpy as np
import sympy
from sympy.matrices import Matrix
from sympy.parsing.sympy_parser import parse_expr

from descent_optimizations import modified_newton_method, newton_method, ravine_method
from fast_descents import accelerated_gradient_descent, constant_step_descent, divergent_series_method, \
    fastest_gradient_descent
from fletcher_reeves import fletcher_reeves_method
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

"""
Minimize f(x)=5 * x ** 2 - 8 * x ** (5 / 4) - 20 * x
====================
naive_method func counter 26
naive_method min: 3.36
====================
dichotomy_method func counter 8
dichotomy_method min: 3.35828125
====================
gold_sech_method func counter 7
gold_sech_method min: 3.350813061875578
====================
tangent_lines_method func counter 7
tangent_lines_method diff counter 8
tangent_lines_method min: 3.352016978583271
====================
newton_raphson_method func counter 0
newton_raphson_method diff counter 2
newton_raphson_method diff2 counter 2
newton_raphson_method min: 3.353210004779323
====================
chord_method func counter 0
chord_method diff counter 3
chord_method min: 3.354781254281020
====================
"""

func_ = all_funcs[117]
x0 = func_.x0
x1, x2 = sympy.symbols('x1 x2')
f = parse_expr(func_.__doc__)
diff_x1 = f.diff(x1)
diff_x2 = f.diff(x2)
a1 = 0.1
a2 = 0.265
a3 = 0.5
eps = 0.00001

func = lambda x: float(f.evalf(subs={x1: x[0], x2: x[1]}))
grad = lambda x: (
    float(diff_x1.evalf(subs={x1: x[0], x2: x[1]})), float(diff_x2.evalf(subs={x1: x[0], x2: x[1]})))


print(f"Minimize f(x)={func_.__doc__}")

print("=" * 20)
print("per_coordinate_descent min:", per_coordinate_descent(func, x0, eps=eps, debug=True))
print("=" * 20)
print("step_partition_descent min:",
      step_partition_descent(func, grad, x0, alpha=1, lr=0.5, d=0.5, eps=eps, debug=True))

print("=" * 20)
print("constant_step_descent with 0.1 min:", constant_step_descent(grad, x0, alpha=0.1, eps=eps, debug=True))
print("=" * 20)
print("constant_step_descent with 0.265 min:", constant_step_descent(grad, x0, alpha=0.265, eps=eps, debug=True))
print("=" * 20)
# print("constant_step_descent with 0.5 min:", constant_step_descent(grad, x0, alpha=0.5, eps=0.01, debug=True))
# print("=" * 20) # Расходится
print("divergent_series_method min:", divergent_series_method(grad, x0, eps=eps, debug=True))
print("=" * 20)
print("fastest_gradient_descent min:", fastest_gradient_descent(func, grad, x0, eps=eps, debug=True))
print("=" * 20)
print("accelerated_gradient_descent min:", accelerated_gradient_descent(func, grad, x0, eps=eps, debug=True))

inv = sympy.simplify(Matrix([[diff_x1.diff(x1), diff_x1.diff(x2)], [diff_x2.diff(x1), diff_x2.diff(x2)]]).inv())
inv_hesse_mat = lambda x: np.array(inv.evalf(subs={x1: x[0], x2: x[1]}), dtype=float)

print("=" * 20)
print("ravine_method min:", ravine_method(func, grad, x0, eps=eps, debug=True))
print("=" * 20)
print("newton_method min:", newton_method(grad, inv_hesse_mat, x0, eps=eps, debug=True))
print("=" * 20)
print("modified_newton_method min:", modified_newton_method(func, grad, inv_hesse_mat, x0, eps=eps, debug=True))
print("=" * 20)
print("fletcher_reeves_method min:", fletcher_reeves_method(func, grad, x0, eps=eps, debug=True))

"""
Minimize f(x)=x1 ** 2 + 2 * x2 ** 2 + exp(x1 + x2)
====================
coordinate_descent minimizations counter 8
per_coordinate_descent min: (0, 0)
====================
step_partition_descent func counter 59
step_partition_descent grad counter 15
step_partition_descent min:           (-0.312763, -0.156384)
====================
constant_step_descent grad counter 43
constant_step_descent with 0.1 min:   (-0.312763, -0.156384)
====================
constant_step_descent grad counter 15
constant_step_descent with 0.265 min: (-0.312768, -0.156382)
====================
divergent_series_method grad counter 36831
divergent_series_method min:          (-0.312770, -0.156382)
====================
fastest_gradient_descent func counter 171
fastest_gradient_descent grad counter 10
fastest_gradient_descent minimizations counter 9
fastest_gradient_descent min:         (-0.312763, -0.156385)
====================
accelerated_gradient_descent func counter 266
accelerated_gradient_descent grad counter 16
accelerated_gradient_descent FGD counter 2
accelerated_gradient_descent min:     (-0.312765, -0.156384)
====================
ravine_method func counter 328
ravine_method grad counter 16
ravine_method FGD counter 4
ravine_method min:                    (-0.312766, -0.156383)
====================
newton_method grad counter 6
newton_method inv_hesse_mat counter 5
newton_method min:                    (-0.312766, -0.156383)
====================
modified_newton_method func counter 100
modified_newton_method grad counter 5
modified_newton_method minimizations counter 4
modified_newton_method inv_hesse_mat counter 4
modified_newton_method min:           (-0.312766, -0.156383)
====================
fletcher_reeves_method func counter 150
fletcher_reeves_method grad counter 7
fletcher_reeves_method minimizations counter 6
fletcher_reeves_method min:           (-0.312766, -0.156383)
"""