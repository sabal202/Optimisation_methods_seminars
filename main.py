from functions import all_funcs
from nonfirst_order_one_arg_methods import tangent_lines_method
from zero_order_methods import dichotomy_method, gold_sech_method, naive_method

func = all_funcs[32]
a = func.a
b = func.b
eps = func.eps

print("Minimize f(x)=5*(x**2) - 8*(x**(5/4)) - 20*x")
print("=" * 20)
print("naive_method min:", naive_method(func, a, b, eps, debug=True))
print("=" * 20)
print("dichotomy_method min:", dichotomy_method(func, a, b, eps, delta_mult=0.25, debug=True))
print("=" * 20)
print("gold_sech_method min:", gold_sech_method(func, a, b, eps, debug=True))
print("=" * 20)
print("tangent_lines_method min:", tangent_lines_method(func, a, b, eps, debug=True))
