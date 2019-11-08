from math import log, tan, sin, sqrt, e, log10, exp, atan2, pi, cos


def with_params_optimisation(**kwargs):
    def deco(func):
        func.__dict__.update(kwargs)
        return func

    return deco


@with_params_optimisation(a=1, b=1.5, eps=0.05)
def func26(x):
    """x ** 2 - 2 * x + exp(-x)"""
    return x ** 2 - 2 * x + exp(-x)


@with_params_optimisation(a=0, b=pi / 4, eps=0.03)
def func27(x):
    """tan(x) - 2 * sin(x)"""
    return tan(x) - 2 * sin(x)


@with_params_optimisation(a=0, b=1, eps=0.1)
def func28(x):
    """sqrt(1 + x ** 2) + exp(-2 * x)"""
    return sqrt(1 + x ** 2) + exp(-2 * x)


@with_params_optimisation(a=1.5, b=2, eps=0.05)
def func29(x):
    """x ** 4 + 4 * x ** 2 - 32 * x + 1"""
    return x ** 4 + 4 * x ** 2 - 32 * x + 1


@with_params_optimisation(a=1, b=1.5, eps=0.05)
def func30(x):
    """1 / 7 * x ** 7 - x ** 3 + 0.5 * x ** 2 - x"""
    return 1 / 7 * x ** 7 - x ** 3 + 0.5 * x ** 2 - x


@with_params_optimisation(a=0.5, b=1, eps=0.05)
def func31(x):
    """x ** 3 - 3 * sin(x)"""
    return x ** 3 - 3 * sin(x)


@with_params_optimisation(a=3, b=3.5, eps=0.02)
def func32(x):
    """5 * x ** 2 - 8 * x ** (5 / 4) - 20 * x"""
    return 5 * x ** 2 - 8 * x ** (5 / 4) - 20 * x


@with_params_optimisation(a=1.5, b=2, eps=0.02)
def func33(x):
    """1 / 3 * x ** 3 - 5 * x + x * log(x)"""
    return 1 / 3 * x ** 3 - 5 * x + x * log(x)


@with_params_optimisation(a=1, b=2, eps=0.02)
def func40(x):
    """x + x ** -2"""
    return x + x ** -2


@with_params_optimisation(a=-5, b=-4, eps=0.02)
def func41(x):
    """x * sin(x) + 2 * cos(x)"""
    return x * sin(x) + 2 * cos(x)


@with_params_optimisation(a=1.5, b=2, eps=0.05)
def func42(x):
    """x ** 4 + 8 * x ** 3 - 6 * x ** 2 - 72 * x + 90"""
    return x ** 4 + 8 * x ** 3 - 6 * x ** 2 - 72 * x + 90


@with_params_optimisation(a=-1, b=0, eps=0.1)
def func43(x):
    """x ** 6 + 3 * x ** 2 + 6 * x - 1"""
    return x ** 6 + 3 * x ** 2 + 6 * x - 1


@with_params_optimisation(a=0.5, b=1, eps=0.05)
def func44(x):
    """10 * x * log(x) - x ** 2 / 2"""
    return 10 * x * log(x) - x ** 2 / 2


def func101(x):
    """4 * x1 ** 2 + x2 ** 2 - 2 * x1 * x2 + 6 * x1 - x2 - 2"""
    x1, x2 = x
    return 4 * x1 ** 2 + x2 ** 2 - 2 * x1 * x2 + 6 * x1 - x2 - 2


def func102(x):
    """sqrt(1 + x1 ** 2 + x2 ** 2)"""
    x1, x2 = x
    return sqrt(1 + x1 ** 2 + x2 ** 2)


def func103(x):
    """x1 ** 2 + x2 ** 2 - cos((x1 - x2) / 2)"""
    x1, x2 = x
    return x1 ** 2 + x2 ** 2 - cos((x1 - x2) / 2)


def func104(x):
    """x1 ** 4 + x2 ** 4 + x1 ** 2 + x2 ** 2 + x1 ** 2 * x2 ** 2"""
    x1, x2 = x
    return x1 ** 4 + x2 ** 4 + x1 ** 2 + x2 ** 2 + x1 ** 2 * x2 ** 2


def func105(x):
    """exp(x1 ** 2 + x2 ** 2 + x3 ** 2)"""
    x1, x2, x3 = x
    return exp(x1 ** 2 + x2 ** 2 + x3 ** 2)


def func106(x):
    """5 * x1 ** 2 + 5 * x2 ** 2 + 4 * x3 ** 2 + 4 * x1 * x2 + 2 * x2 * x3"""
    x1, x2, x3 = x
    return 5 * x1 ** 2 + 5 * x2 ** 2 + 4 * x3 ** 2 + 4 * x1 * x2 + 2 * x2 * x3


@with_params_optimisation(x0=(1, 1))
def func117(x):
    """x1 ** 2 + 2 * x2 ** 2 + exp(x1 + x2)"""
    x1, x2 = x
    return x1 ** 2 + 2 * x2 ** 2 + exp(x1 + x2)


@with_params_optimisation(x0=(0, 0))
def func118(x):
    """2 * x1 ** 2 + x2 ** 2 + x1 * x2 + x1 + x2"""
    x1, x2 = x
    return 2 * x1 ** 2 + x2 ** 2 + x1 * x2 + x1 + x2


@with_params_optimisation(x0=(0, 1, 0))
def func119(x):
    """x1 ** 4 + x2 ** 2 + x3 ** 2 + x1 * x2 + x2 * x3"""
    x1, x2, x3 = x
    return x1 ** 4 + x2 ** 2 + x3 ** 2 + x1 * x2 + x2 * x3


@with_params_optimisation(x0=(1, 1, 1))
def func120(x):
    """exp(x1 ** 2) + (x1 + x2 + x3) ** 2"""
    x1, x2, x3 = x
    return exp(x1 ** 2) + (x1 + x2 + x3) ** 2


@with_params_optimisation(x0=(0, 0))
def func123(x):
    """x1 ** 2 + x2 ** 2 + x1 + x2"""
    x1, x2 = x
    return x1 ** 2 + x2 ** 2 + x1 + x2


@with_params_optimisation(x0=(1, 1))
def func124(x):
    """x1 ** 2 + x1 * x2 + x2 ** 2"""
    x1, x2 = x
    return x1 ** 2 + x1 * x2 + x2 ** 2


@with_params_optimisation(x0=(1, 1))
def func125(x):
    """x1 ** 2 + 2 * x2 ** 2 + exp(x1 + x2)"""
    x1, x2 = x
    return x1 ** 2 + 2 * x2 ** 2 + exp(x1 + x2)


all_funcs = {
    int(name[4:]): f for name, f in globals().items() if name.startswith('func') and name[-1].isdigit()
}