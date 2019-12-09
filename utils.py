class FuncCounter:
    """Класс-обертка для функции, который выглядит как функция

    Метод __call__ вызывается, когда вызываешь экземпляр обертки как ф-ю
    Класс хранит все вызовы функции и их результаты в словаре self.calls
    В атрибуте counter хранится количество уникальных вызовов функции func
    """

    def __init__(self, func, without_memoization=False):
        self.counter = 0
        self.func = func
        self.without_memoization = without_memoization
        self.calls = dict()

    def __call__(self, *args, **kwargs):
        if not self.without_memoization:
            x = args
            if x in self.calls:
                return self.calls[x]

        result = self.func(*args, **kwargs)
        self.counter += 1
        if not self.without_memoization:
            self.calls[x] = result

        return result


def with_params_optimisation(**kwargs):
    def deco(func):
        func.__dict__.update(kwargs)
        return func

    return deco
