import typing as tp


def derivative(func: tp.Callable, point: float) -> float:
    return (func(point + 1e-10) - func(point - 1e-10)) / 2e-10


def multiplicity_newton(func: tp.Callable,
                        interval: tuple[float, float],
                        m: int = 1,
                        eps: float = 1e-5) -> tuple[float, int]:
    x, _ = interval
    cnt = 0
    while abs(func(x)) > eps:
        x -= m * func(x) / derivative(func, x)
        cnt += 1
    return x, cnt


def newton(func: tp.Callable,
           interval: tuple[float, float],
           eps: float = 1e-5) -> tuple[float, int]:
    x_min, x_max = interval
    x = x_max
    cnt = 0
    while abs(func(x)) > eps:
        x -= func(x) / derivative(func, x)
        cnt += 1
    return x, cnt


def bisection(f: tp.Callable, eps: float, interval: tuple[float, float]) -> float:
    a, b = interval
    while abs(a - b) > 2 * eps:
        x = (a + b) / 2
        a_val, x_val = f(a), f(x)
        if a_val * x_val <= 0:
            b = x
        else:
            a = x
    return (a + b) / 2