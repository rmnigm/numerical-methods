import pytest
import numpy as np
import typing as tp
import dataclasses

from utils.diff import deriv


@dataclasses.dataclass
class Case:
    argument: float | np.ndarray
    func: tp.Callable[[float | np.ndarray], float | np.ndarray]
    result: float | np.ndarray

    def __str__(self) -> str:
        return f'{self.func.__name__}({self.argument})'


def square(x):
    return x ** 2


def cube(x):
    return x ** 3


def constant_scalar(x):
    return 1


def constant_vector(x):
    return np.ones_like(x)


DERIV_TEST_CASES = [
    Case(argument=1, func=square, result=2),
    Case(argument=2, func=square, result=4),
    Case(argument=np.array([3, 5]), func=square, result=np.array([6, 10])),
    Case(argument=np.array([0, -2]), func=square, result=np.array([0, -4])),
    Case(argument=1, func=cube, result=3),
    Case(argument=2, func=cube, result=12),
    Case(argument=np.array([3, 5]), func=cube, result=np.array([27, 75])),
    Case(argument=np.array([0, -2]), func=cube, result=np.array([0, 12])),
    Case(argument=1, func=constant_scalar, result=0),
    Case(argument=np.array([0, -2]), func=constant_vector, result=np.array([0, 0])),
]


@pytest.mark.parametrize("t", DERIV_TEST_CASES, ids=str)
def test_deriv(t: Case) -> None:
    assert deriv(t.func, t.argument, eps=1e-10) == pytest.approx(t.result, 1e-5)
