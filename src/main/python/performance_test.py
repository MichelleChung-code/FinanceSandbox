from src.main.python.common.timeit import timeit
import numexpr as ne
import numpy as np
import math
# import numba as nb
from numba import jit


@timeit
def single_threaded(A_np, expression):
    """
    Args:
        A_np: <np.Array> to evaluate expression on
        expression: <str> expression to evaluate

    Returns: Result of single threaded evaluation of given expression
    """
    ne.set_num_threads(1)
    return ne.evaluate(expression)


@timeit
def multi_threaded(A_np, expression, num_threads):
    """
    Args:
        A_np: <np.Array> to evaluate expression on
        expression: <str> expression to evaluate
        num_threads: <int> number of threads to use

    Returns: Result of multi-threaded evaluation of given expression
    """
    # number of threads must be an integer
    assert isinstance(num_threads, int)

    ne.set_num_threads(num_threads)
    return ne.evaluate(expression)


@timeit
def f_python(i, j):
    # python implementation
    result = 0
    for x in range(i):
        for y in range(j):
            result += math.cos(math.log(1))

    return result


@timeit
@jit(nopython=True)
def f_numpy(i, j):
    # vectorized with numpy and use numba for even more optimization
    x = np.ones((i, j))
    return np.sum(np.cos(np.log(x)))


if __name__ == '__main__':
    I = 5000000
    A_np = np.arange(I)  # make array of length I
    ex = 'abs(sin(A_np)) ** 5 + cos(10 + 4 * A_np)'

    res_single = single_threaded(A_np, ex)
    res_multi = multi_threaded(A_np, ex, 10)

    i, j = 5000, 5000
    res_py = f_python(i, j)
    res_np = f_numpy(i, j)
