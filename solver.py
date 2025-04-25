import numpy as np


def solver(cost, n: int, m: int):
    vals = {}
    for i in range(m):
        vals[i] = cost(i)
    return [0] * n
