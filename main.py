import math
import random

import numpy as np


def cost(A, x, b):
    b_exp = np.matmul(A, x)
    return sum(np.abs(b-b_exp))


A_5x3 = np.random.randint(-5, 6, size=(5, 3))  # random integers in [-5, 5]
b_5 = np.random.randint(-10, 11, size=5)  # random integers in [-10, 10]

# 2) 10x7 system (10 equations, 7 unknowns)
A_10x7 = np.random.randint(-5, 6, size=(10, 7))
b_10 = np.random.randint(-10, 11, size=10)

# 3) 12x8 system (12 equations, 8 unknowns)
A_12x8 = np.random.randint(-5, 6, size=(12, 8))
b_12 = np.random.randint(-10, 11, size=12)

# 4) 15x10 system (15 equations, 10 unknowns)
A_15x10 = np.random.randint(-5, 6, size=(15, 10))
b_15 = np.random.randint(-10, 11, size=15)

print("5x3 System:")
print("A_5x3 =\n", A_5x3)
print("b_5   =", b_5, "\n")
lowest_cost = 1000
lowest_vals = []
for i in range(10000):
    vals = [random.randint(-5,6), random.randint(-5,6), random.randint(-5,6)]
    c = cost(A_5x3, np.array(vals), b_5)
    # print(f"Cost: {c}")
    if lowest_cost > c:
        lowest_cost = c
        lowest_vals = vals
print(f"Lowest cost: {lowest_cost}")
print(f"Lowest vals: {lowest_vals}")