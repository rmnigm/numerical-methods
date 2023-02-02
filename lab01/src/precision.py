#!/usr/bin/env python
# coding: utf-8

import numpy as np
import warnings

displayer = {
    'float': np.single,
    'double': np.double,
    'long double': np.longdouble,
}


def experiment(type_name):
    k: int = 0
    num_type = displayer[type_name]
    num = num_type(1)
    while num != 0:
        num = num_type(num / 2)
        k += 1
    print(type_name, "zero is 2^-" + str(k))
    k = 0
    num = num_type(1)
    while num != np.inf:
        num = num_type(num * 2)
        k += 1
    print(type_name, "infinity is 2^" + str(k))
    k = 0
    num = num_type(1)
    while num_type(1.) + num > num_type(1.):
        num = num_type(num / 2)
        k += 1
    print(type_name, "epsilon is 2^-" + str(k))


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for typename in displayer.keys():
        print()
        experiment(typename)
