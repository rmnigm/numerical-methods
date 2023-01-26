#!/usr/bin/env python
# coding: utf-8

import numpy as np
import warnings

displayer = {
    'float16': np.float16,
    'float32': np.float32,
    'float64': np.float64,
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
    while 1. + num > 1.:
        num = num_type(num / 2)
        k += 1
    print(type_name, "epsilon is 2^-" + str(k))


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for typename in displayer.keys():
        print()
        experiment(typename)
