import numpy as np

k: int = 0
a = np.float16(1)
b = np.float32(1)
c = np.float64(1)
while a != 0:
    a = np.float16(a / 2)
    k += 1
print("float16 zero is 1/2^" + str(k))
k = 0
while b != 0:
    b = np.float32(b / 2)
    k += 1
print("float32 zero is 1/2^" + str(k))
k = 0
while c != 0:
    c = np.float64(c / 2)
    k += 1
print("float64 zero is 1/2^" + str(k))

