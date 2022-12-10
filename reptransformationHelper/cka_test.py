import torch
import numpy as np
from CKA import CKA, CudaCKA


np_cka = CKA()

X = np.random.randn(10000, 100)
Y = np.random.randn(10000, 100)

print('Linear CKA, between X and Y: {}'.format(np_cka.linear_CKA(X, Y)))
print('Linear CKA, between X and X: {}'.format(np_cka.linear_CKA(X, X)))