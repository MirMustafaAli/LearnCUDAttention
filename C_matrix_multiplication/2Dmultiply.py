import numpy as np 
import ctypes


lib = ctypes.CDLL("./2dmultiply.so")

lib.multiply_2d_matrix.argtypes = [
    ctypes.POINTER(ctypes.c_float), # A
    ctypes.POINTER(ctypes.c_float), # B
    ctypes.POINTER(ctypes.c_float), # C
    ctypes.c_int, # M
    ctypes.c_int, # N 
    ctypes.c_int # K
]

lib.multiply_2d_matrix.restype = None

M = 2
N = 3
K = 4

A = np.random.rand(M, K).astype(np.float32)
B = np.random.rand(N, K).astype(np.float32)

A = np.ones((M, K), dtype=np.float32)
B = np.ones((N, K), dtype=np.float32)
C = np.ones((M, N), dtype=np.float32)


lib.multiply_2d_matrix(
    A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    B.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    C.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    M, N, K)

out_python = A @ B.T
print("Matrix A =\n ", A)
print("Matrix B =\n ", B)

print("Answer using C =\n ", C)

print("Using Numpy =\n ", A @ B.T)

print("Is this Close  = ",np.allclose(C,out_python ))

