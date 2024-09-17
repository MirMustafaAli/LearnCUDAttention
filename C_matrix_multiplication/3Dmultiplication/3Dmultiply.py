import numpy as np 
import ctypes


lib = ctypes.CDLL("./3dmultiply.so")

lib.multiply_3d_matrix.argtypes = [
    ctypes.POINTER(ctypes.c_float), # A
    ctypes.POINTER(ctypes.c_float), # B
    ctypes.POINTER(ctypes.c_float), # C
    ctypes.c_int, # M
    ctypes.c_int, # N 
    ctypes.c_int # K
]

lib.multiply_3d_matrix.restype = None


B = 2
T = 3
C = 4

# A = np.random.rand(B, T, C).astype(np.float32)
# B = np.random.rand(B, T, C).astype(np.float32)

matrixA = np.ones((B, T, C), dtype=np.float32)
matrixB = np.ones((B, T, C), dtype=np.float32)
out = np.ones((B, T, T), dtype=np.float32)


lib.multiply_3d_matrix(
    matrixA.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    matrixB.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    B, T, C)

print("Matrix A =\n ", matrixA)
print("Matrix B =\n ", matrixB)

print("Answer using C =\n ", out)

print("Using Numpy =\n ", matrixA @ matrixB.transpose(0,2,1))

print("Is this Close  = ",np.allclose(out,matrixA @ matrixB.transpose(0,2,1) ))

