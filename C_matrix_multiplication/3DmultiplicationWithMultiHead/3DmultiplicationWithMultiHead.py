import numpy as np
import ctypes

def multiply_3d_multihead(MatrixA, MatrixB, B, T, C, H):
    """
    Performs matrix multiplication between two input matrices (MatrixA and MatrixB) over multiple heads
    using NumPy operations to speed up computation.

    Parameters:
    •   MatrixA: A NumPy array of shape (B, T, C), where B is the batch size, T is the number of time steps, and C is the number of channels.
    •   MatrixB: A NumPy array of shape (B, T, C), same as MatrixA.
    •   B: Batch size (number of independent sequences).
    •   T: Number of time steps (sequence length).
    •   C: Number of channels (hidden dimension).
    •   H: Number of heads (parallel attention heads).
    
    Returns:
    •   Out: A NumPy array of shape (B, H, T, T), storing the matrix multiplication results across multiple heads.
    """

    hs = C // H  # Each head operates on C/H dimensions
    Out = np.zeros((B, H, T, T))  # Initialize the output array

    # Reshape input matrices to split the heads
    # Reshape to (B, T, H, hs) to separate the head dimensions
    MatrixA_reshaped = MatrixA.reshape(B, T, H, hs)
    MatrixB_reshaped = MatrixB.reshape(B, T, H, hs)

    for b in range(B):
        for h in range(H):
            # For each head, perform matrix multiplication between time steps for each batch
            # MatrixA_reshaped[b, :, h, :] has shape (T, hs)
            # MatrixB_reshaped[b, :, h, :] has shape (T, hs)
            # We want to compute the dot product across `hs` dimensions, so we use np.dot
            
            # Perform matrix multiplication: (T, hs) x (T, hs).T -> (T, T)
            Out[b, h, :, :] = np.dot(MatrixA_reshaped[b, :, h, :], MatrixB_reshaped[b, :, h, :].T)
    
    return Out



lib = ctypes.CDLL("./3dmultiply.so")

lib.multihead_dot_product.argtypes = [
    ctypes.POINTER(ctypes.c_float), # MatrixA
    ctypes.POINTER(ctypes.c_float), # MatrixB
    ctypes.POINTER(ctypes.c_float), # Out
    ctypes.c_int, # B
    ctypes.c_int, # T 
    ctypes.c_int, # C
    ctypes.c_int # H
]

lib.multiply_attention_together.argtypes = [
    ctypes.POINTER(ctypes.c_float), # MatrixA
    ctypes.POINTER(ctypes.c_float), # MatrixB
    ctypes.POINTER(ctypes.c_float), # Out
    ctypes.c_int, # B
    ctypes.c_int, # T 
    ctypes.c_int, # C
    ctypes.c_int # H
]

lib.multiply_attention_together.restype = None
lib.multihead_dot_product.restype = None


# Example usage:
B, T, C = 1, 4, 8  # Batch size, Sequence length, Channel dimension
num_heads = 2

# Randomly initialize matrices
Matrix_A = np.ones((B, T, C), dtype=np.float32)
Matrix_B = np.ones((B, T, C), dtype=np.float32)
Matrix_C = np.ones((B, T, C), dtype=np.float32)


# Perform multihead matrix multiplication
np_attn = multiply_3d_multihead(Matrix_A, Matrix_B, B, T, C, num_heads)
attn = np.zeros(np_attn.shape, dtype=np.float32)

lib.multihead_dot_product(
    Matrix_A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    Matrix_B.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    attn.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    B, T, C, num_heads)

# np_attn = multiply_3d_multihead(Matrix_A, Matrix_B, B, T, C, num_heads)
output = np.zeros((B, T, C), dtype=np.float32)

lib.multiply_attention_together(
    attn.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    Matrix_C.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    B, T, C, num_heads
)

print(f"Batch = {B}, Time steps = {T}, Channels = {C}, Heads = {num_heads}")
print(f"shape of input matrix = {Matrix_A.shape}")
print(f"shape of output matrix = {np_attn.shape}")
print("Output using NumPy:\n", np_attn)
print("Output using C:\n", attn)
print("Are they close = ", np.allclose(np_attn, attn))
print("Output after using Attention together:\n", output, "\n", output.shape)