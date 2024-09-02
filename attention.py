import ctypes
import numpy as np

"""
In this code, we compute the attention mechanism in Python using NumPy and then compare it with a C implementation.
It's easier to debug the NumPy code compared to working directly with C pointers.
We later verify the correctness of the C code by running it through Python's ctypes library.
"""

# Function to compute attention in Python using NumPy
def attention_forward_python(inp, B, T, C, NH):
    """
    Perform the forward pass of the attention mechanism.

    Args:
        inp (np.array): Input tensor of shape (B, T, 3*C).
        B (int): Batch size.
        T (int): Sequence length (number of tokens).
        C (int): Number of channels (embedding dimension).
        NH (int): Number of attention heads.

    Returns:
        out (np.array): Output tensor after applying attention.
        preatt (np.array): Pre-attention scores before softmax.
        att (np.array): Attention weights after softmax.
    """
    C3 = C * 3  # Total number of channels (C for each of query, key, value)
    hs = C // NH  # Head size: number of channels per attention head
    scale = 1.0 / np.sqrt(hs)  # Scaling factor for dot products (improves numerical stability)

    # Initialize output and intermediate matrices
    out = np.zeros((B, T, C))          # Output tensor
    preatt = np.zeros((B, NH, T, T))   # Pre-attention (raw dot-product values)
    att = np.zeros((B, NH, T, T))      # Attention weights after softmax

    # Iterate over each batch
    for b in range(B):
        # Iterate over each time step
        for t in range(T):
            # Iterate over each attention head
            for h in range(NH):
                # Extract the query vector for the current token and head
                query_t = inp[b, t, h*hs:(h+1)*hs]
                maxval = -np.inf  # Initialize max value for numerical stability

                # Calculate query dot product with keys (up to the current token)
                for t2 in range(t+1):
                    key_t2 = inp[b, t2, C+h*hs:C+(h+1)*hs]
                    val = np.dot(query_t, key_t2) * scale  # Scaled dot product
                    preatt[b, h, t, t2] = val  # Store pre-attention score
                    if val > maxval:
                        maxval = val  # Track maximum value for stability

                # Pad with -INFINITY outside of autoregressive region (prevent attention beyond current token)
                for t2 in range(t+1, T):
                    preatt[b, h, t, t2] = -np.inf

                # Calculate the exponentials and their sum
                expsum = 0.0
                for t2 in range(t+1):
                    expv = np.exp(preatt[b, h, t, t2] - maxval)  # Subtract maxval for numerical stability
                    expsum += expv
                    att[b, h, t, t2] = expv  # Store exponential value in attention matrix

                # Invert the sum for normalization (softmax)
                expsum_inv = 1.0 / expsum if expsum != 0.0 else 0.0

                # Normalize to get the softmax output (attention weights)
                for t2 in range(T):
                    if t2 <= t:
                        att[b, h, t, t2] *= expsum_inv
                    else:
                        att[b, h, t, t2] = 0.0  # Zero out values outside autoregressive region

                # Accumulate weighted values into the output of attention
                for i in range(hs):
                    out[b, t, h*hs+i] = 0.0

                # Compute weighted sum of values
                for t2 in range(t+1):
                    value_t2 = inp[b, t2, 2*C+h*hs:2*C+(h+1)*hs]  # Extract the value vector
                    att_btht2 = att[b, h, t, t2]  # Get the corresponding attention weight
                    for i in range(hs):
                        out[b, t, h*hs+i] += att_btht2 * value_t2[i]  # Add weighted value to output

    return out, preatt, att  # Return the final output, pre-attention, and attention weights


if __name__ == '__main__':
    # Load the shared C library containing the attention forward function
    lib = ctypes.CDLL("./libattention.so")

    # Define the argument types and return type for the C function
    lib.attention_forward_cpu.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # out
        ctypes.POINTER(ctypes.c_float),  # preatt
        ctypes.POINTER(ctypes.c_float),  # att
        ctypes.POINTER(ctypes.c_float),  # inp
        ctypes.c_int,  # B
        ctypes.c_int,  # T
        ctypes.c_int,  # C
        ctypes.c_int,  # NH
    ]
    lib.attention_forward_cpu.restype = None  # C function does not return anything

    # Define the dimensions for the input data
    B = 2   # Batch size
    T = 4   # Sequence length
    C = 12  # Number of channels
    NH = 3  # Number of attention heads

    # Create random input tensor with shape (B, T, 3*C)
    inp = np.random.rand(B, T, 3*C).astype(np.float32)

    # Initialize preatt, att, and out arrays to hold intermediate and final results
    preatt = np.zeros((B, NH, T, T), dtype=np.float32)
    att = np.zeros((B, NH, T, T), dtype=np.float32)
    out = np.zeros((B, T, C), dtype=np.float32)

    # Run the attention forward pass in Python
    out_python, preatt_python, att_python = attention_forward_python(inp, B, T, C, NH)

    # Run the C implementation of the attention forward pass using ctypes
    lib.attention_forward_cpu(
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        preatt.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        att.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        inp.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        B, T, C, NH
    )

    # Compare the output of the Python implementation with the C implementation
    res = np.allclose(out_python, out)  # Check if the outputs are close (within a tolerance)

    # Print the results
    print("Output:\n", out[0][0][0])
    print("Are both of them close = ", res)