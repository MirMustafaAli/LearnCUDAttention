#include <stdio.h>
#include <cuda_runtime.h>

template <class T>
__host__ __device__ T ceil_div(T dividend, T divisor)
{
    return (dividend + divisor - 1) / divisor;
}