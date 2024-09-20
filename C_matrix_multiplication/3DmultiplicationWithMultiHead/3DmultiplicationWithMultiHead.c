#include <stdio.h>
#include <stdlib.h>
#include <float.h>

void multihead_dot_product(float *MatrixA, float *MatrixB, float *Out, int B, int T, int C, int H)
{
    /*
    This function performs matrix multiplication between two input matrices (MatrixA and MatrixB) over multiple heads and stores the result in the output matrix (Out). The function operates on 4D tensors, where the dimensions represent batch size, time steps, channels, and heads.

The purpose of this function is to compute matrix products across multiple heads, which is common in multi-head attention mechanisms, such as in transformer models. The computation is done for each head separately, and the results are stored in the corresponding slice of the output tensor.

Parameters:

    •	float* MatrixA: A pointer to the first input matrix. This matrix is of shape (B, T, C), where:
    •	float* MatrixB: A pointer to the second input matrix. This matrix also has shape (B, T, C), and it is multiplied with MatrixA across the last two dimensions (channels and heads).
    •	float* Out: A pointer to the output matrix where the result of the matrix multiplication will be stored. It has the same shape (B, H, T, T).

    •	B: Batch size (number of independent sequences).
    •	T: Number of time steps (sequence length).
    •	C: Number of channels (hidden dimension).
    •	H: Number of heads (parallel attention heads).

    */

    int hs = C / H;
    for (int b = 0; b < B; b++)
    {
        for (int t = 0; t < T; t++)
        {
            for (int h = 0; h < H; h++)
            {
                const float *inp_a = MatrixA + b * T * C + t * C + h * hs;
                float *out_c = Out + b * H * T * T + h * T * T + t * T;

                for (int t2 = 0; t2 < T; t2++)
                {
                    const float *inp_b = MatrixB + b * T * C + t2 * C + h * hs;

                    float value = 0.0f;
                    for (int i = 0; i < hs; i++)
                    {
                        value += inp_a[i] * inp_b[i];
                    }
                    out_c[t2] = value;
                }
            }
        }
    }
}

void multiply_attention_together(float *MatrixA, float *MatrixB, float *Out, int B, int T, int C, int H)
{
    /*

    This function performs matrix multiplication between two input matrices (MatrixA and MatrixB) over multiple heads and stores the result in the output matrix (Out). The function operates on 4D tensors, where the dimensions represent batch size, time steps, channels, and heads.
    MatrixA : (B, H, T, T)
    MatrixB : (B, T, C)
    Out : (B, T, C)
    */

    int hs = C / H;
    for (int b = 0; b < B; b++)
    {
        for (int t = 0; t < T; t++)
        {
            for (int h = 0; h < H; h++)
            {
                const float *inp_a = MatrixA + b * H * T * T + h * T * T + t * T;
                float *out_c = Out + b * T * C + t * C + h * hs;

                for (int t2 = 0; t2 < T; t2++)
                {
                    const float *inp_b = MatrixB + b * T * C + t2 * C + h * hs;
                    float att_btt2 = inp_a[t2];
                    for (int i = 0; i < hs; i++)
                    {
                        out_c[i] += att_btt2 * inp_b[i];
                    }
                }
            }
        }
    }
}

int main()
{
    // Example usages
    int B = 1, T = 2, C = 4, H = 2;
    float MatrixA[B * T * C], MatrixB[B * T * C], Out[B * H * T * T];

    // Initialize MatrixA and MatrixB with example values
    for (int i = 0; i < B * T * C; i++)
    {
        MatrixA[i] = 1.0f;
        MatrixB[i] = 1.0f;
    }

    // Call the multiply_3d_multihead function
    multihead_dot_product(MatrixA, MatrixB, Out, B, T, C, H);

    // Print the output matrix
    for (int b = 0; b < B; b++)
    {
        for (int h = 0; h < H; h++)
        {
            printf("Batch %d, Head %d:\n", b, h);
            for (int t = 0; t < T; t++)
            {
                for (int t2 = 0; t2 < T; t2++)
                {
                    printf("%f ", Out[b * H * T * T + h * T * T + t * T + t2]);
                }
                printf("\n");
            }
        }
    }

    return 0;
}