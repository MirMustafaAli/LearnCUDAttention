#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "common.h"

float *make_random_float(int N)
{
    float *arr = (float *)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++)
    {
        arr[i] = ((float)rand() / (float)RAND_MAX) * 2.0 - 1.0;
    }
    return arr;
}

void attention_forward_cpu(float *out, float *preatt, float *att, const float *inp, int B, int T, int C, int NH)
{
    int C3 = 3 * C;
    int hs = C / NH;
    float scale = 1.0 / sqrtf(hs);
    for (int b = 0; b < B; b++)
    {
        for (int t = 0; t < T; t++)
        {
            for (int h = 0; h < NH; h++)
            {
                const float *query_t = inp + b * T * 3 * C + t * 3 * C + h * hs;
                float *preatt_bth = preatt + b * NH * T * T + h * T * T + t * T;
                float *att_bth = att + b * NH * T * T + h * T * T + t * T;

                float maxval = -FLT_MAX;
                for (int t2 = 0; t2 <= t; t2++)
                {
                    const float *key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C;

                    float val = 0.0f;
                    for (int i = 0; i < hs; i++)
                    {
                        val += query_t[i] * key_t2[i];
                    }
                    val *= scale;
                    if (val > maxval)
                    {
                        maxval = val;
                    }

                    preatt_bth[t2] = val;
                }

                for (int t2 = t + 1; t2 < T; t2++)
                {
                    preatt_bth[t2] = -INFINITY;
                }

                float expsum = 0.0f;
                for (int t2 = 0; t2 <= t; t2++)
                {
                    float expv = expf(preatt_bth[t2] - maxval);
                    expsum += expv;
                    att_bth[t2] = expv;
                }

                float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

                for (int t2 = 0; t2 < T; t2++)
                {
                    if (t2 <= t)
                    {
                        att_bth[t2] *= expsum_inv;
                    }
                    else
                    {
                        att_bth[t2] = 0.0f;
                    }
                }

                float *out_bth = out + b * T * C + t * C + h * hs;
                for (int i = 0; i < hs; i++)
                {
                    out_bth[i] = 0.0f;
                }
                for (int t2 = 0; t2 <= t; t2++)
                {
                    const float *value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C * 2;
                    float att_btht2 = att_bth[t2];
                    for (int i = 0; i < hs; i++)
                    {
                        out_bth[i] += att_btht2 * value_t2[i];
                    }
                }
            }
        }
    }
}

__global__ void attention_query_key_kernels(float *preatt, const float *inp, int B, int T, int C, int NH)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = B * NH * T * T;

    if (idx < total_threads)
    {
        int t2 = idx % T;
        int t = (idx / T) % T;

        if (t2 > t)
        {
            preatt[idx] = -INFINITY;
            return;
        }

        int h = (idx / (T * T)) % NH;
        int b = idx / (NH * T * T);

        int C3 = C * 3;
        int hs = C / NH;

        const float *query_t = inp + b * T * 3 * C + t * 3 * C + h * hs;
        const float *key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C;

        float val = 0.0f;

        for (int i = 0; i < hs; i++)
        {
            val += query_t[i] * key_t2[i];
        }

        val *= 1.0 / sqrtf(hs);
        preatt[idx] = val;
    }
}

__global__ void attention_softmax_kernel(float *att, const float *preatt, int B, int T, int NH)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = B * T * NH;

    if (idx < total_threads)
    {
        int h = idx % NH;
        int t = (idx / NH) / T;
        int b = idx / (NH * T);

        const float *preatt_bth = preatt + b * NH * T * T + h * T * T + t * T;
        float *att_bth = att + b * NH * T * T + h * T * T + t * T;

        float maxval = -FLT_MAX;
        for (int t2 = 0; t2 <= t; t2++)
        {
            if (preatt_bth[t2] > maxval)
            {
                maxval = preatt_bth[t2];
            }
        }

        float expsum = 0.0f;
        for (int t2 = 0; t2 <= t; t2++)
        {
            float expv = expf(preatt_bth[t2] - maxval);
            expsum += expv;
            att_bth[t2] = expv;
        }

        float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

        for (int t2 = 0; t2 < T; t2++)
        {
            if (t2 <= t)
            {
                att_bth[t2] *= expsum_inv;
            }
            else
            {
                att_bth[t2] = 0.0f;
            }
        }
    }
}

__global__ void attention_value_kernel1(float *out, const float *att, const float *inp, int B, int T, int C, int NH)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = B * T * NH;

    if (idx < total_threads)
    {
        int h = idx % NH;
        int t = (idx / NH) % T;
        int b = idx / (NH * T);

        int C3 = C * 3;
        int hs = C / NH;

        float *out_bth = out + b * T * C + t * C + h * hs;
        const float *att_bth = att + b * NH * T * T + h * T * T + t * T;
        for (int i = 0; i < hs; i++)
        {
            out_bth[i] = 0.0f;
        }
        for (int t2 = 0; t2 <= t; t2++)
        {
            const float *value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C * 2;
            float att_btht2 = att_bth[t2];
            for (int i = 0; i < hs; i++)
            {
                out_bth[i] += att_btht2 * value_t2[i];
            }
        }
    }
}

void attention_forward1(float *out, float *preatt, float *att, const float *inp, int B, int T, int C, int NH, const int block_size)
{
    int total_threads = B * NH * T * T;
    int num_blocks = ceil_div(total_threads, block_size);
    attention_query_key_kernels<<<num_blocks, block_size>>>(preatt, inp, B, T, C, NH);

    total_threads = B * T * NH;
    num_blocks = ceil_div(total_threads, block_size);
    attention_softmax_kernel<<<num_blocks, block_size>>>(att, preatt, B, T, NH);

    attention_value_kernel1<<<num_blocks, block_size>>>(out, att, inp, B, T, C, NH);

    // Ensure all kernels have completed before proceeding
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Error in attention_forward1: %s\n", cudaGetErrorString(err));
    }
}

int main(int argc, char **argv)
{
    int B = 8;
    int T = 1024;
    int C = 768;
    int NH = 12;

    float *out = (float *)malloc(B * T * C * sizeof(float));
    float *preattn = (float *)malloc(B * NH * T * T * sizeof(float));
    float *attn = (float *)malloc(B * NH * T * T * sizeof(float));
    float *inp = make_random_float(B * T * 3 * C);

    attention_forward_cpu(out, preattn, attn, inp, B, T, C, NH);
    printf("%f\n", out[0]);

    int block_size = 128;
    attention_forward1(out, preattn, attn, inp, B, T, C, NH, block_size);

    free(out);
    free(preattn);
    free(attn);
    free(inp);

    return 0;
}
