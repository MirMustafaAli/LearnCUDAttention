#include<stdio.h>
#include<math.h>
#include<float.h>
#include<stdlib.h>


float* make_random_float(int N) {
    float * arr = (float *) malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) {
        arr[i] = ((float)rand() / (float) RAND_MAX) * 2.0 - 1.0;
    }
    return arr;
}

void multiply_3d_matrix(float* Matrix_A, float* Matrix_B, float* Out, int B, int T, int C){

    // matrix B will be B * T * T, 

    for (int b = 0 ; b < B; b++){
        for (int t = 0 ; t<T; t++ ){
            float* inp_a = Matrix_A + b * T * C + t * C;
            float* out_c = Out + b * T * T + t * T;
            
            for(int t2 = 0 ; t2 < T; t2++ ){
                float* inp_b = Matrix_B + b * T * C + t * C;
                float value = 0.0f;
                for(int i = 0 ; i < C; i++){
                    value += inp_a[i] * inp_b[i];
                }

                out_c[t2] = value;
            }
        }
    }
}