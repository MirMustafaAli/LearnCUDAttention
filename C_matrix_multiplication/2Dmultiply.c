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




void multiply_2d_matrix(float* A, float* B, float* C, int M, int N, int K){



//output i
// will have to iterate through all the columns of each row for matrix A 
// for each row of matrix A will have to iterative all rows of matrix B
// this will hence to compute (0, 0) of matrix C, will we need values of row1 = 0 of matrix A and all the columns of matrix B for row2 =0 
// henceforth (i, j) of matrix C will be computed by row i of matrix A and column j of matrix B

for (int m = 0 ; m < M; m++){

        float* inp_a = A + m * K; // A matrix pointer moves w.r.t to no of columns i.e. size of K
        float* out_c = C + m * N; // C matrix pointer moves w.r.t to no of columns i.e. size of N

        // will be at starting of the row for matrix A
        // let's get all the valuess of 


        for(int n = 0 ; n < N; n++){

            float* inp_b = B + n * K; // B matrix pointer moves w.r.t to no of columns i.e. size of K

            // right now we are at the starting of the row for matrix B

            // now we will have to iterate through all the columns of matrix B
            // for each column of matrix B, we will have to multiply the values of matrix A with the values of matrix B
            // and then sum them up to get the value of matrix C

            float value = 0.0f;
            for(int i = 0 ; i < K; i++){

                value += inp_a[i] * inp_b[i];
            }
             // this is the value of matrix C at (m, n)
            // Here the head is C which contains the base address of C Matrix. n will give the column number
            out_c[n] = value;
        }   

    }
}


int main(){


int M = 10;
int K = 3;
int N = 3;


// assigning matrix as row ordered vectors
float* A = make_random_float(M * K);
float* B = make_random_float(N * K);
float* C = make_random_float(M * N); // because (M x N ) * (N x M) = (M x M)


multiply_2d_matrix(A, B, C, M, N, K);
return 0;
}