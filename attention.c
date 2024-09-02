
# include <stdio.h>
# include <stdlib.h>
# include <math.h>
# include <float.h>


//the code has been references and taken from 
// https://github.com/karpathy/llm.c/blob/master/dev/cuda/attention_forward.cu 
// all credits to this repo for providing such amazing code consisting of exhaustive pointers
float* make_random_float(int N) {
    float * arr = (float *) malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) {
        arr[i] = ((float)rand() / (float) RAND_MAX) * 2.0 - 1.0;
    }
    return arr;
}

void attention_forward_cpu(float* out, float* preatt, float* att, const float* inp, int B, int T, int C, int NH) {
    // inp ( B, T, 3C) Q, K, V (this includes the weights Q,K,V).
    // attn (B, NH, T, T)
    // preattn (B, NH, T, T)
    // out (B, T, C)

    int C3 = 3 * C;
    int hs = C / NH;
    float scale = 1.0/sqrtf(hs);
    for(int b = 0 ; b < B ; b++) {
        for(int t = 0 ; t < T ; t++) {
            for(int h = 0 ; h < NH ; h++) {
                const float* query_t = inp+ b*T*3*C + t*3*C + h*hs;  //inp[b][t][c][h]
                float* preatt_bth = preatt + b*NH*T*T + h*T*T+ t*T; // because
                float* att_bth = att + b*NH*T*T + h*T*T + t*T;

                float maxval = -FLT_MAX;
                for(int t2=0; t2<= t; t2++) {
                    const float* key_t2 = inp + b*T*C3 + t2*C3 + h*hs + C;

                    float val = 0.0f;
                    for(int i = 0 ; i < hs; i++) {
                        val += query_t[i] * key_t2[i];
                    }
                    val *= scale;
                    if(val > maxval) {
                        maxval = val;
                    }

                    preatt_bth[t2] = val;
                }

                for (int t2=t+1; t2<T; t2++) {
                    preatt_bth[t2] = -INFINITY;
                }

                float expsum = 0.0f;
                for(int t2 =0 ; t2 <= t; t2++) {
                    float expv =expf(preatt_bth[t2] - maxval);
                    expsum += expv;
                    att_bth[t2] = expv;
                }

                float expsum_inv = expsum == 0.0f ? 0.0f: 1.0f/expsum;

                // pass 3 : normalize to get the softmax
                for (int t2=0; t2 < T; t2++) {
                    if(t2 <= t) {
                        att_bth[t2] *= expsum_inv;
                    } else {
                        att_bth[t2] = 0.0f;
                    }
                }

                // pass 4 : accumulate weighted values into the output of attention
                float* out_bth = out + b*T*C+t*C+h*hs;
                for(int i = 0 ; i<hs; i++){ out_bth[i] = 0.0f;}
                for(int t2=0; t2<=t; t2++) {
                    const float* value_t2 = inp + b*T*C3 + t2*C3+ h*hs + C*2; // C*2 because it's value
                    float att_btht2 = att_bth[t2];
                    for (int i = 0 ; i < hs; i++) {
                        out_bth[i] += att_btht2 * value_t2[i];

                    }
                }
            }
        }
    }
}


int main(int argc, char **argv) {

    int B = 8;
    int T = 1024;
    int C = 768;
    int NH = 12;


    float* out = (float*) malloc(B * T*C * sizeof(float));
    float* preattn = (float*) malloc(B * NH * T * T *sizeof(float));
    float* attn = (float*) malloc(B * NH * T * T * sizeof(float));
    float* inp = make_random_float(B * T * 3 * C);

    attention_forward_cpu(out, preattn, attn, inp, B, T, C, NH);

    // printf("%f", out[0]);
    // printf("%lu\n", sizeof(float));
    free(out);
    free(preattn);
    free(attn);
    free(inp);

}