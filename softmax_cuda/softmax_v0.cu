#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cmath>

#include "include/cuda_utils.cuh"


__global__ void softmax_v0_kernel(
    float *__restrict__ x,   // (A, B)
    float *__restrict__ out, // (A, B)
    size_t A,
    size_t B
){
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i >= A) return;

    // 1. find max
    float maxval = -INFINITY;
    for(size_t j = 0; j < B; j++){
        maxval = max(maxval, x[i * B + j]);
    }

    // 2. expf(x - max) for normalization
    float norma = 0.f;
    for(size_t j = 0; j < B; j++){
        norma += expf(x[i * B + j] - maxval);
    }

    // 3. softmax
    for(size_t j = 0; j < B; j++){
        size_t i_j = i * B + j;
        out[i_j] = expf(x[i_j] - maxval) / norma;
    }
}

extern "C" void softmax(float* d_x, float* d_out, size_t A, size_t B) {
    dim3 threads(A);
    dim3 blocks(CEIL_DIV(A, threads.x));
    softmax_v0_kernel<<<blocks, threads>>>(d_x, d_out, A, B);
    cudaDeviceSynchronize();
}

/*
nvcc -arch=sm_86 -Xcompiler -fPIC -shared softmax_v0.cu -o libsoftmax_v0.so
*/