#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cmath>

#include "include/cuda_utils.cuh"
#include "include/rand_utils.cuh"

__global__ void softmax_v1_kernel(
    float* __restrict__ x,   // (A, B)
    float* __restrict__ out, // (A, B)
    size_t A,
    size_t B 
){
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;

    // (1) max and norma
    float new_max = -INFINITY;
    float old_max = -INFINITY;
    float norma = 0.f;
    for(size_t j = 0; j < B; j++){
        size_t i_j = i * B + j;
        new_max = fmaxf(old_max, x[i_j]);
        norma = norma * expf(old_max - new_max) + expf(x[i_j] - new_max);
        old_max = new_max;
    }

    // (2) softmax
    for(size_t j = 0; j < B; j++){
        size_t i_j = i * B + j;
        out[i_j] = expf(x[i_j] - new_max) / norma;
    }
}

extern "C" void softmax(float* d_x, float* d_out, size_t A, size_t B) {
    dim3 threads(A);
    dim3 blocks(CEIL_DIV(A, threads.x));
    softmax_v1_kernel<<<blocks, threads>>>(d_x, d_out, A, B);
    cudaDeviceSynchronize();
}

/*
nvcc -arch=sm_86 -Xcompiler -fPIC -shared softmax_v1.cu -o libsoftmax_v1.so
*/
