#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cmath>

#include "include/cuda_utils.cuh"


/*
One Block per Row
Each Element for a Thread spread over with step_size of `BlockDim`
*/
__global__ void softmax_v2_kernel(
    float* __restrict__ x,   // (A, B)
    float* __restrict__ out, // (A, B)
    size_t A,
    size_t B
){
    const int T = 1024;
    // init shared memory, row, thread index
    __shared__ float shmem[T]; // shared mem is not present in "bulk", so prolly keep it const size
    size_t row = blockIdx.x;
    size_t tid = threadIdx.x;

    // compute local max and local norma (with elements of same `tid`)
    float local_new_max = -INFINITY;
    float local_old_max = -INFINITY;
    float local_norma = 0.f;
    for(size_t col = tid; col < B; col += blockDim.x){
        size_t idx = row * B + col;
        local_new_max = max(local_old_max, x[idx]);
        local_norma = local_norma * expf(local_old_max - local_new_max) + expf(x[idx] - local_new_max);
        local_old_max = local_new_max;
    }
    shmem[tid] = local_new_max;
    __syncthreads(); // let every thread compute it's local max and local norma

    // shared threads -- get row max
    for(size_t strd = blockDim.x / 2; strd > 0; strd /= 2){
        if(tid < strd){
            shmem[tid] = max(shmem[tid], shmem[tid + strd]);
        }
        __syncthreads();
    }
    float row_max = shmem[0];
    __syncthreads();

    // correct local norma using row max
    local_norma = local_norma * expf(local_new_max - row_max);
    shmem[tid] = local_norma;
    __syncthreads();

    // shared threads -- get row norma
    for(size_t strd = blockDim.x / 2; strd > 0; strd /= 2){
        if(tid < strd){
            shmem[tid] += shmem[tid + strd];
        }
        __syncthreads();
    }
    float row_norma = shmem[0];
    __syncthreads();

    // compute softmax
    for(size_t col = tid; col < B; col += blockDim.x){
        size_t idx = row * B + col;
        out[idx] = expf(x[idx] - row_max) / row_norma;
    }
}


extern "C" void softmax(float* d_x, float* d_out, size_t A, size_t B) {
    dim3 threads(1024);
    dim3 blocks(A);
    softmax_v2_kernel<<<blocks, threads>>>(d_x, d_out, A, B);
    cudaDeviceSynchronize();
}