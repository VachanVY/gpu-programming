#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cooperative_groups.h>
#include <cmath>
#include <device_launch_parameters.h>

#include "include/cuda_utils.cuh"


__global__ void softmax_v3_kernel(
    float* __restrict__ x,   // (A, B)
    float* __restrict__ out, // (A, B)
    size_t A,
    size_t B
){
    unsigned int warp_size = 32;
    // init shared memory for storing per warp `local_max` and `local_norm`
    __shared__ float shmem[1024];
    size_t row = blockIdx.x;
    size_t tid = threadIdx.x;

    // "fused" get `local_max` and `local_norma`
    float local_new_max = -INFINITY;
    float local_old_max = -INFINITY;
    float local_norma = 0.f;
    for(size_t col = tid; col < B; col += blockDim.x){
        size_t idx = row * B + col;
        local_new_max = fmaxf(local_old_max, x[idx]);
        local_norma = local_norma * expf(local_old_max - local_new_max) + expf(x[idx] - local_new_max);
        local_old_max = local_new_max;
    }
    __syncthreads(); // let all threads compute their local max

    // get per warp max `local_warp_max`
    float local_warp_max = local_new_max;
    for(size_t strd = warp_size / 2; strd > 0; strd /= 2){
        local_warp_max = fmaxf(local_warp_max, __shfl_down_sync(
            0xffffffff, // 32 (`warp_size`) bit ones
            local_warp_max,
            strd
        ));
    }

    // get `row_max` by reduction of `shmem` which consists of `local_warp_max` for each warp
    if(blockDim.x > warp_size){
        if(tid % warp_size == 0){
            // first thread of each warp only do the below
            shmem[tid / warp_size] = local_warp_max;
        }
        __syncthreads(); // let all "first" threads finish doing this

        if(tid < warp_size){
            //                          // number of warps   // the first warp number of threads will do global reduction
            local_warp_max = (tid < CEIL_DIV(blockDim.x, warp_size)) ? shmem[tid] : -INFINITY;
            for(size_t strd = warp_size / 2; strd > 0; strd /= 2){
                // all threads in the warp exec this function at the same time
                local_warp_max = fmaxf(local_warp_max, __shfl_down_sync(
                    0xffffffff, // 32 (`warp_size`) bit ones
                    local_warp_max,
                    strd
                ));
            }
            if(tid == 0)
                shmem[0] = local_warp_max;
        }
    } else { // if the number of threads are less than the warp size then we don't need to do anything; it's already reduced
        if(tid == 0)
            shmem[0] = local_warp_max;
    }
    __syncthreads(); // let all threads wait for the `row_max` to be written
    float row_max = shmem[0];
    __syncthreads();

    
    // same for norma
    float local_warp_norma = local_norma * expf(local_new_max - row_max);
    for(size_t strd = warp_size / 2; strd > 0; strd /= 2){
        local_warp_norma += __shfl_down_sync(
            0xffffffff, // 32 (`warp_size`) bit ones
            local_warp_norma,
            strd
        );
    }
    if(blockDim.x > warp_size){
        if(tid % warp_size == 0){
            shmem[tid / warp_size] = local_warp_norma;
        }
        __syncthreads();

        if(tid < warp_size){
            local_warp_norma = (tid < CEIL_DIV(blockDim.x, warp_size)) ? shmem[tid] : 0.f;
            for(size_t strd = warp_size / 2; strd > 0; strd /= 2){
                local_warp_norma += __shfl_down_sync(
                    0xffffffff, // 32 (`warp_size`) bit ones
                    local_warp_norma,
                    strd
                );
            }
            if(tid == 0)
                shmem[0] = local_warp_norma;
        }
    } else {
        if(tid == 0)
            shmem[0] = local_warp_norma;
    }
    __syncthreads();
    float row_norma = shmem[0];
    __syncthreads();

    // softmax calculation
    for(size_t col = tid; col < B; col += blockDim.x){
        size_t idx = row * B + col;
        out[idx] = expf(x[idx] - row_max) / row_norma;
    }
}

extern "C" void softmax(float* d_x, float* d_out, size_t A, size_t B) {
    dim3 threads(1024);
    dim3 blocks(A);
    softmax_v3_kernel<<<blocks, threads>>>(d_x, d_out, A, B);
    cudaDeviceSynchronize();
}

/*
nvcc -arch=sm_86 -Xcompiler -fPIC -shared softmax_v3.cu -o libsoftmax_v3.so
*/