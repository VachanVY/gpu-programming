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
        new_max = max(old_max, x[i_j]);
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


// int main(){
//     size_t A = 64;
//     size_t B = 32;

//     dim3 nthreads(A); // num threads per block
//     dim3 nblocks(CEIL_DIV(A, nthreads.x)); // num blocks totaly in the grid
//     std::printf("nthreads.x: %d\n", nthreads.x);
//     std::printf("nblocks.x: %d\n", nblocks.x);

//     // Allocate GPU memory
//     float *x = (float *)malloc(A * B * sizeof(float));
//     fill_random_normal(x, A * B);

//     // cuda_transfer x to gpu
//     float *d_x, *d_out;
//     cudaMalloc(&d_x, A * B * sizeof(float));
//     cudaMemcpy(d_x, x, A * B * sizeof(float), cudaMemcpyHostToDevice);
//     cudaMalloc(&d_out, A * B * sizeof(float));

//     // kernel<<<num_blocks, num_threads>>>(...)
//     softmax_v1_kernel<<<nblocks, nthreads>>>(d_x, d_out, A, B);

//     cudaDeviceSynchronize();

//     // // TORCH COMPARISON
//     // auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

//     // auto torch_custom_out = torch::from_blob(d_out, {(long)A, (long)B}, options);
//     // auto torch_x = torch::from_blob(x, {(long)A, (long)B}, torch::kFloat32).clone();
//     // auto torch_out = torch::softmax(torch_x, 1);

//     // auto diff = (torch_custom_out - torch_out).abs();
//     // std::printf("Max diff: %f\n", diff.max().item<float>());
//     // std::printf("Mean diff: %f\n", diff.mean().item<float>());

//     free(x);
//     cudaFree(d_x);
//     cudaFree(d_out);
//     return 0;
// }
