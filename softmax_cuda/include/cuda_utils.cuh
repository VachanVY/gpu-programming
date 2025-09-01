/*
Ref https://github.com/Maharshi-Pandya/cudacodes/blob/master/softmax/include/cuda_utils.cuh
*/

#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <limits.h>

#define CUDA_CHECK(ans)                        \
    {                                          \
        cudaAssert((ans), __FILE__, __LINE__); \
    }
inline void cudaAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "CUDA error %s: %s at %s: %d\n",
                cudaGetErrorName(code), cudaGetErrorString(code),
                file, line);
        exit(code);
    }
}
#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// warp reduce for any Op
template <typename T, typename Op>
__device__ __forceinline__ T warpReduce(T val, Op op, unsigned int mask = 0xffffffffu)
{
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1)
        val = op(val, __shfl_down_sync(mask, val, offset));
    return val;
}

// Function objects for reduction operations
template <typename T>
struct SumOp
{
    __device__ __forceinline__ T operator()(T a, T b) const
    {
        return a + b;
    }
};

template <typename T>
struct MaxOp
{
    __device__ __forceinline__ T operator()(T a, T b) const
    {
        return a > b ? a : b;
    }
};

// warp reduce for sum and max using function objects
template <typename T>
__device__ __forceinline__ T warpReduceSum(T val)
{
    return warpReduce(val, SumOp<T>());
}

template <typename T>
__device__ __forceinline__ T warpReduceMax(T val)
{
    return warpReduce(val, MaxOp<T>());
}

template <typename T, typename Op>
__device__ __forceinline__ void blockReduce(T val, T *smem, T identity, Op op)
{
    int tx = threadIdx.x;
    int wid = tx / WARP_SIZE;
    int lane = tx % WARP_SIZE;

    val = warpReduce(val, op);

    // when blockDim is greater than 32, we need to do a block level reduction
    // AFTER warp level reductions since we have the 8 maximum values that needs to be reduced again
    // the global max will be stored in the first warp
    if (blockDim.x > WARP_SIZE)
    {
        if (lane == 0)
        {
            // which warp are we at?
            // store the value in its first thread index
            smem[wid] = val;
        }
        __syncthreads();

        // first warp will do global reduction only
        // this is possible because we stored the values in the shared memory
        // so the threads in the first warp will read from it and then reduce
        if (tx < WARP_SIZE)
        {
            val = (tx < (int)CEIL_DIV(blockDim.x, WARP_SIZE)) ? smem[tx] : identity;
            val = warpReduce(val, op);
            if (tx == 0)
                smem[0] = val;
        }
    }
    else
    {
        // this is for when the number of threads in a block are not
        // greater than the warp size, in that case we already reduced
        // so we can store the value
        if (tx == 0)
            smem[0] = val;
    }
}

template <typename T>
__device__ __forceinline__ void blockReduceSum(T val, T *smem, T identity)
{
    return blockReduce(val, smem, identity, SumOp<T>());
}

template <typename T>
__device__ __forceinline__ void blockReduceMax(T val, T *smem, T identity)
{
    return blockReduce(val, smem, identity, MaxOp<T>());
}

#endif // CUDA_UTILS_CUH