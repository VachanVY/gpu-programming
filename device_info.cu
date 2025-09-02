// As is from https://github.com/Maharshi-Pandya/cudacodes/blob/master/query-device/main.cu

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    int dev_count;
    cudaDeviceProp prop;

    cudaGetDeviceCount(&dev_count);
    cudaGetDeviceProperties(&prop, 0);

    printf(">> CUDA enabled devices in the system: %d\n", dev_count);
    printf(">> Compute capability: %d.%d\n", prop.major, prop.minor);

    printf(">> Max grid size: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf(">> Max block size: %d\n", prop.maxThreadsPerBlock);

    printf(">> Number of SMs: %d\n", prop.multiProcessorCount);
    printf(">> Clock rate of the SMs (in kHz): %d\n", prop.clockRate);

    printf(">> Max threads dimension: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf(">> Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);

    printf(">> Registers available per block: %d\n", prop.regsPerBlock);
    printf(">> Registers available per SM: %d\n", prop.regsPerMultiprocessor);

    printf(">> Warp size (threads per warp): %d\n", prop.warpSize);
    printf(">> Shared memory size per block: %zd bytes\n", prop.sharedMemPerBlock);
    printf(">> Shared memory size per SM: %zd bytes\n", prop.sharedMemPerMultiprocessor);

    printf(">> L2 cache size: %d bytes\n", prop.l2CacheSize);

    printf(">> Memory bus width: %d bits\n", prop.memoryBusWidth);
    printf(">> Memory clock rate: %d KHz\n", prop.memoryClockRate);

    int cudaCores = prop.multiProcessorCount * 128;
    float clockGHz = prop.clockRate / 1e6;
    float gflops = cudaCores * clockGHz * 2;

    printf(">> Theoretical Max GFLOPS: %.2f\n", gflops);

    float memoryBandwidth = (2 * prop.memoryClockRate * prop.memoryBusWidth) / (8.0 * 1e6);
    printf(">> Maximum Memory Bandwidth: %.2f GB/s\n", memoryBandwidth);
}

// >> CUDA enabled devices in the system: 1
// >> Compute capability: 8.9
// >> Max grid size: (2147483647, 65535, 65535)
// >> Max block size: 1024
// >> Number of SMs: 128
// >> Clock rate of the SMs (in kHz): 2550000
// >> Max threads dimension: (1024, 1024, 64)
// >> Max threads per SM: 1536
// >> Registers available per block: 65536
// >> Registers available per SM: 65536
// >> Warp size (threads per warp): 32
// >> Shared memory size per block: 49152 bytes
// >> Shared memory size per SM: 102400 bytes
// >> L2 cache size: 75497472 bytes
// >> Memory bus width: 384 bits
// >> Memory clock rate: 10501000 KHz
// >> Theoretical Max GFLOPS: 83558.40
// >> Maximum Memory Bandwidth: -65.65 GB/s