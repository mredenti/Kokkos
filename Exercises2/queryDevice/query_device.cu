#include <cuda_runtime.h>
#include <stdio.h>

int main(int argc, char *argv[])
{

    int devCount;
    cudaDeviceProp devProp;

    cudaGetDeviceCount(&devCount);
    cudaGetDeviceProperties(&devProp, 0);

    fprintf(stdout, "Compute Capability %d.%d\n", devProp.major, devProp.minor);

    fprintf(stdout, "Number of multiprocessors on device %d\n", devProp.multiProcessorCount);
    fprintf(stdout, "Maximum number of resident blocks per multiprocessor %d\n", devProp.maxBlocksPerMultiProcessor);

    fprintf(stdout, "Max Threads/Block: %d\n", devProp.maxThreadsPerBlock);
    fprintf(stdout, "Max Threads/SM: %d\n", devProp.maxThreadsPerMultiProcessor);
    fprintf(stdout, "Max Threads in (x,y,z) dimension: (%d, %d, %d)\n",
            devProp.maxThreadsDim[0],
            devProp.maxThreadsDim[1],
            devProp.maxThreadsDim[2]);
    fprintf(stdout, "Max Size in (x,y,z) dimension grid: (%d, %d, %d)\n", 
            devProp.maxGridSize[0],
            devProp.maxGridSize[1],
            devProp.maxGridSize[2]);

    fprintf(stdout, "\n# (32-bit) registers/SM: %d\n", devProp.regsPerMultiprocessor);
    fprintf(stdout, "\n# (32-bit) registers/Block: %d\n", devProp.regsPerBlock);
    
    fprintf(stdout, "# threads/warp (warp size): %d\n", devProp.warpSize);

    fprintf(stdout, "\n\nMemory Architecture\n");
    fprintf(stdout, "Global Memory: %f [GB]\n", (double) devProp.totalGlobalMem * 1e-9f);
    //fprintf(stdout, "Global Memory: %zu [B]\n", devProp.totalGlobalMem);
    fprintf(stdout, "L2 Cache: %f [MB]\n", (double) devProp.l2CacheSize * 1e-6f);

    fprintf(stdout, "\n\nClock frequency %f [MHz]\n", (double) devProp.clockRate * 1e-3f);
    fprintf(stdout, "Ratio of single precision performance (in floating-point operations per second) to double precision performance %d\n", devProp.singleToDoublePrecisionPerfRatio);

    return EXIT_SUCCESS;
}