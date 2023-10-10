
#include <cuda.h>
#include <cstdlib>

__global__
void kernel(int sz, double* data)
{
    int _beg = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = _beg; i < sz; ++i)
        data[i] += static_cast<double>(i);
}

int main()
{
    double* data = NULL;
    int blocks = 64;
    int grids = 64;
    int ret = cudaMalloc(&data, blocks * grids * sizeof(double));
    if(ret != cudaSuccess)
        return EXIT_FAILURE;
    kernel<<<grids, blocks>>>(blocks * grids, data);
    cudaDeviceSynchronize();
    return EXIT_SUCCESS;
}
