#include <cuda_runtime.h>
#include <stdio.h>

#define N 1 << 20
#define BYTES N*sizeof(float)

__global__ void vectorAddGPU(float* c, const float* a, const float* b){

    // It is important to note that the size of the constant array needs to be known at compile time, therefore the use of the define preprocessor statement.

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < N; i += stride){
        // a,b may be loaded from DRAM or constant memory (it might be the case that constant memory still resides on DRAM?)
        c[idx] = a[idx] + b[idx];
    }

}

int main(int argc, char* argv){

    int i;

    // Host Vectors
    float A_h[N];
    float B_h[N];
    float C_h[N];

    // Device Vectors
    __constant__ float *A_d;
    __constant__ float *B_d;
    float *C_d;

    // Allocate data on device 
    cudaMalloc(&A_d, BYTES);
    cudaMalloc(&B_d, BYTES);
    cudaMalloc(&C_d, BYTES);

    // initialise a,b vectors on host (could use openmptooo)
    for (i=0; i<N; i++){
        A_h[i]=1.0f;
        B_h[i]=2.0f;
    }
    
    // Copy data to GPU memory
    cudaMemcpy(A_d, A_h, BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, BYTES, cudaMemcpyHostToDevice);

    // Invoke vector addition kernel 
    vectorAddGPU<<<(N + 255) / 256, 256>>>(C_d, A_d, B_d);

    // Copy output vector C back to host to verify results
    cudaMemcpy(C_h, C_d, BYTES, cudaMemcpyDeviceToHost);

    // Verify results (could use openmp too)
    // Average value of C_h entries should be equal 3 within error
    float sum = 0;
    for(i=0; i<N; i++){
        sum += C_h[i];
    }
    
    printf("Average %f", sum);
    
    // Release device memory
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    return EXIT_SUCCESS;

}

