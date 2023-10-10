#include <cuda_runtime.h>
#include <stdio.h>

typedef float real;

#define CHECK_ERROR(err) { \
	cudaError_t cuda_err = err; \
	if (cuda_err != cudaSuccess) { \
		fprintf(stderr, "%s in %s at line %d\n", cudaGetErrorString(cuda_err), __FILE__, __LINE__); \
		exit(cuda_err); \
	} \
}

#undef C_ORDER

#ifdef C_ORDER
#define MATRIX_ENTRY(i, j, ncols) ((i) * (ncols) + (j))
#else
#define MATRIX_ENTRY(i, j, nrows) ((i) + (j) * (nrows))
#endif

#define N 4096*2
#define M 4096*2

#define BYTES (M * N * sizeof(real))

void initMat(real *A, int nrows, int ncols, float value);

__global__ void matMulGPU(real *A, real *B, real *C)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride =  blockDim.x * gridDim.x;
    
    for (int i = index; i < N; i += stride)
    {
        for (int j = 0; j < N; j++)
        {
            for (int k = 0; k < M; k++)
            {   
                #ifdef C_ORDER
                C[MATRIX_ENTRY(i, j, N)] += A[MATRIX_ENTRY(i, k, M)] * B[MATRIX_ENTRY(k, j, N)]; // in this current set-up there should be no risk of concurrency
                #else
                C[MATRIX_ENTRY(i, j, N)] += A[MATRIX_ENTRY(i, k, N)] * B[MATRIX_ENTRY(k, j, M)]; // B[MATRIX_ENTRY(j, k, N)]
                #endif
            }
        }
    }
}

int main(int argc, char *argv[])
{

    // Host Array
    real *A_h;
    real *B_h;
    real *C_h;

    // Device Arrays [Layout Left is Fortran style]
    real *A_d;
    real *B_d;
    real *C_d;

    A_h = (real *)malloc(BYTES); // NxM
    B_h = (real *)malloc(BYTES); // MxN
    C_h = (real *)malloc(BYTES); // NxN

    cudaMalloc(&A_d, BYTES);
    cudaMalloc(&B_d, BYTES);
    cudaMalloc(&C_d, BYTES);

    // Initialise matrices to a constant value
    initMat(A_h, N, M, 1.0f);
    initMat(B_h, M, N, 2.0f);
    initMat(C_h, N, N, 0.0f);

    cudaMemcpy(A_d, A_h, BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, BYTES, cudaMemcpyHostToDevice);

    /**
     *
     *
     * A problem with using host-device synchronization points, such as cudaDeviceSynchronize(), is that they stall the GPU pipeline.
     * For this reason, CUDA offers a relatively light-weight alternative to CPU timers via the CUDA event API.
     * The CUDA event API includes calls to create and destroy events, record events, and compute the elapsed time in milliseconds
     * between two recorded events.
     *
     *
     *
     */

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    matMulGPU<<<(N + 255) / 256, 256>>>(A_d, B_d, C_d); // pass N, M if defined at runtime
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    float milliseconds = 0; // compare it to the output of nvprof -- is it the same?
    cudaEventElapsedTime(&milliseconds, start, stop);
    float seconds = milliseconds / 1e3;

    // printf("Max error: %fn", maxError);
    unsigned long int n = N * N;                               // number of C entries
    unsigned long int bytes_r = (M + M) * n; // Bytes read per kernel
    unsigned long int bytes_w = N * n;       // Bytes written per kernel
    double Gflops = (n * M * 2.0) * 1.0e-9f;           // multiply-add instruction could be done in a single cycle?
    // I have checked this against CUDA sample and I am pretty sure of it.
    // I am less sure about the number memory reads and writes (clearer if looking at the compilation?
    // as for example I could put sum in a register and read-write once from C - so not correct bandwidth)
    
    // print to standard output
    fprintf(stdout, "\nElapsed MatMul kernel time %.4f [s]\n", seconds);
    fprintf(stdout, "Gflops %f\n", Gflops);
    fprintf(stdout, "Effective Bandwidth: %f [GB/s]\n", (bytes_r + bytes_w) * sizeof(real) / seconds / 1e9);
    fprintf(stdout, "Computational Throughput (single-precision floating point throughput): %f [GFLOP/s]\n", Gflops / seconds);
    fprintf(stdout, "For each matrix entry computed, MatMul reads %d bytes but performs "
                    "only %d multiply-add instructions, i.e. %d FLOPS per element computed\n",
            M * 2 * sizeof(real), M, M * 2);

    // free device data
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    // free host data
    free(A_h);
    free(B_h);
    free(C_h);

    return 0;
}

void initMat(real *A, int nrows, int ncols, float value)
{

#ifdef C_ORDER
    for (int i = 0; i < nrows; i++)
    {
        for (int j = 0; j < ncols; j++)
        {
            A[MATRIX_ENTRY(i, j, ncols)] = value;
        }
    }
#else
    for (int j = 0; j < ncols; j++)
    {
        for (int i = 0; i < nrows; i++)
        {
            A[MATRIX_ENTRY(i, j, nrows)] = value;
        }
    }
#endif
}