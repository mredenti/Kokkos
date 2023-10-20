#include <cuda_runtime.h>
#include <stdio.h>
#include <helper_functions.h> // cuda helper functions

#define CHECK_CUDA_ERROR(err)                                                                           \
    {                                                                                                   \
        cudaError_t cuda_err = err;                                                                     \
        if (cuda_err != cudaSuccess)                                                                    \
        {                                                                                               \
            fprintf(stderr, "%s in %s at line %d\n", cudaGetErrorString(cuda_err), __FILE__, __LINE__); \
            exit(cuda_err);                                                                             \
        }                                                                                               \
    }

#define TIME(start, stop, stream, kernel, milliseconds)                     \
    {                                                                       \
        CHECK_CUDA_ERROR(cudaEventRecord(start, stream));                   \
        kernel;                                                             \
        CHECK_CUDA_ERROR(cudaEventRecord(stop, stream));                    \
        CHECK_CUDA_ERROR(cudaEventSynchronize(stop));                       \
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop)); \
    }

#undef C_ORDER
#ifdef C_ORDER
#define MATRIX_ENTRY(i, j, ncols) ((i) * (ncols) + (j))
#else
#define MATRIX_ENTRY(i, j, nrows) ((i) + (j) * (nrows))
#endif

typedef float real;

#define N 512
#define M 512
#define K 512

#define A_SIZE N *K
#define B_SIZE K *M
#define C_SIZE N *M

#define A_BYTES A_SIZE * sizeof(real)
#define B_BYTES B_SIZE * sizeof(real)
#define C_BYTES C_SIZE * sizeof(real)

#define TILE_WIDTH_X 16
#define TILE_WIDTH_Y 16

void initMat(real *A, const int size);
void matMulCPU(real *C, const real *A, const real *B);
void print_performance(real arith_int, real elapsed_milliseconds);

__global__ void matMulNaiveGPU(real *A, real *B, real *C)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < N; i += stride)
    {
        for (unsigned int j = 0; j < N; j++)
        {
            for (unsigned int k = 0; k < M; k++)
            {
#ifdef C_ORDER
                C[MATRIX_ENTRY(i, j, N)] += A[MATRIX_ENTRY(i, k, M)] * B[MATRIX_ENTRY(k, j, N)]; // in this current set-up there should be no risk of concurrency
#else
                C[MATRIX_ENTRY(i, j, N)] += A[MATRIX_ENTRY(i, k, N)] * B[MATRIX_ENTRY(k, j, M)];
#endif
            }
        }
    }
}

__global__ void matMulTilingGPU(real *A, real *B, real *C)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // unsigned int tid = threadIdx.y * blockDim.x * threadIdx.x ; // thread index (C-order or column major)
    // unsigned int warpid = tid / warpSize;
    // int stride = blockDim.x * gridDim.x + blockDim.y * gridDim.y; // this is in case you schedule less blocks that are necessary

    real sum = 0; // automatic scalar variable

    // shared memory input matrices
    // __device__ __shared__ float Atile[blockDim.y], Btile[blockDim.x] // can go up to Atile[blockDim.y * <(blockDim.x)], Btile[blockDim.x <(BlockDim.y)]
    // Atile[threadIdx.x, threadIdx.y] = A[row, col] // it's like a local to global coordinate mapping
    // THIS APPROACH SEEMS FEASIBLE BUT VERY INCONVENIENT, ON THE OTHER HAND IT WOULD SEEM MORE CONVENIENT IF
    // IN THE SENSE THAT FOR A THREAD TO LOAD ONLY ONE OF THE MATRICES I WOULD HAVE TO CREATE A SINGLE NUMBERED
    // ARRAY WHERE I WOULD KNOW THE ORDER OF THE ENTRIES A AND B
    // THE BLOCKdim IN EITHER X OR Y WAS 1

    // so i think we have answered the question of why it is sensible to have a square tile when the block size
    // in the x and y dimensions is greater than 1

    // now, the second question I had was whether the tile sizes in the x and y dimensions can be smaller
    // than the blocks sizes in the respective dimensions and what is the benefit --
    // mmmh i get the feeling that the tiles in the input matrices simply corresponds to the block sizes, in fact
    // you would still be loading data in the same pattern across all threads of a block... (we will see later)

    // declare shared memory arrays

    // map local block indices to tile indices [C or Fortran order for these matrices?]
    // I AM PRETTY SURE YOU'D WANT THE MATRICES IN C ORDER SINCE THE THREADS WITHIN A WARP ARE SCHEDULED IN C ORDER...
    // ...AND  mmmh I think you'd want A-fortran order and B C-order
    //__device__ __shared__ float Atile[blockDim.y * blockDim.x], Btile[blockDim.x * blockDim.y]; // shared access across all threads in the block
    //__device__ __shared__ real Atile[TILE_WIDTH_Y * TILE_WIDTH_X], Btile[TILE_WIDTH_X * TILE_WIDTH_Y];
    __device__ __shared__ float Atile[TILE_WIDTH_X][TILE_WIDTH_Y], Btile[TILE_WIDTH_Y][TILE_WIDTH_X]; // I think this and the below approach would
    // only work if you know the sizes at compile time
    // __device__ __shared__ float Atile[blockDim.y][blockDim.x], Btile[blockDim.x][blockDim.y]
    for (int j = 0; j < M / TILE_WIDTH_X; j++)
    {
        // load memory addresses
        // to optimise memory access patterns we need to keep the same
        // Atile[threadIdx.x * blockDim.y + k]
        // Atile[MATRIX_ENTRY(threadIdx.x, threadIdx.y, TILE_WIDTH_X)] = A[MATRIX_ENTRY(row, j, N)] // row*M + TILE_WIDTH_X*j + threadIdx.x C-Order
        // Btile[threadIdx.y * blockDim.x + k]
        // Btile[MATRIX_ENTRY(threadIdx.x, threadIdx.y, TILE_WIDTH_X)] = B[MATRIX_ENTRY(row, j, N)] // col*M(nrows) + TILE_WIDTH_Y*j + threadIdx.y Fortran order
        // Btile[MATRIX_ENTRY(threadIdx.x, threadIdx.y, TILE_WIDTH_X)] = B[MATRIX_ENTRY(row, j, N)] // (TILE_WIDTH_Y*j + threadIdx.y)+TILE_WIDTH_X + col C order
        // READ-AFTER-WRITE DEPENDENCE
#ifdef C_ORDER
        Atile[threadIdx.x][threadIdx.y] = A[(TILE_WIDTH_X * j + threadIdx.x) * N + row]; // this would depend on whether it is C or Fortran order
        Btile[threadIdx.y][threadIdx.x] = B[col * K + TILE_WIDTH_Y * j + threadIdx.y];
#else
        Atile[threadIdx.x][threadIdx.y] = A[(TILE_WIDTH_X * j + threadIdx.x) * N + row]; // this would depend on whether it is C or Fortran order
        Btile[threadIdx.y][threadIdx.x] = B[col * K + j * TILE_WIDTH_Y + threadIdx.y];
#endif
        // wait for all threds in the block to load the data
        __syncthreads();
        // wait for all threads in the warp to load the data
        // __syncwarp(); block size/tile size must be less than warp size
        for (int k = 0; k < TILE_WIDTH_X; k++)
        {
            // c-order
            // sum += Atile[MATRIX_ENTRY(threadIdx.y, k, TILE_WIDTH_X)] * Btile[MATRIX_ENTRY(threadIdx.x, k, TILE_WIDTH_X)];
            // A is fortran order and B is C-order
            // sum += Atile[threadIdx.x * blockDim.y + k] * Btile[threadIdx.y * blockDim.x + k]; // assume Btile is also C-order (you might want it Fortran order)
            sum += Atile[k][threadIdx.y] * Btile[k][threadIdx.x];
            // blockDim.x = TILE_WIDTH_X, blockDim.y = TILE_WIDTH_Y
        }
        // Thus none of the threads would load the elements too early and corrupt
        // the input values of other threads.
        //  WRITE-AFTER-READ DEPENDENCE
        //  wait for all threads in the block to have computed the sum before updating the tile matrix entries
        __syncthreads();
        //  wait for all threads in the warp to have computed the sum before updating the tile matrix entries [I guess this would limit the block size to...]
        //  __syncwarp();
    }
#ifdef C_ORDER
    // int index = row * N + col; // index of matrix C corresponding to row and col
    C[MATRIX_ENTRY(row, col, N)] = sum; // in this current set-up there should be no risk of concurrency
#else
    //  int index = col * N + row; // index of matrix C corresponding to row and col
    C[MATRIX_ENTRY(row, col, N)] = sum;
#endif
}

int main(int argc, char *argv[])
{

    // Host Arrays
    real *A_h;
    real *B_h;
    real *C_h;
    real *C_reference;

    // Device Arrays [Layout Left is Fortran style]
    real *A_d;
    real *B_d;
    real *C_d;

    // Allocate Host Arrays
    A_h = (real *)malloc(A_BYTES);         // NxK
    B_h = (real *)malloc(B_BYTES);         // KxM
    C_h = (real *)malloc(C_BYTES);         // NxM
    C_reference = (real *)malloc(C_BYTES); // NxM

    // Allocate Device Arrays
    CHECK_CUDA_ERROR(cudaMalloc(&A_d, A_BYTES));
    CHECK_CUDA_ERROR(cudaMalloc(&B_d, B_BYTES));
    CHECK_CUDA_ERROR(cudaMalloc(&C_d, C_BYTES));

    // Set seed and initialise matrices with random values
    srand(1);
    initMat(A_h, A_SIZE);
    initMat(B_h, B_SIZE);

    // printf("\nA_h[%d] = %f\n", 0, A_h[0]);
    // compute reference solution
    printf("Computing reference solution on CPU...");
    matMulCPU(C_reference, A_h, B_h); //, matrix_size.uiHA, matrix_size.uiWA, matrix_size.uiWB);
    printf("done.\n\n");

    // Copy data to device
    CHECK_CUDA_ERROR(cudaMemcpy(A_d, A_h, A_BYTES, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(B_d, B_h, B_BYTES, cudaMemcpyHostToDevice));

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

    cudaEvent_t start, stop; // wrap all of it into a macro
    real milliseconds = 0;
    bool result;

    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    printf("Launching Naive MatMul to GPU...");
    TIME(start, stop, NULL, (matMulNaiveGPU<<<(N + 255) / 256, 256>>>(A_d, B_d, C_d)), milliseconds);

    CHECK_CUDA_ERROR(cudaMemcpy(C_h, C_d, C_BYTES, cudaMemcpyDeviceToHost)); // copy results to host
    result = sdkCompareL2fe(C_reference, C_h, C_SIZE, /* TOL */ 1.0e-6f);    // verify results

    if (!result) // write a macro for it
    {
        fprintf(stderr, "Comparing Naive Matrix Multiply with CPU results: %s in %s at line %d\n", "FAIL", __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
    // print performance to file
    print_performance(2.0f / (2 * sizeof(real)), milliseconds);
    // arithmetic intensity, computational throughput, matrix sizes
    /**
     *
     * Keep the consistent view that blockDim.x is along the x-axis
     *
     */
    // tiling is about resource sharing efficiency, it goes threads co-operating
    // in order to achieve data reuse. Note that this was not feasible in the
    // vector addition kernel because the threads were not only working on independent
    // computational workloads but the data accessed was also needed only for just that one
    // thread: maybe vector addition is a better use case where you can show the speedup
    // achieved by declaring the input vector in constant memory given that this memory
    // has higher bandwidth and lower latency than DRAM
    dim3 dimBlock(TILE_WIDTH_X, TILE_WIDTH_Y, 1);
    dim3 dimGrid((M + 255) / 256, (N + 255) / 256, 1);
    TIME(start, stop, NULL, (matMulTilingGPU<<<dimGrid, dimBlock>>>(A_d, B_d, C_d)), milliseconds);
    CHECK_CUDA_ERROR(cudaMemcpy(C_h, C_d, C_BYTES, cudaMemcpyDeviceToHost)); // copy results to host
    result = sdkCompareL2fe(C_reference, C_h, C_SIZE, /* TOL */ 1.0e-6f);    // verify results
    if (!result)
    {
        fprintf(stderr, "Comparing Tiled Matrix Multiply with CPU results: %s in %s at line %d\n", "FAIL", __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
    // print performance to file
    print_performance((TILE_WIDTH_X*2.0f) / (2.0*sizeof(real)), milliseconds);

    // cuda event destroy handle
    // free device data
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    // free host data
    free(A_h);
    free(B_h);
    free(C_h);
    free(C_reference);

    return 0;
}

// Allocate matrix with random entries
void initMat(real *array, const int size)
{
    for (unsigned int i = 0; i < size; ++i)
    {
        array[i] = rand() / (real)RAND_MAX;
    }
}

void matMulCPU(real *C, const real *A, const real *B)
{

    real sum;

    for (unsigned int i = 0; i < N; i++)
    {
        for (unsigned int j = 0; j < M; j++)
        {
            sum = 0;
            for (unsigned int k = 0; k < K; k++)
            {
                sum += A[MATRIX_ENTRY(i, k, N)] * B[MATRIX_ENTRY(k, j, M)];
            }
            C[MATRIX_ENTRY(i, j, N)] = sum;
        }
    }
}

void print_performance(real arith_int, real milliseconds)
{

    // printf("Max error: %fn", maxError);
    unsigned long int n = C_SIZE;                 // number of C entries
    unsigned long int bytes_r = (2 * K) * C_SIZE; // Bytes read per kernel
    unsigned long int bytes_w = C_SIZE;           // Bytes written per kernel
    double Gflops = (C_SIZE * M * 2.0) * 1.0e-9f; // multiply-add instruction could be done in a single cycle?
    // I have checked this against CUDA sample and I am pretty sure of it.
    // I am less sure about the number memory reads and writes (clearer if looking at the compilation?
    // as for example I could put sum in a register and read-write once from C - so not correct bandwidth)
    real seconds = milliseconds / 1e3;

    fprintf(stdout, "%-25s %-25s %-25s %-25s %-25s %-25s %-25s\n",
            "Kernel",
            "Arith. Int. [FLOP/B]",
            "Ideal Arith. Int.",
            "Elapsed [s])",
            "GFLOP",
            "Effective BW [GB/s]",
            "[GFLOP/s]");
    fprintf(stdout, "-------------------------------------------------------------------\n");
    fprintf(stdout, "%-25s %-25.8f %-25.8f %-25.8f %-25.8f %-25.8f %-25.8f\n",
            "<kernel_name>",
            arith_int,
            arith_int,
            seconds,
            Gflops,
            (bytes_r + bytes_w) * sizeof(real) / milliseconds / 1e6,
            Gflops / seconds);

    /*fprintf(stdout, "Effective Bandwidth: %f [GB/s]\n", (bytes_r + bytes_w) * sizeof(real) / milliseconds / 1e6);
    fprintf(stdout, "Computational Throughput (single-precision floating point throughput): %f [GFLOP/s]\n", Gflops / seconds);
    fprintf(stdout, "For each matrix entry computed, MatMul reads %d bytes but performs "
                    "only %d multiply-add instructions, i.e. %d FLOPS per element computed\n",
            M * 2 * sizeof(real), M, M * 2);
    */
}