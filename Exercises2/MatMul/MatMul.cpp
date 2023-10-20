#include "MatMul.hpp"

/**
 *
 * CONSTRUCTORS
 *
 */
template <class T>
MatMul<T>::MatMul(const Kokkos::View<T **> &A, const Kokkos::View<T **> &B, const Kokkos::View<T **> &C)
    : A(A), B(B), C(C){};

template <class T>
MatMul<T>::MatMul(const int n, const int k, const int m) : n(n), k(k), m(m)
{
    A = Kokkos::View<T **, Kokkos::LayoutLeft, Kokkos::CudaSpace>(Kokkos::ViewAllocateWithoutInitializing("A"), n, k);
    B = Kokkos::View<T **, Kokkos::LayoutLeft, Kokkos::CudaSpace>(Kokkos::ViewAllocateWithoutInitializing("B"), n, k);
    C = Kokkos::View<T **, Kokkos::LayoutLeft, Kokkos::CudaSpace>(Kokkos::ViewAllocateWithoutInitializing("C"), n, k);

    // print info about matrix sizes and such ...
    // print info about device information

    sizeC = C.extent(0) * C.extent(0);             // number of C entries
    bytes_r = 2 * A.extent(1) * sizeC * sizeof(T); // Bytes read per kernel
    bytes_w = sizeC * sizeof(T);
    Gflops = sizeC * A.extent(1) * 2 * 1.0e-9f;
};

/**
 *
 * MATRIX MULTIPLICATION KERNELS
 *
 */

template <class T>
void MatMul<T>::outerLoop()
{
    Kokkos::parallel_for("A@B", Kokkos::RangePolicy<OuterLoop>(0, A.extent(0)), *this);
};

template <class T>
KOKKOS_INLINE_FUNCTION void MatMul<T>::operator()(const OuterLoop, const int i) const
{
    // Implementation of operator() here...
    for (int j = 0; j < B.extent(1); j++)
    {
        T sum = 0; // automatic scalar variable created and initialised by each thread 
        // automatic scalar variables are placed into registers 
        // compute a single entry
        for (int k = 0; k < B.extent(0); k++)
        {
            sum += A(i, k) * B(k, j); // in this current set-up there should be no risk of concurrency
        }
        // store the result
        C(i, j) = sum;
    }
};

template <class T>
void MatMul<T>::doubleLoop()
{
    Kokkos::parallel_for("DoubleLoop", Kokkos::MDRangePolicy<DoubleLoop, Kokkos::Rank<2>>({0, 0}, {A.extent(0), B.extent(1)}), *this);
};

template <class T>
KOKKOS_INLINE_FUNCTION void MatMul<T>::operator()(const DoubleLoop, const int i, const int j) const
{
    T sum = 0;
    // compute a single entry
    for (int k = 0; k < B.extent(0); k++)
    {
        sum += A(i, k) * B(k, j); // in this current set-up there should be no risk of concurrency
    }
    // store the result
    C(i, j) = sum;
}

template <class T>
void MatMul<T>::print_performance(T elapsedTotSeconds, int niter){
    
    T elapsedAvgSeconds = elapsedTotSeconds / niter;

    sizeC = C.extent(0) * C.extent(0);             // number of C entries
    bytes_r = 2 * A.extent(1) * sizeC * sizeof(T); // Bytes read per kernel
    bytes_w = sizeC * sizeof(T);
    Gflops = sizeC * A.extent(1) * 2.0 * 1.0e-9f;

    fprintf(stdout, "Avg Computational Throughput (SPFPT):= %.5f [GFLOP/s], Avg Elapsed Time= %.5f [s], Size= %.0f GFLOP\n",
                Gflops / elapsedAvgSeconds, elapsedAvgSeconds, Gflops);

}

// bandwidth is the most important metric to measure and optimize.
//  In more sophisticated computations, measuring performance at the level of FLOPs can be very difficult.
//  Therefore it’s more common to use profiling tools to get an idea of whether computational throughput
//  is a bottleneck. Applications often provide throughput metrics that are problem-specific
//  (rather than architecture specific) and therefore more useful to the user.
// computational throughput: each entry of C is obtained as an inner product [that inspires anothe way for parallelisation]
// an inner product is N multiply and N adds (N-1 truly but implementation wise is N)
// maybe an external routine to print this information to scree
// do a warmup iteration -- thi motivates me to write a functor so that I do not have to write the
// body of the parallel computational kernel multiple times
// Kernel-0