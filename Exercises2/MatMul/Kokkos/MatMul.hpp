// #pragma once
//  MatMul.h
#ifndef _MATMUL_H_
#define _MATMUL_H_
#include <Kokkos_Core.hpp>

struct OuterLoop
{
};
struct DoubleLoop
{
};

struct Tiled
{
};

template <typename ViewType>
class MatMul
{
private:
    using shared_space = Kokkos::DefaultExecutionSpace::scratch_memory_space;
    using shared_1d = Kokkos::View<ViewType *, shared_space, Kokkos::MemoryUnmanaged>;
    using shared_2d = Kokkos::View<ViewType **, shared_space, Kokkos::MemoryUnmanaged>;
    using team_policy = Kokkos::TeamPolicy<>;
    using team_member = team_policy::member_type;

public:
    MatMul(const Kokkos::View<ViewType **> &A, const Kokkos::View<ViewType **> &B, const Kokkos::View<ViewType **> &C);
    MatMul(const int n, const int k, const int m);

    void outerLoop();  // kernel-0
    void doubleLoop(); // kernel-1
    void tiled(const int tile_x, const int tile_y);

    KOKKOS_INLINE_FUNCTION
    void operator()(const OuterLoop, const int i) const;
    KOKKOS_INLINE_FUNCTION
    void operator()(const DoubleLoop, const int i, const int j) const;
    KOKKOS_INLINE_FUNCTION
    void operator()(const Tiled, const team_member& thread) const;

    void print_performance(double elapsedTotSeconds, int niter);

private:
    // matrix orders
    int n, k, m;
    unsigned long int sizeC;
    double Gflops;
    size_t bytes_r, bytes_w;

    Kokkos::View<ViewType **, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace> A;
    Kokkos::View<ViewType **, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace> B;
    Kokkos::View<ViewType **, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace> C;
};

/**
 *
 * CONSTRUCTORS
 *
 */
template <typename ViewType>
MatMul<ViewType>::MatMul(const Kokkos::View<ViewType **> &A, const Kokkos::View<ViewType **> &B, const Kokkos::View<ViewType **> &C)
    : A(A), B(B), C(C){};

template <typename ViewType>
MatMul<ViewType>::MatMul(const int n, const int k, const int m) : n(n), k(k), m(m)
{
    A = Kokkos::View<ViewType **>("A", n, k);
    B = Kokkos::View<ViewType **>("B", n, k);
    C = Kokkos::View<ViewType **>("C", n, k);

    // print info about matrix sizes and such ...
    // print info about device information

    sizeC = C.extent(0) * C.extent(0);                    // number of C entries
    bytes_r = 2 * A.extent(1) * sizeC * sizeof(ViewType); // Bytes read per kernel
    bytes_w = sizeC * sizeof(ViewType);
    Gflops = sizeC * A.extent(1) * 2 * 1.0e-9f;
};

/**
 *
 * MATRIX MULTIPLICATION KERNELS
 *
 */

/* ------------------------- VERY NAIVE MATRIX MULTIPLICATION KERNEL -------------------------------*/

template <typename ViewType>
void MatMul<ViewType>::outerLoop()
{
    Kokkos::parallel_for("A@B", Kokkos::RangePolicy<OuterLoop>(0, A.extent(0)), *this);
};

template <typename ViewType>
KOKKOS_INLINE_FUNCTION void MatMul<ViewType>::operator()(const OuterLoop, const int i) const
{
    // Implementation of operator() here...
    for (int j = 0; j < B.extent(1); j++)
    {
        ViewType sum = 0;
        // compute a single entry
        for (int k = 0; k < B.extent(0); k++)
        {
            sum += A(i, k) * B(k, j); // in this current set-up there should be no risk of concurrency
        }
        // store the result
        C(i, j) = sum;
    }
};

/* ------------------------- NAIVE MATRIX MULTIPLICATION KERNEL -------------------------------*/

template <typename ViewType>
void MatMul<ViewType>::doubleLoop()
{
    using MDPolicy = Kokkos::MDRangePolicy<DoubleLoop, Kokkos::Rank<2>>;
    MDPolicy mdpolicy({0, 0}, {A.extent(0), B.extent(1)}, {32, 4});
    Kokkos::parallel_for("DoubleLoop", mdpolicy, *this);
};

template <typename ViewType>
KOKKOS_INLINE_FUNCTION void MatMul<ViewType>::operator()(const DoubleLoop, const int i, const int j) const
{
    ViewType sum = 0;
    // compute a single entry
    for (int k = 0; k < B.extent(0); k++)
    {
        sum += A(i, k) * B(k, j); // in this current set-up there should be no risk of concurrency
    }
    // store the result
    C(i, j) = sum;
}

/* ------------------------- TILED MATRIX MULTIPLICATION KERNEL -------------------------------*/

template <typename ViewType>
void MatMul<ViewType>::tiled(const int tile_x, const int tile_y)
{
    TeamMDPolicy mdpolicy({0, 0}, {A.extent(0), B.extent(1)}, {tile_x, tile_y});
    //Kokkos::TeamPolicy<ARGS>(Space, league_size, team_size [, vector_length=1])
    const Kokkos::TeamPolicy<> policy();
    Kokkos::parallel_for("Tiled", policy, *this);
};

template <typename ViewType>
KOKKOS_INLINE_FUNCTION void MatMul<ViewType>::operator()(const Tiled, const team_member& thread) const
{
    ViewType sum = 0;
    // Allocate a shared array for the team.
    shared_2d Atile(thread.team_shmem(), data.extent(1)); // size of tile / block
    // compute a single entry [inner product row i of A and column j of B]
    for (int k = 0; k < B.extent(0); k++)
    {
        sum += A(i, k) * B(k, j); // in this current set-up there should be no risk of concurrency
    }
    // store the result
    C(i, j) = sum;
}

template <typename ViewType>
void MatMul<ViewType>::print_performance(double elapsedTotSeconds, int niter)
{

    double elapsedAvgSeconds = elapsedTotSeconds / niter;

    sizeC = C.extent(0) * C.extent(0);                    // number of C entries
    bytes_r = 2 * A.extent(1) * sizeC * sizeof(ViewType); // Bytes read per kernel
    bytes_w = sizeC * sizeof(ViewType);
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
#endif
