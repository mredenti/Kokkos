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

struct Tiling
{
};

template <class T>
class MatMul
{
public:
    MatMul(const Kokkos::View<T **> &A, const Kokkos::View<T **> &B, const Kokkos::View<T **> &C);
    MatMul(const int n, const int k, const int m);
    
    void outerLoop();  // kernel-0
    void doubleLoop(); // kernel-1

    KOKKOS_INLINE_FUNCTION
    void operator()(const OuterLoop, const int i) const;
    KOKKOS_INLINE_FUNCTION
    void operator()(const DoubleLoop, const int i, const int j) const;
    KOKKOS_INLINE_FUNCTION
    void operator()(const Tiling, const int i, const int j) const;

    void print_performance(T elapsedTotSeconds, int niter);

private:
    // matrix orders
    int n, k, m;
    unsigned long int sizeC;
    double Gflops;
    size_t bytes_r, bytes_w;

    Kokkos::View<T **, Kokkos::LayoutLeft, Kokkos::CudaSpace> A;
    Kokkos::View<T **, Kokkos::LayoutLeft, Kokkos::CudaSpace> B;
    Kokkos::View<T **, Kokkos::LayoutLeft, Kokkos::CudaSpace> C;
};

#include "MatMul.cpp"

#endif