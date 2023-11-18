#include <Kokkos_Core.hpp>
#include <stdio.h>
#include <cstdio>

// Define a typedef for the desired data type
typedef float real;

int main(int argc, char **argv)
{

    // vector size
    int N = 1000;
    double time = 0.0;

    Kokkos::initialize(argc, argv);
    Kokkos::Timer timer;

    // allocate data on default memory space
    // On success, returns the pointer to the beginning of newly allocated memory.
    // To avoid a memory leak, the returned pointer must be deallocated with Kokkos::kokkos_free()

    // check but I guess it will simply call cudaMalloc ? --verify
    real *xd = (real *)Kokkos::kokkos_malloc<Kokkos::DefaultExecutionSpace::memory_space>("xd", N * sizeof(real));
    real *yd = (real *)Kokkos::kokkos_malloc<Kokkos::DefaultExecutionSpace::memory_space>("yd", N * sizeof(real));
    real *zh = (real *)Kokkos::kokkos_malloc<Kokkos::HostSpace>("zh", N * sizeof(real));

    using ExecSpace = Kokkos::OpenMP;
    Kokkos::parallel_for(
        "zhInit", Kokkos::RangePolicy<Kokkos::Schedule<Kokkos::Dynamic>, Kokkos::OpenMP>(0, N),
        KOKKOS_LAMBDA(const int64_t i) {
            zh[i] = 0.0;
        });

    Kokkos::fence();

    Kokkos::parallel_for(
        "zhInit", Kokkos::RangePolicy<Kokkos::OpenMP>(Kokkos::OpenMP(), 0, N, Kokkos::ChunkSize(4)),
        KOKKOS_LAMBDA(const int64_t i) {
            zh[i] = 0.0;
        });

    // is there a way to assign the vector to CUDA constant memory?

    // initialise data on device
    Kokkos::parallel_for(
        "xdInit", N, KOKKOS_LAMBDA(const int i) {
            // printf("Greetings from iteration %i\n",i);
            xd[i] = 1.0;
        });

    Kokkos::parallel_for(
        "ydInit", N, KOKKOS_LAMBDA(const int i) {
            // printf("Greetings from iteration %i\n",i);
            yd[i] = 1.0;
        });

    timer.reset();
    Kokkos::parallel_for(
        "x+y", N, KOKKOS_LAMBDA(const int i) {
            // printf("Greetings from iteration %i\n",i);
            yd[i] = xd[i] + yd[i];
        });

    Kokkos::fence();
    time = timer.seconds();

    printf("Time %.10f", time);

    std::cout << *zh << std::endl;

    Kokkos::deep_copy(Kokkos::View<real, Kokkos::HostSpace>(zh, N),
                      Kokkos::View<real>(yd, N));

    Kokkos::kokkos_free<Kokkos::DefaultExecutionSpace::memory_space>(xd);
    Kokkos::kokkos_free<Kokkos::DefaultExecutionSpace::memory_space>(yd);
    Kokkos::kokkos_free<Kokkos::HostSpace>(zh);

    Kokkos::finalize();

    return 0;
}