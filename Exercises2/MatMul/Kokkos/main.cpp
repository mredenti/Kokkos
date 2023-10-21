#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include "MatMul.hpp"
// #include<KokkosBlas.hpp>

#define N (1 << 12)
#define K (1 << 12)
#define M (1 << 12)

#define NITER 15

typedef float real;

int main(int argc, char *argv[])
{

    Kokkos::initialize(argc, argv);
    {
        // Query the default execution space's memory space
        auto memorySpace = Kokkos::DefaultExecutionSpace::memory_space();

        // Print the name of the memory space
        std::cout << "Summary\n"; // perhaps print other usefule informations
        std::cout << "-------\n";
        std::cout << "Default Memory Space: " << memorySpace.name() << std::endl; // perhaps print other usefule informations
        // parse input

        // Device Views
        // Kokkos::View<real **, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace> A(Kokkos::ViewAllocateWithoutInitializing("A"), N, K);
        Kokkos::View<real **> A("A", N, K);
        Kokkos::View<real **> B("B", K, M);
        Kokkos::View<real **> C("C", N, M);

        // Randomly fill Device Views [-2,2]
        Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);
        Kokkos::fill_random(A, random_pool, -2, 2);
        Kokkos::fill_random(B, random_pool, -2, 2);
        Kokkos::fence();

        // Initialize GEMM functor
        MatMul<real> matmul(A, B, C); // initialise matrices witn NxK, KxM = NxM [i.e. with N, K, M]

        Kokkos::Timer timer;
        real elapsedTotSeconds; // perhaps better to use output from nvprof?
        int i;

        // Kokkos::Profiling::popRegion();
        timer.reset(); // reset clock
        for (i = 0; i < NITER; i++)
        {
            matmul.outerLoop(); // it would seem pretty straightforward that it is a bad approach since you are doing things sequentially for each row while you could have done them in parallel. you will probably see a low GPU utilisation. You can maybe improve things by scheduling many blocks mmm
        }
        Kokkos::fence("MatMul Kernel"); // Synchronize to measure kernel execution time and not kernel launch time
        elapsedTotSeconds = timer.seconds();

        matmul.print_performance(elapsedTotSeconds, NITER);
#define DOUBLE_LOOP
#ifdef DOUBLE_LOOP
        timer.reset(); // reset clock
        for (i = 0; i < NITER; i++)
        {
            matmul.doubleLoop();
        }
        Kokkos::fence("MatMul Kernel"); // Synchronize to measure kernel execution time and not kernel launch time
        elapsedTotSeconds = timer.seconds();

        matmul.print_performance(elapsedTotSeconds, NITER);
#endif
/* ------------------ TILED MATRIX MULTIPLICATION KERNEL ----------------------*/
        timer.reset(); // reset clock
        for (i = 0; i < NITER; i++)
        {
            matmul.tiled(32,8); // I am curious to know what the tiling does for the CPU, OpenMP
        }
        Kokkos::fence("Tiled MatMul Kernel"); // Synchronize to measure kernel execution time and not kernel launch time
        elapsedTotSeconds = timer.seconds();

        matmul.print_performance(elapsedTotSeconds, NITER);
        // KokkosBlas::gemm("N","N", 1.0, A, B, 0.0, C);

        // verify ...
        // have Gemm/MatMul print this information - print_performance();

        // fprintf(stdout, "%-40s %f [GB/s]\n", "Effective Bandwidth:", (bytes_r + bytes_w) / seconds / 1e9);
        // fprintf(stdout, "%-40s %d bytes but performs %d multiply-add instructions, i.e. %d FLOPS per element computed\n",
        //        "For each matrix entry computed, MatMul reads", A.extent(1) * 2 * sizeof(real), A.extent(1), A.extent(1) * 2);

        //  CUDA events use the GPU timer and therefore avoid the problems associated with host-device synchronization.
        // But can I use them in Kokkos?

        // the question is maybe: how does this map to cuda?
        // I could also think about looping through i,j entries in C [I am only reading from A,B but writing to one place]
    }
    Kokkos::finalize(); // is this where the data gets freed if allocated on heap?

    return EXIT_SUCCESS;
}
