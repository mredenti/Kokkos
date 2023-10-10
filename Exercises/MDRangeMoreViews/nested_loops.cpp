#include <Kokkos_Core.hpp>
#include <omp.h>
#include <stdio.h>

#define N 100

int main(int argc, char* argv[]){

    Kokkos::initialize(argc, argv);
    {
    
    //Kokkos::View<int[N][N][N], Kokkos::HostSpace> A("A");
    Kokkos::View<int[N][N][N], Kokkos::CudaSpace> A("A");


/*
   
    Kokkos::Profiling::pushRegion("OmpParallelCollapse");
 
#pragma omp parallel for collapse(3)
    for (int64_t i=0; i < N; i++){
        for (int64_t j=0; j < N; j++){
            for (int64_t k=0; k < N; k++){
                A(i, j, k) = 1;
            }
        }
    }

    Kokkos::Profiling::popRegion();
*/
    //Kokkos::Timer timer;
    // dirty trick for having it marked with NVTX (Nvidia Tool Extension Kit) and visible in NSightSystems
    //Kokkos::Profiling::pushRegion("KokkosParallelFor");
    // Multi-Dimensional range policy
    Kokkos::parallel_for("NestedLoop", Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<3>>({0,0,0}, {N, N, N}),
        KOKKOS_LAMBDA(int64_t i, int64_t j, int64_t k){
            A(i, j, k) = 1;
        });
    //Kokkos::Profiling::popRegion();
    Kokkos::fence();
    
    // double time = timer.end(); 
    }
    Kokkos::finalize();

    return 0;

}
