#include <Kokkos_Core.hpp>
#include <omp.h>

#define N 10000

int main(int argc, char* argv[]){

    Kokkos::initialize(argc, argv);
    {
    // Kokkos::LayoutRight -> right-most index is stride 1 (you should be able to infer what happens for 3d arrays)
    Kokkos::View<int[N][N], Kokkos::LayoutRight, Kokkos::HostSpace> A("A");

    Kokkos::Timer timer;

    #pragma omp parallel for
    for (int64_t i=0; i < N; i++){
        for (int64_t j=0; j < N; j++){
            A(i,j) = 1;
            //printf("(threadIdx, i, j, &A(i,j)) = (%d, %d, %d, %p)\n", omp_get_thread_num(), i, j, &A(i,j));
        }
    }
    double time = timer.seconds(); 

    printf("\nExecution time = %f [s]\n\n", time);

    timer.reset();
    #pragma omp parallel for
    for (int64_t i=0; i < N; i++){
        for (int64_t j=0; j < N; j++){
            A(j,i) = 1; // EQUIVALENT TO REVERSING THE ORDER OF THE LOOPS (in one case you are giving the master thread indices which jump by rows, first index runs faster while the storage is 2nd index runs faster)
            // If I change the layout of A from left to right, I should see the opposite behaviour
            //printf("(threadIdx, i, j, &A(i,j)) = (%d, %d, %d, %p)\n", omp_get_thread_num(), i, j, &A(j,i));
        }
    }
    
    time = timer.seconds(); 

    printf("\nExecution time = %f [s]\n", time);

    }
    Kokkos::finalize();

    return 0;

}