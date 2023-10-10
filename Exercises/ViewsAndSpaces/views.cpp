#include <Kokkos_Core.hpp>

#define N 1000

int main(int argc, char* argv[]){

    Kokkos::initialize(argc, argv);
    {
    // dimensions defined at compile time 
    // If no space is provided, the view's data resides in the default memory space of the default execution space.
    //Kokkos::View<double[N]>  x("x"); // "2D" dimensional array
    // equivalent to
    //Kokkos::View<double[N], Kokkos::DefaultExecutionSpace::memory_space> x("x");
    Kokkos::View<double[N], Kokkos::CudaUVMSpace>  x("x");
    
    // //Kokkos::View<double[N], Kokkos::DefaultHostExecutionSpace::memory_space> x("x"); 
    // use Kokkos accessibility to verify that a unit of work in cuda execution space 
    // can not access memory_space on the host
    
    printf("We have allocated a Kokkos::View matrix with label %s\n\n", x.label());
    printf("vector x has rank %d with dimensions (%d, %d)\n\n", 1, x.extent(0));

    /* 
    *
    *   WHERE IS THE DATA? WHICH MEMORY SPACE DOES IT LIVE IN?
    * 
    */
    
    printf("Where is Kokkos::View data stored? In which MEMORY SPACE is data stored by default?\n");
    printf("----> Each view stores its data in a MEMORY SPACE set at compile time.\n");
    printf("----> Each EXECUTION SPACE has a DEFAULT MEMORY SPACE which is used if SPACE provided is actually an execution space.\n");
    printf("----> If no space is provided, the view's data resides in the default memory space of the default execution space.\n");
    printf("----> IS THIS TRUE ALSO FOR THE METADATA? I SEEM TO HAVE UNDERSTOOD THIS IS NOT THE CASE\n");

    // By default data should reside on the device and therefore we should not encounter an illegal memory access

    // initialize data on device 
    Kokkos::parallel_for("xInit", N, KOKKOS_LAMBDA (int64_t i){
        x[i] = 1;
    });

    printf("\n\nThe first x element on the host is: x[0]=%f\n\n", x[0]); // Kokkos::View ERROR: attempt to access inaccessible memory space (because data array lives on GPU but not the metadata)
    
    // sum vector elements 
    double sum = 0;
    Kokkos::parallel_reduce("xSum", N, KOKKOS_LAMBDA (int64_t i, double& partial_sum){
        partial_sum += x[i];
    }, sum);

    printf("\n\nThe sum of %d elements of value 1 is: %f \n", N, sum);

    }

    Kokkos::finalize();

    // Is A still alive after we have called Kokkos::finalize()?
    // assert(x.data() != nullptr); // -> compilation error: identifier "x" is undefined 

}