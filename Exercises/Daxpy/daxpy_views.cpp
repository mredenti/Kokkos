#include<Kokkos_Core.hpp>

#define N 10000
#define NREPEATS 1000

int main(int argc, char* argv[]){

    Kokkos::initialize(argc, argv);
    {
    
    /*
    *
    *   HOW DO I CHANGE KOKKOS DEFAULT EXECUTION SPACE? RATHER THAN PASSING THE EXECUTION SPACE 
    *   AS AN ARGUMENT AND RESULTING MORE VERBOSE...?
    * 
    */

    /*  
    *   
    * Allocate data using Kokkos::View (multi-dimensional array with some metadata)
    *   
    */
    
    double a = 1.0;
    
    Kokkos::View<double*> = x("x", N), y("y", N);
    
    Kokkos::parallel_for("xInit", N, KOKKOS_LAMBDA (int64_t i) {
        x[i] = 2;
    });

    Kokkos::parallel_for("yInit", N, KOKKOS_LAMBDA (int64_t i) {
        y[i] = 2;
    });

    Kokkos::Timer timer; 
    for (int itr =0; itr < NREPEAT; itr++){
        Kokkos::parallel_for("DAXPY", N, KOKKOS_LAMBDA (int64_t i){
            y[i] = a * x[i] + y[i];
        })
    }
    double time = timer.seconds();

    double Gbytes = 1.0e-9 * double(sizeof(double) * (2 * N * N + N));

    printf("  N( %d ) M( %d ) nrepeat ( %d ) problem( %g MB ) time( %g s ) bandwidth( %g GB/s )\n",
            N, N, NREPEAT, Gbytes * 1000, time, Gbytes * NREPEAT / time);

    // free memory

    }
    Kokkos::finalize();

    return 0;
}