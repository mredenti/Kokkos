#include <Kokkos_Core.hpp>
#include <stdio.h>


int main(int argc, char** argv){

    std::ostringstream msg;

    Kokkos::initialize(argc, argv);

    Kokkos::print_configuration(msg, true);

    std::cout << msg.str();

    Kokkos::finalize();

    return 0;
}