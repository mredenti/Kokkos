#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>

#include <stdio.h>

__global__ void test_kernel(void) {
}

void wrapper(void) {
	test_kernel <<<2, 2 >>> ();
}

int main(void) {
	wrapper();
	return 0;
}