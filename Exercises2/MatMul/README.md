Matrix-multiplication
Matrix multiplication occurs in many areas of scientific computing and machine learning..
According to the definition of BLAS libraries, the single-precision general matrix-multiplication (SGEMM) computes the following:
$$
C := \alpha * A * B + \beta * C
$$

In this equation, A is a K by M input matrix, B is an N by K input matrix, C is the M by N output matrix, and alpha and beta are scalar constants. For simplicity, we assume the common case where alpha is equal to 1 and beta is equal to zero, yielding:
C := A * B

This computation is illustrated in the following image: to compute a single element of C (in purple), we need a row of A (in green) and a column of B (in blue).

- The operations involved in Matrix-Matrix multiplication are simple floating point operations of addition and multiplication. 
- Effectively, each entry in the output matrix is the result of the inner product between two vectors ... to this end see the performances obtained in the inner product kernel ...


## Serial Implementation

```c
for (int m=0; m<M; m++) {
    for (int n=0; n<N; n++) {
        float acc = 0.0f;
        for (int k=0; k<K; k++) {
            acc += A[k*M + m] * B[n*K + k];
        }
        C[n*M + m] = acc;
    }
}
```

| HARDWARE | CUDA ABSTRACTION | KOKKOS ABSTRACTION |
| :---         |     :---:      |          ---: |
| git status   | git status     | git status    |
| git diff     | git diff       | git diff      |


## Kokkos simple starting script
```cpp
#include<Kokkos_Core.hpp>

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc,argv);

  Kokkos::finalize();
}
```

Kokkos gives users two options for defining the body of a parallel loop: functors and lambdas. It also lets users control how the parallel operation executes, by specifying an **execution policy**. A good programming abstraction layer makes it relatively 
straightforward to map the abstraction to the actual device backend abstraction and consequently to the hardware. The difficulty 
may come in because there are multiple different backends.

## An paradigmatic Kokkos programs has the following main characteristics
In order to use Kokkos an initialization call is required. That call is responsible for initializing internal objects and acquiring hardware resources such as threads. Typically, this call should be placed right at the start of a program. If you use both MPI and Kokkos, your program should initialize Kokkos right after calling MPI_Init. That way, if MPI sets up process binding masks, Kokkos will get that information and use it for best performance. Your program must also finalize Kokkos when done using it in order to free hardware resources.


A functor is one way to define the body of a parallel loop. It is a class or struct1 with a public operator() instance method. That method’s arguments depend on both which parallel operation you want to execute (for, reduce, or scan), and on the loop’s execution policy (e.g., range or team).