# Building and Installing Kokkos Core as a shared library on Leonardo 

**Enabling CUDA, OpenMP, Serial backends**

<details>
  <summary>Click me</summary>
  
**Clone release 4.1.00**

  ```shell
 git clone --branch 4.1.00 https://github.com/kokkos/kokkos.git kokkos-4.1.00 && cd kokkos-4.1.00
 #mkdir cudaompbuild cudaompinstall
  ```
  
  ```shell
  module load cmake/3.24.3 gcc/11.3.0 cuda/11.8
  ```
  ```shell
cmake -S . -B cudaompbuild -DCMAKE_BUILD_TYPE=Release -DKokkos_ENABLE_OPENMP=ON -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80=ON -DKokkos_ENABLE_CUDA_LAMBDA=ON -DKokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE=ON -DCMAKE_VERBOSE_MAKEFILE=ON -DKokkos_ENABLE_TESTS=OFF -DCMAKE_CXX_COMPILER=g++ -DCMAKE_INSTALL_PREFIX=$WORK/mredenti/Software/KOKKOS-4.1.00/kokkos-4.1.00/cudaompinstall

make -j8
make install
 ```

  1. Foo
  2. Bar
     * Baz
     * Qux

  ### Some Javascript
  ```js
  function logSomething(something) {
    console.log('Something', something);
  }
  ```
</details>

**Enabling HIP, OpenMP, Serial backends**

<details>
  <summary>Click me</summary>
  
Although interesting, I do not think it makes a lot of sense. Tried but failed unsurprisingly perhaps

</details>

**Enabling CUDA, OpenMP, Serial backends with tests**

<details>
  <summary>Click me</summary>
  
The building and installation of Kokkos Core takes more time when bulding the tests as well

We can then build and run the tests on a Leonardo boost compute node as follows 

</details>