# Building and Installing Kokkos Core as a shared library on Leonardo


**Clone Kokkos repository**
Define Kokkos version/tag to clone 
```shell
export KOKKOS_TAG=4.1.00
```
Download repository
```shell
git clone --branch $KOKKOS_TAG https://github.com/kokkos/kokkos.git kokkos-$KOKKOS_TAG 
```
and export the path to the Kokkos folder (you might want to source the full path such that you can activate the full path for every session)
```shell
export KOKKOS_PATH=$PWD/kokkos-$KOKKOS_TAG
```

### Building Kokkos Kernels

### Using Kokkos in-tree
<details>
  <summary>Click me</summary>

For tutorial purposes, it may be preferable to build Kokkos inline so that you can easily switch the default execution space rather than having multiple installations of Kokkos shared libraries with different default execution spaces and features

**Example On Leonardo**

 </details>

### Building and Installing Kokkos Core as a shared library on Leonardo

<details>
  <summary>Click me</summary>

**Enabling CUDA, OpenMP, Serial backends**

**Clone release 4.1.00**

```shell
cd kokkos-$KOKKOS_TAG
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
 </details>