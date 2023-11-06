# Installing Kokkos

There are ... ways to build and install Kokkos
- cmake 
- .. 
- inline with Make

For the tutorial we will employ the latter mode as this will allow us to play around with setting different defaults backends (mm will see)


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

<details>
  <summary>Click me</summary>
## Building Kokkos Tools
**Clone Kokkos Tools repository**
Define Kokkos Tools version/tag to clone 
```shell
export KOKKOS_TOOLS_TAG=2.5.00
```
Download repository
```shell
git clone --branch $KOKKOS_TOOLS_TAG https://github.com/kokkos/kokkos-tools.git kokkos-tools-$KOKKOS_TOOLS_TAG 
```
and export the path to the Kokkos folder (you might want to source the full path such that you can activate the full path for every session)
```shell
export KOKKOS_TOOLS_PATH=$PWD/kokkos-$KOKKOS_TOOLS_TAG
make CUDA_ROOT=$NVHPC_HOME/Linux_x86_64/22.3/cuda/
```

**Vtune connector**
```shell
make VTUNE_HOME=$INTEL_ONEAPI_VTUNE_HOME/vtune/2021.7.1
```

You must enable `Kokkos` wih `Kokkos_ENABLE_LIBDL=ON` to load profiling hooks dynamically. To use one of the tools shipped with this repository you have to compile it, which will generate a dynamic library.

Before executing the Kokkos application you then have to set the environment variable `KOKKOS_TOOLS_LIBS` to point to the dynamic library e.g. in the bash shell:

```shell
export KOKKOS_TOOLS_LIBS=${HOME}/kokkos-tools/src/tools/memory-events/kp_memory_event.so
```

Explicit instrumentation:

```C++
Kokkos::Profiling::pushRegion("foo");
foo();
Kokkos::Profiling::popRegion();
```

</details>

## Building Kokkos Kernels

???+ info
    To know more about Spack and COMPSs:    
    [Spack documentation](https://spack.readthedocs.io/en/latest/)   
    [COMPSs documentation](https://compss-doc.readthedocs.io/en/stable/)


## Useful spack commands
Below is a short list of some useful commands. For the complete list of Spack commands check this [link](https://spack.readthedocs.io/en/latest/command_index.html).      

| Command                        |      Comment      |
|--------------------------------|---------------------------------------|
| `spack env create -d .`        | To create a new environment in the current folder |
| `spack env activate -p .`      | To activate an environment in the current folder |