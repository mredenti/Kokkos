# Building Kokkos inline 

For the tutorial, we will compile our Kokkos programs via a Makefile while **building Kokkos inline**. This allows us to easily swap between different default execution spaces and memory spaces.


???+ note "Instructions: Cloning Kokkos Core repository"

    **Change into your work area on Leonardo...**
    ```shell
    cd $WORK
    ```

    **...define Kokkos release version/tag to clone...** 
    ```shell
    export KOKKOS_TAG=4.1.00
    ```

    **...clone the Kokkos repository...**
    ```shell
    git clone --branch $KOKKOS_TAG https://github.com/kokkos/kokkos.git kokkos-$KOKKOS_TAG 
    ```

    **...and finally export the path to the Kokkos folder.**
    ```shell
    export KOKKOS_PATH=$PWD/kokkos-$KOKKOS_TAG
    ```

    ??? tip
        To avoid having to export this environment variable every time you open a new shell, you might want to add it to your `~/.bashrc` file

    ??? info "Installing Kokkos as shared library/package"
    You may consult the documentation to learn about:    
        [Building Kokkos as an intalled package](https://kokkos.github.io/kokkos-core-wiki/building.html)   
        [Building Kokkos via Spack package manager](https://kokkos.github.io/kokkos-core-wiki/building.html#:~:text=a%20single%20process.-,Spack,-%23)
    but for the tutorial we will compile Kokkos programs inline via a Makefile


???+ note "Instructions: Cloning the tutorial/exercises repository"

    No need as they are within the SYCL repository

    **Change into your work area on Leonardo...**
    ```shell
    cd $WORK
    ```

!!! success "Next"
    Great! We can now turn to executing our first Kokkos program [Tutorial 01: Vector Addition](../vectorAdd/index.md)

        
???+ note
    g 

???+ abstract
    g 

???+ note
    g 

???+ success
    g 

???+ question
    g 

???+ failure
    g 

???+ danger
    g 

???+ bug
    g 

???+ example
    g    

???+ quote
    g 

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



## Useful spack commands
Below is a short list of some useful commands. For the complete list of Spack commands check this [link](https://spack.readthedocs.io/en/latest/command_index.html).      

| Command                        |      Comment      |
|--------------------------------|---------------------------------------|
| `spack env create -d .`        | To create a new environment in the current folder |
| `spack env activate -p .`      | To activate an environment in the current folder |


???+ warning
    STILL WORK IN PROGRESS: no configuration files and job submission script yet!


???+ tip
    To avoid running the same command every time you open a new terminal, you might want to add this line to your ~/.bashrc file:
    ```
    . /<FULLPATH-TO-SPACK-ROOT>/spack/share/spack/setup-env.sh
    ```
