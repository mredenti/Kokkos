


## MPI 

So far it seems Kokkos does not yet support single process multi-gpu paradigm. How do you achieve that in Cuda?

**One MPI rank per GPU**

```shell
srun -N 1 --ntasks-per-node=2 --cpus-per-task=1 --gres=gpu:2 -A cin_staff -p boost_usr_prod --time=00:30:00 --pty /bin/bash
mpirun -n 2 query_device.x --kokkos-map-device-id-by=mpi_rank
```

The `--kokkos-map-device-id-by=mpi_rank` will map each MPI process to a GPU in a round robin fashion