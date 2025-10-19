# Parallel 2D Convolution (MPI + OpenMP)

This project implements a 2D convolution with halo handling and supports hybrid parallelism:

- MPI for process-level parallelism (domain decomposition by rows)
- OpenMP for thread-level parallelism inside each process

## Repository structure

- `conv.c`: MPI + OpenMP implementation of 2D convolution and CLI
- `makefile`: makefile for testing and compiling
- `bench_core.sh`: sweep MPI ranks (process-level scaling)
- `bench_thread.sh`: sweep OpenMP threads with fixed problem size
- `bench_both.sh`: sweep both ranks and threads (hybrid)
- `stress.sh`: stress tests for varying matrix sizes and kernel sizes

## Build

Requirements:

- An MPI stack with wrapper compiler (e.g., `openmpi`, `mpich`, or Cray wrappers)
- OpenMP support from your compiler

In kaya you can build the program using the following

```sh
module load gcc openmpi
make build-mpi-omp
```

And in setonix

```sh
make build-mpi-omp
```

## Running (SLURM examples)

Pure OpenMP (single MPI rank, multiple threads):

```bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun ./conv -W 10000 -H 10000 -kH 5 -kW 5 -sH 2 -sW 2
```

Pure MPI (many ranks, one thread per rank):

```bash
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1

export OMP_NUM_THREADS=1
srun ./conv -W 10000 -H 10000 -kH 5 -kW 5 -sH 2 -sW 2
```

Hybrid MPI + OpenMP:

```bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4     # 4 MPI ranks/node
#SBATCH --cpus-per-task=8       # 8 threads/rank
#SBATCH --ntasks=4

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun ./conv -W 10000 -H 10000 -kH 5 -kW 5 -sH 2 -sW 2
```

Program CLI flags (provided by `conv.c`):

- `-H <int>`: input height
- `-W <int>`: input width
- `-kH <int>`: kernel height
- `-kW <int>`: kernel width
- `-sH <int>`: stride height (default 1)
- `-sW <int>`: stride width (default 1)
- `-f <path>`: input matrix file (optional)
- `-g <path>`: kernel matrix file (optional)
- `-o <path>`: output file to write resulting matrix (optional)

Output format (stdout):

```sh
<total_flops> <max_computation_time_seconds>
```

## Bench and stress helpers (SLURM)

Run core-scaling benchmarks (MPI ranks):

```sh
make bench-core
```

Run thread-scaling benchmarks (OpenMP threads at fixed problem size):

```sh
make bench-thread
```

Run combined rank+thread benchmarks:

```sh
make bench-both
```

Run stress tests (vary matrix and kernel sizes):

```sh
make stress
```

Each helper submits a set of SLURM jobs via `srun` and writes outputs into per-script folders.

## Notes

- Use `--mem` or `--mem-per-cpu` in your SLURM scripts to avoid OOM on very large matrices; memory use can be substantial (input + output + working buffers).
- On systems without `mpicc` in `PATH`, load your site’s MPI module (e.g., `openmpi/5.0.5`) or use vendor wrappers (`cc/CC/ftn` on Cray, `mpiicc` with Intel MPI).
