# Makefile for HPC Convolution Assignment
# 컴파일러 설정
CC = gcc
CFLAGS = -O3

# HPC 관련 설정
OMP_FLAGS = -DHPC -fopenmp
SLURM_FLAGS = -lm

# 타겟 파일들
TARGET = conv
SOURCE = conv.c

# 기본 타겟 (로컬 디버그 버전)
all: $(TARGET)

# HPC 버전 (OpenMP 포함)
hpc: $(SOURCE)
	@echo "Building HPC version with OpenMP"
	@$(CC) $(CFLAGS) $(OMP_FLAGS) $(SOURCE) -o $(TARGET)_hpc $(SLURM_FLAGS)

build-mpi: $(SOURCE)
	@echo "Building MPI version"
	@mpicc -o $(TARGET) $(SOURCE)

build-mpi-omp: $(SOURCE)
	@echo "Building MPI version"
	@mpicc $(OMP_FLAGS) -o $(TARGET) $(SOURCE) $(SLURM_FLAGS)

# Test(local)
test: $(TARGET)
	@echo "Running test"
	@./$(TARGET) -W 100 -H 100 -kH 3 -kW 3 -o 100x100_3x3.txt

test-mpi: mpi
	@echo "Running MPI test"
	@mpiexec -n 4 $(TARGET) -f test/data/f4.txt -g test/data/g4.txt -o mpi_test4.txt

test-mpi-omp: mpi-omp
	@echo "Running MPI & OpenMP test"
	@mpiexec -n 4 $(TARGET) -W 100 -H 100 -kH 3 -kW 3 -sH 2 -sW 2 -o mpi_hpc_test.txt

# Bench(remote)
bench-core: build-mpi-omp
	@echo "Running core benchmark"
	@chmod 744 bench_core.sh
	@./bench_core.sh

bench-thread: build-mpi-omp
	@echo "Running thread benchmark"
	@chmod 744 bench_thread.sh
	@./bench_thread.sh

stress: build-mpi-omp
	@echo "Running stress test"
	@chmod 744 stress.sh
	@./stress.sh

# 정리
clean:
	@echo "Cleaning up..."
	@rm -f $(TARGET) $(TARGET)_hpc
	@rm -f *.o *.out
	@rm -rf stress bench thread_bench
	@rm -f gmon.out

# 가상 타겟 (파일이 아닌 명령)
.PHONY: all hpc test