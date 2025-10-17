# Makefile for HPC Convolution Assignment
# 컴파일러 설정
CC = gcc
CFLAGS = -O3

# HPC 관련 설정
HPC_FLAGS = -DHPC -fopenmp
SLURM_FLAGS = -lm

# 타겟 파일들
TARGET = conv
SOURCE = conv.c

# 기본 타겟 (로컬 디버그 버전)
all: $(TARGET)

# 일반 버전 (OpenMP 없음)
$(TARGET): $(SOURCE)
	@echo "Building standard version"
	@$(CC) $(CFLAGS) $(SOURCE) -o $(TARGET)

# HPC 버전 (OpenMP 포함)
hpc: $(SOURCE)
	@echo "Building HPC version with OpenMP"
	@$(CC) $(CFLAGS) $(HPC_FLAGS) $(SOURCE) -o $(TARGET)_hpc $(SLURM_FLAGS)

mpi: $(SOURCE)
	@echo "Building MPI version"
	@mpicc -o $(TARGET) -DDEBUG $(SOURCE)

mpi-hpc: $(SOURCE)
	@echo "Building MPI version"
	@mpicc $(HPC_FLAGS) -o $(TARGET) $(SOURCE) $(SLURM_FLAGS)


# 테스트 실행 (로컬)
test: $(TARGET)
	@echo "Running local test"
	@./$(TARGET) -W 100 -H 100 -kH 3 -kW 3 -o 100x100_3x3.txt

test-mpi: mpi
	@echo "Running MPI test"
	@mpiexec -n 4 $(TARGET) -f test/data/f4.txt -g test/data/g4.txt -o mpi_test4.txt

test-mpi-hpc: mpi-hpc
	@echo "Running MPI & OpenMP test"
	@mpiexec -n 4 $(TARGET) -W 100 -H 100 -kH 3 -kW 3 -sH 2 -sW 2 -o mpi_hpc_test.txt

# 정리
clean:
	@echo "Cleaning up..."
	@rm -f $(TARGET) $(TARGET)_hpc
	@rm -f *.o *.out output*.txt job_*.sh
	@rm -f result*.txt output*.log
	@rm -f gmon.out
	@rm -f quick_test_output.txt
	@rm -rf test/test_data

# 가상 타겟 (파일이 아닌 명령)
.PHONY: all hpc test test-quick test-comprehensive test-comprehensive-keep test-hpc benchmark slurm-script clean help parallel-hpc stress-hpc