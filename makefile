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

# 테스트 실행 (로컬)
test: $(TARGET)
	@echo "Running local test"
	@./$(TARGET) -W 100 -H 100 -kH 3 -kW 3 -o 100x100_3x3.txt

test-mpi: mpi
	@echo "Running MPI test"
	@echo "Test 1"
	@mpiexec -n 1 $(TARGET) -f test/data/f1.txt -g test/data/g1.txt -sH 3 -sW 2 -o mpi_test1.txt
	@echo "Test 2"
	@mpiexec -n 1 $(TARGET) -W 1000 -H 1000 -kW 3 -kH 3 -o mpi_test2.txt 

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