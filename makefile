# Junepil Lee 25097868
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

# 테스트 실행 (로컬)
test: $(TARGET)
	@echo "Running local test"
	@./$(TARGET) -W 100 -H 100 -kH 3 -kW 3 -o 100x100_3x3.txt

# HPC 테스트 실행
test-hpc: hpc
	@echo "Generating Slurm script for 10000x10000 hpc test"
	@echo "#!/bin/bash" > job_100.sh
	@echo "#SBATCH --job-name=conv2d" >> job_100.sh
	@echo "#SBATCH --cpus-per-task=1" >> job_100.sh
	@echo "#SBATCH --time=00:10:00" >> job_100.sh
	@echo "#SBATCH --partition=cits3402" >> job_100.sh
	@echo "#SBATCH --output=test_result.txt" >> job_100.sh
	@echo "" >> job_100.sh
	@echo "module load gcc" >> job_100.sh
	@echo "./$(TARGET)_hpc -W 10000 -H 10000 -kH 3 -kW 3 -o test.txt" >> job_100.sh
	sbatch job_100.sh
# Slurm 작업 제출용 스크립트 생성
stress-hpc: hpc
	@chmod +x generate_stress.sh
	@./generate_stress.sh

parallel-hpc: hpc
	@chmod +x generate_parallel.sh
	@./generate_parallel.sh

# 정리
clean:
	@echo "Cleaning up..."
	@rm -f $(TARGET) $(TARGET)_hpc
	@rm -f *.o *.out job_*.sh
	@rm -f *.txt *.log
	@rm -f gmon.out
	@rm -rf test/test_data

# 도움말
help:
	@echo "Available targets:"
	@echo "  all                - Build standard version (no OpenMP)"
	@echo "  hpc                - Build HPC version with OpenMP"
	@echo "  test               - Run basic local test"
	@echo "  test-hpc           - Run HPC test on HPC cluster"
	@echo "  stress-hpc         - Run stress test on HPC"
	@echo "  parallel-hpc       - Run parallel test on HPC"
	@echo "  clean              - Remove all generated files"
	@echo "  help               - Show this help"

# 가상 타겟 (파일이 아닌 명령)
.PHONY: all hpc test test-hpc benchmark clean help parallel-hpc stress-hpc