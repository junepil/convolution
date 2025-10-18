#!/bin/bash
mkdir -p bench/thread

# Define test parameters
THREADS=(1 2 4 8 16 32 64 128)
FIXED_MATRIX_SIZE=10000
FIXED_KERNEL_SIZE=5
FIXED_STRIDE=2

# Test OpenMP thread scaling
for threads in "${THREADS[@]}"; do
  echo "Generating Slurm script for ${threads} OpenMP threads"
  
  cat > "bench/thread/threads_${threads}.sh" << EOF
#!/bin/bash
#SBATCH --job-name=threads_${threads}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=${threads}
#SBATCH --mem=128G
#SBATCH --time=00:15:00
#SBATCH --output=bench/thread/threads_${threads}.out
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun ./conv -W ${FIXED_MATRIX_SIZE} -H ${FIXED_MATRIX_SIZE} -kH ${FIXED_KERNEL_SIZE} -kW ${FIXED_KERNEL_SIZE} -sH ${FIXED_STRIDE} -sW ${FIXED_STRIDE}
EOF
  chmod +x "bench/thread/threads_${threads}.sh"
  echo "Submitting benchmark for ${threads} threads"
  sbatch "bench/thread/threads_${threads}.sh"
done

echo "All OpenMP thread scaling tests submitted!"
echo "Total tests: ${#THREADS[@]}"
echo "Thread counts: ${THREADS[*]}"
