#!/bin/bash
mkdir -p bench/both

# Define test parameters
CORES=(1 2 4 8 16 32 64)
THREADS=(2 4 8 16)

echo "Generating comprehensive benchmark tests..."

# Test all combinations of cores, kernels, and strides
for cores in "${CORES[@]}"; do
  for threads in "${THREADS[@]}"; do
    echo "Generating Slurm script for ${cores} cores, ${threads} threads"
    
    cat > "bench/both/${cores}cores_${threads}threads.sh" << EOF
#!/bin/bash
#SBATCH --job-name=bench_${cores}_${threads}
#SBATCH --nodes=${cores}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=${threads}
#SBATCH --mem=128G
#SBATCH --time=00:15:00
#SBATCH --output=bench/both/${cores}cores_${threads}.out

srun ./conv -W 10000 -H 10000 -kH 3 -kW 3
EOF
    chmod +x "bench/both/${cores}cores_${threads}threads.sh"
    echo "Submitting bench for ${cores} cores, ${threads}threads"
    sbatch "bench/both/${cores}cores_${threads}threads.sh"
  done
done

echo "All benchmark tests submitted!"
echo "Total tests: $((${#CORES[@]} * ${#THREADS[@]}))"
