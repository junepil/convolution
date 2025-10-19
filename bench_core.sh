#!/bin/bash
mkdir -p bench/core

# Define test parameters
CORES=(1 2 4 8 16 32 64 128 256)

echo "Generating comprehensive benchmark tests..."

# Test all combinations of cores, kernels, and strides
for cores in "${CORES[@]}"; do
  echo "Generating Slurm script for ${cores} cores"
    
  cat > "bench/core/${cores}cores.sh" << EOF
#!/bin/bash
#SBATCH --job-name=bench_${cores}
#SBATCH --nodes=${cores}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=${cores}
#SBATCH --mem=128G
#SBATCH --time=00:15:00
#SBATCH --output=bench/core/${cores}cores.out

srun ./conv -W 10000 -H 10000 -kH 3 -kW 3
EOF
  chmod +x "bench/core/${cores}cores.sh"
  echo "Submitting bench for ${cores} cores"
  sbatch "bench/core/${cores}cores.sh"
done

echo "All benchmark tests submitted!"
echo "Total tests: $((${#CORES[@]}))"
