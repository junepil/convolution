#!/bin/bash
mkdir -p bench/core

# Define test parameters
CORES=(1 2 4 8 16 32 64 128 256)
KERNELS=(3 5 7 9)

echo "Generating comprehensive benchmark tests..."

# Test all combinations of cores, kernels, and strides
for cores in "${CORES[@]}"; do
  for kernel in "${KERNELS[@]}"; do
    echo "Generating Slurm script for ${cores} cores, ${kernel}x${kernel} kernel"
    
    cat > "bench/core/${cores}cores_${kernel}x${kernel}k.sh" << EOF
#!/bin/bash
#SBATCH --job-name=bench_${cores}_${kernel}
#SBATCH --nodes=${cores}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=${cores}
#SBATCH --mem=128G
#SBATCH --time=00:15:00
#SBATCH --output=bench/core/${cores}cores_${kernel}x${kernel}k.out

srun ./conv -W 10000 -H 10000 -kH ${kernel} -kW ${kernel}
EOF
    chmod +x "bench/core/${cores}cores_${kernel}x${kernel}k.sh"
    echo "Submitting bench for ${cores} cores, ${kernel}x${kernel} kernel"
    sbatch "bench/core/${cores}cores_${kernel}x${kernel}k.sh"
  done
done

echo "All benchmark tests submitted!"
echo "Total tests: $((${#CORES[@]} * ${#KERNELS[@]}))"
