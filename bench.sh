#!/bin/bash
mkdir -p bench

# Define test parameters
CORES=(1 2 4 8 16 32 64 128)
KERNELS=(3 5 7 9)
STRIDES=(1 2 3 4)

echo "Generating comprehensive benchmark tests..."

# Test all combinations of cores, kernels, and strides
for cores in "${CORES[@]}"; do
  for kernel in "${KERNELS[@]}"; do
    for stride in "${STRIDES[@]}"; do
      echo "Generating Slurm script for ${cores} cores, ${kernel}x${kernel} kernel, stride ${stride}x${stride}"
      
      cat > "bench/${cores}cores_${kernel}x${kernel}k_${stride}x${stride}s.sh" << EOF
#!/bin/bash
#SBATCH --job-name=bench_${cores}_${kernel}_${stride}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=${cores}
#SBATCH --ntasks=${cores}
#SBATCH --mem=128G
#SBATCH --time=00:15:00
#SBATCH --output=bench/${cores}cores_${kernel}x${kernel}k_${stride}x${stride}s.out

srun ./conv -W 10000 -H 10000 -kH ${kernel} -kW ${kernel} -sH ${stride} -sW ${stride} -o bench/${cores}cores_${kernel}x${kernel}k_${stride}x${stride}s.txt
EOF
      chmod +x "bench/${cores}cores_${kernel}x${kernel}k_${stride}x${stride}s.sh"
      echo "Submitting bench for ${cores} cores, ${kernel}x${kernel} kernel, stride ${stride}x${stride}"
      sbatch "bench/${cores}cores_${kernel}x${kernel}k_${stride}x${stride}s.sh"
    done
  done
done

echo "All benchmark tests submitted!"
echo "Total tests: $((${#CORES[@]} * ${#KERNELS[@]} * ${#STRIDES[@]}))"
