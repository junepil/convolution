#!/bin/bash
mkdir -p stress

input=(1000 10000 100000)
kernels=(5 15 25 35)

# Rows fixed W=10000
echo "Creating stress tests for varying rows (H) with fixed columns (W=10000)"
for i in "${input[@]}"; do
  for k in "${kernels[@]}"; do
    echo "Generating Slurm script for ${i}x10000 with ${k}x${k} kernel"
    cat > "stress/rows_${i}x10000_${k}x${k}k.sh" << EOF
#!/bin/bash
#SBATCH --job-name=stress_rows_${i}_${k}
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=4
#SBATCH --mem=128G
#SBATCH --time=00:15:00
#SBATCH --output=stress/rows_${i}x10000_${k}x${k}k.out

srun ./conv -W 10000 -H ${i} -kH ${k} -kW ${k}
EOF
    chmod +x "stress/rows_${i}x10000_${k}x${k}k.sh"
    sbatch "stress/rows_${i}x10000_${k}x${k}k.sh"
  done
done

# Cols fixed H=10000
echo "Creating stress tests for varying columns (W) with fixed rows (H=10000)"
for i in "${input[@]}"; do
  for k in "${kernels[@]}"; do
    echo "Generating Slurm script for 10000x${i} with ${k}x${k} kernel"
    cat > "stress/cols_10000x${i}_${k}x${k}k.sh" << EOF
#!/bin/bash
#SBATCH --job-name=stress_cols_${i}_${k}
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=4
#SBATCH --mem=128G
#SBATCH --time=00:15:00
#SBATCH --output=stress/cols_10000x${i}_${k}x${k}k.out

srun ./conv -W ${i} -H 10000 -kH ${k} -kW ${k}
EOF
    chmod +x "stress/cols_10000x${i}_${k}x${k}k.sh"
    sbatch "stress/cols_10000x${i}_${k}x${k}k.sh"
  done
done

echo "All stress tests submitted!"