#!/bin/bash
mkdir -p stress

# Test varying rows with fixed columns = 10000
echo "Creating stress tests for varying rows (H) with fixed columns (W=10000)"
for i in 100 1000 10000 100000 1000000; do
  echo "Generating Slurm script for ${i}x10000 stress test (varying rows)"
  
  cat > "stress/rows_${i}x10000.sh" << EOF
#!/bin/bash
#SBATCH --job-name=stress_rows_${i}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --time=00:15:00
#SBATCH --output=stress/rows_${i}x10000.out

srun ./conv -W 10000 -H ${i} -kH 3 -kW 3 -o stress/rows_${i}x10000.txt
EOF
  chmod +x "stress/rows_${i}x10000.sh"
  echo "Submitting stress for rows ${i}x10000"
  sbatch "stress/rows_${i}x10000.sh"
done

# Test varying columns with fixed rows = 10000
echo "Creating stress tests for varying columns (W) with fixed rows (H=10000)"
for i in 100 1000 10000 100000 1000000; do
  echo "Generating Slurm script for 10000x${i} stress test (varying columns)"
  
  cat > "stress/cols_10000x${i}.sh" << EOF
#!/bin/bash
#SBATCH --job-name=stress_cols_${i}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --time=00:15:00
#SBATCH --output=stress/cols_10000x${i}.out

srun ./conv -W ${i} -H 10000 -kH 3 -kW 3 -o stress/cols_10000x${i}.txt
EOF
  chmod +x "stress/cols_10000x${i}.sh"
  echo "Submitting stress for columns 10000x${i}"
  sbatch "stress/cols_10000x${i}.sh"
done

echo "All stress tests submitted!"