#!/bin/bash
mkdir -p stress

# Test varying rows with fixed columns = 10000
echo "Creating stress tests for varying rows (H) with fixed columns (W=10000)"
for i in 100 1000 10000 100000 1000000; do
  for kernel in 3 5 7 9; do
    echo "Generating Slurm script for ${i}x10000 stress test (varying rows) with ${kernel}x${kernel} kernel"
    
    cat > "stress/rows_${i}x10000_${kernel}x${kernel}k.sh" << EOF
#!/bin/bash
#SBATCH --job-name=stress_rows_${i}_${kernel}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --time=00:15:00
#SBATCH --output=stress/rows_${i}x10000_${kernel}x${kernel}k.out

srun ./conv -W 10000 -H ${i} -kH ${kernel} -kW ${kernel} -o stress/rows_${i}x10000_${kernel}x${kernel}k.txt
EOF
    chmod +x "stress/rows_${i}x10000_${kernel}x${kernel}k.sh"
    echo "Submitting stress for rows ${i}x10000 with ${kernel}x${kernel} kernel"
    sbatch "stress/rows_${i}x10000_${kernel}x${kernel}k.sh"
  done
done

# Test varying columns with fixed rows = 10000
echo "Creating stress tests for varying columns (W) with fixed rows (H=10000)"
for i in 100 1000 10000 100000 1000000; do
  for kernel in 3 5 7 9; do
    echo "Generating Slurm script for 10000x${i} stress test (varying columns) with ${kernel}x${kernel} kernel"
    
    cat > "stress/cols_10000x${i}_${kernel}x${kernel}k.sh" << EOF
#!/bin/bash
#SBATCH --job-name=stress_cols_${i}_${kernel}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --time=00:15:00
#SBATCH --output=stress/cols_10000x${i}_${kernel}x${kernel}k.out

srun ./conv -W ${i} -H 10000 -kH ${kernel} -kW ${kernel} -o stress/cols_10000x${i}_${kernel}x${kernel}k.txt
EOF
    chmod +x "stress/cols_10000x${i}_${kernel}x${kernel}k.sh"
    echo "Submitting stress for columns 10000x${i} with ${kernel}x${kernel} kernel"
    sbatch "stress/cols_10000x${i}_${kernel}x${kernel}k.sh"
  done
done

echo "All stress tests submitted!"
echo "Total tests: $((5 * 4 * 2)) = 40 tests"
echo "  - 5 matrix sizes (100, 1000, 10000, 100000, 1000000)"
echo "  - 4 kernel sizes (3x3, 5x5, 7x7, 9x9)"
echo "  - 2 test types (rows, columns)"