mkdir -p parallel
#!/bin/bash
for i in 1 2 4 8 16 32 64; do
  echo "Generating Slurm script for ${i} threads test"
  
  cat > "job_${i}threads.sh" << EOF
#!/bin/bash
#SBATCH --job-name=${i}threads_conv
#SBATCH --cpus-per-task=${i}
#SBATCH --time=00:10:00
#SBATCH --partition=cits3402
#SBATCH --output=./parallel_result/${i}threads.log

./conv_hpc -W 100000 -H 100000 -kH 3 -kW 3
EOF
  chmod +x "job_${i}threads.sh"
  echo "Submitting job for ${i} threads"
  sbatch "job_${i}threads.sh"
done