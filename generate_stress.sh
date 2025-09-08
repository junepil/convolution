mkdir -p stress
#!/bin/bash
for i in 100 1000 10000 100000; do
  echo "Generating Slurm script for ${i}x${i} stress test"
  
  cat > "job_${i}x${i}.sh" << EOF
#!/bin/bash
#SBATCH --job-name=conv2d_${i}
#SBATCH --cpus-per-task=1
#SBATCH --time=00:60:00
#SBATCH --partition=cits3402
#SBATCH --output=./stress_result/${i}x${i}.log

./conv_hpc -W ${i} -H ${i} -kH 3 -kW 3
EOF
    
  chmod +x "job_${i}x${i}.sh"
  echo "Submitting job for ${i}x${i}"
  sbatch "job_${i}x${i}.sh"
done