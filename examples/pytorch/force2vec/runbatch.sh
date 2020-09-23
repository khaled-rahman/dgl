#!/bin/bash -l

#SBATCH -N 1
#SBATCH -p azad
#SBATCH -t 48:30:00
#SBATCH -J dglF2V
#SBATCH -o dglF2V.o%j

module load gcc
srun -p azad -N 1 -n 1 -c 1 bash runall.sh
