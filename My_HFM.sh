#!/bin/bash -l
#SBATCH --job-name=My_HFM
#SBATCH --time=200:00:00
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --mail-type=end
#SBATCH --mail-user=rcheng15@jhu.edu

source /data/apps/go.sh
ml anaconda
conda activate HFMtf
cd /home/rcheng15/HFM_2.0/My_HFM
python3.8 test.py
