#!/bin/bash
#SBATCH --nodes=1                 # Request one node
#SBATCH --exclusive=user          # Exclusive access to the node
#SBATCH --time=90:00:00
#SBATCH --job-name=Lichess

module load anaconda/anaconda3
module load cuda/cuda-12.1
source /apps/anaconda/anaconda3/etc/profile.d/conda.sh
conda activate base
conda activate chess
which python
hostname
pwd

python dev/data/lichess_uci.py -f -v 202201-moves 202202-moves 202203-moves 202204-moves 202205-moves 202206-moves 202207-moves 202208-moves 202209-moves 202210-moves 202211-moves 202212-moves 202301-moves 202302-moves 202303-moves 202304-moves 202305-moves 202306-moves 202307-moves 202308-moves 202309-moves 202310-moves 202311-moves 202312-moves

