#!/bin/bash
#SBATCH --time=2-05:00:00
#SBATCH --mem=100GB
#SBATCH --gres gpu:1


module load v100
module load cuda
module load mamba
source activate sam

python color_annotations.py --images_dir /home/ludosc/data/stylegan-10000-textile/
conda deactivate    

