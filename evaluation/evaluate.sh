#!/bin/bash
#SBATCH --time=2-03:00:00
#SBATCH --mem=50GB
#SBATCH --gres gpu:1


module load v100
module load mamba
source activate sam

#python evaluation.py --df_separation_vectors ../data/shapleyvec_separation_vector_Color.csv
#python evaluation.py --df_separation_vectors ../data/stylespace_separation_vector_Color.csv

#python evaluation.py --df_separation_vectors ../data/interfaceGAN_separation_vector_V1.csv
#python evaluation.py --df_separation_vectors ../data/interfaceGAN_separation_vector_S1.csv

python evaluation.py --df_separation_vectors ../data/interfaceGAN_separation_vector_Color.csv

conda deactivate    

