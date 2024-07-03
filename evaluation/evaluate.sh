#!/bin/bash
#SBATCH --time=3-00:00:00
#SBATCH --mem=50GB
#SBATCH --gres gpu:1


module load v100
module load cuda
module load mamba
source activate sam

export LD_LIBRARY_PATH=/data/ludosc/conda/envs/sam/lib:$LD_LIBRARY_PATH

python evaluation.py --df_separation_vectors ../data/interfaceGAN_separation_vector_color.csv --annotations_file ../data/seeds00000-10000_color.pkl --max_lambda 25 --model_file /home/ludosc/ludosc/textiles_stylegan2_ada/00002-stylegan2-dataset_styleganada_textiles-gpus2-batch32-gamma0.4096/network-snapshot-014000.pkl
python evaluation.py --df_separation_vectors ../data/stylespace_separation_vector_color.csv --annotations_file ../data/seeds00000-10000_color.pkl --max_lambda 15 --model_file /home/ludosc/ludosc/textiles_stylegan2_ada/00002-stylegan2-dataset_styleganada_textiles-gpus2-batch32-gamma0.4096/network-snapshot-014000.pkl
python evaluation.py --df_separation_vectors ../data/shapleyvec_separation_vector_color.csv --annotations_file ../data/seeds00000-10000_color.pkl --max_lambda 20 --model_file /home/ludosc/ludosc/textiles_stylegan2_ada/00002-stylegan2-dataset_styleganada_textiles-gpus2-batch32-gamma0.4096/network-snapshot-014000.pkl

python rescoring.py --df_modification_vectors ../data/modifications_shapleyvec_separation_vector_color.csv --annotations_file ../data/seeds00000-10000_color.pkl --max_lambda 20
python rescoring.py --df_modification_vectors ../data/modifications_interfaceGAN_separation_vector_color.csv --annotations_file ../data/seeds00000-10000_color.pkl --max_lambda 25
python rescoring.py --df_modification_vectors ../data/modifications_stylespace_separation_vector_color.csv --annotations_file ../data/seeds00000-10000_color.pkl --max_lambda 15

python attribute_dependency.py --df_separation_vectors ../data/shapleyvec_separation_vector_color.csv --df_modification_vectors ../data/modifications_shapleyvec_separation_vector_color.csv --annotations_file ../data/seeds00000-10000_color.pkl --max_lambda 20
python attribute_dependency.py --df_separation_vectors ../data/interfaceGAN_separation_vector_color.csv --df_modification_vectors ../data/modifications_interfaceGAN_separation_vector_color.csv --annotations_file ../data/seeds00000-10000_color.pkl --max_lambda 25
python attribute_dependency.py --df_separation_vectors ../data/stylespace_separation_vector_color.csv --df_modification_vectors ../data/modifications_stylespace_separation_vector_color.csv --annotations_file ../data/seeds00000-10000_color.pkl --max_lambda 15
conda deactivate    

