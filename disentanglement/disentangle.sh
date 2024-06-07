#!/bin/bash
#SBATCH --time=05:00:00
#SBATCH --mem=50GB
#SBATCH --gres gpu:1


module load v100
module load cuda
module load mamba
source activate sam

export LD_LIBRARY_PATH=/data/ludosc/conda/envs/sam/lib:$LD_LIBRARY_PATH

python InterfaceGAN.py --variable color --subfolder interfaceGAN_1/color/ --max_lambda 30 --annotations_file ../data/seeds00000-10000_color.pkl --model_file /home/ludosc/ludosc/textiles_stylegan2_ada/00002-stylegan2-dataset_styleganada_textiles-gpus2-batch32-gamma0.4096/network-snapshot-014000.pkl
python ShapleyVec.py --variable color --subfolder ShapleyVec_1/color/ --max_lambda 20 --annotations_file ../data/seeds00000-10000_color.pkl --model_file /home/ludosc/ludosc/textiles_stylegan2_ada/00002-stylegan2-dataset_styleganada_textiles-gpus2-batch32-gamma0.4096/network-snapshot-014000.pkl
python StyleSpace.py --variable color --subfolder StyleSpace_1/color/ --max_lambda 15 --annotations_file ../data/seeds00000-10000_color.pkl --model_file /home/ludosc/ludosc/textiles_stylegan2_ada/00002-stylegan2-dataset_styleganada_textiles-gpus2-batch32-gamma0.4096/network-snapshot-014000.pkl
python GANSpace.py --subfolder GANSpace/ --max_lambda 9

