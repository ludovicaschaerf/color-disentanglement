#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --mem=50GB
#SBATCH --gres gpu:1

module load v100
module load cuda
module load mamba
source activate sam

export TORCH_EXTENSIONS_DIR=/home/ludosc/data/.cache/torch_extensions
export LD_LIBRARY_PATH=/data/ludosc/conda/envs/sam/lib:$LD_LIBRARY_PATH

python annotate_images.py

###python generate_annotate_images.py --seeds 0-10000  --outdir /home/ludosc/ludosc/old_data/stylegan-10000-textile_ada/ --save true --network /home/ludosc/ludosc/textiles_stylegan2_ada/00002-stylegan2-dataset_styleganada_textiles-gpus2-batch32-gamma0.4096/network-snapshot-014000.pkl

conda deactivate    
