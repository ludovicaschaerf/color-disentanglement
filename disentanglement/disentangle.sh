#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --mem=50GB
#SBATCH --gres gpu:1


module load v100
module load cuda
module load mamba
source activate sam

python ShapleyVec.py --variable Color --subfolder ShapleyVec/color/

###python GANSpace.py --subfolder GANSpace/ --max_lambda 9

##python InterfaceGAN.py --variable Color --subfolder interfaceGAN/color/ --max_lambda 18

##python InterfaceGAN.py --variable Monochromatic --subfolder interfaceGAN/monochromatic/ --max_lambda 9
##python InterfaceGAN.py --variable Triadic --subfolder interfaceGAN/triadic/ --max_lambda 9
##python InterfaceGAN.py --variable Analogous --subfolder interfaceGAN/analogous/ --max_lambda 9
##python InterfaceGAN.py --variable Complementary --subfolder interfaceGAN/complementary/ --max_lambda 9
##python InterfaceGAN.py --variable "Split Complementary" --subfolder interfaceGAN/split_complementary/ --max_lambda 9
##python InterfaceGAN.py --variable "Double Complementary" --subfolder interfaceGAN/double_complementary/ --max_lambda 9

##python InterfaceGAN.py --variable S1 --continuous_experiment true --subfolder interfaceGAN/saturation/ --max_lambda 6
##python InterfaceGAN.py --variable V1 --continuous_experiment true --subfolder interfaceGAN/value/ --max_lambda 6


##python StyleSpace.py --variable Color --subfolder StyleSpace/color/

##python StyleSpace.py --variable Monochromatic --subfolder StyleSpace/monochromatic/
##python StyleSpace.py --variable Triadic --subfolder StyleSpace/triadic/
##python StyleSpace.py --variable Analogous --subfolder StyleSpace/analogous/
##python StyleSpace.py --variable Complementary --subfolder StyleSpace/complementary/
##python StyleSpace.py --variable "Split Complementary" --subfolder StyleSpace/split_complementary/
##python StyleSpace.py --variable "Double Complementary" --subfolder StyleSpace/double_complementary/

conda deactivate    

