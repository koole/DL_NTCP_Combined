#!/bin/bash

#SBATCH --job-name=Xerostomia_1
#SBATCH --mail-type=END
#SBATCH --mail-user=d.macrae@student.rug.nl  # email updates don't work on habrok yet
#SBATCH --time=1:59:59              # shorter time, this script just installs libraries
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --output=slurm-%j.log


# Install:
module purge
#module load fosscuda/2020b
#module load foss/2022b
module load Python/3.11.3-GCCcore-12.3.0              # import Python
#module load OpenCV/4.6.0-foss-2022a-contrib
#module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0 
python3 -m venv /scratch/$USER/.envs/HNC_env             # make a .venv with all the dependencies needed
source /scratch/$USER/.envs/HNC_env/bin/activate
pip3 install torch torchvision torchaudio
pip3 install torchinfo tqdm monai pytz SimpleITK pydicom scikit-image matplotlib numpy 
pip3 install torch_optimizer
pip3 install scikit-learn opencv-python
pip3 install timm
pip install --upgrade pip
pip3 install pandas



# Run
#module purge
#module load fosscuda/2020b
# module load Python/3.7.4-GCCcore-8.3.0
# module load OpenCV/4.2.0-fosscuda-2019b-Python-3.7.4
#module load OpenCV/4.2.0-foss-2020a-Python-3.8.2-contrib
#module load Python/3.8.6-GCCcore-10.2.0
# module load PyTorch/1.10.0-fosscuda-2020b
## Activate local python environment


#source /scratch/$USER/.envs/HNC_env/bin/activate

# Train
#python3 -u main.py

