#!/bin/bash

#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -o /work/ws-tmp/g051176-SB_NDF/ndf/generate_ndf/generate_13.out
#SBATCH -e /work/ws-tmp/g051176-SB_NDF/ndf/generate_ndf/generate_13.err
#SBATCH -t 23:00:00
#SBATCH --mem=230G
#SBATCH --ntasks=1

#SBATCH --mail-type=ALL
#SBATCH --mail-user=suraj.bhardwaj@student.uni-siegen.de

module load GpuModules

module unload cuda11.2/toolkit/11.2.2
module load cuda10.2/blas/10.2.89

eval "$(conda shell.bash hook)"
conda deactivate
conda activate NDF_SB

python /work/ws-tmp/g051176-SB_NDF/ndf/generate.py --config /work/ws-tmp/g051176-SB_NDF/ndf/configs/shapenet_cars.txt
