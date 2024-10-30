#!/bin/bash

#SBATCH -p long
#SBATCH -o /work/ws-tmp/g051176-SB_NDF/ndf/slurm_scripts/out/%j.out
#SBATCH -e /work/ws-tmp/g051176-SB_NDF/ndf/slurm_scripts/out/%j.err
#SBATCH -t 10:00:00

#SBATCH --mail-type=ALL
#SBATCH --mail-user=suraj.bhardwaj@student.uni-siegen.de

python /work/ws-tmp/g051176-SB_NDF/ndf/dataprocessing/create_split.py --config /work/ws-tmp/g051176-SB_NDF/ndf/configs/shapenet_cars.txt
