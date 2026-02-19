#!/bin/bash
#SBATCH --gres=gpu:volta:1
#SBATCH -c 20

# Loading the required modules
source /etc/profile
module load anaconda/2023a

source activate skyscraper

ROOT=data/skyscraper_gdelt_sentinel
CSV=${ROOT}/labels.csv
BACKBONE=remoteclip-14

python -m cpd.eval_cpd \
  --csv ${CSV} \
  --root ${ROOT} \
  --backbone ${BACKBONE} \