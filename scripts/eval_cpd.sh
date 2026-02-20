#!/bin/bash
#SBATCH --gres=gpu:volta:1
#SBATCH -c 20

# Loading the required modules
source /etc/profile
module load anaconda/2023a

source activate skyscraper

LOG_NAME=cpd_$(date +%Y%m%d_%H%M%S).log

ROOT=data/skyscraper_gdelt_sentinel
CSV=${ROOT}/labels.csv
BACKBONE=remoteclip-14
PEN=2
PELT_MODEL=l2

python -m cpd.eval_cpd \
  --csv ${CSV} \
  --root ${ROOT} \
  --backbone ${BACKBONE} \
  --pen ${PEN} \
  --pelt_model ${PELT_MODEL} \
  |& tee logs/${LOG_NAME}