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

for FEAT_TYPE in cls cap_sim class_sim
do
    for TOL in 0 1 2
    do
        for STD in 0.5 0.75
        do
            python -m eval.eval_anomaly \
              --csv ${CSV} \
              --root ${ROOT} \
              --backbone ${BACKBONE} \
              --feat_type ${FEAT_TYPE} \
              --tol ${TOL} \
              --std_thresh ${STD} \
              --output_dir out/anomaly/${BACKBONE}/${FEAT_TYPE}/STD_${STD}/tol_${TOL} \
              |& tee logs/anomaly_$(date +%Y%m%d_%H%M%S).log
        done
    done
done