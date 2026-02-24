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

for FEAT_TYPE in patch cls
do
    for TOL in 1 2
    do
        for PELT_MODEL in l2
        do
            for PEN in 1
            do
                python -m cpd.eval_cpd \
                  --csv ${CSV} \
                  --root ${ROOT} \
                  --backbone ${BACKBONE} \
                  --feat_type ${FEAT_TYPE} \
                  --tol ${TOL} \
                  --pen ${PEN} \
                  --pelt_model ${PELT_MODEL} \
                  --output_dir out/cpd/${BACKBONE}/${FEAT_TYPE}/${PELT_MODEL}_penalty_${PEN}/tol_${TOL} \
                  |& tee logs/cpd_$(date +%Y%m%d_%H%M%S).log
            done
        done
    done
done