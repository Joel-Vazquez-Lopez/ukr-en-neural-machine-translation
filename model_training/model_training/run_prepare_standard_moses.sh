#!/bin/sh
#SBATCH -A uppmax2026-1-123
#SBATCH -M pelle
#SBATCH -n 2
#SBATCH -t 01:00:00
#SBATCH -J prep_standard_moses
#SBATCH -o results/prep_standard_moses_%j.out
#SBATCH -e results/prep_standard_moses_%j.err

cd /home/jova3528/private/MT/project_ukr_en

echo "Starting standard Moses data preparation"
bash data_extraction/prepare_data_general.sh
echo "Finished standard Moses data preparation"
