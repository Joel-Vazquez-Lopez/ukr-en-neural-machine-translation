#!/bin/sh
#SBATCH -A uppmax2026-1-123
#SBATCH -M pelle
#SBATCH -n 2
#SBATCH -t 02:00:00
#SBATCH -J prep_bpe5000
#SBATCH -o results/prep_bpe5000_%j.out
#SBATCH -e results/prep_bpe5000_%j.err

cd /home/jova3528/private/MT/project_ukr_en

echo "Starting BPE 5000 data preparation"
bash data_extraction/prepare_bpe5000_data.sh
echo "Finished BPE 5000 data preparation"
