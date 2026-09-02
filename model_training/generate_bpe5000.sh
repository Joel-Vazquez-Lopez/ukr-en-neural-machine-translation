#!/bin/sh
#SBATCH -A uppmax2026-1-123
#SBATCH -M pelle
#SBATCH -t 02:00:00
#SBATCH -J gen_bpe5000
#SBATCH -o results/bpe5000_generate_%j.out
#SBATCH -e results/bpe5000_generate_%j.err

source ~/.bashrc
conda activate mt26_b

cd /home/jova3528/private/MT/project_ukr_en

mkdir -p results/bpe5000/translations

echo "Generating test translations for BPE 5000"

fairseq-generate data/data-bin-bpe5000 \
  --path results/bpe5000/checkpoints-bpe/checkpoint_best.pt \
  --batch-size 64 \
  --beam 5 \
  --remove-bpe \
  > results/bpe5000/translations/generate-test.txt

echo "BLEU result:"
grep "Generate test" results/bpe5000/translations/generate-test.txt

grep '^H-' results/bpe5000/translations/generate-test.txt | sort -V | cut -f3- > results/bpe5000/translations/test.hyp
grep '^T-' results/bpe5000/translations/generate-test.txt | sort -V | cut -f2- > results/bpe5000/translations/test.ref

echo "Line counts:"
wc -l results/bpe5000/translations/test.hyp results/bpe5000/translations/test.ref
