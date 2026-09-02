#!/bin/sh
#SBATCH -A uppmax2026-1-123
#SBATCH -M pelle
#SBATCH -t 02:00:00
#SBATCH -J gen_bpe10000
#SBATCH -o results/bpe10000_generate_%j.out
#SBATCH -e results/bpe10000_generate_%j.err

source ~/.bashrc
conda activate mt26_b

cd /home/jova3528/private/MT/project_ukr_en

mkdir -p results/bpe10000/translations

echo "Generating test translations for BPE 10000"

fairseq-generate data/data-bin-bpe10000 \
  --path results/bpe10000/checkpoints-bpe/checkpoint_best.pt \
  --batch-size 64 \
  --beam 5 \
  --remove-bpe \
  > results/bpe10000/translations/generate-test.txt

echo "BLEU result:"
grep "Generate test" results/bpe10000/translations/generate-test.txt

grep '^H-' results/bpe10000/translations/generate-test.txt | sort -V | cut -f3- > results/bpe10000/translations/test.hyp
grep '^T-' results/bpe10000/translations/generate-test.txt | sort -V | cut -f2- > results/bpe10000/translations/test.ref

echo "Line counts:"
wc -l results/bpe10000/translations/test.hyp results/bpe10000/translations/test.ref
