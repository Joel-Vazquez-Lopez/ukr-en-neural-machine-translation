#!/bin/bash
#SBATCH -A uppmax2026-1-123
#SBATCH -M pelle
#SBATCH -p gpu
#SBATCH --gpus=l40s:1
#SBATCH -t 04:00:00
#SBATCH -J seq2seq60_infer_b5
#SBATCH -o results/seq2seq_bpe30000_len60_infer_beam5_%j.out
#SBATCH -e results/seq2seq_bpe30000_len60_infer_beam5_%j.err

set -e

source ~/.bashrc
conda activate mt26_b

cd /home/jova3528/private/MT/project_ukr_en

echo "========================================"
echo "Inference started on: $(date)"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "========================================"

nvidia-smi || true

python seq2seq/seq2seq_train_beam_best.py \
  --src_lang uk \
  --tgt_lang en \
  --train_file seq2seq/data/train.bpe30000 \
  --dev_file seq2seq/data/valid.bpe30000 \
  --test_file seq2seq/data/test.bpe30000 \
  --checkpoint_dir results/seq2seq_bpe30000_len60/checkpoints \
  --load_checkpoint results/seq2seq_bpe30000_len60/checkpoints/best_model.pt \
  --hidden_size 256 \
  --seed 1004 \
  --max_length 60 \
  --beam_size 5 \
  --inference \
  --out_file results/seq2seq_bpe30000_len60/test_translations_best_beam5.txt

echo "========================================"
echo "Inference finished on: $(date)"
echo "Output:"
ls -lh results/seq2seq_bpe30000_len60/test_translations_best_beam5.txt
echo "========================================"
