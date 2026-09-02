#!/bin/bash
#SBATCH -A uppmax2026-1-123
#SBATCH -M pelle
#SBATCH -p gpu
#SBATCH --gpus=l40s:1
#SBATCH -t 20:00:00
#SBATCH -J seq2seq_attn_300k
#SBATCH -o results/seq2seq_attention_bpe30000_len60_300k_%j.out
#SBATCH -e results/seq2seq_attention_bpe30000_len60_300k_%j.err

set -e

source ~/.bashrc
conda activate mt26_b

cd /home/jova3528/private/MT/project_ukr_en

echo "========================================"
echo "Attention seq2seq training started on: $(date)"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "========================================"

which python
nvidia-smi || true

python - <<'PY'
import torch, nltk
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
print("NLTK:", nltk.__version__)
PY

python seq2seq/seq2seq_train_attention.py \
  --src_lang uk \
  --tgt_lang en \
  --train_file seq2seq/data/train.bpe30000 \
  --dev_file seq2seq/data/valid.bpe30000 \
  --test_file seq2seq/data/test.bpe30000 \
  --checkpoint_dir results/seq2seq_attention_bpe30000_len60/checkpoints \
  --hidden_size 256 \
  --n_iters 300000 \
  --print_every 10000 \
  --status_every 1000 \
  --checkpoint_every 50000 \
  --initial_learning_rate 0.001 \
  --seed 1004 \
  --max_length 60 \
  --beam_size 1 \
  --dev_eval_size 500 \
  --dropout 0.1 \
  --best_checkpoint_name best_model_attention.pt

echo "========================================"
echo "Attention training finished on: $(date)"
echo "Checkpoint folder size:"
du -sh results/seq2seq_attention_bpe30000_len60/checkpoints
echo "Best checkpoint:"
ls -lh results/seq2seq_attention_bpe30000_len60/checkpoints/best_model_attention.pt || true
echo "========================================"
