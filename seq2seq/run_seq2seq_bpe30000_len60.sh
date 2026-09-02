#!/bin/sh
#SBATCH -A uppmax2026-1-123
#SBATCH -M pelle
#SBATCH -t 35:00:00
#SBATCH -J seq2seq_len60
#SBATCH -o results/seq2seq_bpe30000_len60_train_%j.out
#SBATCH -e results/seq2seq_bpe30000_len60_train_%j.err

source ~/.bashrc
conda activate mt26_b

cd /home/jova3528/private/MT/project_ukr_en

mkdir -p results/seq2seq_bpe30000_len60/checkpoints

which python
python -c "import torch, nltk; print(torch.__version__); print(torch.cuda.is_available())"

python seq2seq/seq2seq_train.py \
  --src_lang uk \
  --tgt_lang en \
  --train_file seq2seq/data/train.bpe30000 \
  --dev_file seq2seq/data/valid.bpe30000 \
  --test_file seq2seq/data/test.bpe30000 \
  --checkpoint_dir results/seq2seq_bpe30000_len60/checkpoints \
  --hidden_size 256 \
  --n_iters 100000 \
  --print_every 10000 \
  --status_every 1000 \
  --checkpoint_every 10000 \
  --initial_learning_rate 0.001 \
  --seed 1004 \
  --max_length 60
