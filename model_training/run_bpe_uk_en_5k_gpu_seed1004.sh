#!/bin/bash
#SBATCH -A uppmax2026-1-123
#SBATCH -M pelle
#SBATCH -p gpu
#SBATCH --gpus=l40s:1
#SBATCH -t 20:00:00
#SBATCH -J bpe_uk_en_5k_40ep_gpu
#SBATCH --output=logs/bpe_uk_en_5k_40ep_%j.out
#SBATCH --error=logs/bpe_uk_en_5k_40ep_%j.err

set -e

mkdir -p logs

source ~/.bashrc
conda activate mt26_b

PROJECT=/home/jova3528/private/MT/project_ukr_en

BPE_SIZE=5000
SEED=1004
MAX_EPOCH=40
DROPOUT=0.3

TEXT=$PROJECT/data/bpe${BPE_SIZE}/bpe-data
DATA_DIR=$PROJECT/data/data-bin-bpe${BPE_SIZE}_seed${SEED}_clean
WORK_DIR=$PROJECT/results/bpe${BPE_SIZE}_clean_${MAX_EPOCH}ep_dropout${DROPOUT}_seed${SEED}
TRAIN_DIR=$WORK_DIR/checkpoints-bpe
GEN_DIR=$WORK_DIR/generation_best

mkdir -p "$DATA_DIR"
mkdir -p "$TRAIN_DIR"
mkdir -p "$GEN_DIR"

echo "========================================"
echo "Job started on: $(date)"
echo "Node: $(hostname)"
echo "BPE size: $BPE_SIZE"
echo "Seed: $SEED"
echo "Max epoch: $MAX_EPOCH"
echo "Dropout: $DROPOUT"
echo "TEXT: $TEXT"
echo "DATA_DIR: $DATA_DIR"
echo "WORK_DIR: $WORK_DIR"
echo "TRAIN_DIR: $TRAIN_DIR"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "========================================"

echo "Checking GPU"
nvidia-smi || true

python - <<'PY'
import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
PY

echo "========================================"
echo "Checking input files"
echo "========================================"

ls -lh "$TEXT"/train.uk "$TEXT"/train.en
ls -lh "$TEXT"/valid.uk "$TEXT"/valid.en
ls -lh "$TEXT"/test.uk "$TEXT"/test.en

echo "========================================"
echo "Preprocessing data if needed"
echo "========================================"

if [ -f "$DATA_DIR/dict.uk.txt" ] && [ -f "$DATA_DIR/dict.en.txt" ]; then
    echo "Preprocessed data already exists in $DATA_DIR"
    echo "Skipping fairseq-preprocess"
else
    echo "Running fairseq-preprocess"
    fairseq-preprocess \
        --source-lang uk \
        --target-lang en \
        --trainpref "$TEXT/train" \
        --validpref "$TEXT/valid" \
        --testpref "$TEXT/test" \
        --destdir "$DATA_DIR" \
        --workers 20 \
        --seed "$SEED"
fi

echo "========================================"
echo "Training / resuming model safely"
echo "========================================"

if [ -f "$TRAIN_DIR/checkpoint_last.pt" ]; then
    echo "Found checkpoint_last.pt"
    echo "Resuming from $TRAIN_DIR/checkpoint_last.pt"
    RESTORE_ARGS="--restore-file $TRAIN_DIR/checkpoint_last.pt"
else
    echo "No checkpoint_last.pt found"
    echo "Starting training from scratch"
    RESTORE_ARGS=""
fi

fairseq-train "$DATA_DIR" \
   $RESTORE_ARGS \
   --seed "$SEED" \
   --arch transformer_iwslt_de_en \
   --share-decoder-input-output-embed \
   --optimizer adam \
   --adam-betas '(0.9, 0.98)' \
   --clip-norm 0.0 \
   --lr 5e-4 \
   --lr-scheduler inverse_sqrt \
   --warmup-updates 4000 \
   --dropout "$DROPOUT" \
   --weight-decay 0.0001 \
   --criterion label_smoothed_cross_entropy \
   --label-smoothing 0.1 \
   --max-tokens 4096 \
   --eval-bleu \
   --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
   --eval-bleu-detok moses \
   --eval-bleu-remove-bpe \
   --save-dir "$TRAIN_DIR" \
   --best-checkpoint-metric bleu \
   --maximize-best-checkpoint-metric \
   --keep-best-checkpoints 1 \
   --no-epoch-checkpoints \
   --max-epoch "$MAX_EPOCH"

echo "========================================"
echo "Cleaning extra checkpoints"
echo "========================================"

find "$TRAIN_DIR" -name "checkpoint[0-9]*.pt" -delete
find "$TRAIN_DIR" -name "checkpoint.best_bleu_*.pt" -delete

echo "Remaining checkpoints:"
ls -lh "$TRAIN_DIR"/checkpoint*.pt

echo "========================================"
echo "Generating translations from best checkpoint"
echo "========================================"

rm -rf "$GEN_DIR"
mkdir -p "$GEN_DIR"

fairseq-generate "$DATA_DIR" \
  --path "$TRAIN_DIR/checkpoint_best.pt" \
  --beam 5 \
  --remove-bpe \
  --batch-size 128 \
  --results-path "$GEN_DIR"

echo "========================================"
echo "Extracting generated translations"
echo "========================================"

grep '^H-' "$GEN_DIR/generate-test.txt" \
  | sort -V \
  | cut -f3- > "$WORK_DIR/translations_best.txt"

echo "Translations saved to:"
echo "$WORK_DIR/translations_best.txt"

echo "========================================"
echo "Final folder sizes"
echo "========================================"
du -sh "$TRAIN_DIR"
du -sh "$WORK_DIR"
du -sh "$PROJECT/results"

echo "========================================"
echo "Job finished on: $(date)"
echo "========================================"
