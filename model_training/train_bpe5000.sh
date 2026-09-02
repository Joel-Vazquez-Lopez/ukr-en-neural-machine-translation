#!/bin/sh
#SBATCH -A uppmax2026-1-123
#SBATCH -M pelle

#SBATCH -t 35:00:00
#SBATCH -J bpe_nmt_uk_en_5000
#SBATCH -o results/bpe5000_train_%j.out
#SBATCH -e results/bpe5000_train_%j.err

source ~/.bashrc
conda activate mt26_b

export CUDA_VISIBLE_DEVICES=0

PROJECT=/home/jova3528/private/MT/project_ukr_en

DATA_DIR=$PROJECT/data/data-bin-bpe5000
WORK_DIR=$PROJECT/results/bpe5000
TRAIN_DIR=$WORK_DIR/checkpoints-bpe
TEXT=$PROJECT/data/bpe5000/bpe-data

mkdir -p $TRAIN_DIR
mkdir -p $DATA_DIR

echo "preparing data"
fairseq-preprocess --source-lang uk --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir $DATA_DIR \
    --workers 20 \
    --seed 1004

echo "Running training"
fairseq-train $DATA_DIR/ \
   --restore-file $TRAIN_DIR/checkpoint_last.pt \
   --seed 1004 \
   --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
   --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
   --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
   --dropout 0.3 --weight-decay 0.0001 \
   --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
   --max-tokens 4096 \
   --eval-bleu \
   --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
   --eval-bleu-detok moses \
   --eval-bleu-remove-bpe \
   --eval-bleu-print-samples \
   --save-dir $TRAIN_DIR \
   --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
   --max-epoch 5
