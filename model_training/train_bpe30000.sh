#!/bin/sh
#SBATCH -A uppmax2026-1-123
#SBATCH -M pelle
#SBATCH -t 20:00:00
#SBATCH -J bpe_nmt_ukr_eng_30000

source ~/.bashrc

# change if needed
conda activate mt26_b 

export CUDA_VISIBLE_DEVICES=0

# change the project path
PROJECT=/home/jova3528/private/MT/project_ukr_en

### change the name of your working directory for your own run, so that you don't overwrite the output directory of your other model!
DATA_DIR=$PROJECT/data/data-bin-bpe30000
WORK_DIR=$PROJECT/results/bpe30000
TRAIN_DIR=$WORK_DIR/checkpoints-bpe
mkdir -p $TRAIN_DIR
mkdir -p $DATA_DIR
TEXT=$PROJECT/data/bpe30000/bpe-data

echo "preparing data"
fairseq-preprocess --source-lang uk --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir $DATA_DIR \
    --workers 20 \
    --seed 1004

echo "Running training"
fairseq-train $DATA_DIR/ \
   --restore-file $TRAIN_DIR/checkpoint4.pt \
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

