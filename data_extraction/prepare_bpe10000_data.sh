#!/usr/bin/env bash

# This is the data preparing for the BPE with 30000k 
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh


echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
git clone https://github.com/rsennrich/subword-nmt.git

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
BPEROOT=subword-nmt/subword_nmt

# Here you change the number of BPE tokens you want
BPE_TOKENS=10000

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

src=uk
tgt=en
lang=uk-en

# change the name here for the folder 
prep=data/bpe10000

tmp=$prep/tmp
orig=data/raw
bpe=$prep/bpe-data

mkdir -p $tmp $prep $bpe

echo "pre-processing train data..."
for l in $src $tgt; do
    f=train.tags.$lang.$l
    tok=train.tags.$lang.tok.$l

    if [ "$l" = "$src" ]; then
        cat $orig/train/Neulab-tedtalks_train-1-eng-ukr.ukr | \
        perl $TOKENIZER -threads 8 -l $l > $tmp/$tok
    else
        cat $orig/train/Neulab-tedtalks_train-1-eng-ukr.eng | \
        perl $TOKENIZER -threads 8 -l $l > $tmp/$tok
    fi
    echo ""
done

perl $CLEAN -ratio 3 $tmp/train.tags.$lang.tok $src $tgt $tmp/train.tags.$lang.clean 1 175

for l in $src $tgt; do
    cp $tmp/train.tags.$lang.clean.$l $tmp/train.tags.$lang.$l
done

echo "pre-processing valid/test data..."
for l in $src $tgt; do
    if [ "$l" = "$src" ]; then
        cat $orig/dev/Neulab-tedtalks_dev-1-eng-ukr.ukr | \
        perl $TOKENIZER -threads 8 -l $l > $tmp/valid.$l

        cat $orig/test/Neulab-tedtalks_test-1-eng-ukr.ukr | \
        perl $TOKENIZER -threads 8 -l $l > $tmp/test.$l
    else
        cat $orig/dev/Neulab-tedtalks_dev-1-eng-ukr.eng | \
        perl $TOKENIZER -threads 8 -l $l > $tmp/valid.$l

        cat $orig/test/Neulab-tedtalks_test-1-eng-ukr.eng | \
        perl $TOKENIZER -threads 8 -l $l > $tmp/test.$l
    fi
    echo ""
done


echo "creating train, valid, test..."
for l in $src $tgt; do
    cp $tmp/train.tags.$lang.$l $tmp/train.$l
done

TRAIN=$tmp/train.uk-en
BPE_CODE=$prep/code
rm -f $TRAIN
for l in $src $tgt; do
    cat $tmp/train.$l >> $TRAIN
done

echo "learn_bpe.py on ${TRAIN}..."
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

for L in $src $tgt; do
    for f in train.$L valid.$L test.$L; do
        echo "apply_bpe.py to ${f}..."
        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/$f > $bpe/$f
    done
done
