#!/usr/bin/env bash
# This is the standard tokenized data preparation for Moses SMT
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

src=uk
tgt=en
lang=uk-en
prep=data/standard
tmp=$prep/tmp
orig=data/raw
tok=$prep/tokenized-data

mkdir -p $tmp $prep $tok

echo "pre-processing train data..."
for l in $src $tgt; do
    f=train.tags.$lang.$l
    tokenized=train.tags.$lang.tok.$l

    if [ "$l" = "$src" ]; then
        cat $orig/train/Neulab-tedtalks_train-1-eng-ukr.ukr | \
        perl $TOKENIZER -threads 8 -l $l > $tmp/$tokenized
    else
        cat $orig/train/Neulab-tedtalks_train-1-eng-ukr.eng | \
        perl $TOKENIZER -threads 8 -l $l > $tmp/$tokenized
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
    cp $tmp/train.$l $tok/train.$l
    cp $tmp/valid.$l $tok/valid.$l
    cp $tmp/test.$l $tok/test.$l
done
