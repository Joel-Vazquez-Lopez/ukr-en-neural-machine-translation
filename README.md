# Ukrainian–English Neural Machine Translation

This repository contains my contribution to the Ukrainian–English machine-translation component of the project. It includes data preparation, byte-pair encoding (BPE) preprocessing, neural sequence-to-sequence training scripts, and tools for inspecting BPE segmentation.

The experimental results and quantitative analysis are reported in the accompanying paper. This repository contains the code and reproducibility material needed to reconstruct the preprocessing and training workflows.

## Contents

- `data/raw/` — raw parallel Ukrainian–English corpus files.
- `data/bpe5000/`, `data/bpe10000/`, `data/bpe30000/` — BPE merge-code files for the three vocabulary sizes.
- `data_extraction/` — corpus tokenization, cleaning, and BPE-preparation scripts.
- `model_training/` — Fairseq training and generation scripts, including seeded configurations.
- `seq2seq/` — custom PyTorch sequence-to-sequence implementations, including attention and beam-search variants.
- `scripts/` — helper scripts for extracting and formatting qualitative examples.
- `extract_bpe_segmentation_items.py` — extracts selected Ukrainian items for BPE analysis.
- `analyse_bpe_items_full_examples.py` — compares the segmentation of selected items across BPE configurations.
- `make_manual_examples_bpe30000.py` — prepares qualitative translation examples for analysis.

Generated checkpoints, Fairseq binary datasets, intermediate preprocessing files, experiment outputs, logs, and third-party dependencies are intentionally excluded from version control.

## Data

The experiments use a parallel Ukrainian–English TED-talk corpus divided into training, development, and test sets. Before redistribution, verify that the corpus may legally be included in a public repository. If redistribution is not permitted, remove `data/raw/` and follow the data-download instructions provided by the project authors.

## Requirements

- Python 3
- PyTorch
- Fairseq for the Fairseq training scripts
- Perl for the Moses tokenization and corpus-cleaning scripts
- A Unix-like shell
- Moses tokenizer scripts
- `subword-nmt`

Moses and `subword-nmt` are third-party projects and are not vendored in this repository. They should be installed or downloaded separately according to their respective licenses.

## Preparing the data

From the repository root, run the appropriate preparation script for the desired BPE vocabulary size:

```bash
bash data_extraction/prepare_bpe5000_data.sh
bash data_extraction/prepare_bpe10000_data.sh
bash data_extraction/prepare_bpe30000_data.sh
```

These scripts tokenize and clean the parallel corpus, learn BPE merge operations, and apply BPE to the training, development, and test data. The scripts assume that Moses and `subword-nmt` are available and that commands are run from the repository root.

## Training

Fairseq training configurations are provided for 5,000, 10,000, and 30,000 BPE merge operations:

```bash
bash model_training/train_bpe5000.sh
bash model_training/train_bpe10000.sh
bash model_training/train_bpe30000.sh
```

The `run_bpe_uk_en_*_gpu_seed1004.sh` scripts document the GPU/HPC training setup and random seed used for the corresponding experiments. Adapt the project root, scheduler directives, resource requests, and environment modules to your system before running them.

The custom sequence-to-sequence experiments are launched with the scripts in `seq2seq/`. These include standard, attention-based, continued-training, and beam-search variants.

## BPE analysis

The analysis scripts inspect how selected Ukrainian words and place names are segmented under different BPE vocabulary sizes. Example tables can be generated after preprocessing has produced the corresponding BPE test files.

## Reproducibility notes

- Run commands from the repository root.
- Generated data and model files are not tracked by Git.
- Training requires substantial compute and storage, especially for checkpoint files.
- HPC scripts contain scheduler-specific settings that may need to be adapted locally.
- The numerical findings should be read together with the accompanying paper.

## Citation

If you use this code, please cite the accompanying paper and acknowledge the original corpus and third-party tools used for tokenization, BPE preprocessing, and model training.

## License

Add the project license here after confirming the licensing terms for the code and the redistribution terms for the corpus. The licenses of Moses, `subword-nmt`, PyTorch, Fairseq, and the corpus remain applicable to their respective components.
