#!/bin/bash

module load StdEnv/2023
#module load ~/.venvs/fl_env/bin/~/.venvs/fl_env/bin/python/3.11.5
module load r-bundle-bioconductor/3.18
module load gcc/12.3
module load cuda
module load cudnn
module load arrow
module load perl

~/.venvs/fl_env/bin/python inference.py

#~/.venvs/fl_env/bin/python inference.py > ./logs/inference.log