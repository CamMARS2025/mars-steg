#!/bin/bash

CUDA_DEVICE=0
SCRIPT=./scripts/4_testing_theory_of_mind_on_generalisation/test/test_theory_of_mind_deepseek_7b_with_penalise.sh
MODEL_NAME=MARS-STEGO_TRAIN_GenPRM_GenPRM-7B_seed_336
FIXED_ARG=0
MAX=96
STEP=96

for ((i=288; i>=$MAX; i-=$STEP)); do
    echo "Running with argument: $i"
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE $SCRIPT $MODEL_NAME $FIXED_ARG $i
done