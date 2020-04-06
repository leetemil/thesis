#!/bin/bash
# normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=1 --cpus-per-task=4 --mem=6000M
# we run on the gpu partition and we allocate 1 titan x
#SBATCH -p gpu --gres=gpu:titanx:1
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=7-00:00:00

# Info:
date -Is
hostname
echo "GPU IDs: $CUDA_VISIBLE_DEVICES"

for filepath in ./args/dataset_args/*
do
    protein_family="$(basename $filepath)"
    echo "Running python-script on ${protein_family}"
    echo "Program output follows:"
    echo ""

    # set up result folder and output file (otherwise tee complains)
    mkdir "./results/${protein_family}"
    touch "./results/${protein_family}/${protein_family}.out"

    # run the model unbuffered
    python3 -u run_vae.py "@${filepath}" -r "${protein_family}" | tee "./results/${protein_family}/${protein_family}.out"
done
