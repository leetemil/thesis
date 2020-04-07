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
    echo ""
    echo ""
    echo "################################################################################"
    echo "    Runnin python-script on ${protein_family}. Program output follows"
    echo "################################################################################"
    echo ""

    # set up result folder and output file (otherwise tee complains)
    mkdir -p "./results/${protein_family}"
    touch "./results/${protein_family}/${protein_family}.out"

    # run the model unbuffered
    python3 -u run_vae.py "@${filepath}" -r "${protein_family}" --patience 30 | tee "./results/${protein_family}/${protein_family}.out"
done
