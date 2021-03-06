#!/bin/bash
# normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=1 --cpus-per-task=4 --mem=6000M
# we run on the gpu partition and we allocate 1 titan x
#SBATCH -p gpu --gres=gpu:titanrtx:1
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=10-00:00:00

# Info:
date -Is
hostname
echo "GPU IDs: $CUDA_VISIBLE_DEVICES"

folder_suffix="$1"
run_script="$2"
model=${run_script:4:-3}
model_folder="${model}_${folder_suffix}"
model_args="${@:3}"

for filepath in ./args/dataset_args/*
do
    protein_family="$(basename $filepath)"
    echo ""
    echo ""
    echo "################################################################################"
    echo "    Running Python script on ${protein_family}. Program output follows"
    echo "################################################################################"
    echo ""

    # set up result folder and output file (otherwise tee complains)
    mkdir -p "./results/${model_folder}/${protein_family}"
    touch "./results/${model_folder}/${protein_family}/${protein_family}.out"

    # run the model unbuffered
    python3 -u "${run_script}" "@${filepath}" -r "${model_folder}/${protein_family}" ${model_args} | tee "./results/${model_folder}/${protein_family}/${protein_family}.out"
done
