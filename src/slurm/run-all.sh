#!/bin/bash
for filepath in ./args/dataset_args/*
do
    protein_family="$(basename $filepath)"
    echo "Runnin python-script on ${protein_family}"

    # set up result folder and output file (otherwise tee complains)
    mkdir "./results/${protein_family}"
    touch "./results/${protein_family}/${protein_family}.out"

    # run the model unbuffered
    unbuffer python3 -u run_vae.py "@${filepath}" -r "${protein_family}" | tee "./results/${protein_family}/${protein_family}.out"
done
