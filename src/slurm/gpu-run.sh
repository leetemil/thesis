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
echo "Program output follows:"
echo ""

# Script:
# -u: Unbuffered output
python3 -u "$@"

# End
echo ""
echo "Program finished"
date -Is
