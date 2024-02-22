#!/bin/bash

epochs=500
bu=4

# 1. Dataset
dset="cifar10"
imgsize=32
bottom_width=4

# dset="stl10"
# imgsize=48
# bottom_width=6

# dset="celeba"
# imgsize=128
# bottom_width=16

# 2. Network Size
# Large network; gfdim=256, Small network; gfdim=128
gfdim=128

# 3. Activation
act="swish"

for seed in 12345, 11111, 22222, 33333, 44444, 55555;
do
    expname="train_swish_small_"$dset"_$seed"
    sbatch --output=Results/$dset/%x_%j.out --error=Results/$dset/%x_%j.err --time=1-12:0:0 --job-name="$seed"_"$expname" --export=SEED=$seed,BU=$bu,DSET=$dset,EPOCH=$epochs,EXPNAME=$expname,IMGSIZE=$imgsize,BW=$bottom_width,GFDIM=$gfdim,ACT=$act rc_scripts/train.sh
done
