#!/bin/bash

epochs=500
bu=4

# 1. Dataset

# dset="cifar10"
# imgsize=32
# bottom_width=4
# gen_bs=128
# dis_bs=128
# eval_bs=50

dset="cifar100"
imgsize=32
bottom_width=4
gen_bs=128
dis_bs=128
eval_bs=50

# dset="stl10"
# imgsize=48
# bottom_width=6
# gen_bs=128
# dis_bs=128
# eval_bs=50

# dset="celeba"
# imgsize=64
# bottom_width=8
# gen_bs=64
# dis_bs=64
# eval_bs=25

n_critic=1

# 2. Network Size
# Large network; gfdim=256, Small network; gfdim=128
network="large"
gfdim=256

# network="small"
# gfdim=128

# 3. Activation
act="pmishact"

for seed in 12345 11111 22222 33333 44444 55555;
do
    expname="train/${dset}/${act}_${network}_${n_critic}_${seed}"
    sbatch --output=Results/%x_%j.out --error=Results/%x_%j.err --time=1-12:0:0 --job-name="$expname" --export=SEED=$seed,BU=$bu,DSET=$dset,EPOCH=$epochs,EXPNAME=$expname,IMGSIZE=$imgsize,BW=$bottom_width,GFDIM=$gfdim,ACT=$act,GEN_BS=$gen_bs,DIS_BS=$dis_bs,EVAL_BS=$eval_bs,N_CRITIC=$n_critic rc_scripts/train.sh
done
