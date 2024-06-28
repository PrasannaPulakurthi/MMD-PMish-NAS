#!/bin/bash

epochs=300

# 1. Dataset
# dset="cifar10"
# imgsize=32
# bottom_width=4

dset="stl10"
imgsize=48
bottom_width=6

# dset="celeba"
# imgsize=128
# bottom_width=16

# 2. Network Size
# Large network; gfdim=256, Small network; gfdim=128
network="large"
gfdim=256

# network="small"
# gfdim=128

# 3. Activation
act="relu"

bu=4
buincrate=2
bu_end=4

for seed in 11111 22222 33333 44444 55555;
do
    trainprocedure="fixed"
    for bu in 1 4 16 64 256 1024;
    do
        expname="train_proc/${trainprocedure}_${bu}_${bu}_${seed}"
        sbatch --output=Results/%x_%j.out --error=Results/%x_%j.err --time=3-0:0:0 --job-name="$expname" --export=SEED=$seed,BU=$bu,DSET=$dset,EPOCH=$epochs,EXPNAME=$expname,IMGSIZE=$imgsize,BW=$bottom_width,GFDIM=$gfdim,ACT=$act,TRAINPROC=$trainprocedure,BUINC=$buincrate,BUEND=$bu_end rc_scripts/train_bu.sh
    done
    
    bu=4
    trainprocedure="saturate"
    for buincrate in 1.41 2 4;
    do  
        expname="train_proc/${trainprocedure}_${bu}_${buincrate}_${seed}"
        sbatch --output=Results/%x_%j.out --error=Results/%x_%j.err --time=3-0:0:0 --job-name="$expname" --export=SEED=$seed,BU=$bu,DSET=$dset,EPOCH=$epochs,EXPNAME=$expname,IMGSIZE=$imgsize,BW=$bottom_width,GFDIM=$gfdim,ACT=$act,TRAINPROC=$trainprocedure,BUINC=$buincrate,BUEND=$bu_end rc_scripts/train_bu.sh
    done
    
    bu=4
    trainprocedure="saturate_linear"
    for buincrate in 1 2 4 8 16;
    do  
        expname="train_proc/${trainprocedure}_${bu}_${buincrate}_${seed}"
        sbatch --output=Results/%x_%j.out --error=Results/%x_%j.err --time=3-0:0:0 --job-name="$expname" --export=SEED=$seed,BU=$bu,DSET=$dset,EPOCH=$epochs,EXPNAME=$expname,IMGSIZE=$imgsize,BW=$bottom_width,GFDIM=$gfdim,ACT=$act,TRAINPROC=$trainprocedure,BUINC=$buincrate,BUEND=$bu_end rc_scripts/train_bu.sh
    done
    
    trainprocedure="linear"
    for bu_end in 8 16 64 256 1024;
    do  
        expname="train_proc/${trainprocedure}_${bu}_${bu_end}_${seed}"
        sbatch --output=Results/%x_%j.out --error=Results/%x_%j.err --time=3-0:0:0 --job-name="$expname" --export=SEED=$seed,BU=$bu,DSET=$dset,EPOCH=$epochs,EXPNAME=$expname,IMGSIZE=$imgsize,BW=$bottom_width,GFDIM=$gfdim,ACT=$act,TRAINPROC=$trainprocedure,BUINC=$buincrate,BUEND=$bu_end rc_scripts/train_bu.sh
    done
        
done
