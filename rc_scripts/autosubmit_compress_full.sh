#!/bin/bash -l

###################################################################
# Dataset configuration
# dset="cifar10"
# imgsize=32
# bottom_width=4

# Uncomment for different datasets
dset="stl10"
imgsize=48
bottom_width=6

# dset="celeba"
# imgsize=128
# bottom_width=16

###################################################################
# Network size configuration
network="large"
# network="small"

###################################################################
# Activation function
act="pmishact"

###################################################################
# Checkpoint location
# checkpoint="train/pmishact_large_cifar10_33333_2024_04_18_19_59_27"
# checkpoint="train/pmishact_small_cifar10_12345_2024_04_19_01_57_42"

checkpoint="train/pmishact_large_stl10_11111_2024_04_22_04_35_07"
# checkpoint="train/pmishact_small_stl10_22222_2024_04_22_19_19_44"

###################################################################
# Define additional required variables
seed=11111  # example seed, adjust as necessary
epochs=300  # number of epochs for compression tasks

# Directory to store results
results_dir="Results/compress_full/${dset}"

###################################################################
# Submit the compression jobs for each ratio using sbatch

# Define penalty rate and ranks
penalty_rate="1by1"
ranks="128 128 256 128 256 128 128 128 512 128 128 128 768 nc 2"
sbatch --output=${results_dir}/%x_%j.out --error=${results_dir}/%x_%j.err --job-name="compression_${penalty_rate}_${dset}" --export=SEED=${seed},BU=${bottom_width},DSET=${dset},EPOCH=${epochs},EXPNAME="compress_full/${penalty_rate}",IMGSIZE=${imgsize},BW=${bottom_width},GFDIM=${gf_dim},ACT=${act},PENALTY_RATE=${penalty_rate},CHECKPOINT=${checkpoint},RANKS="${ranks}",NETWORK=${network} rc_scripts/compress_full.sh

penalty_rate="1by5"
ranks="128 128 768 128 768 128 128 256 512 256 128 128 768 nc 2"
sbatch --output=${results_dir}/%x_%j.out --error=${results_dir}/%x_%j.err --job-name="compression_${penalty_rate}_${dset}" --export=SEED=${seed},BU=${bottom_width},DSET=${dset},EPOCH=${epochs},EXPNAME="compress_full/${penalty_rate}",IMGSIZE=${imgsize},BW=${bottom_width},GFDIM=${gf_dim},ACT=${act},PENALTY_RATE=${penalty_rate},CHECKPOINT=${checkpoint},RANKS="${ranks}",NETWORK=${network} rc_scripts/compress_full.sh

penalty_rate="1by10"
ranks="128 128 768 128 768 128 128 256 768 256 768 768 nc nc 2"
sbatch --output=${results_dir}/%x_%j.out --error=${results_dir}/%x_%j.err --job-name="compression_${penalty_rate}_${dset}" --export=SEED=${seed},BU=${bottom_width},DSET=${dset},EPOCH=${epochs},EXPNAME="compress_full/${penalty_rate}",IMGSIZE=${imgsize},BW=${bottom_width},GFDIM=${gf_dim},ACT=${act},PENALTY_RATE=${penalty_rate},CHECKPOINT=${checkpoint},RANKS="${ranks}",NETWORK=${network} rc_scripts/compress_full.sh

penalty_rate="1by15"
ranks="128 128 768 128 768 128 128 512 nc 256 768 768 nc nc 2"
sbatch --output=${results_dir}/%x_%j.out --error=${results_dir}/%x_%j.err --job-name="compression_${penalty_rate}_${dset}" --export=SEED=${seed},BU=${bottom_width},DSET=${dset},EPOCH=${epochs},EXPNAME="compress_full/${penalty_rate}",IMGSIZE=${imgsize},BW=${bottom_width},GFDIM=${gf_dim},ACT=${act},PENALTY_RATE=${penalty_rate},CHECKPOINT=${checkpoint},RANKS="${ranks}",NETWORK=${network} rc_scripts/compress_full.sh

penalty_rate="1by20"
ranks="128 128 768 128 nc 128 128 nc nc 512 768 768 nc nc 2"
sbatch --output=${results_dir}/%x_%j.out --error=${results_dir}/%x_%j.err --job-name="compression_${penalty_rate}_${dset}" --export=SEED=${seed},BU=${bottom_width},DSET=${dset},EPOCH=${epochs},EXPNAME="compress_full/${penalty_rate}",IMGSIZE=${imgsize},BW=${bottom_width},GFDIM=${gf_dim},ACT=${act},PENALTY_RATE=${penalty_rate},CHECKPOINT=${checkpoint},RANKS="${ranks}",NETWORK=${network} rc_scripts/compress_full.sh

penalty_rate="256_nc_9_13"
ranks="256 256 256 256 256 256 256 256 nc 256 256 256 nc nc 2"
sbatch --output=${results_dir}/%x_%j.out --error=${results_dir}/%x_%j.err --job-name="compression_${penalty_rate}_${dset}" --export=SEED=${seed},BU=${bottom_width},DSET=${dset},EPOCH=${epochs},EXPNAME="compress_full/${penalty_rate}",IMGSIZE=${imgsize},BW=${bottom_width},GFDIM=${gf_dim},ACT=${act},PENALTY_RATE=${penalty_rate},CHECKPOINT=${checkpoint},RANKS="${ranks}",NETWORK=${network} rc_scripts/compress_full.sh

penalty_rate="512_nc_9_13"
ranks="512 512 512 512 512 512 512 512 nc 512 512 512 nc nc 2"
sbatch --output=${results_dir}/%x_%j.out --error=${results_dir}/%x_%j.err --job-name="compression_${penalty_rate}_${dset}" --export=SEED=${seed},BU=${bottom_width},DSET=${dset},EPOCH=${epochs},EXPNAME="compress_full/${penalty_rate}",IMGSIZE=${imgsize},BW=${bottom_width},GFDIM=${gf_dim},ACT=${act},PENALTY_RATE=${penalty_rate},CHECKPOINT=${checkpoint},RANKS="${ranks}",NETWORK=${network} rc_scripts/compress_full.sh

