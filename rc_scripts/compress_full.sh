#!/bin/bash -l

#SBATCH --mail-user=pp4405@rit.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name="${DSET}_${NETWORK}_${ACT}_${PENALTY_RATE}"
#SBATCH --time=3-0:0:0 
#SBATCH --output=Results/compress/%x_%j.out 
#SBATCH --error=Results/compress/%x_%j.err 
#SBATCH --account=vividgan
#SBATCH --partition=tier3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128g
#SBATCH --gres=gpu:a100:1

# Loading the required spack environment
spack env activate tensors-23062101

# Split the RANKS string into an array
IFS=' ' read -r -a ranks_array <<< "$RANKS"

# Debug: Print the ranks array to verify its content
echo "Ranks Array: ${ranks_array[*]}"

# Dynamically set gf_dim based on network size
gf_dim=128  # default for small network
if [ "$NETWORK" = "large" ]; then
    gf_dim=256
fi

# Construct the experiment name dynamically
exp_name="compress_final/${DSET}_${NETWORK}_${PENALTY_RATE}"

# Construct the rank arguments dynamically for the command line
ranks_string="${ranks_array[*]}"
rank_args="--rank $ranks_string"
echo "Ranks Array: ${rank_args}"


# Run the compression model
time python -u MGPU_cpcompress_arch.py \
    --random_seed $SEED \
    --gpu_ids 0 \
    --num_workers 8 \
    --dataset $DSET \
    --bottom_width $BW \
    --img_size $IMGSIZE \
    --arch arch_cifar10 \
    --draw_arch False \
    --freeze_before_compressed \
    --freeze_layers l1 l2 \
    --genotypes_exp arch_cifar10 \
    --latent_dim 120 \
    --gf_dim $gf_dim \
    --df_dim 512 \
    --num_eval_imgs 50000 \
    --checkpoint $CHECKPOINT \
    --exp_name $exp_name \
    --val_freq 5 \
    --gen_bs 128 \
    --dis_bs 128 \
    --beta1 0.0 \
    --beta2 0.9 \
    --byrank \
    $rank_args \
    --layers cell1.c0.ops.0.op.1 cell1.c1.ops.0.op.1 cell1.c2.ops.0.op.1 cell1.c3.ops.0.op.1 \
            cell2.c0.ops.0.op.1 cell2.c2.ops.0.op.1 cell2.c3.ops.0.op.1 cell2.c4.ops.0.op.1 \
            cell3.c0.ops.0.op.1 cell3.c1.ops.0.op.1 cell3.c2.ops.0.op.1 cell3.c3.ops.0.op.1 cell3.c4.ops.0.op.1 \
            l1 l2 \
    --max_epoch_G 300 \
    --act $ACT \
    --eval_batch_size 50
