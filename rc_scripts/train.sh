#!/bin/bash -l

# To send emails, set the adcdress below and remove one of the "#" signs.
#SBATCH --mail-user=pp4405@rit.edu

# notify on state change: BEGIN, END, FAIL or ALL
#SBATCH --mail-type=ALL

# Put the job in the appropriate partition matchine the account and request FOUR cores
#SBATCH --account=vividgan   #this is the name created for your project when you filled out the questionnaire
#SBATCH --partition=tier3  #currently tier3 is the partition where everyone is put.  To get a listing of partitions where the account can run use the command my-accounts
#SBATCH --ntasks 1  #This option advises the Slurm controller that job steps run within the allocation will launch a maximum of number tasks and to provide for sufficient resources. The default is one task per node.
#SBATCH --cpus-per-task=1

# Job memory requirements in MB=m (default),GB=g, or TB=t
#SBATCH --mem=64g
#SBATCH --gres=gpu:a100:1

spack env activate tensors-23062101

# time torchrun --standalone --nproc_per_node=1 train.py
echo "Seed: $SEED"
echo "Dataset: $DSET"
echo "Epochs: $EPOCH"
echo "Bu: $BU"
echo "Expname: $EXPNAME"
echo "Imgsize: $IMGSIZE"
echo "bottom_width: $BW"
echo "gf_dim: $GFDIM"
echo "Activation: $ACT"

time python -u MGPU_train_arch.py --random_seed $SEED --gpu_ids 0 --num_workers 1 --gen_bs 128 --dis_bs 128 --dataset $DSET --bottom_width $BW --img_size $IMGSIZE --max_epoch_G $EPOCH --n_critic 1 --arch arch_cifar10 --draw_arch False --genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim $GFDIM --df_dim 512 --g_lr 0.0002 --d_lr 0.0002 --beta1 0.0 --beta2 0.9 --init_type xavier_uniform --val_freq 5 --num_eval_imgs 50000 --exp_name $EXPNAME --bu $BU --act $ACT
