#!/bin/bash -l

# To send emails, set the adcdress below and remove one of the "#" signs.
#SBATCH --mail-user=pp4405@rit.edu

# notify on state change: BEGIN, END, FAIL or ALL
#SBATCH --mail-type=ALL

#SBATCH --job-name="mish_search"
#SBATCH --time=1-0:0:0 
#SBATCH --output=Results/search/%x_%j.out 
#SBATCH --error=Results/search/%x_%j.err 

# Put the job in the appropriate partition matchine the account and request FOUR cores
#SBATCH --account=vividgan   #this is the name created for your project when you filled out the questionnaire
#SBATCH --partition=debug  #currently tier3 is the partition where everyone is put.  To get a listing of partitions where the account can run use the command my-accounts
#SBATCH --ntasks 1  #This option advises the Slurm controller that job steps run within the allocation will launch a maximum of number tasks and to provide for sufficient resources. The default is one task per node.
#SBATCH --cpus-per-task=9

# Job memory requirements in MB=m (default),GB=g, or TB=t
#SBATCH --mem=64g
#SBATCH --gres=gpu:a100:1

spack env activate tensors-23062101

# time torchrun --standalone --nproc_per_node=1 train.py

time python -u MGPU_search_arch.py --gpu_ids 0 --gen_bs 64 --dis_bs 64 --dataset cifar10 --bottom_width 4 --img_size 32 --max_epoch_G 100 --arch search_both_cifar10 --latent_dim 120 --gf_dim 160 --df_dim 80 --g_spectral_norm False --d_spectral_norm True --g_lr 0.0002 --d_lr 0.0002 --beta1 0.0 --beta2 0.9 --init_type xavier_uniform --n_critic 1 --val_freq 5 --derive_freq 1 --derive_per_epoch 16 --draw_arch False --exp_name search/mish --num_workers 1 --gumbel_softmax True --act mish
