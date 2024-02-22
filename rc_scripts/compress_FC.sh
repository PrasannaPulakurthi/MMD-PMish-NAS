#!/bin/bash -l

# To send emails, set the adcdress below and remove one of the "#" signs.
#SBATCH --mail-user=pp4405@rit.edu

# notify on state change: BEGIN, END, FAIL or ALL
#SBATCH --mail-type=ALL

#SBATCH --job-name="compress_FC"
#SBATCH --time=1-0:0:0 
#SBATCH --output=Results/compress/%x_%j.out 
#SBATCH --error=Results/compress/%x_%j.err 

# Put the job in the appropriate partition matchine the account and request FOUR cores
#SBATCH --account=vividgan   #this is the name created for your project when you filled out the questionnaire
#SBATCH --partition=debug  #currently tier3 is the partition where everyone is put.  To get a listing of partitions where the account can run use the command my-accounts
#SBATCH --ntasks 1  #This option advises the Slurm controller that job steps run within the allocation will launch a maximum of number tasks and to provide for sufficient resources. The default is one task per node.
#SBATCH --cpus-per-task=9

# Job memory requirements in MB=m (default),GB=g, or TB=t
#SBATCH --mem=64g
#SBATCH --gres=gpu:a100:1

spack env activate tensors-23062101

python MGPU_cpcompress_arch.py --gpu_ids 0 --num_workers 8 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --genotypes_exp arch_cifar10  --latent_dim 120 --gf_dim 256 --df_dim 512 --num_eval_imgs 50000 --eval_batch_size 100 --checkpoint compress_train_cifar10_large_2024_01_31_10_07_05  --exp_name compress_train_cifar10_large_FC --val_freq 5  --gen_bs  128 --dis_bs 128 --beta1 0.0 --beta2 0.9  --byrank --rank 4 --layers l2 l3 --freeze_layers l2 l3 --compress-mode "allatonce" --max_epoch_G 1 --eval_before_compression
