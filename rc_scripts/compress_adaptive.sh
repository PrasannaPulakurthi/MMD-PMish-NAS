#!/bin/bash -l

# To send emails, set the adcdress below and remove one of the "#" signs.
#SBATCH --mail-user=pp4405@rit.edu

# notify on state change: BEGIN, END, FAIL or ALL
#SBATCH --mail-type=ALL

#SBATCH --job-name="pmish_compress_6"
#SBATCH --time=1-12:0:0 
#SBATCH --output=Results/compress/%x_%j.out 
#SBATCH --error=Results/compress/%x_%j.err 

# Put the job in the appropriate partition matchine the account and request FOUR cores
#SBATCH --account=vividgan   #this is the name created for your project when you filled out the questionnaire
#SBATCH --partition=tier3  #currently tier3 is the partition where everyone is put.  To get a listing of partitions where the account can run use the command my-accounts
#SBATCH --ntasks=1  #This option advises the Slurm controller that job steps run within the allocation will launch a maximum of number tasks and to provide for sufficient resources. The default is one task per node.
#SBATCH --cpus-per-task=1

# Job memory requirements in MB=m (default),GB=g, or TB=t
#SBATCH --mem=64g
#SBATCH --gres=gpu:a100:1

spack env activate tensors-23062101

# 1
# time python -u MGPU_cpcompress_arch.py --gpu_ids 0 --num_workers 1 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --freeze_before_compressed --genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim 256 --df_dim 512 --num_eval_imgs 50000 --checkpoint train/pmishact_large_cifar10_33333_2024_04_18_19_59_27 --freeze_layers l1 l2 l3 --exp_name compress_full/1.00000 --val_freq 5 --gen_bs 128 --dis_bs 128 --beta1 0.0 --beta2 0.9 --byrank --rank 64 128 256 64 256 64 64 128 512 128 128 128 768 nc 2 2 --layers cell1.c0.ops.0.op.1 cell1.c1.ops.0.op.1 cell1.c2.ops.0.op.1 cell1.c3.ops.0.op.1 cell2.c0.ops.0.op.1 cell2.c2.ops.0.op.1 cell2.c3.ops.0.op.1 cell2.c4.ops.0.op.1 cell3.c0.ops.0.op.1 cell3.c1.ops.0.op.1 cell3.c2.ops.0.op.1 cell3.c3.ops.0.op.1 cell3.c4.ops.0.op.1 l1 l2 l3 --max_epoch_G 300 --act pmishact

# 2
# time python -u MGPU_cpcompress_arch.py --gpu_ids 0 --num_workers 1 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --freeze_before_compressed --genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim 256 --df_dim 512 --num_eval_imgs 50000 --checkpoint train/pmishact_large_cifar10_33333_2024_04_18_19_59_27 --freeze_layers l1 l2 l3 --exp_name compress_full/0.50000 --val_freq 5 --gen_bs 128 --dis_bs 128 --beta1 0.0 --beta2 0.9 --byrank --rank 64 128 256 128 256 64 64 128 512 128 128 128 768 nc 2 2 --layers cell1.c0.ops.0.op.1 cell1.c1.ops.0.op.1 cell1.c2.ops.0.op.1 cell1.c3.ops.0.op.1 cell2.c0.ops.0.op.1 cell2.c2.ops.0.op.1 cell2.c3.ops.0.op.1 cell2.c4.ops.0.op.1 cell3.c0.ops.0.op.1 cell3.c1.ops.0.op.1 cell3.c2.ops.0.op.1 cell3.c3.ops.0.op.1 cell3.c4.ops.0.op.1 l1 l2 l3 --max_epoch_G 300 --act pmishact

# 3
# time python -u MGPU_cpcompress_arch.py --gpu_ids 0 --num_workers 1 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --freeze_before_compressed --genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim 256 --df_dim 512 --num_eval_imgs 50000 --checkpoint train/pmishact_large_cifar10_33333_2024_04_18_19_59_27 --freeze_layers l1 l2 l3 --exp_name compress_full/0.25000 --val_freq 5 --gen_bs 128 --dis_bs 128 --beta1 0.0 --beta2 0.9 --byrank --rank 64 128 512 128 768 64 128 256 512 128 128 128 768 nc 2 2 --layers cell1.c0.ops.0.op.1 cell1.c1.ops.0.op.1 cell1.c2.ops.0.op.1 cell1.c3.ops.0.op.1 cell2.c0.ops.0.op.1 cell2.c2.ops.0.op.1 cell2.c3.ops.0.op.1 cell2.c4.ops.0.op.1 cell3.c0.ops.0.op.1 cell3.c1.ops.0.op.1 cell3.c2.ops.0.op.1 cell3.c3.ops.0.op.1 cell3.c4.ops.0.op.1 l1 l2 l3 --max_epoch_G 300 --act pmishact

# 4
# time python -u MGPU_cpcompress_arch.py --gpu_ids 0 --num_workers 1 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --freeze_before_compressed --genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim 256 --df_dim 512 --num_eval_imgs 50000 --checkpoint train/pmishact_large_cifar10_33333_2024_04_18_19_59_27 --freeze_layers l1 l2 l3 --exp_name compress_full/0.12500 --val_freq 5 --gen_bs 128 --dis_bs 128 --beta1 0.0 --beta2 0.9 --byrank --rank 64 128 768 128 768 64 128 256 768 256 768 768 nc nc 2 2 --layers cell1.c0.ops.0.op.1 cell1.c1.ops.0.op.1 cell1.c2.ops.0.op.1 cell1.c3.ops.0.op.1 cell2.c0.ops.0.op.1 cell2.c2.ops.0.op.1 cell2.c3.ops.0.op.1 cell2.c4.ops.0.op.1 cell3.c0.ops.0.op.1 cell3.c1.ops.0.op.1 cell3.c2.ops.0.op.1 cell3.c3.ops.0.op.1 cell3.c4.ops.0.op.1 l1 l2 l3 --max_epoch_G 300 --act pmishact

# 5
# time python -u MGPU_cpcompress_arch.py --gpu_ids 0 --num_workers 1 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --freeze_before_compressed --genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim 256 --df_dim 512 --num_eval_imgs 50000 --checkpoint train/pmishact_large_cifar10_33333_2024_04_18_19_59_27 --freeze_layers l1 l2 l3 --exp_name compress_full/0.06250 --val_freq 5 --gen_bs 128 --dis_bs 128 --beta1 0.0 --beta2 0.9 --byrank --rank 64 128 768 128 nc 128 128 512 nc 256 768 768 nc nc 2 8 --layers cell1.c0.ops.0.op.1 cell1.c1.ops.0.op.1 cell1.c2.ops.0.op.1 cell1.c3.ops.0.op.1 cell2.c0.ops.0.op.1 cell2.c2.ops.0.op.1 cell2.c3.ops.0.op.1 cell2.c4.ops.0.op.1 cell3.c0.ops.0.op.1 cell3.c1.ops.0.op.1 cell3.c2.ops.0.op.1 cell3.c3.ops.0.op.1 cell3.c4.ops.0.op.1 l1 l2 l3 --max_epoch_G 300 --act pmishact

# 6
time python -u MGPU_cpcompress_arch.py --gpu_ids 0 --num_workers 1 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --freeze_before_compressed --genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim 256 --df_dim 512 --num_eval_imgs 50000 --checkpoint train/pmishact_large_cifar10_33333_2024_04_18_19_59_27 --freeze_layers l1 l2 l3 --exp_name compress_full/0.03125 --val_freq 5 --gen_bs 128 --dis_bs 128 --beta1 0.0 --beta2 0.9 --byrank --rank 64 128 768 128 nc 128 128 nc nc nc 768 768 nc nc 2 8 --layers cell1.c0.ops.0.op.1 cell1.c1.ops.0.op.1 cell1.c2.ops.0.op.1 cell1.c3.ops.0.op.1 cell2.c0.ops.0.op.1 cell2.c2.ops.0.op.1 cell2.c3.ops.0.op.1 cell2.c4.ops.0.op.1 cell3.c0.ops.0.op.1 cell3.c1.ops.0.op.1 cell3.c2.ops.0.op.1 cell3.c3.ops.0.op.1 cell3.c4.ops.0.op.1 l1 l2 l3 --max_epoch_G 300 --act pmishact

