#!/bin/bash -l

# To send emails, set the adcdress below and remove one of the "#" signs.
#SBATCH --mail-user=pp4405@rit.edu

# notify on state change: BEGIN, END, FAIL or ALL
#SBATCH --mail-type=ALL

#SBATCH --job-name="test"
#SBATCH --time=0-12:0:0 
#SBATCH --output=Results/test/%x_%j.out 
#SBATCH --error=Results/test/%x_%j.err 

# Put the job in the appropriate partition matchine the account and request FOUR cores
#SBATCH --account=vividgan   #this is the name created for your project when you filled out the questionnaire
#SBATCH --partition=debug  #currently tier3 is the partition where everyone is put.  To get a listing of partitions where the account can run use the command my-accounts
#SBATCH --ntasks=1  #This option advises the Slurm controller that job steps run within the allocation will launch a maximum of number tasks and to provide for sufficient resources. The default is one task per node.
#SBATCH --cpus-per-task=1

# Job memory requirements in MB=m (default),GB=g, or TB=t
#SBATCH --mem=64g
#SBATCH --gres=gpu:p4:1

spack env activate tensors-23062101

# ReLU
# time python -u MGPU_test_arch.py --gpu_ids 0 --num_workers 1 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --checkpoint train_relu_large_cifar10_11111_2023_09_19_03_11_50 --genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim 256 --num_eval_imgs 50000 --eval_batch_size 100 --exp_name test/relu_11111 --act relu
# time python -u MGPU_test_arch.py --gpu_ids 0 --num_workers 1 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --checkpoint train_relu_large_cifar10_22222_2023_09_19_03_12_13 --genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim 256 --num_eval_imgs 50000 --eval_batch_size 100 --exp_name test/relu_22222 --act relu
# time python -u MGPU_test_arch.py --gpu_ids 0 --num_workers 1 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --checkpoint train_relu_large_cifar10_33333_2023_09_19_03_15_22 --genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim 256 --num_eval_imgs 50000 --eval_batch_size 100 --exp_name test/relu_33333 --act relu
# time python -u MGPU_test_arch.py --gpu_ids 0 --num_workers 1 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --checkpoint train_relu_large_cifar10_44444_2023_09_19_03_16_46 --genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim 256 --num_eval_imgs 50000 --eval_batch_size 100 --exp_name test/relu_44444 --act relu
# time python -u MGPU_test_arch.py --gpu_ids 0 --num_workers 1 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --checkpoint train_relu_large_cifar10_55555_2023_09_19_03_17_47 --genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim 256 --num_eval_imgs 50000 --eval_batch_size 100 --exp_name test/relu_55555 --act relu
# time python -u MGPU_test_arch.py --gpu_ids 0 --num_workers 1 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --checkpoint train_relu_large_cifar10_12345 --genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim 256 --num_eval_imgs 50000 --eval_batch_size 100 --exp_name test/relu_12345 --act relu

# Silu
# time python -u MGPU_test_arch.py --gpu_ids 0 --num_workers 1 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --checkpoint train_silu_large_cifar10_11111_2024_02_11_19_12_32 --genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim 256 --num_eval_imgs 50000 --eval_batch_size 100 --exp_name test/silu_11111 --act silu
# time python -u MGPU_test_arch.py --gpu_ids 0 --num_workers 1 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --checkpoint train_silu_large_cifar10_22222_2024_02_11_19_17_20 --genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim 256 --num_eval_imgs 50000 --eval_batch_size 100 --exp_name test/silu_22222 --act silu
# time python -u MGPU_test_arch.py --gpu_ids 0 --num_workers 1 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --checkpoint train_silu_large_cifar10_33333_2024_02_11_19_30_19 --genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim 256 --num_eval_imgs 50000 --eval_batch_size 100 --exp_name test/silu_33333 --act silu
# time python -u MGPU_test_arch.py --gpu_ids 0 --num_workers 1 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --checkpoint train_silu_large_cifar10_44444_2024_02_11_19_31_19 --genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim 256 --num_eval_imgs 50000 --eval_batch_size 100 --exp_name test/silu_44444 --act silu
# time python -u MGPU_test_arch.py --gpu_ids 0 --num_workers 1 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --checkpoint train_silu_large_cifar10_55555_2024_02_11_19_37_46 --genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim 256 --num_eval_imgs 50000 --eval_batch_size 100 --exp_name test/silu_55555 --act silu
# time python -u MGPU_test_arch.py --gpu_ids 0 --num_workers 1 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --checkpoint train_silu_large_cifar10_12345_2024_02_11_19_12_32 --genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim 256 --num_eval_imgs 50000 --eval_batch_size 100 --exp_name test/silu_12345 --act silu

time python -u MGPU_test_arch.py --gpu_ids 0 --num_workers 1 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --checkpoint arch_train_large_cifar10_44444_2024_01_27_04_29_26 --genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim 256 --num_eval_imgs 50000 --eval_batch_size 100 --exp_name test/silu_12345 --act silu

# Swish
# time python -u MGPU_test_arch.py --gpu_ids 0 --num_workers 1 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --checkpoint train_swish_large_cifar10_11111_2024_02_15_03_36_03 --genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim 256 --num_eval_imgs 50000 --eval_batch_size 100 --exp_name test/swish_11111 --act swish
# time python -u MGPU_test_arch.py --gpu_ids 0 --num_workers 1 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --checkpoint train_swish_large_cifar10_22222_2024_02_15_06_24_54 --genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim 256 --num_eval_imgs 50000 --eval_batch_size 100 --exp_name test/swish_22222 --act swish
# time python -u MGPU_test_arch.py --gpu_ids 0 --num_workers 1 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --checkpoint train_swish_large_cifar10_33333_2024_02_15_06_38_45 --genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim 256 --num_eval_imgs 50000 --eval_batch_size 100 --exp_name test/swish_33333 --act swish
# time python -u MGPU_test_arch.py --gpu_ids 0 --num_workers 1 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --checkpoint train_swish_large_cifar10_44444_2024_02_15_08_49_43 --genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim 256 --num_eval_imgs 50000 --eval_batch_size 100 --exp_name test/swish_44444 --act swish
# time python -u MGPU_test_arch.py --gpu_ids 0 --num_workers 1 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --checkpoint train_swish_large_cifar10_55555_2024_02_15_08_50_53 --genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim 256 --num_eval_imgs 50000 --eval_batch_size 100 --exp_name test/swish_55555 --act swish
# time python -u MGPU_test_arch.py --gpu_ids 0 --num_workers 1 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --checkpoint train_swish_large_cifar10_12345_2024_02_15_03_31_50 --genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim 256 --num_eval_imgs 50000 --eval_batch_size 100 --exp_name test/swish_12345 --act swish