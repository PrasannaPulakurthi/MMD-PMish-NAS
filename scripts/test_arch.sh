
## Test CIFAR10
# Large Network
python MGPU_test_arch.py --random_seed 33333 --gpu_ids 0 --num_workers 1 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --checkpoint train/pmishact_large_cifar10_33333_2024_04_18_19_59_27 --genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim 256 --num_eval_imgs 50000 --exp_name test/pmishact_large_cifar10 --act pmishact

# Small Network
python MGPU_test_arch.py --random_seed 12345 --gpu_ids 0 --num_workers 1 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --checkpoint train/pmishact_small_cifar10_12345_2024_04_19_01_57_42 --genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim 128 --num_eval_imgs 50000 --exp_name test/pmishact_small_cifar10 --act pmishact

## Test CIFAR100
# Large Network
python MGPU_test_arch.py --random_seed 12345 --gpu_ids 0 --num_workers 1 --dataset cifar100 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --checkpoint train/pmishact_large_cifar100_12345_2024_07_04_01_39_05 --genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim 256 --num_eval_imgs 50000 --exp_name test/pmishact_large_cifar100 --act pmishact

# Small Network
python MGPU_test_arch.py --random_seed 22222 --gpu_ids 0 --num_workers 1 --dataset cifar100 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --checkpoint train/pmishact_small_cifar100_22222_2024_09_30_14_50_04 --genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim 128 --num_eval_imgs 50000 --exp_name test/pmishact_small_cifar100 --act pmishact

## Test STL10
# Large Network
python MGPU_test_arch.py --random_seed 11111 --gpu_ids 0 --num_workers 1  --dataset stl10 --bottom_width 6 --img_size 48 --arch arch_cifar10 --draw_arch False --checkpoint train/pmishact_large_stl10_11111_2024_04_22_04_35_07 --genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim 256 --num_eval_imgs 50000 --exp_name test/pmishact_large_stl10 --act pmishact --eval_batch 50

# Small Network
python MGPU_test_arch.py --random_seed 22222 --gpu_ids 0 --num_workers 1  --dataset stl10 --bottom_width 6 --img_size 48 --arch arch_cifar10 --draw_arch False --checkpoint train/pmishact_small_stl10_22222_2024_04_22_19_19_44 --genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim 128 --num_eval_imgs 50000 --exp_name test/pmishact_small_stl10 --act pmishact --eval_batch 50


## Test CelebA
# Small Network
python MGPU_test_arch.py --random_seed 55555 --gpu_ids 0 --num_workers 1 --dataset celeba --bottom_width 8 --img_size 64 --arch arch_cifar10 --draw_arch False --checkpoint train/pmishact_small_celeba64_22222_2024_09_23_20_07_38 --genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim 128 --num_eval_imgs 50000 --exp_name test/arch_test_celeba_large --act pmishact --eval_batch 25
