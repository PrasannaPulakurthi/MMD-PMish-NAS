
# Large Network
python MGPU_test_arch.py --random_seed 12345 --gpu_ids 0 --num_workers 1 --dataset cifar100 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --checkpoint train/pmishact_large_cifar100_12345_2024_07_04_01_39_05 --genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim 256 --num_eval_imgs 50000 --exp_name test/pmishact_large_cifar100 --act pmishact

# Small Network
python MGPU_test_arch.py --random_seed 22222 --gpu_ids 0 --num_workers 1 --dataset cifar100 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --checkpoint train/pmishact_small_cifar100_22222_2024_09_30_14_50_04 --genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim 128 --num_eval_imgs 50000 --exp_name test/pmishact_small_cifar100 --act pmishact
