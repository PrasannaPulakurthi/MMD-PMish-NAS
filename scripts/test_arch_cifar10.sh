
# Large Network
python MGPU_test_arch.py --random_seed 33333 --gpu_ids 0 --num_workers 1 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --checkpoint train/pmishact_large_cifar10_33333_2024_04_18_19_59_27 --genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim 256 --num_eval_imgs 50000 --exp_name test/pmishact_large_cifar10 --act pmishact

# Small Network
python MGPU_test_arch.py --random_seed 12345 --gpu_ids 0 --num_workers 1 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --checkpoint pmishact_small_cifar10_12345_2024_04_19_01_57_42 --genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim 128 --num_eval_imgs 50000 --exp_name test/pmishact_small_cifar10 --act pmishact