# Large Network
python MGPU_test_arch.py --random_seed 11111 --gpu_ids 0 --num_workers 1  --gen_bs 128 --dis_bs 128 --dataset stl10 --bottom_width 6 --img_size 48 --arch arch_cifar10 --draw_arch False --checkpoint pmishact_large_stl10_11111_2024_04_22_04_35_07 --genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim 256 --num_eval_imgs 50000 --exp_name test/pmishact_large_stl10_11111  --act pmishact

# Small Network
python MGPU_test_arch.py --random_seed 22222 --gpu_ids 0 --num_workers 1  --gen_bs 128 --dis_bs 128 --dataset stl10 --bottom_width 6 --img_size 48 --arch arch_cifar10 --draw_arch False --checkpoint pmishact_small_stl10_22222_2024_04_22_19_19_44 --genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim 128 --num_eval_imgs 50000 --exp_name test/pmishact_small_stl10_22222 --act pmishact
