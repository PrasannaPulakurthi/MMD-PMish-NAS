
# Large Network
python MGPU_test_arch.py --random_seed 55555 --gpu_ids 0 --num_workers 1 --dataset celeba --bottom_width 8 --img_size 64 --arch arch_cifar10 --draw_arch False --checkpoint train/pmishact_large_celeba_55555_2024_06_19_13_54_25 --genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim 256 --num_eval_imgs 50000 --eval_batch_size 50 --exp_name test/arch_test_celeba_large --act pmishact
