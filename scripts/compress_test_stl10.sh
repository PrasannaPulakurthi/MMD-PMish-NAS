
# Train Large Network
python MGPU_test_cpcompress.py --gpu_ids 0 --num_workers 8 --dataset stl10 --bottom_width 6 --img_size 48 --arch arch_cifar10 --draw_arch False --checkpoint compress_train_stl10_large --genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim 256 --num_eval_imgs 50000 --eval_batch_size 100 --exp_name compress_test_stl10_large  --byrank

# Train Small Network
python MGPU_test_cpcompress.py --gpu_ids 0 --num_workers 8 --dataset stl10 --bottom_width 6 --img_size 48 --arch arch_cifar10 --draw_arch False --checkpoint compress_train_stl10_small --genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim 128 --num_eval_imgs 50000 --eval_batch_size 100 --exp_name compress_test_stl10_small  --byrank
