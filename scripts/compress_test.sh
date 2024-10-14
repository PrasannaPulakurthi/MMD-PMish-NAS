## CIFAR10
# Compress Small Network
python MGPU_test_cpcompress.py --gpu_ids 0 --num_workers 1 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --checkpoint Compress/cifar10_small_1by15_2024_05_05_02_38_59 --genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim 128 --num_eval_imgs 50000 --eval_batch_size 100 --exp_name Compress/test_cifar10_small  --byrank

## STL10
# Train Small Network
python MGPU_test_cpcompress.py --gpu_ids 0 --num_workers 1 --dataset stl10 --bottom_width 6 --img_size 48 --arch arch_cifar10 --draw_arch False --checkpoint Compress/stl10_small_1by20_2024_05_06_23_32_55 --genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim 128 --num_eval_imgs 50000 --eval_batch_size 100 --exp_name Compress/test_stl10_small  --byrank
