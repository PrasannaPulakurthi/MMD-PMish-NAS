## CIFAR10
python MGPU_test_cpcompress.py --gpu_ids 0 --num_workers 1 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --checkpoint compress/cifar10_small_1by15_2024_05_05_02_38_59 --genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim 128 --num_eval_imgs 50000 --eval_batch_size 100 --exp_name test/compress_cifar10_small --act pmishact --byrank

## CIFAR100
python MGPU_test_cpcompress.py --gpu_ids 0 --num_workers 1 --dataset cifar100 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --checkpoint compress/cifar100_small_1by15_2024_10_19_02_24_51 --genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim 128 --num_eval_imgs 50000 --eval_batch_size 100 --exp_name test/compress_cifar100_small --act pmishact --byrank

## STL10
python MGPU_test_cpcompress.py --gpu_ids 0 --num_workers 1 --dataset stl10 --bottom_width 6 --img_size 48 --arch arch_cifar10 --draw_arch False --checkpoint compress/stl10_small_1by20_2024_05_06_23_32_55 --genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim 128 --num_eval_imgs 50000 --eval_batch_size 50 --exp_name test/compress_stl10_small --act pmishact --byrank

## CELEBA
python MGPU_test_cpcompress.py --gpu_ids 0 --num_workers 1 --dataset celeba --bottom_width 8 --img_size 64 --arch arch_cifar10 --draw_arch False --checkpoint compress/celeba_small_1by20_2024_09_30_12_43_51 --genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim 128 --num_eval_imgs 50000 --eval_batch_size 25 --exp_name test/compress_celeba_small --act pmishact --byrank  --gen_bs 64 --dis_bs 64
