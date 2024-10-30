
## CIFAR10 and CIFAR100
'''
best_ranks = [
    [128, 128, 256, 128, 256, 128, 128, 128, 512, 128, 128, 128, 768, 'nc', 2, 2],             # PF = 1/1
    [128, 128, 768, 128, 768, 128, 128, 256, 512, 256, 128, 128, 768, 'nc', 2, 2],             # PF = 1/5
    [128, 128, 768, 128, 768, 128, 128, 256, 768, 256, 768, 768, 'nc', 'nc', 2, 2],            # PF = 1/10
    [128, 128, 768, 128, 768, 128, 128, 512, 'nc', 256, 768, 768, 'nc', 'nc', 2, 2],           # PF = 1/15
    [128, 128, 768, 128, 'nc', 128, 128, 'nc', 'nc', 512, 768, 768, 'nc', 'nc', 2, 2],         # PF = 1/20
    ]
'''

# Compress the CIFAR10 small network with 1/15
python MGPU_cpcompress_arch.py --random_seed 12345 --gpu_ids 0 --num_workers 1 --dataset cifar10  --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --freeze_before_compressed --freeze_layers l1 l2 l3 --genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim 128 --df_dim 512 --num_eval_imgs 50000 --checkpoint train/pmishact_small_cifar10_12345_2024_04_19_01_57_42 --exp_name compress/cifar10_small_1by15 --val_freq 1 --gen_bs 128 --dis_bs 128 --beta1 0.0 --beta2 0.9 --byrank --rank 128 128 768 128 768 128 128 512 nc 256 768 768 nc nc 2 2 --layers cell1.c0.ops.0.op.1 cell1.c1.ops.0.op.1 cell1.c2.ops.0.op.1 cell1.c3.ops.0.op.1 cell2.c0.ops.0.op.1 cell2.c2.ops.0.op.1 cell2.c3.ops.0.op.1 cell2.c4.ops.0.op.1 cell3.c0.ops.0.op.1 cell3.c1.ops.0.op.1 cell3.c2.ops.0.op.1 cell3.c3.ops.0.op.1 cell3.c4.ops.0.op.1 l1 l2 l3 --max_epoch_G 300 --act pmishact

# Compress the CIFAR100 small network with 1/15
python MGPU_cpcompress_arch.py --random_seed 22222 --gpu_ids 0 --num_workers 1 --dataset cifar100 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --freeze_before_compressed --freeze_layers l1 l2 --genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim 128 --df_dim 512 --num_eval_imgs 50000 --checkpoint train/pmishact_small_cifar100_22222_2024_09_30_14_50_04 --exp_name compress/cifar100_small_1by15 --val_freq 1 --gen_bs 128 --dis_bs 128 --beta1 0.0 --beta2 0.9 --byrank --rank 128 128 768 128 768 128 128 512 nc 256 768 768 nc nc 2 --layers cell1.c0.ops.0.op.1 cell1.c1.ops.0.op.1 cell1.c2.ops.0.op.1 cell1.c3.ops.0.op.1 cell2.c0.ops.0.op.1 cell2.c2.ops.0.op.1 cell2.c3.ops.0.op.1 cell2.c4.ops.0.op.1 cell3.c0.ops.0.op.1 cell3.c1.ops.0.op.1 cell3.c2.ops.0.op.1 cell3.c3.ops.0.op.1 cell3.c4.ops.0.op.1 l1 l2 --max_epoch_G 300 --act pmishact


## STL10, CELEBA
'''
best_ranks = [
    [128, 128, 256, 128, 256, 128, 128, 128, 512, 128, 128, 128, 768, 'nc', 2],             # PF = 1/1
    [128, 128, 768, 128, 768, 128, 128, 256, 512, 256, 128, 128, 768, 'nc', 2],             # PF = 1/5
    [128, 128, 768, 128, 768, 128, 128, 256, 768, 256, 768, 768, 'nc', 'nc', 2],            # PF = 1/10
    [128, 128, 768, 128, 768, 128, 128, 512, 'nc', 256, 768, 768, 'nc', 'nc', 2],           # PF = 1/15
    [128, 128, 768, 128, 'nc', 128, 128, 'nc', 'nc', 512, 768, 768, 'nc', 'nc', 2],         # PF = 1/20
    ]
'''

# Compress the STL10 small network with 1/20
python MGPU_cpcompress_arch.py --random_seed 22222 --gpu_ids 0 --num_workers 1 --dataset stl10  --bottom_width 6 --img_size 48 --arch arch_cifar10 --draw_arch False --freeze_before_compressed --freeze_layers l1 l2 --genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim 128 --df_dim 512 --num_eval_imgs 50000 --checkpoint train/pmishact_small_stl10_22222_2024_04_22_19_19_44    --exp_name compress/stl10_small_1by20 --val_freq 1 --gen_bs 128 --dis_bs 128 --beta1 0.0 --beta2 0.9 --byrank --rank 128 128 768 128 nc 128 128 nc nc 512 768 768 nc nc 2 --layers cell1.c0.ops.0.op.1 cell1.c1.ops.0.op.1 cell1.c2.ops.0.op.1 cell1.c3.ops.0.op.1 cell2.c0.ops.0.op.1 cell2.c2.ops.0.op.1 cell2.c3.ops.0.op.1 cell2.c4.ops.0.op.1 cell3.c0.ops.0.op.1 cell3.c1.ops.0.op.1 cell3.c2.ops.0.op.1 cell3.c3.ops.0.op.1 cell3.c4.ops.0.op.1 l1 l2 --max_epoch_G 300 --act pmishact --eval_batch 50

# Compress the CelebA small network with 1/20
python MGPU_cpcompress_arch.py --random_seed 22222 --gpu_ids 0 --num_workers 1 --dataset celeba --bottom_width 8 --img_size 64 --arch arch_cifar10 --draw_arch False --freeze_before_compressed --freeze_layers l1 l2 --genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim 128 --df_dim 512 --num_eval_imgs 50000 --checkpoint train/pmishact_small_celeba64_22222_2024_09_23_20_07_38 --exp_name compress/celeba_small_1by20 --val_freq 1 --gen_bs 128 --dis_bs 128 --beta1 0.0 --beta2 0.9 --byrank --rank 128 128 768 128 nc 128 128 nc nc 512 768 768 nc nc 2 --layers cell1.c0.ops.0.op.1 cell1.c1.ops.0.op.1 cell1.c2.ops.0.op.1 cell1.c3.ops.0.op.1 cell2.c0.ops.0.op.1 cell2.c2.ops.0.op.1 cell2.c3.ops.0.op.1 cell2.c4.ops.0.op.1 cell3.c0.ops.0.op.1 cell3.c1.ops.0.op.1 cell3.c2.ops.0.op.1 cell3.c3.ops.0.op.1 cell3.c4.ops.0.op.1 l1 l2 --max_epoch_G 300 --act pmishact --eval_batch 25  --gen_bs 64 --dis_bs 64
