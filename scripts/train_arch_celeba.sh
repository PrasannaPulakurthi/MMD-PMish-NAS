# celeba 64
# Train Large Generator
python MGPU_train_arch.py --gpu_ids 0 --num_workers 8 --gen_bs 64 --dis_bs 64 --dataset celeba --bottom_width 8 --img_size 64 --max_epoch_G 500 --n_critic 1 --arch arch_cifar10 --draw_arch False --genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim 256 --df_dim 512 --g_lr 0.0002 --d_lr 0.0002 --beta1 0.0 --beta2 0.9 --init_type xavier_uniform --val_freq 5 --num_eval_imgs 50000 --exp_name arch_train_celeba_large --act pmishact --trainprocedure saturate --eval_batch_size 25

# Train Small Generator
python MGPU_train_arch.py --gpu_ids 0 --num_workers 8 --gen_bs 64 --dis_bs 64 --dataset celeba --bottom_width 8 --img_size 64 --max_epoch_G 500 --n_critic 1 --arch arch_cifar10 --draw_arch False --genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim 128 --df_dim 512 --g_lr 0.0002 --d_lr 0.0002 --beta1 0.0 --beta2 0.9 --init_type xavier_uniform --val_freq 5 --num_eval_imgs 50000 --exp_name arch_train_celeba_small --act pmishact --trainprocedure saturate --eval_batch_size 25


# celeba 128
# Train Large Generator
python MGPU_train_arch.py --gpu_ids 0 --num_workers 8 --gen_bs 64 --dis_bs 64 --dataset celeba --bottom_width 16 --img_size 128 --max_epoch_G 500 --n_critic 1 --arch arch_cifar10 --draw_arch False --genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim 256 --df_dim 512 --g_lr 0.0002 --d_lr 0.0002 --beta1 0.0 --beta2 0.9 --init_type xavier_uniform --val_freq 5 --num_eval_imgs 50000 --exp_name arch_train_celeba_large --act pmishact --trainprocedure saturate --eval_batch_size 25

# Train Small Generator
python MGPU_train_arch.py --gpu_ids 0 --num_workers 8 --gen_bs 64 --dis_bs 64 --dataset celeba --bottom_width 16 --img_size 128 --max_epoch_G 500 --n_critic 1 --arch arch_cifar10 --draw_arch False --genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim 128 --df_dim 512 --g_lr 0.0002 --d_lr 0.0002 --beta1 0.0 --beta2 0.9 --init_type xavier_uniform --val_freq 5 --num_eval_imgs 50000 --exp_name arch_train_celeba_small --act pmishact --trainprocedure saturate --eval_batch_size 25