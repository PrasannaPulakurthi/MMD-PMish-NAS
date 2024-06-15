import os
import sys

'''
network_size = "small"
gf_dim = "128"
checkpoint = "train_swish_large_cifar10_12345_2024_02_19_20_07_43"
'''

network_size = "large"
gf_dim = "256"
checkpoint = "train/pmishact_large_cifar10_33333_2024_04_18_19_59_27"

freeze_layers = ["l1", "l2", "l3"]

# FC Layers
layers = ["l1","l2","l3"]
ranks = [2, 4, 8, 16, 32, 'nocompression']

command_template = (
    "python MGPU_find_rank.py --gpu_ids 0 --num_workers 1 --dataset cifar10 "
    "--bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --freeze_before_compressed "
    "--genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim {gf_dim} --df_dim 512 "
    "--num_eval_imgs 50000 --checkpoint {checkpoint} --freeze_layers {freeze_layers} "
    "--exp_name compress_rank/{rank}_{layer_no} --val_freq 5 --gen_bs 128 --dis_bs 128 "
    "--beta1 0.0 --beta2 0.9 --byrank --rank {rank} --layers {layer} --act pmishact"
)

for layer_no, layer in enumerate(layers,start=1):
    for rank in ranks:
        if rank == 'nocompression':
            rank = 40
        # Format the command with the current layer name and rank
        command = command_template.format(layer=layer, rank=rank, layer_no=layer_no, gf_dim=gf_dim, checkpoint=checkpoint, freeze_layers=' '.join(freeze_layers))
        print(command)
        # Execute the command to Compress each layer with Rank R
        os.system(command)

# Conv Layers
layers = [
    "cell1.c0.ops.0.op.1", "cell1.c1.ops.0.op.1", "cell1.c2.ops.0.op.1",
    "cell1.c3.ops.0.op.1", "cell2.c0.ops.0.op.1", "cell2.c2.ops.0.op.1",
    "cell2.c3.ops.0.op.1", "cell2.c4.ops.0.op.1", "cell3.c0.ops.0.op.1",
    "cell3.c1.ops.0.op.1", "cell3.c2.ops.0.op.1", "cell3.c3.ops.0.op.1",
    "cell3.c4.ops.0.op.1"
]
ranks = [64, 128, 256, 512, 768, 'nocompression']
kernel = [
          3, 5, 5, 
          5, 5, 3, 
          3, 3, 5, 
          3, 5, 5, 
          5
]

command_template = (
    "python MGPU_find_rank.py --gpu_ids 0 --num_workers 1 --dataset cifar10 "
    "--bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --freeze_before_compressed "
    "--genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim {gf_dim} --df_dim 512 "
    "--num_eval_imgs 50000 --checkpoint {checkpoint} --freeze_layers {freeze_layers} "
    "--exp_name compress_rank/{rank}_{layer_no} --val_freq 5 --gen_bs 128 --dis_bs 128 "
    "--beta1 0.0 --beta2 0.9 --byrank --rank {rank} --layers {layer} --act pmishact"
)

for layer_no, layer in enumerate(layers,start=1):
    for rank in ranks:
        if rank == 'nocompression':
            if kernel[layer_no-1] == 3:
                rank1 = 1024
            elif kernel[layer_no-1] == 5:
                rank1 = 2048
        else:
            rank1 = rank
        # Format the command with the current layer name and rank
        command = command_template.format(layer=layer, rank=rank1, layer_no=layer_no, gf_dim=gf_dim, checkpoint=checkpoint, freeze_layers=' '.join(freeze_layers))
        print(command)
        # Execute the command to Compress each layer with Rank R
        os.system(command)

quit()

command_template = (
    "python MGPU_cpcompress_arch.py --gpu_ids 0 --num_workers 1 --dataset cifar10 "
    "--bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False "
    "--genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim {gf_dim} --df_dim 512 "
    "--num_eval_imgs 50000 --eval_batch_size 100 --checkpoint {checkpoint} "
    "--exp_name Nocompress_{rank}_{layer_no} --val_freq 5 --gen_bs 128 --dis_bs 128 "
    "--beta1 0.0 --beta2 0.9 --byrank --rank {rank} --layers {layer} "
    "--compress-mode allatonce --max_epoch_G 1 --act swish"
)

ranks = [1024,1024,5,5,5,3,3,3,5,3,5,5,5]
layers = ["cell1.c1.ops.0.op.1"]
ranks = [1536]
# Kernel Size 3,5,5,5,5,3,3,3,5,3,5,5,5
# No Compression
for layer_no, layer in enumerate(layers,start=1):
    # Format the command with the current layer name and rank
    command = command_template.format(layer=layer, rank=ranks[layer_no-1], layer_no=layer_no, gf_dim=gf_dim, checkpoint=checkpoint)
    # print(command)
    # Execute the command to Compress each layer with Rank R
    os.system(command)


# Compress the all layers large network 
command_template = (
    "python MGPU_cpcompress_arch.py --gpu_ids 0 --num_workers 1 --dataset cifar10 "
    "--bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False "
    "--genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim {gf_dim} --df_dim 512 "
    "--num_eval_imgs 50000 --eval_batch_size 100 --checkpoint {checkpoint} "
    "--exp_name compress_{rank}_all --val_freq 5 --gen_bs 128 --dis_bs 128 "
    "--beta1 0.0 --beta2 0.9 --byrank --rank {rank} --layers {layer} "
    "--compress-mode allatonce --max_epoch_G 1 --act swish"
)
# Compress and finetune all the convolutional layers
layer_str = ' '.join(layers)

for rank in ranks:
    # Rank R
    command = command_template.format(layer=layer_str, rank=rank, gf_dim=gf_dim, checkpoint=checkpoint)
    # print(command)
    # os.system(command)


# Compress the all layers large network 
command_template = (
    "python MGPU_cpcompress_arch.py --gpu_ids 0 --num_workers 1 --dataset cifar10 "
    "--bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False "
    "--genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim {gf_dim} --df_dim 512 "
    "--num_eval_imgs 50000 --eval_batch_size 100 --checkpoint {checkpoint} "
    "--exp_name compress_best_rank --val_freq 5 --gen_bs 128 --dis_bs 128 "
    "--beta1 0.0 --beta2 0.9 --byrank --rank {rank} --layers {layer} "
    "--compress-mode allatonce --max_epoch_G 1 --act swish"
)
# The best compression Rank was found to be [64, 64, 256, 64, 256, 64, 64, 128, 256, 128, 128, 128, 512]
best_rank = [64, 64, 256, 64, 256, 64, 64, 128, 256, 128, 128, 128, 512]
# best_rank = [64, 64, 128, 64, 256, 64, 64, 128, 256, 128, 128, 128, 512]
best_rank_str = ' '.join([str(rank) for rank in best_rank])
command = command_template.format(layer=layer_str, rank=best_rank_str, gf_dim=gf_dim, checkpoint=checkpoint)
# print(command)
# os.system(command)

# Skip Layer 13
best_rank = [64, 64, 256, 64, 256, 64, 64, 128, 256, 128, 128, 128]
# best_rank = [64, 64, 128, 64, 256, 64, 64, 128, 256, 128, 128, 128]
best_rank_str = ' '.join([str(rank) for rank in best_rank])
layer_str = ' '.join(layers[:-1])
command = command_template.format(layer=layer_str, rank=best_rank_str, gf_dim=gf_dim, checkpoint=checkpoint)
# print(command)
# os.system(command)

best_rank = [64, 64, 256, 64, 256, 64, 64, 128, 256, 128, 128, 128]
# best_rank = [64, 64, 128, 64, 256, 64, 64, 128, 256, 128, 128, 128]
best_rank_str = ' '.join([str(rank) for rank in best_rank])
layer_str = ' '.join(layers[:-1])
command = command_template.format(layer=layer_str, rank=best_rank_str, gf_dim=gf_dim, checkpoint=checkpoint)
# print(command)
# os.system(command)


layer_str = ' '.join(layers)
# The best compression Rank was found to be [128, 128, 256, 128, 512, 128, 128, 256, 512, 256, 256, 256, 1024]
compression_Rate = [1,1/2,1/4,1/8,1/16]
best_ranks = [
    [64, 64, 256, 64, 256, 64, 64, 128, 512, 256, 128, 256, 512], #1
    [64, 128, 256, 128, 512, 64, 64, 256, 512, 256, 256, 256, 1024], #1/2
    [64, 128, 512, 128, 512, 128, 64, 256, 1024, 512, 512, 256, 1024], #1/4
    [64, 256, 1024, 128, 1024, 128, 64, 256, 1024, 512, 512, 512, 1024], #1/8
    [64, 256, 1024, 128, 1024, 256, 64, 1024, 1024, 1024, 1024, 1024, 1024] #1/16
]

# Compress the all layers large network 
command_template = (
    "python MGPU_cpcompress_arch.py --gpu_ids 0 --num_workers 1 --dataset cifar10 "
    "--bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False "
    "--genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim {gf_dim} --df_dim 512 "
    "--num_eval_imgs 50000 --eval_batch_size 100 --checkpoint {checkpoint} "
    "--exp_name compress_{compression_Rate} --val_freq 5 --gen_bs 128 --dis_bs 128 "
    "--beta1 0.0 --beta2 0.9 --byrank --rank {rank} --layers {layer} "
    "--compress-mode allatonce --max_epoch_G 1 --act swish"
)

print('\n')
for i, best_rank in enumerate(best_ranks):   
    # print(f"Best Rank Configuration #{i}: {best_rank}")
    best_rank_str = ' '.join([str(rank) for rank in best_rank])
    command = command_template.format(layer=layer_str, rank=best_rank_str, gf_dim=gf_dim, checkpoint=checkpoint,compression_Rate=compression_Rate[i])
    # print(command)
    # os.system(command)