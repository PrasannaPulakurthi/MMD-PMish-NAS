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

command_template = (
    "python MGPU_cpcompress_arch.py --gpu_ids 0 --num_workers 1 --dataset cifar10 "
    "--bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False "
    "--genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim {gf_dim} --df_dim 512 "
    "--num_eval_imgs 50000 --eval_batch_size 100 --checkpoint {checkpoint} "
    "--exp_name Nocompress_{rank}_{layer_no} --val_freq 5 --gen_bs 128 --dis_bs 128 "
    "--beta1 0.0 --beta2 0.9 --byrank --rank {rank} --layers {layer} "
    "--compress-mode allatonce --max_epoch_G 1 --act swish"
)
