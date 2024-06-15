
import os

layers = [
    "cell1.c0.ops.0.op.1", "cell1.c1.ops.0.op.1", "cell1.c2.ops.0.op.1", 
    "cell1.c3.ops.0.op.1", "cell2.c0.ops.0.op.1", "cell2.c2.ops.0.op.1", 
    "cell2.c3.ops.0.op.1", "cell2.c4.ops.0.op.1", "cell3.c0.ops.0.op.1", 
    "cell3.c1.ops.0.op.1", "cell3.c2.ops.0.op.1", "cell3.c3.ops.0.op.1", 
    "cell3.c4.ops.0.op.1"
]

ranks = [64 128 256 512]

command_template = (
    "python MGPU_cpcompress_arch.py --gpu_ids 0 --num_workers 1 --dataset cifar10 "
    "--bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False "
    "--genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim 256 --df_dim 512 "
    "--num_eval_imgs 50000 --eval_batch_size 100 --checkpoint train_swish_large_cifar10_12345_2024_02_19_20_07_43 "
    "--exp_name compress_{rank}_{layer} --val_freq 5 --gen_bs 128 --dis_bs 128 "
    "--beta1 0.0 --beta2 0.9 --byrank --rank {rank} --layers {layer} "
    "--compress-mode individual --max_epoch_G 0 --act swish"
)

import os

for rank in ranks:
    for layer in layers:
        # Format the command with the current layer name and rank
        command = command_template.format(layer=layer, rank=rank)
        
        # Execute the command to Compress each layer with Rank R
        os.system(command)



# Compress the large network 

command_template = (
    "python MGPU_cpcompress_arch.py --gpu_ids 0 --num_workers 1 --dataset cifar10 "
    "--bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False "
    "--genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim 256 --df_dim 512 "
    "--num_eval_imgs 50000 --eval_batch_size 100 --checkpoint train_swish_large_cifar10_12345_2024_02_19_20_07_43 "
    "--exp_name compress_{rank}_all --val_freq 5 --gen_bs 128 --dis_bs 128 "
    "--beta1 0.0 --beta2 0.9 --byrank --rank {rank} --layers {layer} "
    "--compress-mode individual --max_epoch_G 0 --act swish"
)

# Compress and finetune all the convolutional layers
# Rank 64
command = command_template.format(layer=layers, rank=64)
os.system(command)
# Rank 128
command = command_template.format(layer=layers, rank=128)
os.system(command)
# Rank 256
command = command_template.format(layer=layers, rank=256)
os.system(command)
# Rank 512
command = command_template.format(layer=layers, rank=512)
os.system(command)
