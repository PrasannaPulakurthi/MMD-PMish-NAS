import os
import sys

'''
network_size = "small"
gf_dim = "128"
checkpoint = "train/pmishact_small_cifar10_12345_2024_04_19_01_57_42"
'''

network_size = "large"
gf_dim = "256"
checkpoint = "train/pmishact_large_cifar10_33333_2024_04_18_19_59_27"

freeze_layers = ["l1", "l2", "l3"]

# Conv & FC Layers
layers = [
    "cell1.c0.ops.0.op.1", "cell1.c1.ops.0.op.1", "cell1.c2.ops.0.op.1",
    "cell1.c3.ops.0.op.1", "cell2.c0.ops.0.op.1", "cell2.c2.ops.0.op.1",
    "cell2.c3.ops.0.op.1", "cell2.c4.ops.0.op.1", "cell3.c0.ops.0.op.1",
    "cell3.c1.ops.0.op.1", "cell3.c2.ops.0.op.1", "cell3.c3.ops.0.op.1",
    "cell3.c4.ops.0.op.1", "l1", "l2", "l3"
]
PenaltyRate = ['Nocomp', 'NC680', 'F680']
best_ranks = [
    ['nc', 'nc', 'nc', 'nc', 'nc', 'nc', 'nc', 'nc', 'nc', 'nc', 'nc', 'nc', 'nc', 'nc', 'nc', 'nc'],
    [680, 680, 680, 680, 680, 680, 680, 680, 'nc', 680, 680, 680, 'nc', 'nc', 2, 2],           #NC 680
    [680, 680, 680, 680, 680, 680, 680, 680, 680, 680, 680, 680, 680, 'nc', 2, 2],             #680
]
'''
PenaltyRate = ['1by1','1by5','1by10','1by15','1by20', 'F128','F256','F512','NC128','NC256','NC512']
best_ranks = [
    [128, 128, 256, 128, 256, 128, 128, 128, 512, 128, 128, 128, 768, 'nc', 2, 2],             #1/1
    [128, 128, 768, 128, 768, 128, 128, 256, 512, 256, 128, 128, 768, 'nc', 2, 2],             #1/5
    [128, 128, 768, 128, 768, 128, 128, 256, 768, 256, 768, 768, 'nc', 'nc', 2, 2],            #1/10
    [128, 128, 768, 128, 768, 128, 128, 512, 'nc', 256, 768, 768, 'nc', 'nc', 2, 2],           #1/15
    [128, 128, 768, 128, 'nc', 128, 128, 'nc', 'nc', 512, 768, 768, 'nc', 'nc', 2, 2],         #1/20
    [128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 'nc', 2, 2],             #128
    [256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 'nc', 2, 2],             #256
    [512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 'nc', 2, 2],             #512
    [128, 128, 128, 128, 128, 128, 128, 128, 'nc', 128, 128, 128, 'nc', 'nc', 2, 2],           #NC 128
    [256, 256, 256, 256, 256, 256, 256, 256, 'nc', 256, 256, 256, 'nc', 'nc', 2, 2],           #NC 256
    [512, 512, 512, 512, 512, 512, 512, 512, 'nc', 512, 512, 512, 'nc', 'nc', 2, 2],           #NC 512
]
'''
command_template = (
    "python MGPU_find_rank.py --gpu_ids 0 --num_workers 1 --dataset cifar10 "
    "--bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --freeze_before_compressed "
    "--genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim {gf_dim} --df_dim 512 "
    "--num_eval_imgs 50000 --checkpoint {checkpoint} --freeze_layers {freeze_layers} "
    "--exp_name compress_large_full/{PenaltyRate} --val_freq 5 --gen_bs 128 --dis_bs 128 "
    "--beta1 0.0 --beta2 0.9 --byrank --rank {rank} --layers {layer} --act pmishact"
)

layer_str = ' '.join(layers)
for i, best_rank in enumerate(best_ranks):   
    # print(f"Best Rank Configuration #{i}: {best_rank}")
    best_rank_str = ' '.join([str(rank) for rank in best_rank])
    command = command_template.format(layer=layer_str, rank=best_rank_str, gf_dim=gf_dim, checkpoint=checkpoint,PenaltyRate=PenaltyRate[i], freeze_layers=' '.join(freeze_layers))
    print(command,'\n')
    os.system(command)



'''
    [64, 128, 256, 64, 256, 64, 64, 128, 512, 128, 128, 128, 768, 'nc', 2, 2],               #1
    [64, 128, 256, 128, 256, 64, 64, 128, 512, 128, 128, 128, 768, 'nc', 2, 2],              #1/2
    [64, 128, 512, 128, 768, 64, 128, 256, 512, 128, 128, 128, 768, 'nc', 2, 2],             #1/4
    [64, 128, 768, 128, 768, 64, 128, 256, 768, 256, 768, 768, 'nc', 'nc', 2, 2],            #1/8
    [64, 128, 768, 128, 'nc', 128, 128, 512, 'nc', 256, 768, 768, 'nc', 'nc', 2, 2],         #1/16
    [64, 128, 768, 128, 'nc', 128, 128, 'nc', 'nc', 'nc', 768, 768, 'nc', 'nc', 2, 2],       #1/32
'''