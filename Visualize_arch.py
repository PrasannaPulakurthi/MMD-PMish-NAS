from __future__ import absolute_import, division, print_function

import cfg
import archs
import datasets
from network import train, validate, LinearLrDecay, load_params, copy_params
from utils.utils import set_log_dir, save_checkpoint, create_logger, count_parameters_in_MB, set_seed
from utils.inception_score import _init_inception
from utils.fid_score import create_inception_graph, check_or_download_inception
from utils.flop_benchmark import print_FLOPs

import torch
import os
import numpy as np
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from copy import deepcopy
from utils.genotype import draw_graph_G, draw_graph_D

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def main():
    args = cfg.parse_args()
    set_seed(args.random_seed)

    # set visible GPU ids
    if len(args.gpu_ids) > 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
        
    # set TensorFlow environment for evaluation (calculate IS and FID)
    _init_inception([args.eval_batch_size,args.img_size,args.img_size,3])
    inception_path = check_or_download_inception('./tmp/imagenet/')
    create_inception_graph(inception_path)

    # the first GPU in visible GPUs is dedicated for evaluation (running Inception model)
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for id in range(len(str_ids)):
        if id >= 0:
            args.gpu_ids.append(id)
    if len(args.gpu_ids) > 1:
      args.gpu_ids = args.gpu_ids[1:]
    else:
      args.gpu_ids = args.gpu_ids
    

    # create new log dir
    assert args.exp_name
    args.path_helper = set_log_dir('exps', args.exp_name)
    logger = create_logger(args.path_helper['log_path'])

    for i in range(76):
        arch_genotype_name = f'{i}_G'
        # genotype G
        genotypes_root = os.path.join('exps', args.genotypes_exp, 'Genotypes')
        genotype_G = np.load(os.path.join(genotypes_root, f'{arch_genotype_name}.npy'))

        # import network from genotype
        basemodel_gen = eval('archs.' + args.arch + '.Generator')(args, genotype_G)
        gen_net = torch.nn.DataParallel(basemodel_gen, device_ids=args.gpu_ids).cuda(args.gpu_ids[0])

        draw_graph_G(genotype_G, save=True, file_path=os.path.join(args.path_helper['graph_vis_path'], arch_genotype_name))
        Flops_G = print_FLOPs(basemodel_gen, (1, args.latent_dim))

        # model size and FLOPS
        print(f'Iteration: {i}, Param size of G = {count_parameters_in_MB(gen_net)} MB, FLOPS of G = {Flops_G}' )


    basemodel_dis = eval('archs.' + args.arch + '.Discriminator')(args)
    dis_net = torch.nn.DataParallel(basemodel_dis, device_ids=args.gpu_ids).cuda(args.gpu_ids[0])
    print(f'Param size of D = {count_parameters_in_MB(dis_net)} MB')
    print_FLOPs(basemodel_dis, (1, 3, args.img_size, args.img_size))
        
if __name__ == '__main__':
    main()
