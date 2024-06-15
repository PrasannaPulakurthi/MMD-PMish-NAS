from __future__ import absolute_import, division, print_function

import cfg_compress
import archs
import datasets
from network import train, validate, LinearLrDecay, load_params, copy_params
from utils.utils import set_log_dir, save_checkpoint, create_logger, count_parameters_in_MB, set_seed
from utils.inception_score import _init_inception
from utils.fid_score import create_inception_graph, check_or_download_inception
from utils.flop_benchmark import print_FLOPs
from utils.compress_utils import *

import torch
import os
import numpy as np
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from copy import deepcopy
from decompose import Compression, DecompositionInfo, CompressionInfo
from utils.metrics import PerformanceStore

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def main():
    args = cfg_compress.parse_args()
    validate_args(args)
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

    # genotype G
    genotypes_root = os.path.join('exps', args.genotypes_exp, 'Genotypes')
    genotype_G = np.load(os.path.join(genotypes_root, 'latest_G.npy'))

    # import network from genotype
    basemodel_gen = eval('archs.' + args.arch + '.Generator')(args, genotype_G)
    gen_net = torch.nn.DataParallel(basemodel_gen, device_ids=args.gpu_ids).cuda(args.gpu_ids[0])
    
    # set up data_loader
    dataset = datasets.ImageDataset(args)
    train_loader = dataset.train
    
    # fid stat
    if args.dataset.lower() == 'cifar10':
        fid_stat = 'fid_stat/fid_stats_cifar10_train.npz'
    elif args.dataset.lower() == 'stl10':
        fid_stat = 'fid_stat/stl10_train_unlabeled_fid_stats_48.npz'
    else:
        raise NotImplementedError(f'no fid stat for {args.dataset.lower()}')
    assert os.path.exists(fid_stat)
    
    # initial
    gen_avg_param = copy_params(gen_net)
    start_epoch = 0

    # model size
    gen_params0 = count_parameters_in_MB(gen_net)
    gen_flops0 = print_FLOPs(basemodel_gen, (1, args.latent_dim))

    # Instanciate the Compression object
    compress_obj = Compression(gen_params0, gen_flops0)

    # set writer
    if args.checkpoint:
        # resuming
        print(f'=> resuming from {args.checkpoint}')
        print(os.path.join('exps', args.checkpoint))
        assert os.path.exists(os.path.join('exps', args.checkpoint))
        checkpoint_file = os.path.join('exps', args.checkpoint, 'Model', 'checkpoint_best.pth')
        assert os.path.exists(checkpoint_file)
        checkpoint = torch.load(checkpoint_file)

        start_epoch = checkpoint['epoch']

        if 'decomposition_info' in checkpoint.keys():
            print('Applying decomposition to generator architecture from the checkpoint...')
            try:
                compression_info = checkpoint['compression_info']
            except:
                compression_info = None
            compress_obj.apply_decomposition_from_checkpoint(args, gen_net, checkpoint['decomposition_info'], compression_info, replace_only=True)  # apply decomposition before loading checkpoint
        else:
            # starting from pretrained model
            # re-set the best_fid and best_is, otherwise, 
            # the best checkpoint will not be saved due to 
            # the performance degradation caused by the compression
            args.resume = False ## saves to different folders
            start_epoch = 0
        
        if 'performance_store' in checkpoint.keys():
            performance_store = checkpoint['performance_store']
            print('Loaded performance store from the checkpoint')
            print(performance_store)
        else:
            performance_store = None
        
        gen_net.load_state_dict(checkpoint['gen_state_dict'])
        avg_gen_net = deepcopy(gen_net)
        avg_gen_net.load_state_dict(checkpoint['avg_gen_state_dict'])
        gen_avg_param = copy_params(avg_gen_net)
        del avg_gen_net

        if args.resume:
            args.path_helper = checkpoint['path_helper']
        else:
            args.path_helper = set_log_dir('exps', args.exp_name)

        logger = create_logger(args.path_helper['log_path'])
        logger.info(f'=> loaded checkpoint {checkpoint_file} (epoch {start_epoch})')
    else:
        # create new log dir
        assert args.exp_name
        args.path_helper = set_log_dir('exps', args.exp_name)
        logger = create_logger(args.path_helper['log_path'])

    # logger.info(args)
    writer_dict = {
        'writer': SummaryWriter(args.path_helper['log_path']),
        'train_global_steps': start_epoch * len(train_loader),
        'valid_global_steps': start_epoch // args.val_freq,
    }

    logger.info('Initial Param size of G = %fM', gen_params0)
    logger.info('Initial FLOPs of G = %fM', gen_flops0)

    if performance_store is None:
        performance_store = PerformanceStore()
    
    # for visualization
    if args.draw_arch:
        from utils.genotype import draw_graph_G, draw_graph_D
        draw_graph_G(genotype_G, save=True, file_path=os.path.join(args.path_helper['graph_vis_path'], 'latest_G'))

    fixed_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (100, args.latent_dim)))

    # Apply compression on all layers of the model (one-shot)
    logger.info(f'args.layers:{args.layers}')
    removed_params = {}
    for name, param in gen_net.named_parameters():
        # logger.info(f'scanning for:{name}')
        if any([name[:len('module.'+layer)]=='module.'+layer for layer in args.layers]):
            logger.info(f'found:{name}')
            removed_params[name]=param
    logger.info(f'Removed params:{removed_params.keys()}')

    gen_avg_param, compression_info, decomposition_info = compress_obj.apply_compression(args, gen_net, gen_avg_param, args.layers, args.rank, logger)
    
    if args.freeze_before_compressed:
        if args.freeze_layers:
            for i in range(len(args.freeze_layers)):
                for layer, param in gen_net.named_parameters():
                    if args.freeze_layers and (args.freeze_layers[i] in layer.split('.')):
                        param.requires_grad = False

    if args.freeze_activations:
        for name, param in gen_net.named_parameters():
            if 'activation_fn' in name:
                param.requires_grad = False

    '''
    logger.info('------------------Uncompressed------------------------')
    for name, param in uncompressed_gen.named_parameters():
        logger.info(f"{name}-{param.requires_grad}")
    '''
    logger.info('------------------Compressed--------------------------')
    for name, param in gen_net.named_parameters():
        logger.info(f"{name}-{param.requires_grad}")

    # Evaluate after compression
    logger.info('------------------------------------------')
    logger.info('Performance Evaluation After compression')
    backup_param = copy_params(gen_net)
    load_params(gen_net, gen_avg_param)
    
    inception_score, std, fid_score = validate(args, fixed_z, fid_stat, gen_net, writer_dict)
    logger.info(f'Inception score mean: {inception_score}, Inception score std: {std}, '
                f'FID score: {fid_score} || after compression.')
    load_params(gen_net, backup_param)

if __name__ == '__main__':
    main()
