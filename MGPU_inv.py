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
from torchvision.utils import save_image

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def invert_all_images(gen_net, train_loader, args, device='cuda'):
    gen_net.eval()
    args.num_inversion_steps = 200000
    args.lr_z = 0.1
    args.log_interval = 1000

    fid_real_dir = os.path.join(args.path_helper['sample_path'], 'fid_real')
    fid_gen_dir = os.path.join(args.path_helper['sample_path'], 'fid_gen')
    os.makedirs(fid_real_dir, exist_ok=True)
    os.makedirs(fid_gen_dir, exist_ok=True)

    all_inverted_zs = []

    for iter_idx, (imgs, _) in enumerate(tqdm(train_loader)):
        real_imgs = imgs.type(torch.cuda.FloatTensor).to(device)

        z = torch.randn(real_imgs.size(0), args.latent_dim, requires_grad=True, device=device)
        optimizer = torch.optim.Adam([z], lr=args.lr_z)

        for iteration in range(args.num_inversion_steps):
            fake_imgs = gen_net(z)

            loss_l2 = torch.nn.functional.mse_loss(fake_imgs, real_imgs)
            loss_l1 = torch.nn.functional.l1_loss(fake_imgs, real_imgs)
            loss = loss_l1 + loss_l2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # z.data.clamp_(-3, 3)

            if iteration % args.log_interval == 0:
                print(f"Iteration {iteration}, Loss1: {loss_l1.item()}, Loss2: {loss_l2.item()}")

        all_inverted_zs.append(z.detach())

        # Save the real and generated images from the last iteration
        save_image(real_imgs, os.path.join(fid_real_dir, f'real_images_batch_{iter_idx}.png'), normalize=True, scale_each=True)
        save_image(fake_imgs, os.path.join(fid_gen_dir, f'fake_images_batch_{iter_idx}.png'), normalize=True, scale_each=True)
        break

    all_inverted_zs = torch.cat(all_inverted_zs, dim=0)
    return all_inverted_zs


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
    
    # genotype G
    genotypes_root = os.path.join('exps', args.genotypes_exp, 'Genotypes')
    genotype_G = np.load(os.path.join(genotypes_root, 'latest_G.npy'))

    # import network from genotype
    basemodel_gen = eval('archs.' + args.arch + '.Generator')(args, genotype_G)
    gen_net = torch.nn.DataParallel(basemodel_gen, device_ids=args.gpu_ids).cuda(args.gpu_ids[0])
    basemodel_dis = eval('archs.' + args.arch + '.Discriminator')(args)
    dis_net = torch.nn.DataParallel(basemodel_dis, device_ids=args.gpu_ids).cuda(args.gpu_ids[0])

    # weight init
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            if args.init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif args.init_type == 'orth':
                nn.init.orthogonal_(m.weight.data)
            elif args.init_type == 'xavier_uniform':
                nn.init.xavier_uniform_(m.weight.data, 1.)
            else:
                raise NotImplementedError('{} unknown inital type'.format(args.init_type))
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)
            
    gen_net.apply(weights_init)
    dis_net.apply(weights_init)
    
    # set up data_loader
    dataset = datasets.ImageDataset(args)
    train_loader = dataset.train
    print(len(train_loader))
    
    # epoch number for dis_net
    args.max_epoch_D = args.max_epoch_G * args.n_critic
    if args.max_iter_G:
        args.max_epoch_D = np.ceil(args.max_iter_G * args.n_critic / len(train_loader))
    max_iter_D = args.max_epoch_D * len(train_loader)
    
    # set optimizer
    gen_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gen_net.parameters()),
                                     args.g_lr, (args.beta1, args.beta2))
    dis_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, dis_net.parameters()),
                                     args.d_lr, (args.beta1, args.beta2))
    gen_scheduler = LinearLrDecay(gen_optimizer, args.g_lr, 0.0, 0, max_iter_D)
    dis_scheduler = LinearLrDecay(dis_optimizer, args.d_lr, 0.0, 0, max_iter_D)

    # fid stat
    if args.dataset.lower() == 'cifar10':
        fid_stat = 'fid_stat/fid_stats_cifar10_train.npz'
    elif args.dataset.lower() == 'stl10':
        fid_stat = 'fid_stat/stl10_train_unlabeled_fid_stats_48.npz'
    elif args.dataset.lower() == 'celeba':
        fid_stat = 'fid_stat/fid_stats_celebA_train.npz'
    else:
        raise NotImplementedError(f'no fid stat for {args.dataset.lower()}')
    assert os.path.exists(fid_stat)
    
    # initial
    gen_avg_param = copy_params(gen_net)
    start_epoch = 0
    best_fid = 1e4
    best_is = 0

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
        best_fid = checkpoint['best_fid']
        gen_net.load_state_dict(checkpoint['gen_state_dict'])
        dis_net.load_state_dict(checkpoint['dis_state_dict'])
        gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
        dis_optimizer.load_state_dict(checkpoint['dis_optimizer'])
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

    logger.info(args)
    writer_dict = {
        'writer': SummaryWriter(args.path_helper['log_path']),
        'train_global_steps': start_epoch * len(train_loader),
        'valid_global_steps': start_epoch // args.val_freq,
    }
    
    # for visualization
    if args.draw_arch:
        from utils.genotype import draw_graph_G, draw_graph_D
        draw_graph_G(genotype_G, save=True, file_path=os.path.join(args.path_helper['graph_vis_path'], 'latest_G'))
        # draw_graph_D(genotype_D, save=True, file_path=os.path.join(args.path_helper['graph_vis_path'], 'latest_D'))

    # model size
    logger.info('Param size of G = %fMB', count_parameters_in_MB(gen_net))
    logger.info('Param size of D = %fMB', count_parameters_in_MB(dis_net))
    print_FLOPs(basemodel_gen, (1, args.latent_dim), logger)
    print_FLOPs(basemodel_dis, (1, 3, args.img_size, args.img_size), logger)
    
    fixed_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (100, args.latent_dim)))
    
    improvement_count = 6
    best_epoch = 0
    icounter = improvement_count
    logger.info(f'Upper bound: {args.bu} and Lower Bound: {args.bl}.')
    logger.info(f'Best FID score: {best_fid}. Best IS score: {best_is}.')

    # Invert
    all_inverted_zs = invert_all_images(gen_net, train_loader, args)
        
if __name__ == '__main__':
    main()


