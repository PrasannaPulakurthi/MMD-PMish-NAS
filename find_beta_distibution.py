import cfg
import archs
from network import validate, load_params, copy_params
from utils.utils import set_log_dir, create_logger, count_parameters_in_MB, set_seed
from utils.inception_score import _init_inception
from utils.fid_score import create_inception_graph, check_or_download_inception
from utils.flop_benchmark import print_FLOPs

import torch
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy

import matplotlib.pyplot as plt
import seaborn as sns

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def main():
    args = cfg.parse_args()
    set_seed(args.random_seed)

    # set visible GPU ids
    if len(args.gpu_ids) > 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids

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
    # basemodel_dis = eval('archs.' + args.arch + '.Discriminator')(args)
    # dis_net = torch.nn.DataParallel(basemodel_dis, device_ids=args.gpu_ids).cuda(args.gpu_ids[0])

    # set writer
    print(f'=> resuming from {args.checkpoint}')
    assert os.path.exists(os.path.join('exps', args.checkpoint))
    checkpoint_file = os.path.join('exps', args.checkpoint, 'Model', 'checkpoint_best.pth')
    assert os.path.exists(checkpoint_file)
    checkpoint = torch.load(checkpoint_file)
    epoch = checkpoint['epoch'] - 1
    gen_net.load_state_dict(checkpoint['gen_state_dict'])
    # dis_net.load_state_dict(checkpoint['dis_state_dict'])
    avg_gen_net = deepcopy(gen_net)
    avg_gen_net.load_state_dict(checkpoint['avg_gen_state_dict'])
    gen_avg_param = copy_params(avg_gen_net)
    del avg_gen_net
    assert args.exp_name
    args.path_helper = set_log_dir('exps', args.exp_name)
    logger = create_logger(args.path_helper['log_path'])
    logger.info(f'=> loaded checkpoint {checkpoint_file} (epoch {epoch})')
    
    logger.info(args)
    writer_dict = {
        'writer': SummaryWriter(args.path_helper['log_path']),
        'valid_global_steps': epoch // args.val_freq,
    }
    
    # model size
    logger.info('Param size of G = %fMB', count_parameters_in_MB(gen_net))
    # logger.info('Param size of D = %fMB', count_parameters_in_MB(dis_net))
    print_FLOPs(basemodel_gen, (1, args.latent_dim), logger)
    # print_FLOPs(basemodel_dis, (1, 3, args.img_size, args.img_size), logger)
    
    # for visualization
    if args.draw_arch:
        from utils.genotype import draw_graph_G
        draw_graph_G(genotype_G, save=True, file_path=os.path.join(args.path_helper['graph_vis_path'], 'latest_G'))
    fixed_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (100, args.latent_dim)))
    
    # test
    load_params(gen_net, gen_avg_param)

    import re
    # Function to collect beta values and their corresponding layer numbers
    def collect_beta_values(model):
        beta_values = []
        layer_name = []
        pattern = re.compile(r'module\.(\w+\.\w+)')
        for name, module in model.named_modules():
            if hasattr(module, 'activation_fn'):
                if hasattr(module.activation_fn, 'beta'):
                    beta_values.append(module.activation_fn.beta.item())
                    match = pattern.search(name)
                    name1 = match.group(1)
                    layer_name.append(name1)
                    print('Name:' + name1 + ', Beta Value: ' + str(module.activation_fn.beta.item()))
        return layer_name, beta_values

    # Collect beta values and their corresponding layer numbers
    layer_name, beta_values = collect_beta_values(gen_net)

    # Plot the bar plot of beta values
    plt.figure(figsize=(10, 6)) 
    plt.bar(layer_name, beta_values)
    plt.title('Beta Values Across Different Layers')
    plt.xlabel('Layer Name')
    plt.ylabel('Beta Value')
    plt.xticks(ticks=layer_name, rotation=45)
    # plt.xticks(ticks=np.arange(1, 21, 1))
    plt.tight_layout()
    plt.savefig('pmish_beta.png')
    # plt.show()

    # Plot the histogram of beta values
    plt.figure(figsize=(10, 6))
    plt.hist(beta_values, bins=8, edgecolor='black')
    plt.title('Histogram of Beta Values Across Layers')
    plt.xlabel('Beta Value')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('pmish_hist.png')

if __name__ == '__main__':
    main()
