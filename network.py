import os
import numpy as np
import scipy.io
import math
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from imageio import imsave
from tqdm import tqdm
from copy import deepcopy
import logging
import shutil

from utils.inception_score import get_inception_score
from utils.fid_score import calculate_fid_given_paths
from utils.genotype import alpha2genotype, beta2genotype, draw_graph_D, draw_graph_G


logger = logging.getLogger(__name__)


import torch
import torch.nn as nn

class MMD_loss(nn.Module):
    def __init__(self, bu=4, bl=1/4):
        super(MMD_loss, self).__init__()
        self.fix_sigma = 1
        self.bl = bl
        self.bu = bu

    def phi(self, x, y):
        # Efficient computation of pairwise distances using broadcasting
        x_norm = x.pow(2).sum(dim=1, keepdim=True)
        y_norm = y.pow(2).sum(dim=1, keepdim=True)
        dists = x_norm + y_norm.T - 2.0 * torch.mm(x, y.T)
        dists.clamp_(min=0)  # Ensure non-negativity
        return dists

    def forward(self, source, target, type):
        M = source.size(0)
        N = target.size(0)
        if M != N:
            target = target[:M, :]
        L2_XX = self.phi(source, source)
        L2_YY = self.phi(target, target)

        alpha = 1 / (2 * self.fix_sigma)
        m = M

        if type == "critic":
            XX_u = torch.exp(-alpha * torch.min(L2_XX, self.bu * torch.ones_like(L2_XX)))
            YY_l = torch.exp(-alpha * torch.max(L2_YY, self.bl * torch.ones_like(L2_YY)))
            XX = (1 / (m * (m - 1))) * (XX_u.sum() - XX_u.diagonal().sum())
            YY = (1 / (m * (m - 1))) * (YY_l.sum() - YY_l.diagonal().sum())
            lossD = XX - YY
            return lossD
        elif type == "gen":
            L2_XY = self.phi(source, target)
            XX_u = torch.exp(-alpha * L2_XX)
            YY_u = torch.exp(-alpha * L2_YY)
            XY_l = torch.exp(-alpha * L2_XY)
            XX = (1 / (m * (m - 1))) * (XX_u.sum() - XX_u.diagonal().sum())
            YY = (1 / (m * (m - 1))) * (YY_u.sum() - YY_u.diagonal().sum())
            XY = XY_l.mean()
            lossmmd = XX + YY - 2 * XY
            lossmmd.clamp_(min=0)  # Ensure non-negativity
            return lossmmd
      
def train(args, gen_net: nn.Module, dis_net: nn.Module, gen_optimizer, dis_optimizer, gen_avg_param, train_loader, epoch,
          writer_dict, lr_schedulers, architect_gen=None, architect_dis=None):
    writer = writer_dict['writer']
    gen_step = 0
    mmd_rep_loss = MMD_loss(args.bu, args.bl)
    
    # train mode
    gen_net = gen_net.train()
    dis_net = dis_net.train()
    
    for iter_idx, (imgs, _) in enumerate(tqdm(train_loader)):
        global_steps = writer_dict['train_global_steps']
        real_imgs = imgs.type(torch.cuda.FloatTensor)

        # search arch of D
        if architect_dis:  
            real_imgs_w = real_imgs[:imgs.shape[0] // 2]
            real_imgs_arch = real_imgs[imgs.shape[0] // 2:]
            # sample noise
            search_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (imgs.shape[0] // 2, args.latent_dim)))
            if args.amending_coefficient:
                architect_dis.step(dis_net, real_imgs_arch, gen_net, search_z, real_imgs_train=real_imgs_w, train_z=z, eta=args.amending_coefficient)
            else:
                architect_dis.step(dis_net, real_imgs_arch, gen_net, search_z)
            # sample noise
            z = torch.cuda.FloatTensor(np.random.normal(0, 1, (imgs.shape[0] // 2, args.latent_dim)))
        else:
            real_imgs_w = real_imgs
            # sample noise
            z = torch.cuda.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim)))
            
        # train weights of D
        dis_optimizer.zero_grad()
        real_validity = dis_net(real_imgs_w)
        fake_imgs = gen_net(z).detach()
        assert fake_imgs.size() == real_imgs_w.size()
        fake_validity = dis_net(fake_imgs)
        d_loss = mmd_rep_loss(real_validity, fake_validity,"critic")
        d_loss.backward()
        dis_optimizer.step()

        writer.add_scalar('d_loss', d_loss.item(), global_steps)

        # sample noise
        gen_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.gen_bs, args.latent_dim)))
        # search arch of G
        if architect_gen:
            if global_steps % args.n_critic == 0:
                # sample noise
                search_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.gen_bs, args.latent_dim)))
                if args.amending_coefficient:
                    architect_gen.step(dis_net, real_imgs, gen_net, search_z, train_z=gen_z, eta=args.amending_coefficient)
                else:
                    architect_gen.step(dis_net, real_imgs, gen_net, search_z)

        # train weights of G
        if global_steps % args.n_critic == 0:
            gen_optimizer.zero_grad()
            gen_imgs = gen_net(gen_z)
            real_validity = dis_net(real_imgs_w)
            fake_validity = dis_net(gen_imgs)
            g_loss = mmd_rep_loss(real_validity, fake_validity,"gen")
            g_loss.backward()
            gen_optimizer.step()

            # learning rate
            if lr_schedulers:
                gen_scheduler, dis_scheduler = lr_schedulers
                g_lr = gen_scheduler.step(global_steps)
                d_lr = dis_scheduler.step(global_steps)
                writer.add_scalar('LR/g_lr', g_lr, global_steps)
                writer.add_scalar('LR/d_lr', d_lr, global_steps)

            # moving average weight
            for p, avg_p in zip(gen_net.parameters(), gen_avg_param):
                avg_p.mul_(0.999).add_(0.001, p.data)

            writer.add_scalar('g_loss', g_loss.item(), global_steps)
            gen_step += 1

        # verbose
        if gen_step and iter_idx % args.print_freq == 0:
            tqdm.write(
                '[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]' %
                (epoch, args.max_epoch_D, iter_idx % len(train_loader), len(train_loader), d_loss.item(), g_loss.item()))

        writer_dict['train_global_steps'] = global_steps + 1

        if architect_gen:
            # deriving arch of G/D during searching
            derive_freq_iter = math.floor((args.max_iter_D / args.max_epoch_D) / args.derive_per_epoch)
            if (args.derive_per_epoch > 0) and (iter_idx % derive_freq_iter == 0):
                genotype_G = alpha2genotype(gen_net.module.alphas_normal, gen_net.module.alphas_up, save=True,
                                            file_path=os.path.join(args.path_helper['genotypes_path'], str(epoch)+'_'+str(iter_idx)+'_G.npy'))
                genotype_D = beta2genotype(dis_net.module.alphas_normal, dis_net.module.alphas_down, save=True,
                                           file_path=os.path.join(args.path_helper['genotypes_path'], str(epoch)+'_'+str(iter_idx)+'_D.npy'))
                if args.draw_arch:
                    draw_graph_G(genotype_G, save=True,
                                 file_path=os.path.join(args.path_helper['graph_vis_path'], str(epoch)+'_'+str(iter_idx)+'_G'))
                    draw_graph_D(genotype_D, save=True,
                                 file_path=os.path.join(args.path_helper['graph_vis_path'], str(epoch)+'_'+str(iter_idx)+'_D'))


def validate(args, fixed_z, fid_stat, gen_net: nn.Module, writer_dict):
    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']

    # eval mode
    gen_net = gen_net.eval()

    # generate images
    sample_imgs = gen_net(fixed_z)
    img_grid = make_grid(sample_imgs, nrow=10, normalize=True, scale_each=True)
    
    writer.add_image('sampled_images', img_grid, global_steps)
    
    file_name = os.path.join(args.path_helper['sample_path'], 'img_grid.png')
    imsave(file_name, img_grid.mul_(255).clamp_(0.0, 255.0).permute(1, 2, 0).to('cpu', torch.uint8).numpy())

    # get fid and inception score
    fid_buffer_dir = os.path.join(args.path_helper['sample_path'], 'fid_buffer')
    os.makedirs(fid_buffer_dir, exist_ok=True)

    eval_iter = args.num_eval_imgs // args.eval_batch_size
    img_list = list()
    for iter_idx in tqdm(range(eval_iter), desc='sample images'):
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.eval_batch_size, args.latent_dim)))
        # generate a batch of images
        gen_imgs = gen_net(z).mul_(127.5).add_(127.5).clamp_(0.0, 255.0).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
        for img_idx, img in enumerate(gen_imgs):
            file_name = os.path.join(fid_buffer_dir, f'iter{iter_idx}_b{img_idx}.png')
            imsave(file_name, img)
        img_list.extend(list(gen_imgs))

    # get inception score
    logger.info('=> calculate inception score')
    mean, std = get_inception_score(img_list)

    # get fid score
    logger.info('=> calculate fid score')
    fid_score = calculate_fid_given_paths([fid_buffer_dir, fid_stat], inception_path=None)
    
    # del buffer in linux
    os.system('rm -r {}'.format(fid_buffer_dir))

    # Assuming fid_buffer_dir is the path to the directory you want to delete in Windows
    try:
        shutil.rmtree(fid_buffer_dir)
        print(f"Successfully deleted {fid_buffer_dir}")
    except OSError as e:
        print(f"Error: {e.strerror}")
    
    writer.add_scalar('Inception_score/mean', mean, global_steps)
    writer.add_scalar('Inception_score/std', std, global_steps)
    writer.add_scalar('FID_score', fid_score, global_steps)

    writer_dict['valid_global_steps'] = global_steps + 1

    return mean, std, fid_score


class LinearLrDecay(object):
    def __init__(self, optimizer, start_lr, end_lr, decay_start_step, decay_end_step):

        assert start_lr > end_lr
        self.optimizer = optimizer
        self.delta = (start_lr - end_lr) / (decay_end_step - decay_start_step)
        self.decay_start_step = decay_start_step
        self.decay_end_step = decay_end_step
        self.start_lr = start_lr
        self.end_lr = end_lr

    def step(self, current_step):
        if current_step <= self.decay_start_step:
            lr = self.start_lr
        elif current_step >= self.decay_end_step:
            lr = self.end_lr
        else:
            lr = self.start_lr - self.delta * (current_step - self.decay_start_step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        return lr


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)


def copy_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten
