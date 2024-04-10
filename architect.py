import torch
import torch.nn as nn

class MMD_loss(nn.Module):
    def __init__(self, bu = 4, bl = 1/4):
      super(MMD_loss, self).__init__()
      self.fix_sigma = 1
      self.bl = bl
      self.bu = bu
      return
  
    def phi(self,x,y):
      total0 = x.unsqueeze(0).expand(int(x.size(0)), int(x.size(0)), int(x.size(1)))
      total1 = y.unsqueeze(1).expand(int(y.size(0)), int(y.size(0)), int(y.size(1)))
      return(((total0-total1)**2).sum(2))
    
    def forward(self, source, target, type):
      M = source.size(dim=0)
      N = target.size(dim=0)
      # print(M,N)
      if M!=N:
        target = target[:M,:]
      L2_XX = self.phi(source, source)
      L2_YY = self.phi(target, target)
      L2_XY = self.phi(source, target)
      # print(source, target)
      bu = self.bu*torch.ones(L2_XX.size()).type(torch.cuda.FloatTensor)
      bl = self.bl*torch.ones(L2_YY.size()).type(torch.cuda.FloatTensor)
      alpha = (1/(2*self.fix_sigma))*torch.ones(1).type(torch.cuda.FloatTensor)
      m = M*torch.ones(1).type(torch.cuda.FloatTensor)
      if type == "critic":
        XX_u = torch.exp(-alpha*torch.min(L2_XX,bu))
        YY_l = torch.exp(-alpha*torch.max(L2_YY,bl))
        XX = (1/(m*(m-1))) * (torch.sum(XX_u) - torch.sum(torch.diagonal(XX_u, 0)))
        YY = (1/(m*(m-1))) * (torch.sum(YY_l) - torch.sum(torch.diagonal(YY_l, 0)))
        lossD = XX - YY # + 0.001*loss_b
        return lossD
      elif type == "gen":
        XX_u = torch.exp(-alpha*L2_XX)
        YY_u = torch.exp(-alpha*L2_YY)
        XY_l = torch.exp(-alpha*L2_XY)
        XX = (1/(m*(m-1))) * (torch.sum(XX_u) - torch.sum(torch.diagonal(XX_u, 0)))
        YY = (1/(m*(m-1))) * (torch.sum(YY_u) - torch.sum(torch.diagonal(YY_u, 0)))
        XY = torch.mean(XY_l)
        lossmmd = XX + YY - 2 * XY
        eps = 1e-10*torch.tensor(1).type(torch.cuda.FloatTensor)
        lossG = torch.sqrt(torch.max(lossmmd,eps))
        return lossG
      
def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])

class Architect_gen(object):
    def __init__(self, model, args):
        self.args = args
        self.mmd_rep_loss = MMD_loss(args.bu, args.bl)
        if isinstance(model, torch.nn.DataParallel):
            self.model = model.module
        else:
            self.model = model
        self.optimizer = torch.optim.Adam(self.model.arch_parameters(), lr=0.0003, betas=(0.5, 0.999), weight_decay=0.0001)

    def step(self, dis_net, real_imgs, gen_net, search_z):
        self.optimizer.zero_grad()
        self._backward_step(dis_net, real_imgs, gen_net, search_z)
        self.optimizer.step()

    def _backward_step(self, dis_net, real_imgs, gen_net, search_z):
        real_validity = dis_net(real_imgs)
        fake_imgs = gen_net(search_z).detach()
        fake_validity = dis_net(fake_imgs)
        d_loss = self.mmd_rep_loss(real_validity, fake_validity,"gen")
        d_loss.backward()


# --------------------------------------------------------------------------------------- #
class Architect_dis(object):
    def __init__(self, model, args):
        self.args = args
        self.mmd_rep_loss = MMD_loss(args.bu, args.bl)
        if isinstance(model, torch.nn.DataParallel):
            self.model = model.module
        else:
            self.model = model
        self.optimizer = torch.optim.Adam(self.model.arch_parameters(), lr=0.0003, betas=(0.5, 0.999), weight_decay=0.0001)

    def step(self, dis_net, real_imgs, gen_net, search_z):
        self.optimizer.zero_grad()
        self._backward_step(dis_net, real_imgs, gen_net, search_z)
        self.optimizer.step()

    def _backward_step(self, dis_net, real_imgs, gen_net, search_z):
        real_validity = dis_net(real_imgs)
        fake_imgs = gen_net(search_z).detach()
        fake_validity = dis_net(fake_imgs)
        d_loss = self.mmd_rep_loss(real_validity, fake_validity,"critic")
        d_loss.backward()
