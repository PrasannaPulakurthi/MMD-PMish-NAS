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
        lossD = XX - YY
        return lossD
      elif type == "gen":
        XX_u = torch.exp(-alpha*L2_XX)
        YY_u = torch.exp(-alpha*L2_YY)
        XY_l = torch.exp(-alpha*L2_XY)
        XX = (1/(m*(m-1))) * (torch.sum(XX_u) - torch.sum(torch.diagonal(XX_u, 0)))
        YY = (1/(m*(m-1))) * (torch.sum(YY_u) - torch.sum(torch.diagonal(YY_u, 0)))
        XY = torch.mean(XY_l)
        lossMMD2 = XX + YY - 2 * XY
        # lossMMD = torch.sqrt(torch.clamp(lossMMD2, min=0))
        return lossMMD2
      
class Modified_MMD_loss(nn.Module):
    def __init__(self, bu = 4, bl = 1/4, lambda_m=0.001):
      super(Modified_MMD_loss, self).__init__()
      self.fix_sigma = 1
      self.bl = bl
      self.bu = bu
      self.lambda_m = lambda_m
      return
  
    def phi(self,x,y):
      total0 = x.unsqueeze(0).expand(int(x.size(0)), int(x.size(0)), int(x.size(1)))
      total1 = y.unsqueeze(1).expand(int(y.size(0)), int(y.size(0)), int(y.size(1)))
      return(((total0-total1)**2).sum(2))
    
    def phi_abs(self,x,y):
      total0 = x.unsqueeze(0).expand(int(x.size(0)), int(x.size(0)), int(x.size(1)))
      total1 = y.unsqueeze(1).expand(int(y.size(0)), int(y.size(0)), int(y.size(1)))
      return((torch.abs(total0-total1)).sum(2))
        
    def forward(self, source, target, type):
      M = source.size(dim=0)
      N = target.size(dim=0)
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
        L2_YY = self.phi_abs(target, target)
        L2_YY = torch.sqrt(torch.clamp(L2_YY,min=bl))
        YY_dist = (1/(m*(m-1))) * (torch.sum(L2_YY) - torch.sum(torch.diagonal(L2_YY, 0)))
        lossD = XX - YY + self.lambda_m * YY_dist
        return lossD
      elif type == "gen":
        XX_u = torch.exp(-alpha*L2_XX)
        YY_u = torch.exp(-alpha*L2_YY)
        XY_l = torch.exp(-alpha*L2_XY)
        XX = (1/(m*(m-1))) * (torch.sum(XX_u) - torch.sum(torch.diagonal(XX_u, 0)))
        YY = (1/(m*(m-1))) * (torch.sum(YY_u) - torch.sum(torch.diagonal(YY_u, 0)))
        XY = torch.mean(XY_l)
        lossMMD2 = XX + YY - 2 * XY
        # lossMMD = torch.sqrt(torch.clamp(lossMMD2, min=0))
        return lossMMD2