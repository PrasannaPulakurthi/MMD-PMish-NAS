import torch
from torch import nn
import torch.nn.functional as F

# Define the Activation Function class
class Activation(nn.Module): 
    def __init__(self, act): 
        super(Activation, self).__init__()
        if act == 'relu':
            self.activation_fn = nn.ReLU()
        elif act == 'silu':
            self.activation_fn = nn.SiLU()
        elif act == 'swish':
            self.activation_fn = SwishActivation()
        elif act == 'mish':
            self.activation_fn = nn.Mish()
        elif act == 'pmishact':
            self.activation_fn = PMish()
        else:
            raise NotImplementedError(f'No activation function found for {act}')
		
    def forward(self, x): 
        return self.activation_fn(x)

# Define the Swish activation function 
class SwishActivation(nn.Module): 
	def __init__(self): 
		super(SwishActivation, self).__init__() 
		self.beta = nn.Parameter(torch.ones(1).type(torch.cuda.FloatTensor))
		self.sigmoid = nn.Sigmoid()
		
	def forward(self, x): 
		return x * self.sigmoid(self.beta*x)
    
# Define the PMish activation function 
class PMish(nn.Module):
    def __init__(self):
        super(PMish, self).__init__()
	self.beta = nn.Parameter(torch.ones(1).type(torch.cuda.FloatTensor))

    def forward(self, x):
        return x * torch.tanh(F.softplus(self.beta * x) / self.beta)

	
