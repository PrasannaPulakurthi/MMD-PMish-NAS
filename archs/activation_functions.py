import torch
from torch import nn

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
            self.activation_fn = PMishActivation()
        else:
            raise NotImplementedError(f'No activation function found for {act}')
		
    def forward(self, x): 
        return self.activation_fn(x)

'''
elif act == 'mish_a':
	self.activation_fn = MishActivationA()
elif act == 'mish_b':
	self.activation_fn = MishActivationB()
elif act == 'pmish':
	self.activation_fn = ParametricMishActivation()
elif act == 'new':
	self.activation_fn = NewActivation()
elif act == 'pnew':
	self.activation_fn = ParametricNewActivation()
elif act == 'prmmish':
	self.activation_fn = ParametricMishActivation3()
'''

# Define the Swish activation function 
class SwishActivation(nn.Module): 
	def __init__(self): 
		super(SwishActivation, self).__init__() 
		self.beta = nn.Parameter(torch.ones(1).type(torch.cuda.FloatTensor))
		self.sigmoid = nn.Sigmoid()
		
	def forward(self, x): 
		return x * self.sigmoid(self.beta*x)
    
# Define the Mish activation function 
class PMishActivation(nn.Module): 
	def __init__(self): 
		super(PMishActivation, self).__init__() 
		self.beta = nn.Parameter(torch.ones(1).type(torch.cuda.FloatTensor))
		self.tanh_fn = nn.Tanh()
		self.softplus_fn = nn.Softplus()
		
	def forward(self, x): 
		return x * self.tanh_fn((1/self.beta) * self.softplus_fn(self.beta*x)) 
	
'''
# Define the Mish activation function 
class MishActivationA(nn.Module): 
	def __init__(self): 
		super(MishActivationA, self).__init__() 
		self.beta = nn.Parameter(torch.ones(1).type(torch.cuda.FloatTensor))
		self.tanh_fn = nn.Tanh()
		self.softplus_fn = nn.Softplus()
		
	def forward(self, x): 
		return x * self.tanh_fn(self.softplus_fn(self.beta*x))
   
# Define the Mish activation function 
class MishActivationB(nn.Module): 
	def __init__(self): 
		super(MishActivationB, self).__init__() 
		self.beta = nn.Parameter(torch.ones(1).type(torch.cuda.FloatTensor))
		self.tanh_fn = nn.Tanh()
		self.softplus_fn = nn.Softplus()
		
	def forward(self, x): 
		return (1/self.beta) * x * self.tanh_fn(self.softplus_fn(self.beta*x)) 
    

# Define the Mish activation function 
class ParametricMishActivation(nn.Module): 
	def __init__(self): 
		super(ParametricMishActivation, self).__init__() 
		self.alpha = nn.Parameter(torch.ones(1).type(torch.cuda.FloatTensor))
		self.beta = nn.Parameter(torch.ones(1).type(torch.cuda.FloatTensor))
		self.tanh_fn = nn.Tanh()
		self.softplus_fn = nn.Softplus()
		
	def forward(self, x): 
		return self.alpha * x * self.tanh_fn(self.softplus_fn(self.beta*x))
    
class ExpoTanh(nn.Module):
    def __init__(self):
        super(ExpoTanh, self).__init__()

    def forward(self, x):
        return (1 - torch.exp(-x)) / (1 + torch.exp(-x))
    
# Define the New activation function 
class NewActivation(nn.Module): 
	def __init__(self): 
		super(NewActivation, self).__init__() 
		self.expo_tanh_fn = ExpoTanh()
		self.softplus_fn = nn.Softplus()
		
	def forward(self, x): 
		return x * self.expo_tanh_fn(self.softplus_fn(x))
   
# Define the New activation function 
class ParametricNewActivation(nn.Module): 
	def __init__(self): 
		super(ParametricNewActivation, self).__init__() 
		self.alpha = nn.Parameter(torch.ones(1).type(torch.cuda.FloatTensor))
		self.beta = nn.Parameter(torch.ones(1).type(torch.cuda.FloatTensor))
		self.expo_tanh_fn = ExpoTanh()
		self.softplus_fn = nn.Softplus()
		
	def forward(self, x): 
		return self.alpha * x * self.expo_tanh_fn(self.softplus_fn(self.beta*x))
	


# Define the Parametric Mish activation function 
class ParametricMishActivation3(nn.Module): 
	def __init__(self): 
		super(ParametricMishActivation3, self).__init__() 
		self.alpha = nn.Parameter(torch.ones(1).type(torch.cuda.FloatTensor))
		self.beta = nn.Parameter(torch.ones(1).type(torch.cuda.FloatTensor))
		self.gamma = nn.Parameter(torch.ones(1).type(torch.cuda.FloatTensor))
		self.tanh_fn = nn.Tanh()
		self.softplus_fn = nn.Softplus()
		
	def forward(self, x): 
		return self.alpha * x * self.tanh_fn(self.gamma * self.softplus_fn(self.beta*x))
'''