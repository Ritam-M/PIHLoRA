import pickle
import scipy.io
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import OrderedDict

# set random seed
np.random.seed(1234)
torch.manual_seed(1234)

# CUDA support 
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class Base_Net(nn.Module):
    def __init__(self):
        super(Base_Net, self).__init__()
        self.fc1 = nn.Linear(2, 64)  # Input layer
        self.fc2 = nn.Linear(64, 64)  
        self.fc3 = nn.Linear(64, 64) 
        self.fc4 = nn.Linear(64, 64)  
        self.fc5 = nn.Linear(64, 64) 
        self.fc6 = nn.Linear(64, 1)
        
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = torch.tanh(self.fc5(x))
        x = self.fc6(x)
        
        return x

class LoRA_Net(nn.Module):
    def __init__(self, base_model, m):
        super(LoRA_Net, self).__init__()
        
        self.base_model = base_model
        
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        self.fc1_A = nn.Linear(2, m, bias=False)
        self.fc1_B = nn.Linear(m, 64, bias=False)
        
        self.fc2_A = nn.Linear(64, m, bias=False)
        self.fc2_B = nn.Linear(m, 64, bias=False)

        self.fc3_A = nn.Linear(64, m, bias=False)
        self.fc3_B = nn.Linear(m, 64, bias=False)

        self.fc4_A = nn.Linear(64, m, bias=False)
        self.fc4_B = nn.Linear(m, 64, bias=False)

        self.fc5_A = nn.Linear(64, m, bias=False)
        self.fc5_B = nn.Linear(m, 64, bias=False)
        
        self.fc6_A = nn.Linear(64, m, bias=False)
        self.fc6_B = nn.Linear(m, 1, bias=False)       

    def forward(self, x):
        
        x = torch.tanh(self.base_model.fc1(x) + self.fc1_B(self.fc1_A(x)))
        x = torch.tanh(self.base_model.fc2(x) + self.fc2_B(self.fc2_A(x)))
        x = torch.tanh(self.base_model.fc3(x) + self.fc3_B(self.fc3_A(x)))
        x = torch.tanh(self.base_model.fc4(x) + self.fc4_B(self.fc4_A(x)))
        x = torch.tanh(self.base_model.fc5(x) + self.fc5_B(self.fc5_A(x)))
        x = self.base_model.fc6(x) + self.fc6_B(self.fc6_A(x)) # This layer is not changed

        return x

  class VPNSFNet():
    def __init__(self, xb_train, yb_train, u_train, x_train, y_train, layers, nu = 1e-2):  
        
        self.xb = torch.tensor(xb_train[:, 0:1], requires_grad=True).float().to(device)
        self.yb = torch.tensor(yb_train[:, 0:1], requires_grad=True).float().to(device)       
        
        self.ub = torch.tensor(ub_train[:, 0:1], requires_grad=True).float().to(device)
        
        self.x = torch.tensor(x_train[:, 0:1], requires_grad=True).float().to(device)
        self.y = torch.tensor(y_train[:, 0:1], requires_grad=True).float().to(device)
        
        self.layers = layers
        
        # deep neural networks
        self.dnn1 = Base_Net().to(device)
        self.dnn1.load_state_dict(torch.load("model.pt"))
        
        self.dnn = LoRA_Net(self.dnn1,4).to(device)
        
        # optimizers: using the same settings
        
        self.losses = []
        self.optimizer1 = torch.optim.Adam(self.dnn.parameters(), lr=1e-3)
        self.optimizer2 = torch.optim.LBFGS(self.dnn.parameters(), lr=1,max_iter=20,tolerance_grad=1e-10)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer1, milestones=[], gamma=0.1)
        
        self.nu = nu
        self.count = 0
        self.loss = 1e10
        self.path = "model1.pt"
        
        self.iter = 0
    
    def net_u(self, x, y):  
        u = self.dnn(torch.cat([x, y], dim=1))
        return u
        
    def net_f(self, x, y):
        u = self.net_u(x,y)
        
        u_y = torch.autograd.grad(u,y,grad_outputs=torch.ones_like(u),retain_graph=True,create_graph=True)[0]
        u_x_ = torch.autograd.grad(u**2,x,grad_outputs=torch.ones_like(u),retain_graph=True,create_graph=True)[0]
        
        u_x = torch.autograd.grad(u,x,grad_outputs=torch.ones_like(u),retain_graph=True,create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x,x,grad_outputs=torch.ones_like(u_x),retain_graph=True,create_graph=True)[0]
        
        f_u = u_y + u_x_/2 - self.nu*(u_xx)
        
        return u, f_u
    
    def loss_func(self):
        
        self.optimizer1.zero_grad() 
        
        ub = self.net_u(self.xb, self.yb)
        
        loss_ub = torch.mean((ub - self.ub)**2) 
        
        upred, uf = self.net_f(self.x, self.y)
        
        loss_uf = torch.mean(uf**2)
        
        loss = 1*loss_ub + loss_uf 
        
        self.losses.append(loss.detach().cpu().float())
      
        if loss<self.loss:
            self.loss = loss
            torch.save(self.dnn.state_dict(), self.path)
            self.count = 0
        else:
          self.count += 1
    
        loss.backward()
        
        if self.iter%100==0:
            print("Iter:{},Loss:{}".format(self.iter, loss))
            print("Iter:{},Loss_bc:{}".format(self.iter, loss_ub))
            print("Iter:{},Loss_f:{}".format(self.iter, loss_uf))
            print("-------------------------------------------")
                
        return loss        
            
    def train(self):
        self.dnn.train()
        for _ in range(20001):
            self.optimizer1.step(self.loss_func)
            self.scheduler.step()
            self.iter+=1

            if self.count == 1000:
              break
        self.optimizer2.step(self.loss_func)        
            
    def predict(self, x, y):   
        #self.dnn.load_state_dict(torch.load(self.path))
        return self.net_u(x,y)

data = scipy.io.loadmat("datasets/burgers_pino.mat")
x_data = data['input']
y_data = data['output'].transpose(0,2,1)

N_train = 10001
layers = [2, 256, 256, 256, 256, 256, 1]

x = np.linspace(0, 1.0, 128)
t = np.linspace(0, 1.0, 101)

A = np.meshgrid(x,t)
X = np.concatenate([A[0].reshape(-1,1),A[1].reshape(-1,1)],axis=1)

xb_train = x.reshape(-1,1)
tb_train = np.array([0.0]*128).reshape(-1,1)
ub_train = np.array(x_data[2]).reshape(-1,1)

x_train = np.random.rand(N_train, 1)
t_train = np.random.rand(N_train, 1)

model = VPNSFNet(xb_train, tb_train, ub_train, x_train, t_train, layers)
model.train()

A = np.meshgrid(x,t)
X = np.concatenate([A[0].reshape(-1,1),A[1].reshape(-1,1)],axis=1)

X_test = torch.tensor(X[:,0:1], requires_grad=True).float().to(device)
Y_test = torch.tensor(X[:,1:2], requires_grad=True).float().to(device)
u = model.predict(X_test, Y_test)

u = u.detach().cpu().numpy().reshape(1,-1)
u_true = y_data[2].reshape(1,-1,order='F')

print(np.linalg.norm(u_true-u,2)/np.linalg.norm(u_true,2))
print("u:",np.mean((u_true-u)**2))
