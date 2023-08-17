import pickle
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
        self.fc1 = nn.Linear(3, 64)  # Input layer
        self.fc2 = nn.Linear(64, 64)  
        self.fc3 = nn.Linear(64, 64) 
        self.fc4 = nn.Linear(64, 64)  
        self.fc5 = nn.Linear(64, 64) 
        self.fc6 = nn.Linear(64, 2)
        
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = torch.tanh(self.fc5(x))
        x = self.fc6(x)
        
        u, v, w = x[:,0:1], x[:,1:2], x[:,2:3]
        return u,v,w

class PINN():
    def __init__(self, X_BC, U_BC, X_F, X_interface, layers):  
        
        self.X_bc = X_BC
        self.U_bc = U_BC
        
        self.X_f = X_F
        self.X_interface = X_interface

        self.layers = layers
        self.losses = []
        
        self.mlp1 = Base_Net().to(device)
        
        self.optimizer = torch.optim.Adam(self.mlp1.parameters(), lr=1e-2)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,milestones=[5000,10000,15000,20000],gamma=0.1)
        
        self.loss = 1e10
        self.path1 = 'model_mlp1.pt'
        
        self.iter = 0
                        
    def net_u(self, x, y, t, nn):  
        
        sol = nn(torch.cat([x, y, t], dim=1))
        u = sol[:,0:1]
        v = sol[:,1:2]
        return u, v
    
    def net_f(self, x, y, t, nn):
        u, v = self.net_u(x,y,t,nn)
        
        u_t = torch.autograd.grad(u,t,grad_outputs=torch.ones_like(u),retain_graph=True,create_graph=True)[0]
        v_t = torch.autograd.grad(v,t,grad_outputs=torch.ones_like(v),retain_graph=True,create_graph=True)[0]
                
        u_x = torch.autograd.grad(u,x,grad_outputs=torch.ones_like(u),retain_graph=True,create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x,x,grad_outputs=torch.ones_like(u_x),retain_graph=True,create_graph=True)[0]
        u_y = torch.autograd.grad(u,y,grad_outputs=torch.ones_like(u),retain_graph=True,create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y,y,grad_outputs=torch.ones_like(u_y),retain_graph=True,create_graph=True)[0]
        
        v_x = torch.autograd.grad(v,x,grad_outputs=torch.ones_like(v),retain_graph=True,create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x,x,grad_outputs=torch.ones_like(v_x),retain_graph=True,create_graph=True)[0]
        v_y = torch.autograd.grad(v,y,grad_outputs=torch.ones_like(v),retain_graph=True,create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y,y,grad_outputs=torch.ones_like(v_y),retain_graph=True,create_graph=True)[0]
        
        u_f = u_t + u*u_x + v*u_y - nu*(u_xx + u_yy) 
        v_f = v_t + u*v_x + v*v_y - nu*(v_xx + v_yy) 
        
        return u_f, v_f

    def loss_func(self):
        self.optimizer.zero_grad()
        
        loss = 0.0
        bc1 = 0.0        
        u = 0.0
        v = 0.0
        
        self.X_bc_x = torch.tensor(self.X_bc[i][:, 0:1], requires_grad=True).float().to(device)
        self.X_bc_y = torch.tensor(self.X_bc[i][:, 1:2], requires_grad=True).float().to(device)
        self.X_bc_t = torch.tensor(self.X_bc[i][:, 2:3], requires_grad=True).float().to(device)       

        self.U_bc_ =  torch.tensor(self.U_bc[i], requires_grad=True).float().to(device)           

        self.X_f_x = torch.tensor(self.X_f[i][:, 0:1], requires_grad=True).float().to(device)
        self.X_f_y = torch.tensor(self.X_f[i][:, 1:2], requires_grad=True).float().to(device)
        self.X_f_t = torch.tensor(self.X_f[i][:, 2:3], requires_grad=True).float().to(device)
        
        Sol1_u, Sol1_v = self.net_u(self.X_bc_x, self.X_bc_y, self.X_bc_t, self.mlp1)         
        u_f, v_f = self.net_f(self.X_f_x, self.X_f_y, self.X_f_t, self.mlp1)
    
        loss_bc1 = torch.mean((Sol1_u - self.U_bc_[:,0:1])**2 + (Sol1_v - self.U_bc_[:,1:2])**2)
        loss_u = torch.mean(u_f**2)
        loss_v = torch.mean(v_f**2)

        loss += 100*(loss_bc1) + loss_u + loss_v
        bc1 += loss_bc1
        u += loss_u
        v += loss_v
        
        if loss<self.loss:
            self.loss = loss
            torch.save(self.mlp1.state_dict(), self.path1)
                        
        loss.backward()
        self.losses.append(loss.detach().cpu().float())
        
        if self.iter%100==0:
            print("Iter:{},Loss:{}".format(self.iter, loss))
            print("Iter:{},Loss_bc1:{}".format(self.iter, bc1))
            print("Iter:{},Loss_u:{}".format(self.iter, u))
            print("Iter:{},Loss_v:{}".format(self.iter, v))
            print("-------------------------------------------")
        self.iter+=1
        
        return loss        
            
    def train(self):
        self.mlp1.train()
                
        for _ in range(30001):
            self.optimizer.step(self.loss_func)
            self.scheduler.step()
            
    def predict(self,sets):
        
        self.mlp1.load_state_dict(torch.load(self.path1)) 
        
        x = torch.tensor(sets[i][:,0:1],requires_grad=True).float().to(device)
        y = torch.tensor(sets[i][:,1:2],requires_grad=True).float().to(device)
        t = torch.tensor(sets[i][:,2:3],requires_grad=True).float().to(device)
        
        u1, v1 = self.net_u(x,y,t,self.mlp1)
      
        return u1,v1

        
