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

class Hyper_Model(nn.Module):
    def __init__(self,outputs=643):
        super(Hyper_Model, self).__init__()
        
        self.layers = nn.Sequential(nn.Linear(128,512),
                      nn.ReLU(),             
                      nn.Linear(512,512),
                      nn.ReLU(),
                      nn.Linear(512,256),
                      nn.ReLU(),
                      nn.Linear(256,256),
                      nn.ReLU(),
                      nn.Linear(256,128),
                      nn.ReLU(),
                      nn.Linear(128,128),
                      nn.ReLU(),              
                      nn.Linear(128,outputs))    
        
    def forward(self,param):        
        weights = self.layers(param)
        return weights    

class Main_Model_DNN(nn.Module):
    def __init__(self):
        super(Main_Model_DNN, self).__init__()
    
    def forward(self,original_weights,lora_weights,x):
        # [2-64-64-64-64-64-1]
        
        i = 2*64+64
        idx = 2+64
        matrix = lora_weights[0,2:66].view(64,1)*lora_weights[0,:2].view(1,2)
             
        h = F.tanh(F.linear(x, weight = original_weights[0, :128].view(64, 2)+matrix, bias = original_weights[0, 128:192].view(64)))
        
        for _ in range(4):
            matrix = lora_weights[0,idx+64:idx+128].view(64,1)*lora_weights[0,idx:idx+64].view(1,64)
            h = F.tanh(F.linear(h, weight = original_weights[0, i:i+64*64].view(64,64)+matrix, bias = original_weights[0,i+64*64:i+64*64+64].view(64)))
            i += 64*65
            idx += 2*64
            
        matrix = lora_weights[0,idx+64:idx+65].view(1,1)*lora_weights[0,idx:idx+64].view(1,64)    
        
        h = F.linear(h, weight = original_weights[0, i:i+64*1].view(1, 64)+matrix, bias = original_weights[0, i+64*1:i+64*1+1].view(1))
        
        return h

class HyperNetwork(nn.Module):
    def __init__(self,train_X_BC, train_U_BC, Xf_train, val_X_BC, val_U_BC, test_X_BC, test_U_BC, train_weights=None, valid_weights=None, m=0):  
        
        super(HyperNetwork, self).__init__()
        self.train_X_bc = train_X_BC     
        self.train_U_bc = train_U_BC
        
        self.val_X_bc = val_X_BC     
        self.val_U_bc = val_U_BC
        
        self.test_X_bc = val_X_BC     
        self.test_U_bc = val_U_BC
        
        self.train_weights = train_weights
        self.valid_weights = valid_weights
        
        self.X_f_x = torch.tensor(Xf_train[:, 0:1], requires_grad=True).float()
        self.X_f_t = torch.tensor(Xf_train[:, 1:2], requires_grad=True).float()
        
        self.hypernetwork = Hyper_Model().to(device)  ## Input the number of parameters of PINN
        
        self.base_net = Base_Net().to(device)
        self.base_net.load_state_dict(torch.load("model.pt"))   # Insert base model path
        self.base_weights = torch.cat([param.view(-1) for param in self.base_net.parameters()])
        self.base_weights = self.base_weights.reshape(1,-1)
        
        self.m = m
        
        self.mainnetwork = Main_Model_DNN().to(device)
        
        self.optimizer = torch.optim.Adam(self.hypernetwork.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,milestones=[5000,10000],gamma=0.1)
            
        self.loss = 1e10
        self.losses = []
        self.path = "model_hypernetwork{}.pt".format(m)
                
        self.iter = 0
    
    def net_u(self,param,x,y):  
        weights = self.hypernetwork(param)
        X = torch.cat([x,y],dim=1)
        u = self.mainnetwork(self.base_weights,weights[:,0:643],X)
        return u,weights
        
    def net_f(self,param,x,y):
        u, _ = self.net_u(param,x,y)
        
        u_x = torch.autograd.grad(u,x,grad_outputs=torch.ones_like(u),retain_graph=True,create_graph=True)[0]
        u_y = torch.autograd.grad(u,y,grad_outputs=torch.ones_like(u),retain_graph=True,create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x,x,grad_outputs=torch.ones_like(u_x),retain_graph=True,create_graph=True)[0]
        u_x_ = torch.autograd.grad(u**2,x,grad_outputs=torch.ones_like(u_x),retain_graph=True,create_graph=True)[0]
        
        f_u = u_y + u_x_/2 - 1e-2*(u_xx)
        
        return u, f_u

    def loss_func(self):
        #self.optimizer.zero_grad()
        
        train_loss = 0.0
        
        loss_train = 0.0
        loss_train_bc = 0.0
        loss_train_f = 0.0
        loss_val = 0.0
        loss_val_bc = 0.0
        loss_val_f = 0.0
        
        self.hypernetwork.train()
        for i,param in enumerate(self.train_U_bc):
            
            self.hypernetwork.zero_grad()
        
            self.X_bc1_x = torch.tensor(self.train_X_bc[i][:, 0:1], requires_grad=True).float().to(device)
            self.X_bc1_t = torch.tensor(self.train_X_bc[i][:, 1:2], requires_grad=True).float().to(device)
                 
            self.U_bc1 =  torch.tensor(self.train_U_bc[i][:, 0:1], requires_grad=True).float().to(device)
            
            self.param = torch.tensor(param[:128],requires_grad=True).view(1,128).float().to(device)
            
            u, weights = self.net_u(self.param,self.X_bc1_x, self.X_bc1_t)
            loss = torch.mean((weights-self.train_weights[i].float().to(device))**2)/len(self.train_Res)
            
            #loss = myloss(weights,self.train_weights[i].float().to(device))
            
            train_loss += loss
            
        train_loss.backward()
        self.optimizer.step()
        
        self.scheduler.step()

      self.hypernetwork.eval()
      for i,param in enumerate(self.val_U_bc):
            
            self.hypernetwork.zero_grad()
        
            self.X_bc1_x = torch.tensor(self.val_X_bc[i][:, 0:1], requires_grad=True).float().to(device)
            self.X_bc1_t = torch.tensor(self.val_X_bc[i][:, 1:2], requires_grad=True).float().to(device)
                 
            self.U_bc1 =  torch.tensor(self.val_U_bc[i][:, 0:1], requires_grad=True).float().to(device)
            
            self.param = torch.tensor(param[:128],requires_grad=True).view(1,128).float().to(device)
            
            u, weights = self.net_u(self.param,self.X_bc1_x, self.X_bc1_t)
            
            #loss = myloss(weights,self.val_weights[i].float().to(device))
            loss = torch.mean((weights-self.val_weights[i].float().to(device))**2)/len(self.train_Res)
                        
            val_loss += (loss)/len(self.val_U_bc)
            
        train_loss.backward()
        self.optimizer.step()
        
        self.scheduler.step()
        
        if self.iter%100==0:
            print("Iter:{},Train_Loss_total:{}".format(self.iter, train_loss))
            print("Iter:{},Weight_Loss:{}".format(self.iter, loss))
            print("-------------------------------------------")

            print("Iter:{},Val_Loss_total:{}".format(self.iter, train_loss))
            print("Iter:{},Weight_Loss:{}".format(self.iter, loss))
            print("-------------------------------------------")

        if val_loss<self.loss:
            self.loss = train_val
            torch.save(self.hypernetwork.state_dict(), self.path)
        
        self.iter+=1
            
    def train(self):
        for _ in range(15001):
            self.loss_func()
            
    def predict(self,X):
        
        x_grid = torch.tensor(X[:,0:1], requires_grad=True).float().to(device)
        t_grid = torch.tensor(X[:,1:2], requires_grad=True).float().to(device)
        
        self.hypernetwork.load_state_dict(torch.load(self.path))
        
        u_predict = []
        fu_predict = []
        weights = []
        
        ## Train_tasks
        for i,p in enumerate(self.train_U_bc):            
            
            param = torch.tensor(p[:128]).view(1,128).float().to(device)
            u_domain, fu = self.net_f(param,x_grid,t_grid)    
            u_domain = u_domain.detach().cpu().numpy().reshape(-1)
            fu = fu.detach().cpu().numpy().reshape(-1)
            
            u_predict.append(u_domain)
            fu_predict.append(fu)
            
            weight = self.hypernetwork(param)
            weights.append(weight.detach())
            
        ## Valid_tasks
        for i,p in enumerate(self.val_U_bc):            
            
            param = torch.tensor(p[:128]).view(1,128).float().to(device)
            u_domain, fu = self.net_f(param,x_grid,t_grid)    
            u_domain = u_domain.detach().cpu().numpy().reshape(-1)
            fu = fu.detach().cpu().numpy().reshape(-1)
            
            u_predict.append(u_domain)
            fu_predict.append(fu)
            
            weight = self.hypernetwork(param)
            weights.append(weight.detach())
    
        ## Test_tasks
        for i,p in enumerate(self.test_U_bc):            
            
            param = torch.tensor(p[:128]).view(1,128).float().to(device)
            u_domain, fu = self.net_f(param,x_grid,t_grid)    
            u_domain = u_domain.detach().cpu().numpy().reshape(-1)
            fu = fu.detach().cpu().numpy().reshape(-1)
            
            u_predict.append(u_domain)
            fu_predict.append(fu)
            
            weight = self.hypernetwork(param)
            weights.append(weight.detach())
            
        return u_predict, fu_predict, weights

N_train = 1000
layers = [2, 64, 64, 64, 64, 64, 1]

x = np.linspace(0, 1.0, 128)
t = np.linspace(0, 1.0, 101)

xb_train = x.reshape(-1,1)
tb_train = np.array([0.0]*128).reshape(-1,1)
ub_train = np.array(x_data[0]).reshape(-1,1)

x_train = np.random.rand(N_train, 1)
t_train = np.random.rand(N_train, 1)

x = np.linspace(0, 1.0, 128)
t = np.linspace(0, 1.0, 101)

A = np.meshgrid(x,t)
X = np.concatenate([A[0].reshape(-1,1),A[1].reshape(-1,1)],axis=1)

# Generate Data
X_BC = []
U_BC = []
U_true = []

x = np.linspace(0, 1.0, 128)
t = np.linspace(0, 1.0, 101)

A = np.meshgrid(x,t)
X = np.concatenate([A[0].reshape(-1,1),A[1].reshape(-1,1)],axis=1)

for i in range(2,202):

    X_BC.append(X)
    U_BC.append(y_data[i].reshape(-1,1,order='F'))
    U_true.append(y_data[i].reshape(-1,1,order='F'))
    
no1 = 120
no2 = 40
train_X_BC, valid_X_BC, test_X_BC = X_BC[:no1], X_BC[no1:no2], X_BC[no2:] 
train_U_BC, valid_U_BC, test_U_BC = U_BC[:no1], U_BC[no1:no2], U_BC[no2:]

A = np.meshgrid(x,t)
X = np.concatenate([A[0].reshape(-1,1),A[1].reshape(-1,1)],axis=1)

model_hypernetwork = HyperNetwork(train_X_BC, train_U_BC, X, valid_X_BC, valid_U_BC, test_X_BC, test_U_BC)
model_hypernetwork.train()

A = np.meshgrid(x,t)
X = np.concatenate([A[0].reshape(-1,1),A[1].reshape(-1,1)],axis=1)

u_predict, fu_predict, _ = model_hypernetwork.predict(X)

upredict = []

## Train task evaluation
train_mses = []
for i in range(120):
    upredict.append(u_predict[i])
    train_mses.append(np.mean((u_predict[i].reshape(-1,1)-U_true[i])**2))

## Valid task evaluation
valid_mses = []
for i in range(40):
    upredict.append(u_predict[i+no1])
    valid_mses.append(np.mean((u_predict[i+no1].reshape(-1,1)-U_true[i+no1])**2))

## Test task evaluation
test_mses = []
for i in range(40):
    upredict.append(u_predict[i+no2])
    test_mses.append(np.mean((u_predict[i+no2].reshape(-1,1)-U_true[i+no2])**2))
