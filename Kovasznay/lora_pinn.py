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
        self.fc1 = nn.Linear(2, 64)  # Input layer
        self.fc2 = nn.Linear(64, 64)  
        self.fc3 = nn.Linear(64, 64) 
        self.fc4 = nn.Linear(64, 64)  
        self.fc5 = nn.Linear(64, 64) 
        self.fc6 = nn.Linear(64, 3)
        
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = torch.tanh(self.fc5(x))
        x = self.fc6(x)
        
        u, v, w = x[:,0:1], x[:,1:2], x[:,2:3]
        return u,v,w

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
        self.fc6_B = nn.Linear(m, 3, bias=False)       

    def forward(self, x):
        
        x = torch.tanh(self.base_model.fc1(x) + self.fc1_B(self.fc1_A(x)))
        x = torch.tanh(self.base_model.fc2(x) + self.fc2_B(self.fc2_A(x)))
        x = torch.tanh(self.base_model.fc3(x) + self.fc3_B(self.fc3_A(x)))
        x = torch.tanh(self.base_model.fc4(x) + self.fc4_B(self.fc4_A(x)))
        x = torch.tanh(self.base_model.fc5(x) + self.fc5_B(self.fc5_A(x)))
        x = self.base_model.fc6(x) + self.fc6_B(self.fc6_A(x)) # This layer is not changed

        u, v, w = x[:,0:1], x[:,1:2], x[:,2:3]
        return u, v, w

class VPNSFNet():
    def __init__(self, xb_train, yb_train, ub_train, vb_train, pb_train, x_train, y_train, layers, Re):  
        
        self.xb = torch.tensor(xb_train[:, 0:1], requires_grad=True).float().to(device)
        self.yb = torch.tensor(yb_train[:, 0:1], requires_grad=True).float().to(device)       
        
        self.ub = torch.tensor(ub_train[:, 0:1], requires_grad=True).float().to(device)
        self.vb = torch.tensor(vb_train[:, 0:1], requires_grad=True).float().to(device)
        self.pb = torch.tensor(pb_train[:, 0:1], requires_grad=True).float().to(device)

        self.x = torch.tensor(x_train[:, 0:1], requires_grad=True).float().to(device)
        self.y = torch.tensor(y_train[:, 0:1], requires_grad=True).float().to(device)
        
        self.layers = layers
        
        self.Re = Re
        # deep neural networks
        self.dnn1 = Base_Net().to(device)
        self.dnn1.load_state_dict(torch.load("model_12.pt"))
        
        self.dnn = LoRA_Net(self.dnn1,1).to(device)
        # self.dnn.load_state_dict(torch.load("/kaggle/input/base-kovasznay-models/model_57.pt"))
        
        # optimizers: using the same settings
        
        self.losses = []
        self.optimizer1 = torch.optim.Adam(self.dnn.parameters(), lr=1e-2)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer1, milestones=[5000,15000,25000], gamma=0.1)

        self.loss = 1e10
        self.path = "model_{}.pt".format(self.Re)

        self.stop = 0
        # self.dnn.load_state_dict(torch.load(self.path))
        
        self.iter = 0
    
    def net_u(self, x, y):  
        u, v, p = self.dnn(torch.cat([x, y], dim=1))
        return u,v,p
        
    def net_f(self, x, y):
        u, v, p = self.net_u(x,y)

        u_x = torch.autograd.grad(u,x,grad_outputs=torch.ones_like(u),retain_graph=True,create_graph=True)[0]
        u_y = torch.autograd.grad(u,y,grad_outputs=torch.ones_like(u),retain_graph=True,create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x,x,grad_outputs=torch.ones_like(u_x),retain_graph=True,create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y,y,grad_outputs=torch.ones_like(u_y),retain_graph=True,create_graph=True)[0]
        
        v_x = torch.autograd.grad(v,x,grad_outputs=torch.ones_like(v),retain_graph=True,create_graph=True)[0]
        v_y = torch.autograd.grad(v,y,grad_outputs=torch.ones_like(v),retain_graph=True,create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x,x,grad_outputs=torch.ones_like(v_x),retain_graph=True,create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y,y,grad_outputs=torch.ones_like(v_y),retain_graph=True,create_graph=True)[0]
        
        p_x = torch.autograd.grad(p,x,grad_outputs=torch.ones_like(p),retain_graph=True,create_graph=True)[0]
        p_y = torch.autograd.grad(p,y,grad_outputs=torch.ones_like(p),retain_graph=True,create_graph=True)[0]
        
        f_u = (u*u_x + v*u_y) + p_x - (1.0/self.Re) * (u_xx + u_yy)
        f_v = (u*v_x + v*v_y) + p_y - (1.0/self.Re) * (v_xx + v_yy)
        f_e = u_x + v_y
        
        return u, v, p, f_u, f_v, f_e

    def loss_func(self):
        
        self.optimizer1.zero_grad()
        
        ub, vb, pb = self.net_u(self.xb, self.yb)
        
        loss_ub = torch.mean((ub - self.ub)**2) 
        loss_vb = torch.mean((vb - self.vb)**2)
        loss_pb = torch.mean((pb - self.pb)**2)

        upred, vpred, ppred, uf, vf, ef = self.net_f(self.x, self.y)
        
        loss_uf = torch.mean(uf**2)
        loss_vf = torch.mean(vf**2)
        loss_ef = torch.mean(ef**2)
        
        loss = loss_ub + loss_vb + loss_pb + loss_uf + loss_vf + loss_ef 
        
        self.losses.append(loss.detach().cpu().float())
        if loss<self.loss:
            self.loss = loss
            torch.save(self.dnn.state_dict(), self.path)
            self.stop = 0
        else:
          self.stop += 1
        loss.backward()
        
        if self.iter%100==0:
            print("Iter:{},Loss:{}".format(self.iter, loss))
            print("Iter:{},Loss_bc:{}".format(self.iter, loss_ub + loss_vb + loss_pb))
            print("Iter:{},Loss_f:{}".format(self.iter, loss_uf + loss_vf + loss_ef))
            print("-------------------------------------------")
        
        self.iter+=1
        
        return loss        
            
    def train(self):
        self.dnn.train()
        for _ in range(30001):
            self.optimizer1.step(self.loss_func)
            self.scheduler.step()

            if self.stop==1000:
              break
        self.optimizer2.step()
            
    def predict(self, x, y):   
        self.dnn.load_state_dict(torch.load(self.path))
        return self.net_u(x,y)

N_train = 2601
layers = [2, 20, 20, 20, 20, 20, 20, 3]

# Load Data
lam = 0.5 * Re - np.sqrt(0.25 * (Re ** 2) + 4 * (np.pi ** 2))

x = np.linspace(-0.5, 1.0, 101)
y = np.linspace(-0.5, 1.5, 101)

yb1 = np.array([-0.5] * 100)
yb2 = np.array([1] * 100)
xb1 = np.array([-0.5] * 100)
xb2 = np.array([1.5] * 100)

y_train1 = np.concatenate([y[1:101], y[0:100], xb1, xb2], 0)
x_train1 = np.concatenate([yb1, yb2, x[0:100], x[1:101]], 0)

xb_train = x_train1.reshape(x_train1.shape[0], 1)
yb_train = y_train1.reshape(y_train1.shape[0], 1)
ub_train = 1 - np.exp(lam * xb_train) * np.cos(2 * np.pi * yb_train)
vb_train = lam / (2 * np.pi) * np.exp(lam * xb_train) * np.sin(2 * np.pi * yb_train)
pb_train = (1 - np.exp(2*lam*xb_train))/2

x_train = np.random.rand(N_train, 1)*1.5-0.5
y_train = np.random.rand(N_train, 1)*2-0.5

model = VPNSFNet(xb_train, yb_train, ub_train, vb_train, pb_train, x_train, y_train, layers, Re)
model.train()

A = np.meshgrid(x,y)
X = np.concatenate([A[0].reshape(-1,1),A[1].reshape(-1,1)],axis=1)

X_test = torch.tensor(X[:,0:1], requires_grad=True).float().to(device)
Y_test = torch.tensor(X[:,1:2], requires_grad=True).float().to(device)
u, v, p = model.predict(X_test, Y_test)

u = u.detach().cpu().numpy().reshape(-1)
v = v.detach().cpu().numpy().reshape(-1)
p = p.detach().cpu().numpy().reshape(-1)

u_true = 1 - np.exp(lam*X[:,0])*np.cos(2*np.pi*X[:,1])
v_true = lam*np.exp(lam*X[:,0])*np.sin(2*np.pi*X[:,1])/(2*np.pi)
p_true = (1-np.exp(2*lam*X[:,0]))/2

print(np.linalg.norm(u_true-u,2)/np.linalg.norm(u_true,2))
print(np.linalg.norm(v_true-v,2)/np.linalg.norm(v_true,2))
print(np.linalg.norm(p_true-p,2)/np.linalg.norm(p_true,2))

print("u:",np.mean((u_true-u)**2))
print("v:",np.mean((v_true-v)**2))
print("p:",np.mean((p_true-p)**2))
