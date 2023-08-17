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
        self.dnn = Base_Net().to(device)
        #self.dnn.load_state_dict(torch.load("model.pt"))
        #self.dnn.load_state_dict(torch.load("/kaggle/input/base-kovasznay-models/model_12.pt"))
        
        #self.dnn = LoRA_Net(self.dnn1,32).to(device)
        #self.dnn.load_state_dict(torch.load("/kaggle/input/base-kovasznay-models/model_57.pt"))
        self.dnn.load_state_dict(torch.load("model1.pt"))
        
        # optimizers: using the same settings
        
        self.losses = []
        self.optimizer1 = torch.optim.Adam(self.dnn.parameters(), lr=1e-3)
        self.optimizer2 = torch.optim.LBFGS(self.dnn.parameters(), lr=1,max_iter=20,tolerance_grad=1e-10)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer1, milestones=[], gamma=0.1)
        
        self.nu = nu
        
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
    
        loss.backward()
        
        if self.iter%100==0:
            print("Iter:{},Loss:{}".format(self.iter, loss))
            print("Iter:{},Loss_bc:{}".format(self.iter, loss_ub))
            print("Iter:{},Loss_f:{}".format(self.iter, loss_uf))
            print("-------------------------------------------")
                
        return loss        
            
    def train(self):
        self.dnn.train()
        for _ in range(201):
            self.optimizer1.step(self.loss_func)
            self.scheduler.step()
            self.iter+=1
        #for _ in range(5001):
        #    self.optimizer2.step(self.loss_func)
        #    self.iter+=1
            
    def predict(self, x, y):   
        #self.dnn.load_state_dict(torch.load(self.path))
        return self.net_u(x,y)

data = scipy.io.loadmat("/kaggle/input/burgers-pino/burgers_pino.mat")
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
#model = VPNSFNet(X[:,0:1], X[:,1:2], ub_train, x_train, t_train, layers)
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

class Hyper_Model(nn.Module):
    def __init__(self,outputs=643):
        super(Hyper_Model, self).__init__()
        
        self.layers = nn.Sequential(nn.Linear(128,512),
                      nn.ReLU(),             
                      nn.Linear(512,512),
                      nn.ReLU(),
                      nn.Linear(512,512),
                      nn.ReLU(),
                      nn.Linear(512,512),
                      nn.ReLU(),
                      nn.Linear(512,outputs))    
        
    def forward(self,param):        
        weights = self.layers(param)
        return weights    

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
        self.base_net.load_state_dict(torch.load("/kaggle/working/model.pt"))
        self.base_weights = torch.cat([param.view(-1) for param in self.base_net.parameters()])
        self.base_weights = self.base_weights.reshape(1,-1)
        
        self.m = m
        
        if m==0:
            pass
        else:
            self.hypernetwork.load_state_dict(torch.load("model_hypernetwork{}.pt".format(m-1)))
            
        #self.hypernetwork.load_state_dict(torch.load("model_hypernetwork{}.pt".format(m)))
        
        self.mainnetwork = Main_Model_DNN().to(device)
        
        self.optimizer = torch.optim.Adam(self.hypernetwork.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,milestones=[],gamma=0.1)
            
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
            #loss = torch.mean((weights-self.train_weights[i].float().to(device))**2)/len(self.train_Res)
            
            #loss = myloss(weights,self.train_weights[i].float().to(device))
            
            loss_u_bc1 = torch.mean((u - self.U_bc1)**2)
            
            #loss_fu = 0.0 
            
            #for k in range(13):
                
            #    self.fx = self.X_f_x[k*1000:(k+1)*1000,:].to(device)
            #    self.ft = self.X_f_t[k*1000:(k+1)*1000,:].to(device)
            #    _,fu = self.net_f(self.param,self.fx, self.ft)
            #    loss_fu += torch.mean(fu**2)
                
            #loss_train += (loss_u_bc1 + loss_fu)/len(self.train_U_bc)
            loss_train_bc += (loss_u_bc1)/len(self.train_U_bc)
            #loss_train_f += (loss_fu)/len(self.train_U_bc)
            
            # a = 20-i%20
            # train_loss += (loss_u_bc1 + loss_fu)/len(self.train_U_bc)
            train_loss += (loss_u_bc1)/len(self.train_U_bc)
            
        train_loss.backward()
        self.optimizer.step()
        
        self.scheduler.step()
        
        if self.iter%100==0:
            print("Iter:{},Train_Loss_total:{}".format(self.iter, train_loss))
            #print("Iter:{},Weight_Loss:{}".format(self.iter, loss))
            #print("Iter:{},Train_Loss:{}".format(self.iter, loss_train))
            print("Iter:{},Train_Loss_bc:{}".format(self.iter, loss_train_bc))
            #print("Iter:{},Train_Loss_f:{}".format(self.iter, loss_train_f))
            print("-------------------------------------------")

        if train_loss<self.loss:
            self.loss = train_loss
            torch.save(self.hypernetwork.state_dict(), self.path)
        
        self.iter+=1
            
    def train(self):
        for _ in range(10001):
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

for i in range(2,142):
    
    """xb_train = x.reshape(-1,1)
    yb_train = np.array([0.0]*128).reshape(-1,1)
    x_bc = np.concatenate([xb_train, yb_train], axis=1) 
    ub_train = np.array(x_data[i]).reshape(-1,1)
    
    u_true = y_data[i].reshape(128,101)
    
    X_BC.append(x_bc)
    U_BC.append(ub_train)
    U_true.append(u_true.reshape(-1,))"""
    
    X_BC.append(X)
    U_BC.append(y_data[i].reshape(-1,1,order='F'))
    U_true.append(y_data[i].reshape(-1,1,order='F'))
    
no1 = 100
no2 = 120
train_X_BC, valid_X_BC, test_X_BC = X_BC[:no1], X_BC[no1:no2], X_BC[no2:] 
train_U_BC, valid_U_BC, test_U_BC = U_BC[:no1], U_BC[no1:no2], U_BC[no2:]

A = np.meshgrid(x,t)
X = np.concatenate([A[0].reshape(-1,1),A[1].reshape(-1,1)],axis=1)

model_hypernetwork = HyperNetwork(train_X_BC, train_U_BC, X, valid_X_BC, valid_U_BC, test_X_BC, test_U_BC)
model_hypernetwork.train()

A = np.meshgrid(x,t)
X = np.concatenate([A[0].reshape(-1,1),A[1].reshape(-1,1)],axis=1)

model_hypernetwork = HyperNetwork(train_X_BC, train_U_BC, X, valid_X_BC, valid_U_BC, test_X_BC, test_U_BC)
u_predict, fu_predict, _ = model_hypernetwork.predict(X)

upredict = []

## Train task evaluation
train_mses = []
for i in range(100):
    upredict.append(u_predict[i])
    train_mses.append(np.mean((u_predict[i].reshape(-1,1)-U_true[i])**2))

## Valid task evaluation
valid_mses = []
for i in range(20):
    upredict.append(u_predict[i+no1])
    valid_mses.append(np.mean((u_predict[i+no1].reshape(-1,1)-U_true[i+no1])**2))

## Test task evaluation
test_mses = []
for i in range(20):
    upredict.append(u_predict[i+no2])
    test_mses.append(np.mean((u_predict[i+no2].reshape(-1,1)-U_true[i+no2])**2))

print(np.mean(train_mses[:100]),np.mean(valid_mses),np.mean(test_mses))
