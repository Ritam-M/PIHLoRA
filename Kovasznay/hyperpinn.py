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

class Hyper_Model(nn.Module):
    def __init__(self,outputs=17027):
        super(Hyper_Model, self).__init__()
        
        self.layers = nn.Sequential(nn.Linear(1,512),
                      nn.ReLU(),             
                      nn.Linear(512,256),
                      nn.ReLU(),
                      nn.Linear(256,256),
                      nn.ReLU(),
                      nn.Linear(256,128),
                      nn.ReLU(),
                      nn.Linear(128,outputs))    
        
    def forward(self,param):        
        weights = self.layers(param)
        return weights    

class Main_Model_DNN(nn.Module):
    def __init__(self):
        super(Main_Model_DNN, self).__init__()
    
    def forward(self,original_weights,lora_weights,x):
        # [2-64-64-64-64-64-3]
        
        i = 2*64+64
        idx = 2+64
        matrix = lora_weights[0,2:66].view(64,1)*lora_weights[0,:2].view(1,2)
             
        h = F.tanh(F.linear(x, weight = original_weights[0, :128].view(64, 2)+matrix, bias = original_weights[0, 128:192].view(64)))
        
        for _ in range(4):
            matrix = lora_weights[0,idx+64:idx+128].view(64,1)*lora_weights[0,idx:idx+64].view(1,64)
            h = F.tanh(F.linear(h, weight = original_weights[0, i:i+64*64].view(64,64)+matrix, bias = original_weights[0,i+64*64:i+64*64+64].view(64)))
            i += 64*65
            idx += 2*64
            
        matrix = lora_weights[0,idx+64:idx+67].view(3,1)*lora_weights[0,idx:idx+64].view(1,64)    
        
        h = F.linear(h, weight = original_weights[0, i:i+64*3].view(3, 64)+matrix, bias = original_weights[0, i+64*3:i+64*3+3].view(3))
        
        u, v, p = h[:,0:1], h[:,1:2], h[:,2:3]
        return u, v, p
    
class Main_Model_DNN(nn.Module):
    def __init__(self):
        super(Main_Model_DNN, self).__init__()
    
    def forward(self,original_weights,lora_weights,x):
        # [2-64-64-64-64-64-3]
        
        i = 2*64+64
        idx = 2+64
        #matrix = lora_weights[0,2:66].view(64,1)*lora_weights[0,:64].view(1,64)
             
        h = F.tanh(F.linear(x, weight = lora_weights[0, :128].view(64, 2), bias = lora_weights[0, 128:192].view(64)))
        
        for _ in range(4):
            #matrix = lora_weights[0,idx+64:idx+128].view(64,1)*lora_weights[0,idx:idx+64].view(1,64)
            h = F.tanh(F.linear(h, weight = lora_weights[0, i:i+64*64].view(64,64), bias = lora_weights[0,i+64*64:i+64*64+64].view(64)))
            i += 64*65
            idx += 2*64
            
        #matrix = lora_weights[0,idx+64:idx+67].view(3,1)*lora_weights[0,idx:idx+64].view(1,64)    
        
        h = F.linear(h, weight = lora_weights[0, i:i+64*3].view(3, 64), bias = lora_weights[0, i+64*3:i+64*3+3].view(3))
        
        u, v, p = h[:,0:1], h[:,1:2], h[:,2:3]
        return u, v, p

class HyperNetwork(nn.Module):
    def __init__(self,train_X_BC, train_U_BC, train_V_BC, train_P_BC, Xf_train, train_Res, val_X_BC, val_U_BC, val_V_BC, val_P_BC, val_Res, test_Res, train_weights, valid_weights, m):  
        
        super(HyperNetwork, self).__init__()
        self.train_X_bc = train_X_BC     
        self.train_U_bc = train_U_BC
        self.train_V_bc = train_V_BC
        self.train_P_bc = train_P_BC
        self.train_Res = train_Res
        
        self.val_X_bc = val_X_BC     
        self.val_U_bc = val_U_BC
        self.val_V_bc = val_V_BC
        self.val_P_bc = val_P_BC
        self.valid_Res = val_Res
        
        self.test_Res = test_Res
        
        self.train_weights = train_weights
        self.valid_weights = valid_weights
        
        self.X_f_x = torch.tensor(Xf_train[:, 0:1], requires_grad=True).float()
        self.X_f_y = torch.tensor(Xf_train[:, 1:2], requires_grad=True).float()
        
        self.hypernetwork = Hyper_Model().to(device)  ## Input the number of parameters of PINN
        
        self.base_net = Base_Net().to(device)
        self.base_net.load_state_dict(torch.load("model_12.pt"))   ## Insert your base net path
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
        weights = self.hypernetwork(torch.cat([param],dim=1))
        X = torch.cat([x,y],dim=1)
        u, v, p = self.mainnetwork(self.base_weights,weights[:,0:17027],X)
        return u,v,p,weights
        
    def net_f(self,param,x,y):
        val = 100*param[0][0].detach()
        u, v, p,_ = self.net_u(param,x,y)
        
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
        
        f_u = (u*u_x + v*u_y) + p_x - (1.0/val) * (u_xx + u_yy)
        f_v = (u*v_x + v*v_y) + p_y - (1.0/val) * (v_xx + v_yy)
        f_e = u_x + v_y
        
        return u, v, p, f_u, f_v, f_e


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
        for i,re in enumerate(self.train_Res):
            
            self.hypernetwork.zero_grad()
        
            m = re/100
            
            self.X_bc1_x = torch.tensor(self.train_X_bc[i][:, 0:1], requires_grad=True).float().to(device)
            self.X_bc1_y = torch.tensor(self.train_X_bc[i][:, 1:2], requires_grad=True).float().to(device)
                 
            self.U_bc1 =  torch.tensor(self.train_U_bc[i][:, 0:1], requires_grad=True).float().to(device)
            self.V_bc1 =  torch.tensor(self.train_V_bc[i][:, 0:1], requires_grad=True).float().to(device)
            self.P_bc1 =  torch.tensor(self.train_P_bc[i][:, 0:1], requires_grad=True).float().to(device)
            
            self.param = torch.tensor([m],requires_grad=True).view(1,1).float().to(device)
            
            u, v, p, weights = self.net_u(self.param,self.X_bc1_x, self.X_bc1_y)
            #loss = torch.mean((weights-self.train_weights[i].float().to(device))**2)/len(self.train_Res)
            
            #loss = myloss(weights,self.train_weights[i].float().to(device))
            
            loss_u_bc1 = torch.mean((u - self.U_bc1)**2)
            loss_v_bc1 = torch.mean((v - self.V_bc1)**2)
            loss_p_bc1 = torch.mean((p - self.P_bc1)**2)
            
            loss_fu = 0.0 
            loss_fv = 0.0 
            loss_fe = 0.0 
            
            for k in range(11):
                
                self.fx = self.X_f_x[k*100:(k+1)*100,:].to(device)
                self.fy = self.X_f_y[k*100:(k+1)*100,:].to(device)
                _,_,_,fu,fv,fe = self.net_f(self.param,self.fx, self.fy)
                loss_fu += torch.mean(fu**2)
                loss_fv += torch.mean(fv**2)
                loss_fe += torch.mean(fe**2)

            loss_train += (loss_u_bc1 + loss_v_bc1 + loss_p_bc1 + loss_fu + loss_fv + loss_fe)/len(self.train_Res)
            loss_train_bc += (loss_u_bc1 + loss_v_bc1 + loss_p_bc1)/len(self.train_Res)
            loss_train_f += (loss_fu + loss_fv + loss_fe)/len(self.train_Res)
            
            a = 20-i%20
            train_loss += a*(loss_u_bc1 + loss_v_bc1 + loss_p_bc1 + loss_fu + loss_fv + loss_fe)/len(self.train_Res)

        self.hypernetwork.eval()
        for i,re in enumerate(self.val_Res):
            
            m = re/100
            
            self.X_bc1_x = torch.tensor(self.train_X_bc[i][:, 0:1], requires_grad=True).float().to(device)
            self.X_bc1_y = torch.tensor(self.train_X_bc[i][:, 1:2], requires_grad=True).float().to(device)
                 
            self.U_bc1 =  torch.tensor(self.train_U_bc[i][:, 0:1], requires_grad=True).float().to(device)
            self.V_bc1 =  torch.tensor(self.train_V_bc[i][:, 0:1], requires_grad=True).float().to(device)
            self.P_bc1 =  torch.tensor(self.train_P_bc[i][:, 0:1], requires_grad=True).float().to(device)
            
            self.param = torch.tensor([m],requires_grad=True).view(1,1).float().to(device)
            
            u, v, p, weights = self.net_u(self.param,self.X_bc1_x, self.X_bc1_y)
            #loss = torch.mean((weights-self.val_weights[i].float().to(device))**2)/len(self.val_Res)
            
            #loss = myloss(weights,self.val_weights[i].float().to(device))
            
            loss_u_bc1 = torch.mean((u - self.U_bc1)**2)
            loss_v_bc1 = torch.mean((v - self.V_bc1)**2)
            loss_p_bc1 = torch.mean((p - self.P_bc1)**2)
            
            loss_fu = 0.0 
            loss_fv = 0.0 
            loss_fe = 0.0 
            
            for k in range(11):
                
                self.fx = self.X_f_x[k*100:(k+1)*100,:].to(device)
                self.fy = self.X_f_y[k*100:(k+1)*100,:].to(device)
                _,_,_,fu,fv,fe = self.net_f(self.param,self.fx, self.fy)
                loss_fu += torch.mean(fu**2)
                loss_fv += torch.mean(fv**2)
                loss_fe += torch.mean(fe**2)

            loss_val += (loss_u_bc1 + loss_v_bc1 + loss_p_bc1 + loss_fu + loss_fv + loss_fe)/len(self.val_Res)
            loss_val_bc += (loss_u_bc1 + loss_v_bc1 + loss_p_bc1)/len(self.val_Res)
            loss_val_f += (loss_fu + loss_fv + loss_fe)/len(self.val_Res)
            
            a = 20-i%20
            val_loss += a*(loss_u_bc1 + loss_v_bc1 + loss_p_bc1 + loss_fu + loss_fv + loss_fe)/len(self.val_Res)
        
        train_loss.backward()
        self.optimizer.step()
        
        self.scheduler.step()
        
        if self.iter%100==0:
            print("Iter:{},Train_Loss_total:{}".format(self.iter, train_loss))
            #print("Iter:{},Weight_Loss:{}".format(self.iter, loss))
            print("Iter:{},Train_Loss:{}".format(self.iter, loss_train))
            print("Iter:{},Train_Loss_bc:{}".format(self.iter, loss_train_bc))
            print("Iter:{},Train_Loss_f:{}".format(self.iter, loss_train_f))
            print("Iter:{},Valid_Loss_total:{}".format(self.iter, val_loss))
            #print("Iter:{},Valid_Weight_Loss:{}".format(self.iter, loss))
            print("Iter:{},Valid_Loss:{}".format(self.iter, loss_val))
            print("Iter:{},Valid_Loss_bc:{}".format(self.iter, loss_val_bc))
            print("Iter:{},Valid_Loss_f:{}".format(self.iter, loss_val_f))
            print("-------------------------------------------")

        if val_loss<self.loss:
            self.loss = val_loss
            torch.save(self.hypernetwork.state_dict(), self.path)
        
        self.iter+=1
            
    def train(self):
        for _ in range(15001):
            self.loss_func()
            
    def predict(self,X):
        
        x_grid = torch.tensor(X[:,0:1], requires_grad=True).float().to(device)
        y_grid = torch.tensor(X[:,1:2], requires_grad=True).float().to(device)
        
        #self.hypernetwork.load_state_dict(torch.load(self.path))
        
        u_predict = []
        v_predict = []
        p_predict = []
        fu_predict = []
        fv_predict = []
        fe_predict = []
        weights = []
        
        ## Train_tasks
        for i, re in enumerate(self.train_Res):            
            
            m = re/100
            param = torch.tensor([m]).view(1,1).float().to(device)
            u_domain, v_domain, p_domain, fu, fv, fe = self.net_f(param,x_grid,y_grid)    
            u_domain = u_domain.detach().cpu().numpy().reshape(-1)
            v_domain = v_domain.detach().cpu().numpy().reshape(-1)
            p_domain = p_domain.detach().cpu().numpy().reshape(-1)
            fu = fu.detach().cpu().numpy().reshape(-1)
            fv = fv.detach().cpu().numpy().reshape(-1)
            fe = fe.detach().cpu().numpy().reshape(-1)
            
            u_predict.append(u_domain)
            v_predict.append(v_domain)
            p_predict.append(p_domain)        
            fu_predict.append(fu)
            fv_predict.append(fv)
            fe_predict.append(fe)
            
            weight = self.hypernetwork(torch.cat([param],dim=1))
            weights.append(weight.detach())
            
        ## Valid_tasks
        for i, re in enumerate(self.valid_Res):            
            
            m = re/100
            param = torch.tensor([m]).view(1,1).float().to(device)
            u_domain, v_domain, p_domain, fu, fv, fe = self.net_f(param,x_grid,y_grid)    
            u_domain = u_domain.detach().cpu().numpy().reshape(-1)
            v_domain = v_domain.detach().cpu().numpy().reshape(-1)
            p_domain = p_domain.detach().cpu().numpy().reshape(-1)
            fu = fu.detach().cpu().numpy().reshape(-1)
            fv = fv.detach().cpu().numpy().reshape(-1)
            fe = fe.detach().cpu().numpy().reshape(-1)
            
            u_predict.append(u_domain)
            v_predict.append(v_domain)
            p_predict.append(p_domain)        
            fu_predict.append(fu)
            fv_predict.append(fv)
            fe_predict.append(fe)
            
            weight = self.hypernetwork(torch.cat([param],dim=1))
            weights.append(weight.detach())
    
        ## Test_tasks
        for i, re in enumerate(self.test_Res):            
            
            m = re/100
            param = torch.tensor([m]).view(1,1).float().to(device)
            u_domain, v_domain, p_domain, fu, fv, fe = self.net_f(param,x_grid,y_grid)    
            u_domain = u_domain.detach().cpu().numpy().reshape(-1)
            v_domain = v_domain.detach().cpu().numpy().reshape(-1)
            p_domain = p_domain.detach().cpu().numpy().reshape(-1)
            fu = fu.detach().cpu().numpy().reshape(-1)
            fv = fv.detach().cpu().numpy().reshape(-1)
            fe = fe.detach().cpu().numpy().reshape(-1)
            
            u_predict.append(u_domain)
            v_predict.append(v_domain)
            p_predict.append(p_domain)        
            fu_predict.append(fu)
            fv_predict.append(fv)
            fe_predict.append(fe)
            
            weight = self.hypernetwork(torch.cat([param],dim=1))
            weights.append(weight.detach())
            
        return u_predict, v_predict, p_predict, fu_predict, fv_predict, fe_predict, weights

import glob
base_net = Base_Net()
base_net.load_state_dict(torch.load("model_12.pt"))   ## Load your base model path

train_Res = []
train_weights = []

valid_Res = []
valid_weights = []

for i,file in enumerate(sorted(glob.glob("/kovasznaydataset/*.pt"))):   ## Load the weights of the pretrained files here
    
    dnn = LoRA_Net(base_net,1)
    dnn.load_state_dict(torch.load(file))
    weights = torch.cat([param.view(-1) for param in dnn.parameters()])
    re = float(file.split('/')[-1][6:-3])
    
    if i%2!=0:
        train_Res.append(re)
        train_weights.append(weights[-645:])
    else:
        valid_Res.append(re)
        valid_weights.append(weights[-645:])

test_Res = list(np.sort(1/np.random.uniform(0.02,0.05,(10,))))+list(np.sort(1/np.random.uniform(0.01,0.02,(10,))))

Res = train_Res+valid_Res+test_Res

with open('train_Res0.pkl','wb') as file:
    pickle.dump(train_Res,file)
    file.close()
    
with open('valid_Res.pkl','wb') as file:
    pickle.dump(valid_Res,file)
    file.close()
    
with open('test_Res.pkl','wb') as file:
    pickle.dump(test_Res,file)
    file.close()


# Generate Data
X_BC = []
U_BC = []
V_BC = []
P_BC = []
U_true = []
V_true = []
P_true = []

x = np.linspace(-0.5, 1.0, 101)
y = np.linspace(-0.5, 1.5, 101)

x_col = np.random.uniform(-0.5,1,100)
y_col = np.random.uniform(-0.5,1.5,100)
A = np.meshgrid(x_col,y_col)
X_col = np.concatenate([A[0].reshape(-1,1),A[1].reshape(-1,1)],axis=1)

A = np.meshgrid(x,y)
X = np.concatenate([A[0].reshape(-1,1),A[1].reshape(-1,1)],axis=1)

for Re in Res:
    
    lam = 0.5 * Re - np.sqrt(0.25 * (Re ** 2) + 4 * (np.pi ** 2))

    yb1 = np.array([-0.5] * 100)
    yb2 = np.array([1] * 100)
    xb1 = np.array([-0.5] * 100)
    xb2 = np.array([1.5] * 100)

    y_train1 = np.concatenate([y[1:101], y[0:100], xb1, xb2], 0)
    x_train1 = np.concatenate([yb1, yb2, x[0:100], x[1:101]], 0)

    xb_train = x_train1.reshape(x_train1.shape[0], 1)
    yb_train = y_train1.reshape(y_train1.shape[0], 1)
    
    x_bc = np.concatenate([xb_train, yb_train], axis=1) 
    ub_train = 1 - np.exp(lam * xb_train) * np.cos(2 * np.pi * yb_train)
    vb_train = lam / (2 * np.pi) * np.exp(lam * xb_train) * np.sin(2 * np.pi * yb_train)
    pb_train = (1 - np.exp(2*lam*xb_train))/2
    
    u_true = 1-np.exp(lam*X[:,0:1])*np.cos(2*np.pi*X[:,1:2])
    v_true = lam/(2*np.pi)*np.exp(lam*X[:,0:1])*np.sin(2*np.pi*X[:,1:2])
    p_true = (1-np.exp(2*lam*X[:,0:1]))/2
    
    X_BC.append(x_bc)
    U_BC.append(ub_train)
    V_BC.append(vb_train)
    P_BC.append(pb_train)
    U_true.append(u_true.reshape(-1,))
    V_true.append(v_true.reshape(-1,))
    P_true.append(p_true.reshape(-1,))
    
no1 = 20
no2 = 40
train_Res, valid_Res, test_Res = Res[:no1], Res[no1:no2], Res[no2:]
train_X_BC, valid_X_BC, test_X_BC = X_BC[:no1], X_BC[no1:no2], X_BC[no2:] 
train_U_BC, valid_U_BC, test_U_BC = U_BC[:no1], U_BC[no1:no2], U_BC[no2:]
train_V_BC, valid_V_BC, test_V_BC = V_BC[:no1], V_BC[no1:no2], V_BC[no2:]
train_P_BC, valid_P_BC, test_P_BC = P_BC[:no1], P_BC[no1:no2], P_BC[no2:]

# Train
model_hypernetwork = HyperNetwork(train_X_BC, train_U_BC, train_V_BC, train_P_BC, X, train_Res, valid_X_BC, valid_U_BC, valid_V_BC, valid_P_BC, valid_Res, test_Res, train_weights, valid_weights, 0)
model_hypernetwork.train()

u_predict, v_predict, p_predict, fu_predict, fv_predict, fe_predict, _ = model_hypernetwork.predict(X)

upredict, vpredict, ppredict = [], [], []

## Train task evaluation
train_mses = []
for i,Re in enumerate(train_Res):
    upredict.append(u_predict[i])
    vpredict.append(v_predict[i])
    ppredict.append(p_predict[i])
    train_mses.append((np.mean((u_predict[i]-U_true[i])**2)+
                     np.mean((v_predict[i]-V_true[i])**2)+
                     np.mean((p_predict[i]-P_true[i])**2))/3)

## Valid task evaluation
valid_mses = []
for i,Re in enumerate(valid_Res):
    upredict.append(u_predict[i+no1])
    vpredict.append(v_predict[i+no1])
    ppredict.append(p_predict[i+no1])
    valid_mses.append((np.mean((u_predict[i+no1]-U_true[i+no1])**2)+
                     np.mean((v_predict[i+no1]-V_true[i+no1])**2)+
                     np.mean((p_predict[i+no1]-P_true[i+no1])**2))/3)

## Test task evaluation
test_mses = []
for i,Re in enumerate(test_Res):
    upredict.append(u_predict[i+no2])
    vpredict.append(v_predict[i+no2])
    ppredict.append(p_predict[i+no2])
    test_mses.append((np.mean((u_predict[i+no2]-U_true[i+no2])**2)+
                     np.mean((v_predict[i+no2]-V_true[i+no2])**2)+
                     np.mean((p_predict[i+no2]-P_true[i+no2])**2))/3)

print(np.mean(train_mses[:20]),np.mean(valid_mses),np.mean(test_mses))
