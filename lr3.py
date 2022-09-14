from importlib.resources import path
import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
import torch.optim as optim
import torch.nn.functional as F
from unet import UNetlr3
from scipy.ndimage import zoom
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device1 = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

class SRDataset(Dataset): 
    def __init__(self, input_dir, input1_dir, input2_dir, output1_dir, output_dir, output2_dir, train=True):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.input1_dir = input1_dir
        self.output1_dir = output1_dir
        self.input2_dir = input2_dir
        self.output2_dir = output2_dir
        self.listdir = []
        for i in range(1000):
            for j in range(22,41):
                self.listdir.append(str(i)+'-'+str(j)+'.mat')
        
        if train:
            self.data_dir = self.listdir[:int(len(self.listdir)*0.8)]
        else:
            self.data_dir = self.listdir[int(len(self.listdir)*0.8):]

    def __len__(self):
        return len(self.data_dir)
 
    def __getitem__(self, idx):
        lrpath = self.input_dir + '/' + self.data_dir[idx]
        hrpath = self.output_dir + '/' + self.data_dir[idx]
        lr_now = sio.loadmat(str(lrpath))['curl']
        hr_now = sio.loadmat(str(hrpath))['curl']
        index = self.data_dir[idx].find('-')
        temporal = int(self.data_dir[idx][index+1:index+3])
        if temporal == 22 and self.data_dir[idx][index+3] == '.':
        # if idx % 29 == 1:
            lbpath = self.input1_dir + '/' + self.data_dir[idx][0:index] + '-21.mat'
            hbpath = self.output1_dir + '/' + self.data_dir[idx][0:index] + '-21.mat'
            lr_before = sio.loadmat(str(lbpath))['curl']
            hr_before = sio.loadmat(str(hbpath))['curl']
            lr_before = np.expand_dims(lr_before,0)
            hr_before = np.expand_dims(hr_before,0)

            lbpath1 = self.input2_dir + '/' + self.data_dir[idx][0:index] + '-20.mat'
            hbpath1 = self.output2_dir + '/' + self.data_dir[idx][0:index] + '-20.mat'
            lr_before1 = sio.loadmat(str(lbpath1))['curl']
            hr_before1 = sio.loadmat(str(hbpath1))['curl']
            lr_before1 = np.expand_dims(lr_before1,0)
            hr_before1 = np.expand_dims(hr_before1,0)
        else:
            lrpath_before = self.input_dir + '/' + self.data_dir[idx-1]
            hrpath_before = self.output_dir + '/' + self.data_dir[idx-1]
            lrpath_before1 = self.input_dir + '/' + self.data_dir[idx-2]
            hrpath_before1 = self.output_dir + '/' + self.data_dir[idx-2]
            lr_before = sio.loadmat(str(lrpath_before))['curl']
            hr_before = sio.loadmat(str(hrpath_before))['curl']
            lr_before = np.expand_dims(lr_before,0)
            hr_before = np.expand_dims(hr_before,0)
            
            lr_before1 = sio.loadmat(str(lrpath_before1))['curl']
            hr_before1 = sio.loadmat(str(hrpath_before1))['curl']
            lr_before1 = np.expand_dims(lr_before1,0)
            hr_before1 = np.expand_dims(hr_before1,0)
        lr = np.expand_dims(lr_now,0)
        hr = np.expand_dims(hr_now,0)
        return lr, hr, lr_before, hr_before, lr_before1, hr_before1

def train(model,train_loader,optimizer,epoch):
    model.train()
    total_loss = 0.
    # pre_loss = 0.
    for i, (lr, hr, lr_before, hr_before, lr_before1, hr_before1) in enumerate(train_loader):
        lr, hr, lr_before, hr_before, lr_before1, hr_before1 = lr.type(torch.FloatTensor).to(device), hr.type(torch.FloatTensor).to(device), lr_before.type(torch.FloatTensor).to(device), hr_before.type(torch.FloatTensor).to(device), lr_before1.type(torch.FloatTensor).to(device), hr_before1.type(torch.FloatTensor).to(device)
        optimizer.zero_grad()
        output = model(lr,lr_before,lr_before1)
        loss = loss_function(output.to(device),hr)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * lr.size(0)
    total_loss = total_loss / len(train_loader.dataset)
    # pre_loss = pre_loss / len(train_loader.dataset)
    print("Train Epoch: {}, train_loss: {}, learning_rate: {}".format(epoch,total_loss,optimizer.param_groups[0]['lr']))

def test(model,test_loader,epoch):
    model.eval()
    total_loss = 0.
    # pre_loss = 0.
    with torch.no_grad():
        for i, (lr, hr, lr_before, hr_before, lr_before1, hr_before1) in enumerate(test_loader):
            lr, hr, lr_before, hr_before, lr_before1, hr_before1 = lr.type(torch.FloatTensor).to(device), hr.type(torch.FloatTensor).to(device), lr_before.type(torch.FloatTensor).to(device), hr_before.type(torch.FloatTensor).to(device), lr_before1.type(torch.FloatTensor).to(device), hr_before1.type(torch.FloatTensor).to(device)
            output = model(lr,lr_before,lr_before1)
            loss = loss_function(output.to(device),hr)
            total_loss += loss.item() * lr.size(0)
    total_loss = total_loss / len(test_loader.dataset)
    # pre_loss = pre_loss / len(test_loader.dataset)
    print("Test Epoch: {}, test_loss: {}".format(epoch,total_loss))
    return total_loss

batch_size = 16
train_data = SRDataset('../LBM/dataset1000/LR','../LBM/dataset1000/LR_before','../LBM/dataset1000/LR_before1','../LBM/dataset1000/HR_before','../LBM/dataset1000/HR','../LBM/dataset1000/HR_before1',train=True)
test_data = SRDataset('../LBM/dataset1000/LR','../LBM/dataset1000/LR_before','../LBM/dataset1000/LR_before1','../LBM/dataset1000/HR_before','../LBM/dataset1000/HR','../LBM/dataset1000/HR_before1',train=False)
# train_data = SRDataset('dataset1/LR_new','dataset1/LR_before','dataset1/HR_before','dataset1/HR_new',train=True)
# test_data = SRDataset('dataset1/LR_new','dataset1/LR_before','dataset1/HR_before','dataset1/HR_new',train=False)
train_loader = DataLoader(train_data,batch_size = batch_size,shuffle=True,num_workers=20) 
test_loader = DataLoader(test_data,batch_size = batch_size,num_workers=20)
loss_function = nn.L1Loss()
net = UNetlr3(256, 128, in_channels=3, num_classes=1).to(device)
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
# optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=1e-04)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=5, verbose=False)

num_epochs = 150
best_loss = 2.9
for epoch in range(num_epochs):
    train(net,train_loader,optimizer,epoch)
    loss = test(net,test_loader,epoch)
    scheduler.step(loss)
    if loss < best_loss:
        best_loss = loss
        torch.save(net.state_dict(),"combine_lr3.pth")