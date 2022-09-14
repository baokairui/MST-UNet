from importlib.resources import path
from turtle import position
from more_itertools import first
import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
import torch.optim as optim
import torch.nn.functional as F
from unet import UNetlr,UNetupsample
from scipy.ndimage import zoom
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device1 = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
class SRDataset(Dataset): 
    def __init__(self, input_dir, input1_dir, output_dir, train=True):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.input1_dir = input1_dir
        self.listdir = []
        for i in range(1000):
            for j in range(21,31):
                self.listdir.append(str(i)+'-'+str(j)+'.mat')
        self.index = []
        for k in range(10000):
            self.index.append(k)

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
        position = int(self.data_dir[idx][:index])
        temporal = int(self.data_dir[idx][index+1:index+3])
        number = position * 10 + temporal -20
        if temporal == 21 and self.data_dir[idx][index+3] == '.':
        # if idx % 29 == 1:
            lbpath = self.input1_dir + '/' + self.data_dir[idx][0:index] + '-20.mat'
            lr_before = sio.loadmat(str(lbpath))['curl']
            lr_before = np.expand_dims(lr_before,0)

        else:
            lrpath_before = self.input_dir + '/' + self.data_dir[idx-1]
            lr_before = sio.loadmat(str(lrpath_before))['curl']
            lr_before = np.expand_dims(lr_before,0)

        lr = np.expand_dims(lr_now,0)
        hr = np.expand_dims(hr_now,0)
        return lr, hr, lr_before, number

def train(model,train_loader,optimizer,epoch,dict1):
    model.train()
    total_loss = 0.
    model_loss = 0.
    for i, (lr, hr, lr_before, idx) in enumerate(train_loader):
        lr, hr, lr_before = lr.type(torch.FloatTensor).to(device), hr.type(torch.FloatTensor).to(device), lr_before.type(torch.FloatTensor).to(device)
        optimizer.zero_grad()
        if epoch == 0:
            output1 = interp(lr_before)
            output = model(lr,output1)
            dict1.update(list(zip(idx.tolist(),output.detach())))
        else:
            # dl = [dict1[i.item()] for i in idx]
            dl = []
            for i,item in enumerate(idx):
                if (item.item() % 10) == 1:
                    dl.append(interp(lr_before[i].unsqueeze (0))[0])
                else:
                    dl.append(dict1[item.item()-1])               
            output1 = torch.stack(dl, dim=0)
            output = model(lr,output1)
            dict1.update(list(zip(idx.tolist(),output.detach())))
        loss = loss_function(output.to(device),hr) * output.shape[0]
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        # model_loss += loss1.item()
    total_loss = total_loss / len(train_loader.dataset)
    # model_loss = model_loss / len(train_loader.dataset)
    # pre_loss = pre_loss / len(train_loader.dataset)
    print("Train Epoch: {}, train_loss: {}, model_loss: {}, learning_rate: {}".format(epoch,total_loss,model_loss,optimizer.param_groups[0]['lr']))


def test(model,test_loader,epoch):
    model.eval()
    total_loss = 0.
    with torch.no_grad():
        for i, (lr, hr, lr_before, idx) in enumerate(test_loader):
            lr, hr, lr_before = lr.type(torch.FloatTensor).to(device), hr.type(torch.FloatTensor).to(device), lr_before.type(torch.FloatTensor).to(device)
            if (idx.item() % 10) == 1:
                first = interp(lr_before)
            output = model(lr,first)
            loss = loss_function(output.to(device),hr)
            total_loss += loss.item()
            first = output
    total_loss = total_loss / len(test_loader.dataset)
    # pre_loss = pre_loss / len(test_loader.dataset)
    print("Test Epoch: {}, test_loss: {}".format(epoch,total_loss))
    return total_loss

dict1 = {}
batch_size = 16
interp = nn.Upsample(size=[128,256],mode='bicubic')
train_data = SRDataset('dataset/LR_pre','../LBM/dataset1000/LR_before1','dataset/HR',train=True)
test_data = SRDataset('dataset/LR_pre','../LBM/dataset1000/LR_before1','dataset/HR',train=False)
# train_data = SRDataset('dataset1/LR_new','dataset1/LR_before','dataset1/HR_before','dataset1/HR_new',train=True)
# test_data = SRDataset('dataset1/LR_new','dataset1/LR_before','dataset1/HR_before','dataset1/HR_new',train=False)
train_loader = DataLoader(train_data,batch_size = batch_size,shuffle=True,num_workers=20) 
test_loader = DataLoader(test_data,batch_size = 1,num_workers=20)
loss_function = nn.L1Loss()
net = UNetupsample(256, 128, in_channels=2, num_classes=1).to(device)
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
# optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=1e-04)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=5, verbose=False)

num_epochs = 150
best_loss = 2.9
for epoch in range(num_epochs):
    train(net,train_loader,optimizer,epoch,dict1)
    loss = test(net,test_loader,epoch)
    scheduler.step(loss)
    if loss < best_loss:
        best_loss = loss
        torch.save(net.state_dict(),"tsnet10.pth")