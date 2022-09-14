import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
import torch.optim as optim
import torch.nn.functional as F
from unet import UNetV2, UNetsingle
from scipy.ndimage import zoom
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SRDataset(Dataset): 
    def __init__(self, input_dir, output_dir, train=True):
        self.input_dir = input_dir
        self.output_dir = output_dir

        self.listdir = []
        for i in range(1000):
            self.listdir.append(str(i)+'.mat')
        
        if train:
            self.data_dir = self.listdir[:int(len(self.listdir)*0.8)]
        else:
            self.data_dir = self.listdir[int(len(self.listdir)*0.8):]

    def __len__(self):
        return len(self.data_dir)
 
    def __getitem__(self, idx):
        lrpath = self.input_dir + '/' + self.data_dir[idx]
        hrpath = self.output_dir + '/' + self.data_dir[idx]
        lr_now = sio.loadmat(str(lrpath))['curl'] #t
        hr_now = sio.loadmat(str(hrpath))['curl']
        return lr_now, hr_now

def train(model,train_loader,optimizer,epoch):
    model.train()
    total_loss = 0.
    # MRE = 0
    for i, (lr, hr) in enumerate(train_loader):
        lr, hr = lr.type(torch.FloatTensor).to(device), hr.type(torch.FloatTensor).to(device)
        optimizer.zero_grad()
        output = model(lr)
        loss = loss_function(output.to(device),hr)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * lr.size(0)
    epoch_loss = total_loss / len(train_loader.dataset)
    # MMRE = MRE / len(train_loader.dataset)
    # tr_writer.add_scalar(tags[0], epoch_loss, epoch)
    # tr_writer.add_scalar(tags[1], optimizer.param_groups[0]['lr'], epoch)
    print("Train Epoch: {}, train_loss: {}, learning_rate: {}".format(epoch,epoch_loss,optimizer.param_groups[0]['lr']))

def test(model,test_loader,epoch):
    model.eval() # 不改变其权值
    total_loss = 0.
    # MRE = 0
    with torch.no_grad():
        for i, (lr,hr) in enumerate(test_loader):
            lr, hr = lr.type(torch.FloatTensor).to(device), hr.type(torch.FloatTensor).to(device)
            output = model(lr) 
            loss = loss_function(output.to(device),hr)
            # MRE += torch.mean(torch.abs(output-label) / label)
            total_loss += loss.item() * hr.size(0)
    total_loss /= len(test_loader.dataset)
    # MMRE = MRE / len(train_loader.dataset)
    # te_writer.add_scalar(tags1[0], total_loss, epoch)
    print("Test Epoch: {}, test_loss: {}".format(epoch,total_loss))
    return total_loss

batch_size = 16
train_data = SRDataset('dataset/LR_stinput','dataset/LR_stoutput',train=True)
test_data = SRDataset('dataset/LR_stinput','dataset/LR_stoutput',train=False)
# train_data = SRDataset('dataset1/LR_new','dataset1/HR_new',train=True)
# test_data = SRDataset('dataset1/LR_new','dataset1/HR_new',train=False)
train_loader = DataLoader(train_data, batch_size = batch_size, shuffle=True, num_workers=10) 
test_loader = DataLoader(test_data, batch_size = batch_size, num_workers=10)
loss_function = nn.L1Loss()
net = UNetV2(in_channels=2, num_classes=8).to(device)
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
# optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=1e-04)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=5, verbose=False)

num_epochs = 300
best_loss = 2.9
for epoch in range(num_epochs):
    train(net,train_loader,optimizer,epoch)
    loss = test(net,test_loader,epoch)
    scheduler.step(loss)
    if loss < best_loss:
        best_loss = loss
        torch.save(net.state_dict(),"temperal.pth")