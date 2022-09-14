from importlib.resources import path
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
    def __init__(self, input_dir1, input_dir2, output_dir, train=True):
        self.input_dir1 = input_dir1
        self.input_dir2 = input_dir2
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
        inputpath1 = self.input_dir1 + '/' + self.data_dir[idx]
        inputpath2 = self.input_dir2 + '/' + self.data_dir[idx]
        outputpath = self.output_dir + '/' + self.data_dir[idx]
        data_input1 = sio.loadmat(str(inputpath1))['curl']
        data_input2 = sio.loadmat(str(inputpath2))['curl']
        data_output = sio.loadmat(str(outputpath))['curl']
        data_input1 = np.expand_dims(data_input1,0)
        data_input2 = np.expand_dims(data_input2,0)
        return data_input1, data_input2, data_output

def train(model,model1,train_loader,optimizer,epoch):
    model.train()
    model1.eval()
    total_loss = 0.
    for i, (data_input1,data_input2,data_output) in enumerate(train_loader):
        data_input1,data_input2,data_output = data_input1.type(torch.FloatTensor).to(device),data_input2.type(torch.FloatTensor).to(device),data_output.type(torch.FloatTensor).to(device)
        optimizer.zero_grad()
        output1 = model1(data_input1)
        output2 = model1(data_input2)
        model_input = torch.cat([output1,output2],1)
        output = model(model_input)
        loss = loss_function(output.to(device),data_output)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    total_loss = total_loss / len(train_loader.dataset)
    # pre_loss = pre_loss / len(train_loader.dataset)
    print("Train Epoch: {}, train_loss: {}, learning_rate: {}".format(epoch,total_loss,optimizer.param_groups[0]['lr']))


def test(model,model1,test_loader,epoch):
    model.eval()
    model1.eval()
    total_loss = 0.
    with torch.no_grad():
        for i, (data_input1,data_input2,data_output) in enumerate(test_loader):
            data_input1,data_input2,data_output = data_input1.type(torch.FloatTensor).to(device),data_input2.type(torch.FloatTensor).to(device),data_output.type(torch.FloatTensor).to(device)
            optimizer.zero_grad()
            output1 = model1(data_input1)
            output2 = model1(data_input2)
            model_input = torch.cat([output1,output2],1)
            output = model(model_input)
            loss = loss_function(output.to(device),data_output)
            total_loss += loss.item()
    total_loss = total_loss / len(test_loader.dataset)
    print("Test Epoch: {}, test_loss: {}".format(epoch,total_loss))
    return total_loss

batch_size = 8
train_data = SRDataset('dataset/LR_st21','dataset/LR_st30','dataset/HR_stoutput',train=True)
test_data = SRDataset('dataset/LR_st21','dataset/LR_st30','dataset/HR_stoutput',train=False)
train_loader = DataLoader(train_data,batch_size = batch_size,num_workers=20) 
test_loader = DataLoader(test_data,batch_size = batch_size,num_workers=20)
loss_function = nn.L1Loss()
net1 = UNetsingle(256, 128, in_channels=1, num_classes=1).to(device)
net1.load_state_dict(torch.load("single.pth"))
net = UNetV2(in_channels=2, num_classes=8).to(device)
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
# optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=1e-04)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=5, verbose=False)

num_epochs = 150
best_loss = 2.9
for epoch in range(num_epochs):
    train(net,net1,train_loader,optimizer,epoch)
    loss = test(net,net1,test_loader,epoch)
    scheduler.step(loss)
    if loss < best_loss:
        best_loss = loss
        torch.save(net.state_dict(),"spatio-temporalcompare.pth")