import torch.nn as nn
import torch
import torch.optim as optim

import tqdm
from torch.utils.data import Dataset, DataLoader
import os
import cv2

from model import Autoencoder_2d
from dataset import VD_dataset


num_epochs = 100
batch_size = 20
epochs = 100




model = Autoencoder_2d().cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

mydataset = VD_dataset('VD_dataset2')
dataloader = torch.utils.data.DataLoader(mydataset, batch_size=batch_size, shuffle=True)

for epoch in range(epochs):
    loss = 0
    with tqdm.tqdm(dataloader) as pbar:
        for n_iter, (inputs, targets) in enumerate(pbar):

            x = inputs.float().cuda()/255.0
            y = targets.float().cuda()/255.0

            optimizer.zero_grad()

            assert x.shape[1] == 3

            outputs = model(x)

            train_loss = criterion(outputs, y)
            train_loss.backward()
            optimizer.step()

            loss += train_loss.item()

        loss = loss / len(dataloader)


        print("epoch : {}/{}, loss = {:.8f}".format(epoch + 1, epochs, loss))
        #torch.save(model.state_dict(), 'latest.pth')
