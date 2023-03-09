import torch.nn as nn

import os
import cv2
from model import Autoencoder_2d
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from utils import handle_close, waitforbuttonpress




dataroot = 'Test_dataset'

# Select a random image part from the data root folder

filenames = os.listdir(dataroot)

num_samples = len(filenames)
model = Autoencoder_2d().cuda()
model.load_state_dict(torch.load('latest.pth'))
criterion = nn.MSELoss()





f, axarr = plt.subplots(1, 2, figsize=(12,12))
f.canvas.mpl_connect('close_event', handle_close)



while True:
    #Randomly choosing one image pair.
    ind = random.randint(0, num_samples-1)

    """
    Rlplacing your own image path here
    Rlplacing your own image path here
    """
    img_path_source = '{}/{}'.format(dataroot, filenames[ind])

    """
    Resize the image arbitrarily
    """
    h = random.randint(128, 1280) # up to 1280 to avoid OOM
    w = random.randint(128, 1280)  # up to 1280 to avoid OOM

    input = cv2.imread(img_path_source)
    input = cv2.resize(input, (w, h))

    input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)

    x = torch.from_numpy(input.transpose([2, 0, 1])).float().cuda()/255.0

    outputs = model(x.unsqueeze(0))

    """
    Visualizing results
    """
    axarr[0].imshow(input)
    axarr[1].imshow((outputs.cpu().detach().numpy()[0,...].transpose([1,2,0])[:,:,::-1]*255).astype('uint'))
    axarr[0].set_title('Source image')
    axarr[1].set_title('Model results')

    plt.draw()
    if not waitforbuttonpress():
        break
    print (ind)
