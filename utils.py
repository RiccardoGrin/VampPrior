import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Utils
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
import cv2
import csv
import sys
import json
import math
import imageio
import numpy as np
import pandas as pd
import scipy.ndimage
from tqdm import tnrange
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython.display import clear_output


LATENT_DIM = 20
BATCH_SIZE = 256
MAX_EPOCHS = 50
RESIZE = 128
BETA = 10
LR = 1e-3
    

if not os.path.isdir('sweep'):
    os.system('mkdir sweep')

if not os.path.isdir('data'):
    os.system('mkdir data')

if not os.path.isdir('checkpoints'):
    os.system('mkdir checkpoints')

        
def gen_data_list(name="Pokemon"):
    if not os.path.isfile(name+'List'):
        if os.path.exists('./'+name):
            with open(name+'List', 'w') as myfile:
                wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                data = [os.path.abspath("./"+name+"/")+"/"+file for file in os.listdir("./"+name+"/")]
                wr.writerow(data)
            print("Generated list of", name, "image paths")
        else:
            print("Missing", name, "folder with images")
    else:
        print(name, "image list available")
        

class MultiSet(Utils.Dataset):
    """
        Dataloader for the model. Can easily add more datasets.
    """
    def __init__(self, name='Pokemon'):
        self.list = pd.read_csv(name+"List", header=None, delimiter=',').values[0]
    
    def __len__(self):
        return len(self.list)
    
    def __getitem__(self, index):
        data = mpimg.imread(self.list[index])
        data = cv2.resize(data, (RESIZE,RESIZE))/255
        return data


def criterion(x_out, target, z_mean, z_logvar, alpha=1, beta=BETA):
    """
        Criterion for VAE done analytically
        output: loss, bce, KL Divergence
    """
    bce = F.mse_loss(x_out, target, size_average=False) # Use MSE loss for images
    kl = -0.5 * torch.sum(1 + z_logvar - (z_mean**2) - torch.exp(z_logvar)) # Analytical KL Divergence - Assumes p(z) is Gaussian Distribution
    loss = ((alpha * bce) + (beta * kl)) / x_out.size(0)    
    return loss, bce, kl


def load_checkpoint(filename, net, LR):
    '''
        Loading function for the model before and during training
        From a checkpoint file, it loads and returns all necessary data 
        input: filename -> The name of the checkpoint file to be opened (.pth or .pt)
        output: net -> The saved model, including weights and biases
        output: epoch -> The epoch number at which the training was saved
        output: loss_save -> An array of all the saved batch losses during training
        output: optimizer -> The current state of the optimiser with its updated LR
    '''
    checkpoint = torch.load(filename) # load checkpoint data
    net.load_state_dict(checkpoint['state_dict'])
    epoch = checkpoint['epoch']
    losses = checkpoint['losses']
    bces = checkpoint['bces']
    kls = checkpoint['kls']
    cs = checkpoint['cs']
    optimizer = optim.Adam(net.parameters(), lr=LR, amsgrad=True)
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler = SGDRScheduler(optimizer, min_lr=1e-5, max_lr=LR, cycle_length=500, current_step=cs)
    
    print("Loaded checkpoint:", filename)
    return net, epoch, losses, bces, kls, optimizer, scheduler


def generate_animation(path, label):
    images = []
    files = sorted(os.listdir(path))
    for file in files:
        if file[-4:] == '.png':
            images.append(mpimg.imread(path + file))
    imageio.mimsave(path + label + '_animation.gif', images, fps=15)


def multi_plot(images, model, ROW=4, COL=4):
    """
        To plot an array of images
        Need batch size of row*col and a screen
        input: batch of image arrays
    """
    try:
        f, axarr = plt.subplots(ROW, COL, figsize=(15, ROW*4))
        for row in range(ROW//2):
            for col in range(COL):
                image = images[col+(COL*row),:,:,:].unsqueeze(0)
                axarr[2*row,col].imshow(image.squeeze().numpy())
                image = image.permute(0,3,1,2)
                x_out, _, _ = model(Variable(image.float().cuda()))
                x_out = x_out.permute(0,2,3,1)
                axarr[2*row+1,col].imshow(x_out.data.cpu().squeeze().numpy())
        plt.show()
    except:
        pass

    
def data_train(model, path, epoch):
    try:
        data = mpimg.imread(path)
        data = cv2.resize(data, (RESIZE,RESIZE))/255
        x_in = Variable(torch.FloatTensor(data).unsqueeze(0).permute(0,3,1,2).cuda())
        x_out, _, _ = model(x_in)
        im = np.floor(x_out.permute(0,2,3,1).data.cpu().squeeze().numpy()*255).astype(np.uint8)
        plt.imsave("data/img_{:04d}".format(epoch//10) + ".png", im)
        print("Saved checkpoint image")
    except:
        raise
        pass
    

### The following was taken from https://github.com/A-Jacobson/tacotron2

def adjust_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

        
class SGDRScheduler:
    """
        Implements STOCHASTIC GRADIENT DESCENT WITH WARM RESTARTS (SGDR)
        with cosine annealing from https://arxiv.org/pdf/1608.03983.pdf.
    """

    def __init__(self, optimizer, min_lr, max_lr, cycle_length, warmup_steps=5, current_step=0):
        self.optimizer = optimizer
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr = optimizer.param_groups[0]['lr']
        self.cycle_length = cycle_length
        self.current_step = current_step
        self.warmup_steps = warmup_steps

    def calculate_lr(self):
        """
            Calculates new learning rate with cosine annealing
        """
        step = self.current_step % self.cycle_length  # get step in current cycle
        self.lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * \
                  (1 + np.cos((step / self.cycle_length) * np.pi))

    def step(self):
        self.current_step += 1
        self.calculate_lr()
        if self.current_step in range(self.warmup_steps):
            self.lr /= 10.0  # take a few steps with a lower lr to "warmup"
        adjust_lr(self.optimizer, self.lr)