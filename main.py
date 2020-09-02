import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import torchvision
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
import torch.optim as optim
import datasets_ver_2
from datasets_ver_2 import TripletDataset
from torchvision import transforms
from new_resnet24 import resnet
from new_np1 import NpairLoss
from mtrainer import fit
cuda = torch.cuda.is_available()

if __name__=='__main__':
    # We'll create mini batches by sampling labels that will be present in the mini batch and number of examples from each class
    transform = transforms.Compose([
    # you can add other transformations in this list
    transforms.ToTensor()
])
    triplet_train_dataset = TripletDataset("/content/gdrive/My Drive/main/Fi/fi_trainimg_label.txt","train",transform) # Returns triplets of images
    triplet_valid_dataset = TripletDataset("/content/gdrive/My Drive/main/Fi/fi_valimg_label.txt","valid",transform)
    triplet_test_dataset = TripletDataset("/content/gdrive/My Drive/main/Fi/fi_testimg_label.txt","test",transform)
    batch_size = 64
    triplet_train_loader = torch.utils.data.DataLoader(triplet_train_dataset, batch_size=batch_size,shuffle=True)
    triplet_valid_loader = torch.utils.data.DataLoader(triplet_valid_dataset, batch_size=batch_size,shuffle=True)
    triplet_test_loader = torch.utils.data.DataLoader(triplet_test_dataset, batch_size=batch_size,shuffle=False)
    margin = 1.
    #embedding_net = EmbeddingNet()
    model = resnet()
    if cuda:
      model.cuda()
    loss_fn1 = NpairLoss()
    loss_fn2 = NpairLoss(0.98)
    lr = 1e-3
    m=0.9
    w=0.0005
    optimizer = optim.SGD(model.parameters(), lr=lr,weight_decay=w,momentum=m)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    n_epochs = 1
    log_interval = 100


    fit(triplet_train_loader, triplet_valid_loader, triplet_test_loader, model, loss_fn1,loss_fn2, optimizer, scheduler, n_epochs, log_interval, cuda)
