import sys
import os
import argparse
import json
import re
import random 
import numpy as np
import torch

from torch.utils.data import DataLoader, Dataset
from utils import *
from PIL import Image
import matplotlib.pyplot as plt

################################################################################
################################################################################
## here the clases or types are divided in folder 
def ud_MakeDatasetFiles (directory, token, split= (.5, .25, .25), shuffle= True):

    files       = u_listFileAll_(directory, token)
    # each folder contains one class
    n_clases    = len(files) 
    split       = np.array(split)

    train = []
    valid = [] 
    test  = []

    #...........................................................................
    for dir, lfile in files:
        if shuffle:
            random.shuffle(lfile)

        bounds  = ( split * len(lfile) ).astype(int)
        
        train.append( [ lfile[:bounds[0]], 
                        os.path.basename(dir) ] )
        valid.append( [ lfile[bounds[0]: bounds[0] + bounds[1]], 
                        os.path.basename(dir) ] )
        test.append( [ lfile[bounds[0] + bounds[1]:], 
                       os.path.basename(dir) ] )

    return train, valid, test

################################################################################
################################################################################
'''  
    this function creates id folds given a vector size 
'''
def ud_MkFolds(nfiles, folds = 5, shuffle= True):

    s_fold  = int( np.floor( nfiles/ folds ) )
    idx     = np.array(list(range(nfiles)))
    idx_f   = []

    if shuffle:
        random.shuffle(idx)

    for i in range(0, nfiles, s_fold):
        idx_f.append(idx[i:(i+s_fold)])
        prev    = i + s_fold

    if prev < nfiles:
        np.append(idx_f[folds-1], idx[prev:(nfiles)] )

    return idx_f
    
################################################################################
################################################################################
def ud_saveDbOrder(file_name, lst):
    info    = {}
    nfiles  = 0

    for folder, name in lst:
        info[name]  = folder
        nfiles     += len(folder)

    u_saveDict2File(file_name, info )
    return nfiles 

################################################################################
################################################################################
def ud_parseData():

    data_dir    = 'db/'
    data_lst    = 'lst/'

    u_mkdir(data_lst)

    tr, vl, ts = u_make_dataset_files(data_dir, '.jpg')
    
    nfiles = saveDbOrder(data_lst + 'train.json', tr)
    nfiles += saveDbOrder(data_lst + 'valid.json', vl)
    nfiles += saveDbOrder(data_lst + 'test.json', ts)

    print(nfiles)
    
################################################################################
################################################################################
def ud_data2loader(file, transformations= None, graph = True):
    data = u_loadJson(file)

    #...........................................................................
    # visualizing db data
    if (graph):
        hist    = []
        names   = []

        for item in data:
            hist.append(len(data[item]))
            names.append(item)

        values  = hist
        ind = np.arange(len(names)) 
        plt.bar(ind, values)
        plt.xticks(ind, names, rotation='vertical')
        plt.show()

    return DbFromDict(data, transform= transformations)

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
class DbFromDict(Dataset):
    
    def __init__(self, info, transform=None):
        self.data_path, self.data_labels  = self.formatInput(info)
        self.transform  = transform  
        self.src        = info

    def __len__(self):
        return len(self.data_path)
    
    def __getitem__(self, index):
        # load image as ndarray type (Height * Width * Channels)
        # be carefull for converting dtype to np.uint8 [Unsigned integer (0 to 255)]
        # so you can convert numpy ndarray shape to tensor in PyTorch (H, W, C) --> (C, H, W)
        image = Image.open(self.data_path[index])
        
        if len(image.getbands()) == 1:
            image = np.stack((image,)*3, axis=-1)
            image = Image.fromarray(image)
        
        if self.transform is not None:
            image = self.transform(image)

        return image, self.data_labels[index]
  
    # utils ....................................................................
    def formatInput(self, info):
        data_path   = []
        data_labels = []
        
        cls = 0
        for lbl in info:
            for item in info[lbl]: 
                data_labels.append(cls)
                data_path.append(item)

            cls += 1

        return data_path, torch.tensor(data_labels)

