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
from PIL import Image, ImageFilter

import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import torchvision.transforms as T

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

    prev = 0
    for i in range(folds):
        idx_f.append(idx[prev:(prev+s_fold)])
        prev    += s_fold

    if prev < nfiles:
        idx_f[folds-1]  = np.append(idx_f[folds-1], idx[prev:(nfiles)] )

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

################################################################################
################################################################################
class DbLoader(Dataset):
    
    def __init__(self, data_path, data_lbl, transform=None):
        self.data_path      = data_path
        self.data_labels    = data_lbl
        self.transform  = transform  
        
    def __len__(self):
        return len(self.data_path)
    
    def __getitem__(self, index):
        
        image = Image.open(self.data_path[index])
        
        if self.transform is not None:
            image = self.transform(image)

        return image, self.data_labels[index]

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
class DbSegment(Dataset):
    
    def __init__(self, info, dtp = None, lbl = None):

        self.data_path_train  = []
        self.data_lbl_train   = []
        self.data_path_test   = []
        self.data_lbl_test    = []

        self.info = info
        self.formatInput()

        if dtp and lbl:
            self.data_path_train    = dtp
            self.data_lbl_train     = lbl

    #---------------------------------------------------------------------------
    def __len__(self):
        return len(self.data_path_train)
    
    #---------------------------------------------------------------------------
    def __getitem__(self, index):
        sflag   = self.info['salience']
            
        if self.info['bypatient']:
            lst     = self.data_path_train[index]
            img     = []
            gt      = []

            for imgp, gtp, pgm in lst:
                img_    = Image.open(imgp)
                gt_     = Image.open(gtp)
                gt_     = gt_.convert('L')
                pgm_    = Image.open(pgm)

                img_, gt_ = self.transform(img_, gt_, pgm_, sflag)
                img.append(img_)
                gt.append(gt_)
            
        else:
            imgp, gtp, pgm  = self.data_path_train[index]
            img     = Image.open(imgp)
            gt      = Image.open(gtp)
            pgm     = Image.open(pgm)

            img, gt = self.transform(img, gt, pgm, sflag)
        
        return img, gt, self.data_lbl_train[index] 
  
    #---------------------------------------------------------------------------
    ## utils
    def formatInput(self):
        patients    = self.info['patients']
        iroot       = self.info['iroot']
        proot       = self.info['proot']
        img_ext     = self.info['img_ext']
        gt_ext      = self.info['gt_ext']
        p_ext       = self.info['p_ext']
        nfolds      = self.info['nfolds']
        

        data_path       = []
        data_lbl        = []

        patients_   = sorted(patients)

        #......................................................................
        # array list or single image
        if self.info['bypatient']:
            for frms in patients_:
                for frm in patients[frms]: 
                    pt_path = []
                    for id in patients[frms][frm]:
                        img_path = iroot + '/' + u_fillVecToken( [frms, frm, id] , '_'  ) + img_ext
                        gt_path  = iroot + '/' + u_fillVecToken( [frms, frm, 'gt', id] , '_'  ) + gt_ext
                        p_path   = proot + '/' + u_fillVecToken( [frms, frm, id] , '_'  ) + p_ext

                        pt_path.append((img_path, gt_path, p_path)) 

                    data_path.append(pt_path)
                    data_lbl.append( frms + '_' + frm )

        #......................................................................
        else:
            for frms in patients_:
                for frm in patients[frms]: 
                    for id in patients[frms][frm]:
                        img_path = iroot + '/' + u_fillVecToken( [frms, frm, id] , '_'  ) + img_ext
                        gt_path  = iroot + '/' + u_fillVecToken( [frms, frm, 'gt', id] , '_'  ) + gt_ext
                        p_path   = proot + '/' + u_fillVecToken( [frms, frm, id] , '_'  ) + p_ext

                        data_path.append((img_path, gt_path, p_path))
                        data_lbl.append( frms + '_' + frm + '_' + id )

        #......................................................................

        n           = len (data_path)
        self.folds  = ud_MkFolds(n, nfolds)


        for f in range(nfolds - 1):
            for i in self.folds[f]:
                self.data_path_train.append(data_path[i])
                self.data_lbl_train.append(data_lbl[i])

        for i in self.folds[nfolds - 1]:
            self.data_path_test.append(data_path[i])
            self.data_lbl_test.append(data_lbl[i])
        

    #---------------------------------------------------------------------------
    def transform(self, image, mask, sali, sflag):
        mask  = mask.convert('RGB')

        step = 40 
        side = 184

        #.......................................................................
        if not sflag:
            # crop image
            image   = TF.crop(image, step, step, side, side)
            mask    = TF.crop(mask, step, step, side, side)
            
            # Random horizontal flipping
            if random.random() > 0.5:
                image   = TF.hflip(image)
                mask    = TF.hflip(mask)                  

            # Random rotation
            if random.random() > 0.5:
                angle   = random.random() * 15
                image   = TF.rotate(image, angle)
                mask    = TF.rotate(mask, angle)

            # Transform to tensor
            image   = TF.to_tensor(image)
            mask    = TF.to_tensor(mask)

        #.......................................................................
        else:
            # crop image
            image   = TF.crop(image, step, step, side, side)
            mask    = TF.crop(mask, step, step, side, side)

            resize  = T.Resize(size=(224, 224))
            sali    = sali.convert('L')
            sali    = sali.filter(ImageFilter.MaxFilter(5))
            sali    = resize(sali)
            sali    = TF.crop(sali, step, step, side, side)
            
            
            # Random horizontal flipping
            if random.random() > 0.5:
                image   = TF.hflip(image)
                mask    = TF.hflip(mask)                  
                sali    = TF.hflip(sali)                  

            # Random rotation
            if random.random() > 0.5:
                angle   = random.random() * 15
                image   = TF.rotate(image, angle)
                mask    = TF.rotate(mask, angle)
                sali    = TF.rotate(sali, angle)

            
    
            # Transform to tensor
            image   = TF.to_tensor(image)
            mask    = TF.to_tensor(mask)
            sali    = TF.to_tensor(sali)

            image   = torch.cat((image, sali), dim = 0)

        #.......................................................................
        return image, mask
       
 