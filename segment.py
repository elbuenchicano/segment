import os 
import subprocess 
import shlex
import time

import pickle
import random

from timeit import default_timer as timer
from PIL import Image
from utils import *
from utilsd_db import *
from torchvision import transforms, utils
from model import *

import torchvision.transforms as T

from sklearn.cluster import KMeans
    
################################################################################
################################################################################
def trainModel(general, individual):
    path        = general['prefix_path'][general['path_op']]
    directory   = path + '/' + general['directory']

    #...........................................................................
    if not individual['train_info_pos']:
        train_info      = individual['train_info'][individual['train_info_pos']]
    else:
        train_info_file = individual['train_info'][individual['train_info_pos']]
        train_info      = u_loadJson(path + '/' + train_info_file)

    n_epochs    = individual['n_epochs']
    save_step   = individual['save_step']
    batch_size  = individual['batch_size']
    chk_point   = individual['chk_point']
    model_name  = individual['model_name']

    #...........................................................................
    model_list  = {
        'unet_salience'     : [UNet(4, 1), True],
        'unet_vanilla'      : [UNet(3, 1), False],
        'unet_2str'         : [UNet2Stream((3, 1), 1), True, True],
        'tunet'             : [TuNet(3, 1, 1), True, True]
        }
    
    #...........................................................................
    out_dir     = directory + '/lists/'
    out_dir_w   = directory + '/models/' + model_name + '/'    
    
    u_mkdir(out_dir)
    u_mkdir(out_dir_w)
    
    #...........................................................................
    if model_name == 'unet_2str' or model_name == 'tunet':
        separated = True
    else:
        separated = False

    #...........................................................................
    if not chk_point[0]: # this part will be deprecated 
        u_look4PtInDict(train_info, path)
        if len(model_list[model_name]) > 2:  
            train_dataset   = DbSegment(train_info, model_list[model_name][1], 
                                        separate_sal_flag = separated)
        else:
            train_dataset   = DbSegment(train_info, model_list[model_name][1])

        train_loader    = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        if not individual['train_info_pos']:
            train_info_file = out_dir + train_info['name'] + '.json', train_dataset.info
            u_saveDict2File(train_info_file)
        
        i_epoch         = 0
        loss_list       = []
        unet            = model_list[model_name][0].cuda()
        
    else: #.....................................................................
        train_dataset   = DbSegment(train_info, model_list[model_name][1])
        train_loader    = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        unet, optimizer, i_epoch, loss_list, criterion = ud_loadCheckpoint(out_dir_w + str(chk_point[1]) + '.pth')
        unet            = model_list[model_name][0].cuda()
        criterion       = criterion.cuda()
        
    #...........................................................................
    criterion       = torch.nn.MSELoss(reduction='sum').cuda()
    optimizer       = torch.optim.Adam(unet.parameters(), lr=1e-4)
    
    for epoch in range(i_epoch, n_epochs):
        for i, (img, gt, _) in enumerate(train_loader):
            
            sloss = 0
            for j in range(1, len(img)):
                gt_     = gt[j].cuda()
                outputs = call4Model(img[j], unet, separated)    
                
                loss    = criterion(outputs, gt_)
                loss_list.append(loss.data.cpu().numpy())
                optimizer.zero_grad()       
                loss.backward()
                optimizer.step()
                sloss   += loss.data.cpu().numpy()
        
        
            # Track training progress
            msg = 'epoch:' + str(epoch) + '| loss:' + str(float(loss.data.cpu().numpy()))                       
            u_progress(i, len(train_loader)-1, msg)
            

        if not (epoch+1) % save_step:
            state = {
                        'model'                 : model_list[model_name][0],
                        'optimizer'             : torch.optim.Adam(unet.parameters(), lr=1e-4),
                        'criterion'             : torch.nn.MSELoss(reduction='sum'),
                        'epoch'                 : epoch,
                        'model_state_dict'      : unet.state_dict(),
                        'optimizer_state_dict'  : optimizer.state_dict(),
                        'losslogger'            : loss_list}

            torch.save(state, out_dir_w + str(epoch) + '.pt')

    state= {'model'       : model_list[model_name][0],
            'state_dict'  : unet.state_dict()
           }
    #...........................................................................
    model_file_out =  out_dir_w + 'final.pt'
    torch.save(state, model_file_out)
    ud_plotLoss(out_dir_w,  loss_list)
    
    train_data = {
        'model_name'        : model_name,
        'final_model_pt'    : model_file_out.replace(path, ''),
        'train_info_pt'     : train_info_file.replace(path, ''),
        'salience'          : model_list[model_name][1]
        }
    u_saveDict2File(out_dir_w + 'train_data.json', train_data)

################################################################################
################################################################################
def testModel(general, individual):
    path        = general['prefix_path'][general['path_op']]
    directory   = path + '/' + general['directory']
    original_gt = path + '/' + individual['original_gt']
    test_flag   = individual['test_flag']

    #...........................................................................
    train_data  = u_loadJson(path + '/' + individual['train_data'])
    u_look4PtInDict(train_data, path)

    model_name  = train_data['model_name']     
    train_info  = u_loadJson(train_data['train_info_pt']) 
    model_file  = train_data['final_model_pt']     
    salience    = train_data['salience']     
        
    out_dir     = directory + '/outputs/' + model_name + '/'
    u_mkdir(out_dir)

    #...........................................................................
    if model_name == 'unet_2str' or model_name == 'tunet':
        separated = True
    else:
        separated = False

    #...........................................................................
    test_dataset    = DbSegment(train_info, salience, False, 
                                separate_sal_flag = separated)

    if test_flag:
        test_dataset.swap_()

    test_loader     = DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    #...........................................................................
    unet            = ud_loadModel(model_file)
    unet            = unet.cuda()
    files           = []
    for i, (img, gt, lbl) in enumerate(test_loader):
        u_progress(i, len(test_loader))

        im      = Image.open(original_gt + '/' + lbl[0] + '_gt_0.png')
        w, h    = im.size
        for j in range(0, len(img)):
            
            gt_     = gt[j].cuda()
            img_    = img[j][0].cuda()
            
            outputs = call4Model(img[j], unet,  separated)    

            resize  = T.Resize(size=(h, w))
            
            #show_tensor(img_)
            #show_tensor(gt_)
            #show_tensor(outputs)

            file_name =  out_dir + lbl[0] + '_' + str(j) + '_t'+ str(test_flag) + '.png'
            save_tensor_batch(file_name, outputs, resize)
            files.append([lbl, file_name])

    #...........................................................................    
    out_data = {
        'files'         : files,
        'test_flag'     : test_flag,
        'original_gt'   : original_gt
    }
    str_flag  = 'test' if test_flag  else 'train'
    u_saveDict2File(out_dir + str_flag +'.json', out_data)
            
################################################################################
################################################################################
def postPro(general, individual):
    path        = general['prefix_path'][general['path_op']]
    directory   = path + '/' + general['directory']
    
    info_data   = u_loadJson(path + '/' + individual['info_data'])
    test_flag   = info_data['test_flag']
    files       = info_data['files']

    out_dir     = os.path.dirname(files[0][1]) + '/'
    
    #...........................................................................
    if not test_flag:
        clusters = []
        for i, (_ , file) in enumerate(files):
            u_progress(i, len(files))

            im      = np.array(Image.open(file) )
            x, y    = im.shape
            im      = im.reshape(-1,1) 
            
            kmeans  = KMeans(n_clusters=4, init='k-means++', max_iter=200, n_init=10)
            select  = im[random.sample(range(im.shape[0]), 8000)]

            kmeans.fit(select)
            
            clusters.append(kmeans.cluster_centers_)

        #......................................................................
        kmeans  = KMeans(n_clusters=4, init='k-means++', max_iter=200, n_init=10)
        clus    = np.array(clusters).reshape(-1,1)
        kmeans.fit(clus)

        u_saveArrayTuple2File(out_dir+'clusters.txt', kmeans.cluster_centers_)
        
    #...........................................................................
    else:
        clusters    = np.array(u_fileList2array(out_dir+'clusters.txt')).reshape(-1, 1)
        kmeans      = KMeans(n_clusters=4, init='k-means++', max_iter=200, n_init=10)
        kmeans.fit(clusters)

        for i, (_ , file) in enumerate(files):
            u_progress(i, len(files))
            im      = np.array(Image.open(file) )
            x, y    = im.shape
            im      = im.reshape(-1,1) 

            out     = kmeans.predict(im)
            out     = out.reshape(x,y)

            file    = file.replace('.png', '_cl.png')
            im      = Image.fromarray(out)
            im.save(file)


################################################################################
################################################################################
################################################################################
################################################################################
# SUPPORT FUNCTIONS ############################################################
################################################################################
################################################################################
def call4Model(img, model, flag):
    if not flag:
        img_    = img.cuda()
        outputs = model(img_)
    else:
        img_, sal_  = img[0].cuda(), img[1].cuda()
        outputs     = model(img_, sal_)
    
    return outputs

################################################################################
################################################################################
def show_tensor(img):
    npimg = img[0].cpu().detach().numpy()
    npimg = np.transpose(npimg, (1, 2, 0) )
    plt.imshow(npimg[:,:,0])
    plt.show()

################################################################################
################################################################################
def save_tensor_batch(file_name, img, resize):
    print('File saved in :', file_name)
    for im in img:
        x   = im.cpu().detach().numpy()
        x   = np.transpose(x, (1, 2, 0) )[:,:,0]
        xmax, xmin = x.max(), x.min()
        x   = (x - xmin)/(xmax - xmin)*255
        #x   = np.pad(x, pad_width=20, mode='constant', constant_values=0)
        #plt.imshow(x)
        #plt.show()
        x   = Image.fromarray(x)
        x   = resize(x).convert('L')

        x.save(file_name)
    