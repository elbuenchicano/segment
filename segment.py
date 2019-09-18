import os 
import subprocess 
import shlex
import time

import pickle

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
# This function Reads all files in png format from a especific directory, 
# and transfroms the image into ppm file, the output is a list with alphanumeric
# order list
def preprocessingPpm(general, individual):

    path        = general['prefix_path'][general['path_op']]
    directory   = path + '/' + general['directory']

    im_size     = individual['im_size'] 
    im_dir      = path + '/' + individual['im_dir']

    #...........................................................................
    out_dir_list = directory + '/lists/'
    out_dir_imgs = directory + '/scale/'
    u_mkdir(out_dir_list)
    u_mkdir(out_dir_imgs)

    #...........................................................................
    file_list_ori       = []
    file_list_gt        = []
    file_list_ori_ppm   = []

    file_list           = u_listFileAll(im_dir, 'png')
        
    #...........................................................................
    label = str(im_size[0]) + '_' + str(im_size[1])  
    out_dir_imgs = out_dir_imgs + label + '/' 
    u_mkdir(out_dir_imgs)

    n = len(file_list)

    m0 = [], m1 = [], m2 = [], s0 = [], s1 = [], s2 = []

    for i in range(n):
        u_progress(i, n)

        file    = file_list[i] 
        img     = Image.open(file)
        img     = img.resize(im_size, Image.ANTIALIAS)            
        img     = img.convert('RGB')
        nimg    = np.array(img)
        nimg    = np.moveaxis(nimg, 2, 0)


        if file.find('gt') < 0:
            file_list_ori.append(file_list[i])            

            file_name   = file.replace('png', 'ppm')
            full_name   = out_dir_imgs + os.path.basename(file_name)

            img.save(full_name, 'ppm')
            file_list_ori_ppm.append(file_name)

        else:

            m0.append(np.mean(nimg[0]))
            m1.append(np.mean(nimg[1]))
            m2.append(np.mean(nimg[2]))
            s0.append(np.std(nimg[0]))
            s1.append(np.std(nimg[1]))
            s2.append(np.std(nimg[2]))

            file_list_gt.append(file)
            full_name   = out_dir_imgs + file
            img.save(full_name, 'png')


    m_s = ([np.mean(m0), np.mean(m1), np.mean(m2)],
           [np.mean(s0), np.mean(s1), np.mean(s2)])
    
    u_saveArrayTuple2File(  out_dir_list + label + '_mean_std.txt', 
                            m_s)
    
    
    u_saveArray2File(   out_dir_list + label + '_file_list_original.lst', 
                        file_list_ori)

    u_saveFlist2File(out_dir_list + label + '_file_list_gt.lst', 
                        out_dir_imgs, file_list_gt)

    u_saveFlist2File(out_dir_list + label + '_file_list_ppm.lst', 
                        out_dir_imgs, file_list_ori_ppm)

################################################################################
################################################################################
################################################################################
################################################################################
def callGraphSegment(out_dir, im_list, cmd1, cmd2):
    graph_files = []
    pgm_files   = []

    root, flist = u_fileList2array_(im_list)

    n = len (flist)
    #...........................................................................
    for i in range(n) :
        u_progress(i,n)
        file = flist[i]

        name_file_  = os.path.basename(file).replace('ppm', 'graph') 
        
        name_file   = out_dir + '/' + name_file_

        os.system(u_fillVecToken([cmd1, root + file, '4', name_file]) )

        out_pgm_file    = name_file.replace('graph', 'pgm')
        
        subprocess.call(
            shlex.split(
                u_fillVecToken(['bash', cmd2, name_file, out_pgm_file, '2 100 0.1'])
                )
        )

        graph_files.append(name_file_)
        pgm_files.append(os.path.basename(out_pgm_file))

    return graph_files, pgm_files

################################################################################
################################################################################
def preprocessingTree(general, individual):
    path        = general['prefix_path'][general['path_op']]
    directory   = path + '/' + general['directory']

    im_list     = path + '/' + individual['im_list']
    cmd1        = path + '/' + individual['cmd1']
    cmd2        = path + '/' + individual['cmd2']

    #...........................................................................
    out_dir_list    = directory + '/lists/'
    u_mkdir(out_dir_list)

    out_dir_graphs  = u_joinPath([directory, 'segbase'])
    u_mkdir(out_dir_graphs)

    #...........................................................................
    graph_list, pgm_files = callGraphSegment(out_dir_graphs, im_list, cmd1, cmd2)
    
    u_saveFlist2File(   u_joinPath([out_dir_list, 'graphs.lst']), 
                        out_dir_graphs, 
                        graph_list)

    u_saveFlist2File(   u_joinPath([out_dir_list, 'pgm_files.lst']), 
                        out_dir_graphs, 
                        pgm_files)
    
################################################################################
################################################################################
def createTrainValTestList(general, individual):
    path        = general['prefix_path'][general['path_op']]
    directory   = path + '/' + general['directory']

    image_list      = individual['image_list']
    image_list_gt   = individual['image_list_gt']
    
    #...........................................................................
    out_dir_list    = directory + '/lists/'
    u_mkdir(out_dir_list)                              

    #...........................................................................
    _, vimage_list = u_fileList2array_(image_list, )
    
    patient_list = {}
    for file in vimage_list:
        
        patient, frame, n = os.path.splitext(file)[0].split('_')
        if not patient in patient_list:
            patient_list[patient] = {}

        if not frame in patient_list[patient]:
            patient_list[patient][frame] = []

        patient_list[patient][frame].append(n)

    u_saveDict2File(out_dir_list + 'patients.json', patient_list)
    
################################################################################
################################################################################
def trainModel(general, individual):
    path        = general['prefix_path'][general['path_op']]
    directory   = path + '/' + general['directory']

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

    model_list  = {
        'unet_salience'     : [UNet(4,1), True],
        'unet_vanilla'      : [UNet(3,1), False],
        }
    
    #...........................................................................
    out_dir     = directory + '/lists/'
    out_dir_w   = directory + '/models/' + model_name + '/'    
    
    u_mkdir(out_dir)
    u_mkdir(out_dir_w)
    
    #...........................................................................
    if not chk_point[0]:
        u_look4PtInDict(train_info, path)
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
    summary(unet, (unet.in_channel, 184, 184), batch_size = batch_size)
        
    #...........................................................................
    criterion       = torch.nn.MSELoss(reduction='sum').cuda()
    optimizer       = torch.optim.Adam(unet.parameters(), lr=1e-4)
    


    for epoch in range(i_epoch, n_epochs):
        for i, (img, gt, _) in enumerate(train_loader):
            for j in range(1, len(img)):
                img_    = img[j].cuda()
                gt_     = gt[j].cuda()

                outputs = unet(img_)
                loss    = criterion(outputs, gt_)
                loss_list.append(loss.data.cpu().numpy())
                optimizer.zero_grad()       
                loss.backward()
                optimizer.step()
                
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

    train_data  = u_loadJson(path + '/' + individual['train_data'])
    u_look4PtInDict(train_data, path)

    model_name  = train_data['model_name']     
    train_info  = u_loadJson(train_data['train_info_pt']) 
    model_file  = train_data['final_model_pt']     
    salience    = train_data['salience']     

    
    out_dir     = directory + '/outputs/' + model_name + '/'
    u_mkdir(out_dir)

    #...........................................................................
    test_dataset    = DbSegment(train_info, salience, False)
    test_dataset.swap_()
    test_loader     = DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    unet            = ud_loadModel(model_file)
    unet            = unet.cuda()

    for i, (img, gt, lbl) in enumerate(test_loader):
        u_progress(i, len(test_loader))

        im      = Image.open(original_gt + '/' + lbl[0] + '_gt_0.png')
        w, h    = im.size

        for j in range(0, len(img)):
            img_    = img[j].cuda()
            gt_     = gt[j].cuda()
            outputs = unet(img_)

            resize  = T.Resize(size=(h, w))

            npimg   = outputs[0].cpu().detach().numpy()
            npimg   = np.transpose(npimg, (1, 2, 0) )

            npimg   = npimg.reshape(-1, 1)

            kmeans  = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10)
            
            kmeans.fit(npimg) 
            out     = kmeans.predict(npimg)
            out     = (out.reshape(224,224) /4 )* 255  
                                          
            out     = Image.fromarray(out)
            out     = resize(out).convert('L')

            #show_tensor(img_)
            #show_tensor(gt_)
            #show_tensor(outputs)

            file_name =  out_dir + lbl[0] + '_' + str(j) + '.png'
            out.save(file_name)
            
################################################################################
################################################################################
def show_tensor(img):
    npimg = img[0].cpu().detach().numpy()
    npimg = np.transpose(npimg, (1, 2, 0) )
    plt.imshow(npimg[:,:,0])
    plt.show()

################################################################################
################################################################################
def save_tensor_batch(file_name, img):
    print('File saved in :', file_name)
    for im in img:
        x   = im.cpu().detach().numpy()
        x   = np.transpose(x, (1, 2, 0) )[:,:,0]
        xmax, xmin = x.max(), x.min()
        x   = (x - xmin)/(xmax - xmin)*255
        #x   = np.pad(x, pad_width=20, mode='constant', constant_values=0)
        #plt.imshow(x)
        #plt.show()
        x   = Image.fromarray(x).convert('L')
        x.save(file_name)
    
