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

    m0 = []
    m1 = []
    m2 = []
    s0 = []
    s1 = []
    s2 = []


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
        train_info  = individual['train_info'][individual['train_info_pos']]
    else:
        train_info = u_loadJson(
                        individual['train_info'][individual['train_info_pos']])

    n_epochs    = individual['n_epochs']
    save_step   = individual['save_step']
    batch_size  = individual['batch_size']

    #...........................................................................
    out_dir     = directory + '/lists/'
    out_dir_w   = directory + '/models/' + train_info['name'] + '/'    
    
    u_mkdir(out_dir)
    u_mkdir(out_dir_w)
    
    #...........................................................................
    u_look4PtInDict(train_info, path)

    train_dataset   = DbSegment(train_info)
    train_loader    = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    if not individual['train_info_pos']:
        u_saveDict2File(out_dir + train_info['name'] + '.json', train_dataset.info)
    
    unet    = UNet(4,1).cuda()
    summary(unet, (4, 184, 184), batch_size = batch_size)

    criterion = torch.nn.MSELoss(reduction='sum').cuda()
    optimizer = torch.optim.Adam(unet.parameters(), lr=1e-4)
    
    #...........................................................................
    for epoch in range(n_epochs):
        for i, (img, gt, _) in enumerate(train_loader):
            for j in range(1, len(img)):
                img_    = img[j].cuda()
                gt_     = gt[j].cuda()

                outputs = unet(img_)
                loss = criterion(outputs, gt_)
                optimizer.zero_grad()       
                loss.backward()
                optimizer.step()
                
                # Track training progress
                print ('Epoch:{} \t loss{}.'.format(epoch, loss))

        if not (epoch+1) % save_step:
            checkpoint = {  'model': UNet(4,1).cuda(),
                        'state_dict': unet.state_dict(),
                        'optimizer' : optimizer.state_dict()}

            torch.save(checkpoint, out_dir_w + 'epoch_' + str(epoch) + '.pth')

    torch.save(checkpoint, out_dir_w + 'final.pth')



################################################################################
################################################################################
def testModel(general, individual):
    path        = general['prefix_path'][general['path_op']]
    directory   = path + '/' + general['directory']

    model_pt    = individual['model']
    info        = u_loadJson(individual['info_pt'])

    out_dir     = directory + '/outputs/' +  info['name'] + '/'
    u_mkdir(out_dir)

    #...........................................................................
    test_dataset    = DbSegment(info)
    test_dataset.swap_()
    test_loader     = DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    unet            = ud_loadCheckpoint(model_pt)
    summary(unet, (4, 184, 184))

    for i, (img, gt, lbl) in enumerate(test_loader):
        for j in range(0, len(img)):
            img_    = img[j].cuda()
            gt_     = gt[j].cuda()
            outputs = unet(img_)
            #show_tensor(outputs)
            #show_tensor(gt_)
            file_name =  out_dir + lbl[0] + '_' + str(j) + '.png'
            save_tensor_batch(file_name, outputs)
            
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
        x   = Image.fromarray(x).convert('L')
        x.save(file_name)
    
