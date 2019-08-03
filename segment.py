import os 
import subprocess 
import cv2
import shlex

from PIL import Image
from utils import *
from utilsd_db import *

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
    for i in range(n):
        u_progress(i, n)

        file    = file_list[i] 
        img     = Image.open(file)
        img     = img.resize(im_size, Image.ANTIALIAS)            

        #nimg    = np.array(img)

        img     = cv2.imread(file)    
        img     = cv2.resize(img, im_size, interpolation= cv2.INTER_AREA)
        
        file    = os.path.basename(file)




        if file.find('gt') < 0:
            file_list_ori.append(file_list[i])            

            file_name   = file.replace('png', 'ppm')
            full_name   = out_dir_imgs + file_name   

            img.save(full_name, 'ppm')
            file_list_ori_ppm.append(file_name)

        else:
            file_list_gt.append(file)
            full_name   = out_dir_imgs + file
            img.save(full_name, 'png')


    
    
    #u_saveArray2File(   out_dir_list + label + '_file_list_original.lst', 
    #                    file_list_ori)

    #u_saveFlist2File(out_dir_list + label + '_file_list_gt.lst', 
    #                    out_dir_imgs, file_list_gt)

    #u_saveFlist2File(out_dir_list + label + '_file_list_ppm.lst', 
    #                    out_dir_imgs, file_list_ori_ppm)

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
################################################################################
# Treee segmentation based on edus model
def trainModel(general, individual):
    path        = general['prefix_path'][general['path_op']]
    directory   = path + '/' + general['directory']

    im_list     = path + '/' + individual['im_list']
    cmd1        = path + '/' + individual['cmd1']
    cmd2        = path + '/' + individual['cmd2']


################################################################################
################################################################################
def createTrainValTestList(general, individual):
    path        = general['prefix_path'][general['path_op']]
    directory   = path + '/' + general['directory']

    image_list      = individual['image_list']
    image_list_gt   = individual['image_list_gt']
    k_folds         = individual['k_folds']
    
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

    n       = len(patient_list)
    folds   = ud_MkFolds(n, k_folds)

    u_saveDict2File(out_dir_list + '/patients.json', patient_list)
    u_saveArrayTuple2File(out_dir_list + '/folds.lst', folds)
    
################################################################################
################################################################################
def trainModel(general, individual):
    path        = general['prefix_path'][general['path_op']]
    directory   = path + '/' + general['directory']

    patient_list    = individual['patient_list']
    folds_saved     = individual['fold_list']
    image_root      = individual['image_root']

    #...........................................................................
    patients    = u_loadJson(patient_list)   
    folds       = u_fileNumberList2array(folds_saved)

    #...........................................................................
    info        = {
        "patients"  : patients,
        "folds"     : folds,
        "ite_fold"  : 0,
        "iroot"     : image_root,
        "bypatient" : True,
        "train"     : True
    }

    image_transforms = {
        # Train uses data augmentation
        'train':
        transforms.Compose([
            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=224),  # Image net standards
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])  # Imagenet standards
        ]),
        # Test does not use augmentation
        'test':
        transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    train = DbSegment(info, image_transforms)
    


################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
class DbSegment(Dataset):
    
    def __init__(self, info, transform=None):

        if info["bypatient"]:
            self.data_path, self.data_labels  = self.formatInputPatient()
        else: 
            self.data_path, self.data_labels  = self.formatInputImage()

        if info["train"]:
            self.transform = transform["train"]
        else:
            self.transform = transform["test"]





    #---------------------------------------------------------------------------
    def __len__(self):
        return len(self.data_labels)
    
    #---------------------------------------------------------------------------
    def __getitem__(self, index):
        image = Image.open(self.data_path[index])
        
        if len(image.getbands()) == 1:
            image = np.stack((image,)*3, axis=-1)
            image = Image.fromarray(image)
        
        if self.transform is not None:
            image = self.transform(image)

        return image, self.data_labels[index]
  
    #---------------------------------------------------------------------------
    ## utils
    def formatInputPatient(self, patients):
        data_path   = []
        data_labels = []
        
        cls = 0
        values = patients.values()

        for i in range( len( values) ):

            for frm in patients[pt]: 
               for im in frm:
                    data_labels.append(cls)
                
                    data_path.append(item)

            cls += 1

        return data_path, torch.tensor(data_labels) 
    
    
        

            
            



    




