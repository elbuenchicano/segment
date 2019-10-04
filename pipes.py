from segment import * 

################################################################################
################################################################################
def pipe_1(general, individual):
    path        = general['prefix_path'][general['path_op']]
    directory   = path + '/' + general['directory']

    #..........................................................................
    train_data = {
        'train_info_pos'	: 1,
        'train_info'		:[
                {
                'name'			: 'unet_info_01',
                'patients_pt'	: 'research/segmentation/lists/patients.json',
                'iroot_pt'		: 'research/segmentation/scale/224_224',
                'proot_pt'		: 'research/segmentation/segbase',
                'bypatient'		: 1,
                'img_ext'		: '.ppm',
                'gt_ext'		: '.png',
                'p_ext'			: '.pgm',	 
                'nfolds'		: 7
                },
                'research/segmentation/lists/unet_info_01.json'
            ],
        'n_epochs'		: 10,
        'save_step'		: 5,
        'batch_size'	: 8,
        'model_name'	: individual['model'],
        'chk_point'		: [0, 4]
      }

    test_data = {
        'train_data'	: 'research/segmentation/models/' +individual['model'] +'/train_data.json',
        'original_gt'	: 'datasets/image_files',
        'test_flag'		: 0
        }

    pos_data = {
	    'info_data'		: 'research/segmentation/outputs/' +individual['model'] +'/train.json'
    }	

    functor = []
    functor.append((trainModel, (general, train_data))) ######  0

    functor.append((testModel, (general, test_data)))   ######  1

    test_data_  = test_data.copy()
    test_data_['test_flag'] = 1
    functor.append((testModel, (general, test_data_)))   ######  2

    functor.append((postPro, (general, pos_data)))      ######  3
    
    pos_data_   = pos_data.copy()
    pos_data_['info_data'] = 'research/segmentation/outputs/' +individual['model'] +'/test.json'
    functor.append((postPro, (general, pos_data_)))      ######  4

    fl = individual['functions']
    for i, flag in enumerate(fl):
        if flag:
            print('Performing:', functor[i][0].__name__, i)
            functor[i][0](*functor[i][1])


################################################################################
################################################################################
################################################################################
################################################################################
def pipe(general, individual):
    pipes = {
        'pipe_1' : pipe_1
        }

    pipes[individual['function']](general, individual['args'])
    
