import json

from utils     import u_getPath
from segment   import *
from preData   import *
from pipes     import pipe

################################################################################
################################################################################
################################ Main controler ################################
def _main():

    funcdict = { 'trainModel'               : trainModel,
                 'preprocessingPpm'         : preprocessingPpm,
                 'preprocessingTree'        : preprocessingTree,
                 'createTrainValTestList'   : createTrainValTestList,
                 'trainModel'               : trainModel,
                 'testModel'                : testModel,
                 'postPro'                  : postPro,
                 'pipe'                     : pipe

                }
    
    conf                = u_getPath('segmentation.json')
    confs               = json.load(open(conf))

    confs['general']['path_op']    = u_whichOS()

    #...........................................................................
    print('Main Function: ', confs['function'])
    funcdict[confs['function']](general     = confs['general'],
                                individual  = confs[confs['function']])
   


################################################################################
################################################################################
############################### MAIN ###########################################
if __name__ == '__main__':
    _main()