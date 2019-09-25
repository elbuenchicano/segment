import json

from utils     import u_getPath
from segment   import *

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
                 'postPro'                  : postPro

                }
    
    conf                = u_getPath('segmentation.json')
    confs               = json.load(open(conf))

    confs['general']['path_op']    = u_whichOS()

    #...........................................................................
    print(confs['function'])
    funcdict[confs['function']](general     = confs['general'],
                                individual  = confs[confs['function']])
   
################################################################################
################################################################################
############################### MAIN ###########################################
if __name__ == '__main__':
    _main()