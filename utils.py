
import sys
import os
import argparse
import json
import re
import random 
import numpy as np


################################################################################
################################################################################
def u_loadJson(file_name):
    with open(file_name) as f:
        data = json.load(f)
    return data

################################################################################
################################################################################
def u_fileList2array(file_name):
    print('Loading data from: ' + file_name)
    F = open(file_name,'r') 
    lst = []
    for item in F:
        item = item.replace('\\', '/').rstrip()
        lst.append(item)
    F.close()
    return lst

################################################################################
################################################################################
'''
It loads data from file list which has the first row as the main root
'''
def u_fileList2array_(file_name):
    print('Loading data from: ' + file_name)
    F       = open(file_name,'r') 
    root    = F.readline().strip() 
    lst = []
    for item in F:
        #item = item.replace('\\', '/').rstrip()
        lst.append(item)
    F.close()
    return root, lst

################################################################################
################################################################################
def u_save2File(file_name, data):
    print('Saving data in: ' + file_name)
    F = open(file_name,'w') 
    F.write(data)
    F.close()

################################################################################
################################################################################
def u_saveList2File(file_name, data):
    print('Saving data in: ' + file_name)
    F = open(file_name,'w') 
    for item in data:
        item = item.strip()
        F.write(item + '\n')
    F.close()

################################################################################
################################################################################
def u_fileNumberList2array(file_name):
    print('Loading data from: ' + file_name)
    F = open(file_name,'r') 
    lst = []
    for item in F:
        if len(item) > 0:
            lst.append(float(item))
    F.close()
    return lst

################################################################################
################################################################################
def u_fileNumberMat2array(file_name):
    print('Loading data from: ' + file_name)
    F = open(file_name,'r') 
    lst = []
    for item in F:
        if len(item) > 0:
            item = item.split(' ')
            sub  = []
            for i in item:
                sub.append(float(i))
            lst.append(sub)
    F.close()
    return lst

################################################################################
################################################################################
def u_saveArray2File(file_name, data):
    print('Saving data in: ' + file_name)
    F = open(file_name,'w') 
    for item in data:
        F.write(str(item))
        F.write('\n')
    F.close()

################################################################################
################################################################################
def u_saveFlist2File(file_name, root, data):
    print('Saving data in: ' + file_name)
    F = open(file_name,'w') 
    F.write(root)
    for item in data:
        F.write('\n')
        F.write(str(item))
        
    F.close()

################################################################################
################################################################################
def u_saveArrayTuple2File(file_name, data):
    print('Saving data in: ' + file_name)
    F = open(file_name,'w') 
    for item in data:
        line = ''
        for tup in item:
            line += str(tup) + ' '
        F.write(line.strip())
        F.write('\n')
    F.close()
################################################################################
################################################################################
'''
Save dict into file, recommendably [.json]
'''
def u_saveDict2File(file_name, data):
    print ('Saving data in: ', file_name)
    with open(file_name, 'w') as outfile:  
        json.dump(data, outfile)

################################################################################
################################################################################
def u_mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

################################################################################
################################################################################
'''
it returns the complete file list in a list
'''
def u_listFileAll(directory, token):
    list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(token):
                list.append(root +'/'+ file)
    
    return sorted (list, key = u_stringSplitByNumbers)
    
################################################################################
################################################################################
'''
it returns a vector with separate root and files
'''
def u_listFileAll_(directory, token, wdir = True):
    list = []
    for root, dirs, files in os.walk(directory):
        sub_list =[root.replace(directory, ''),[]]
        for file in files:
            if file.endswith(token):
                if wdir:
                    file = root + '/' + file
                sub_list[1].append(file)
        if len(files) > 0:
            sub_list[1] = sorted (sub_list[1], key = u_stringSplitByNumbers)
            list.append(sub_list)

    return list

################################################################################
################################################################################
def u_getPath(file):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('inputpath', nargs='?', 
                        help='The input path. Default = conf.json')
    args = parser.parse_args()
    return args.inputpath if args.inputpath is not None else file

################################################################################
################################################################################
def u_loadFileManager(directive, token = ''):
    print(directive)
    if os.path.isfile(directive):
        file_list = []
        file = open(directive)
        for item in file:
            file_list.append(item)
    else:
        file_list   = u_listFileAll(directive, token)

    return sorted(file_list, key = u_stringSplitByNumbers)

################################################################################
################################################################################
def u_progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush() 
    
################################################################################
################################################################################
def u_init_list_of_objects(size):
    list_of_objects = list()
    for i in range(0,size):
        list_of_objects.append( list() ) #different object reference each time
    return list_of_objects

################################################################################
################################################################################
def u_replaceStrList(str_list, token1, token2):
    for i in range(len(str_list)):
        str_list = str_list.replace(token1, token2)
    return str_list

################################################################################
################################################################################
def u_stringSplitByNumbers(x):
    r = re.compile('(\d+)')
    l = r.split(x)
    return [int(y) if y.isdigit() else y for y in l]

################################################################################
################################################################################
def u_fillVecToken(names, token = ' '):
    ans = names[0]
    for i in range(1, len(names)):
        if names[i] is not '':
            ans += token + names[i] 
    return ans

################################################################################
################################################################################
def u_joinPath(names):
    return u_fillVecToken(names, '/')

