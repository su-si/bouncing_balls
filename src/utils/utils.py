import numpy as np
import re
import os
import sys
import json
import logging
import resource
import fcntl

class Object(object):
    '''https://stackoverflow.com/questions/2827623/how-can-i-create-an-object-and-add-attributes-to-it#2827664'''
    pass

########### math-like operations ###################

def rand_float(low, high, shape=()):
    '''returns a random array of floats between low and high, with shape :param:shape.
        Convenience function.
        :param low, high: two numbers. Required.
    '''
    try:
        result = np.random.rand(*shape) * (high - low) + low
    except TypeError as e:
        print("Argument 'shape' to rand_float has to be a tuple; something like (2) is not allowed "
              "(that's just an int in brackets). Replace with (2,) instead. (But the tuple can be '()', i.e., empty).")
        raise e

    return result



########### (image-)file and path operations ##############

def ensure_dir(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError:
            if not os.path.isdir(directory):
                raise


# tests whether all files in the given list (of strings) exist
def filesExist(topDir, filelist):
    for file in filelist:
        if(not os.path.isfile(os.path.join(topDir,file))):
            return False
    return True


    # Get last number from files' path/name string.
    # Returns last number found in string.
def getLastNumber(pathName):
    inNm = pathName.split("/")[-1]
    imgnumber_str = re.findall(r'\d+', inNm)[-1]     #get last number as a string, including any zeros in the beginning
    imgnumber = int(imgnumber_str)
    #imgnumber = int(imgnumber_str.lstrip("0"))
    return imgnumber


    # Cut off "the_top_path" from the beginning of "the_path".
    # Returns shortened pathname with no "/" in the beginning.
def getRelativePath(the_path, the_top_path):

    the_path = the_path.replace("\\", "/")
    the_top_path = the_top_path.replace("\\", "/")

    if( not (the_path.startswith(the_top_path))):
        print("Cannot cut off top directory of path string, path string does not start with '/the/top/dir'", the_top_path+".")
        return
    the_relative_path = the_path[len(the_top_path):]
    while the_relative_path.startswith("/"):   #cut off any /'s in the beginning so that we always have the same format
          the_relative_path = the_relative_path[1:]

    return the_relative_path

def getRelativeSubPath(the_path, partial_top_path):
    ''' find 'partial_top_path' in the_path, then return only the path after partial_top_path in the_path.
        Use case: different top paths that mean the same, namely ../.. etc and some absolute path '''
    assert partial_top_path in the_path
    idx = the_path.find(partial_top_path)
    return getRelativePath(the_path[idx:], partial_top_path)


def getPath(pathtoandfile):
    fn_split = pathtoandfile.replace("\\", "/").split("/")
    return "/".join(fn_split[:-1]) + "/"



def seperateFilenameFromPath(pathtofile):
    spl = pathtofile.split("/")
    filenm = spl[-1]
    pathnm = "/".join(spl[:-1])
    return pathnm, filenm



    # ignoring the actual filename in the end, compares "path" parts of firstPath, secPath
def pathEqual(firstPath, secPath):
    first_split = firstPath.replace('\\','/').split("/")
    sec_split = secPath.replace('\\','/').split("/")
    first_red = "/".join(first_split[:-1])
    sec_red = "/".join(sec_split[:-1])
    return first_red == sec_red


def replaceBasePath(path_list, old_top_path, new_top_path):
    ''' replace the 'olt_top_path' base path by 'new_top_path', e.g. if weights were trained elsewehere'''
    new_paths = []
    for path in path_list:
        sub_path = getRelativePath(path, old_top_path)
        new_path = os.path.join(new_top_path, sub_path)
        new_paths.append(new_path)
    return new_paths


################# user input ###############################

# For usage, look for example at cropHelper in /home/susi/Code/heatmapTest
#
def readUserInputUntilOk(textBefore, textError, comparisonFunction, inputTransformationFun):
    while(True):
        user_input = input(textBefore)
        if(comparisonFunction(user_input)):
            return inputTransformationFun(user_input)
        else:
            print(textError)



################## matrix operations #######################

#/* Return the x-y-indices of the top value in matrix Mat.
# * x (i.e., row) index is the first Point coordinate. */
def argmaxXY(Mat):
    x_maxvec = np.argmax(Mat, axis=1)
    helper = [(i, j) for i, j in enumerate(list(x_maxvec))]
    Maxvec = np.array([Mat[helper[i][0],helper[i][1]] for i in range(Mat.shape[0])])
    y_argmax = np.argmax(Maxvec)
    return x_maxvec[y_argmax], y_argmax

# returns the indices of the N highest values in Vec, sorted.
# matorvec: 1-d numpy array (doesn't work yet, but should rewrite
#       this once so that it can also be any other array; then, argmax and sorting should
#       be done along the 0th axis, as well as the returned maxids)
# TLDR: only use with vectors, not matrices!
def argmax(Vec, N):
    assert(len(Vec) >= N)
    maxids_unord = np.argpartition(Vec, -N, axis=0)[-N:]   # gives indices within Mat so that (-N)th element would be in
    sort = np.argsort(Vec[maxids_unord], axis=0)           #  correct position in ordered array, and all lover values below,
    maxids = maxids_unord[sort]                               #  all higher values above it, but in random order.
    return maxids

def argmin(Vec, N):
    assert(len(Vec) >= N)
    minids_unord = np.argpartition(Vec, N-1, axis=0)[:N]
    sort = np.argsort(Vec[minids_unord], axis=0)
    minids = minids_unord[sort]
    return minids

# take out N middle performing indices: sort by performance and return ~N/2 inds on either side of the median
def argmedian(Vec, N):

    assert(len(Vec) >= N)
    Nb2_lower = int(N/2)
    Nb2_higher = N - Nb2_lower
    m = len(Vec) / 2

    ids_ord = np.argsort(Vec, axis=0)
    meadian_ids = np.hstack((ids_ord[m - Nb2_lower : m], ids_ord[m : m + Nb2_higher] ))
    return meadian_ids

    #                                                                           ,--multfact--,
    # return matrix; if original was (1,2,3,5), "stetches" along given axis to (1,1,1,......1,2,2,...2,3,3,...3,5,...5)
    # multfact: multiplication factor, resulting array will be multfact times as big. Has to be an integer.
def stretchMatrix(mat, multfact, axis=0):
    assert(type(multfact)==int)
    if len(mat)==0:
        return mat
    dims = len(mat.shape)
    newshape = [mat.shape[s] for s in range(len(mat.shape))]
    newshape[axis] *= multfact
    result = np.empty((newshape), dtype=mat.dtype)
    for m in range(multfact):
        idxtuple = [slice(None)]*(axis) + [slice(m, mat.shape[axis]*multfact, multfact)] + [slice(None)]*(dims-axis-1) # slice(None) means to take all along that dimension, it seems.
        result[idxtuple] = mat
    return result


def where(anitem, alist):
    ''' https://stackoverflow.com/questions/364621/how-to-get-items-position-in-a-list'''
    position = [i for i, x in enumerate(alist) if x == anitem]
    if len(position) == 0:
        position = [-1]
    assert len(position) == 1, "noobs' where() does not support duplicates in the list"
    return position[0]


#def get_one_hot(targets, num_classes):
#    '''https://stackoverflow.com/questions/38592324/one-hot-encoding-using-numpy'''
#    res = np.eye(num_classes)[np.array(targets)]
#    return res.reshape(list(targets.shape)+[num_classes])


def check_validity(arr, key="unknown array"):
    if np.nan in arr:
        raise ValueError("nan found in array "+str(key))
    if np.inf in arr or -np.inf in arr:
        raise ValueError("+-inf found in array "+str(key))
        # if it warns about something here, it might have been an array with strings.
        #  Nothing to worry about.


## ----  dict ops ----- ##

def subset_dict(dict0, key_list):
    ''' Credits to
            https://stackoverflow.com/questions/5352546/extract-subset-of-key-value-pairs-from-python-dictionary-object#5352630'''
    for key in key_list:
        assert key in dict0.keys(), "Error: Key "+key+" is no key in this dictionary."
    return {k: dict0[k] for k in key_list}

def test_subset_dict():
    dict0 = {key: val for key, val in zip('abcdefgh', np.arange(8))}
    key_list = ['a', 'b', 'c']
    dict_red = subset_dict(dict0, key_list)
    for key in 'abc':
        assert key in dict_red.keys()
    for key in 'defgh':
        assert key not in dict_red.keys()



def vstack_array_dicts(dict1, dict2, allow_new_keys=False):
    '''
    :param dict1, dict2: dictionaries with arbitrary keys, with matching arrays as values. The array shapes should
            match (if the key exists in both dicts), in the sense that they will be stacked along the first axis.
    :param allow_new_keys: if False, raise an error if both dicts are nonempty, but their keys are not identical
    :return: a dict that contains the same keys as union(dict1.keys(), dict2.keys()), and their array contents stacked
            as np.vstack((array_dict1, array_dict2))
    '''
    if not allow_new_keys:
        assert len(dict1.keys()) == 0 or len(dict2.keys()) == 0  or set(dict1.keys()) == set(
            dict2.keys()), "Keys in dict1, dict2 have to match if allow_new_keys is False."

    combined_dict = dict1.copy()
    for key, value in dict2.items():
        if key in dict1.keys():
            combined_dict[key] = np.vstack((dict1[key], dict2[key]))
        else:
            combined_dict[key] = dict2[key]
    return combined_dict

# todo: test this
def tfstack_array_dicts(dict1, dict2, allow_new_keys=False):
    '''
    :param dict1, dict2: dictionaries with arbitrary keys, with matching arrays as values. The array shapes should
            match (if the key exists in both dicts), in the sense that they will be stacked along the first axis.
    :param allow_new_keys: if False, raise an error if both dicts are nonempty, but their keys are not identical
    :return: a dict that contains the same keys as union(dict1.keys(), dict2.keys()), and their array contents stacked
            as np.vstack((array_dict1, array_dict2))
    '''
    import tensorflow as tf
    if not allow_new_keys:
        assert len(dict1.keys()) == 0 or len(dict2.keys()) == 0  or set(dict1.keys()) == set(
            dict2.keys()), "Keys in dict1, dict2 have to match if allow_new_keys is False."

    combined_dict = dict1.copy()
    for key, value in dict2.items():
        if key in dict1.keys():
            combined_dict[key] = tf.stack((dict1[key], dict2[key]))
        else:
            combined_dict[key] = dict2[key]
    return combined_dict

####################  Colors & Images ##########################

def giveMeNColors(N):
    if(N > 30):
        print("N too big to get N colours.")
    cols = []
    for num in range(N):
        b = (num*20) % 256
        g = (10 + num*50) % 256
        r = (200 + num*30) % 256
        cols.append((b,g,r))
    return cols




def RGB2gray_nd(image_array):
    '''
    factors are from https://stackoverflow.com/questions/10261440/how-can-i-make-a-greyscale-copy-of-a-surface-in-pygame/10693616#10693616
    and from https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python#12201744
    :param image_array: an array of shape (L,M,N,... , width, height, 3) [n-d array of 3-d image arrays]
    :return: an array of shape (L,M,N, ..., width, height with gray level values
    '''
    if not type(image_array) == np.ndarray:
        image_array = np.array(image_array)
    assert image_array.shape[-1] == 3, "look at the name, this is meant for 3-channel images"
    #graylevel = np.zeros(shape=image_array.shape[:-1], dtype=image_array.dtype)
    r, g, b = (image_array[...  , 0], image_array[..., 1], image_array[..., 2])
    graylevel = r * 0.299 + g * 0.587 + b * 0.114
    return graylevel.astype(image_array.dtype)




def RGB2bw_nd(image_array):
    '''
    Transform to a "binary" black white image; 0=False(?) for black, 1=True for white
    :param image_array: an array of shape (L,M,N,... , width, height, 3) [n-d array of 3-d image arrays]
    :return: an array of shape (L,M,N, ..., width, height with gray level values
    '''
    if not type(image_array) == np.ndarray:
        image_array = np.array(image_array)
    assert image_array.shape[-1] == 3, "look at the name, this is meant for 3-channel images"
    r, g, b = (image_array[...  , 0], image_array[..., 1], image_array[..., 2])
    bw = r.astype(np.uint16) + g.astype(np.uint16) + b.astype(np.uint16)
    bw[bw < 255 * 1.5] = 0
    bw[bw != 0] = 1
    bw = bw.astype(np.bool)
    return bw




# ---- logging, storing parameters and other information ----#
# partially from: MP project skeleton & https://github.com/cs230-stanford/cs230-code-examples/blob/master/tensorflow/vision/model/utils.py



# Disable printing-to-stdout
# https://stackoverflow.com/questions/8391411/suppress-calls-to-print-python
def blockPrint():
    sys.stdout = open(os.devnull, 'w')
    # logging seems to write to stderr, so need to disable this, too
    sys.stderr = open(os.devnull, 'w')
# Restore printing-to-stdout
def enablePrint():
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

def export_config_txt(config, output_file):
    """
    Write the configuration parameters into a human readable file.
    :param config: the configuration dictionary
    :param output_file: the output text file
    """
    if not output_file.endswith('.txt'):
        output_file.append('.txt')
    max_key_length = np.amax([len(k) for k in config.keys()])
    with open(output_file, 'w') as f:
        for k in sorted(config.keys()):
            out_string = '{:<{width}}: {}\n'.format(k, config[k], width=max_key_length)
            f.write(out_string)

def export_config_pickle(config, output_file):
    import pickle
    if not output_file.endswith('.pkl'):
        output_file += '.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(config, f, 0) #pickle.HIGHEST_PROTOCOL)  # 0: text format, supposedly
    return output_file #+ '.pkl'

def reload_config_pickle(config_file):
    import pickle
    if not config_file.endswith('.pkl'):
        config_file += '.pkl'
    with open(config_file, 'rb') as f:
        result = pickle.load(f)
    return result

def export_config_json(config, output_file):
    assert os.path.isdir(getPath(output_file)), "Target file >>path<< does not exist. Path was: "+getPath(output_file)
    import json
    if not output_file.endswith('.json'):
        output_file += '.json'
    with open(output_file, 'w') as f:
        json.dump(config, f, indent=2)
    return output_file #+ '.pkl' # .pkl? No.


def reload_config_json(config_file):
    import json
    if not config_file.endswith('.json'):
        config_file += '.json'
    #with open('/'+config_file, 'r') as f:
    with open(config_file, 'r') as f:
        result = json.load(f)
    return result

def obj2dict(obj, ignore_start='_'):
    ''' Return a dict with all of the obj's attributes, keyed by their names.
        Exclude attributes starting with :param ignore_start.'''
    attr_names = dir(obj)
    objdict = {nm: getattr(obj, nm) for nm in attr_names if not nm.startswith(ignore_start)}
    return objdict


# Todo: Careful, I'm not sure this closes the old logfile when set_logger(...) is used to *change* the logfile.
#       Don't call it a huge number of times.
def set_logger(log_path):
    """Sets the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    print("... Setting logger to "+str(log_path))
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        ### Tried to remove the red colors, and failed miserably.
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)



def get_open_fds():
    '''https://stackoverflow.com/questions/4386482/too-many-open-files-in-python#4386502'''
    fds = []
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    for fd in range(3, soft):
            try:
                flags = fcntl.fcntl(fd, fcntl.F_GETFD)
            except IOError:
                    continue
            fds.append(fd)
    return fds

def get_file_names_from_file_number(fds):
    names = []
    for fd in fds:
        names.append(os.readlink('/proc/self/fd/%d' % fd))
    return names



# if __name__ == '__main__':

