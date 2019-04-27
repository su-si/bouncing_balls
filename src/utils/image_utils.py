from __future__ import division
import site
from PIL import Image as Ie
import math
import imageio
#import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

#import filemagic
from src.utils import *






def create_gif(frame_array, path, dt=0.1): #, scale_to_255_if_float=False):
    ''' :param scale_to_255_if_float: if True, rescale input depending on its min and max.
                If False, take * 255 or do nothing.
        :param dt: time between frames in gif, in seconds '''
    if frame_array.dtype == bool:
        imageio.mimwrite(path+'.gif', [f for f in frame_array.astype(np.uint8)*255],
                         duration=dt)
    #elif frame_array.dtype in [float, np.float16, np.float32, np.float64] and scale_to_255_if_float:
    #        #assert np.min(frame_array) >= 0, "Need another scaling, values might be between -1 and 1?"
    #        #assert np.max(frame_array) <= 1., "So you've got a float array with values between 0 and 255? Check: dtype was: "+str(frame_array.dtype)
    #        frame_array = scale_values_to_range(frame_array, range_min=0, range_max=255)
    #        imageio.mimwrite(path+'.gif', [f for f in (frame_array * 255).astype(np.uint8)],
    #                         duration=dt)
    else:
        #assert np.min(frame_array) >= 0, "Need another scaling, values might be between -1 and 1?"
        #assert frame_array.dtye == int, "It's not that easy; see how to check for values between 0 and 255 vs the rest"
        imageio.mimwrite(path+'.gif', [f for f in frame_array.astype(np.uint8)],
                         duration=dt)
    #writeGif(path, [f for f in frame_array.astype(np.uint8)], duration=0.05)


def show_sequence(array_of_frames, rgb=False, dt=0.001):
    '''
    :param array_of_frames:
    :param rgb:
    :param dt: time interval to sleep between frames, in seconds
    :return:
    '''
    if not rgb:
        assert len(array_of_frames.shape) == 3 or len(array_of_frames.shape) == 4 and array_of_frames.shape[-1] == 1
    else:
        assert len(array_of_frames.shape) == 4 and array_of_frames.shape[-1] == 3

    for frame in array_of_frames:
        plt.imshow(frame, cmap='gray')
        plt.pause(dt + 0.00001)



def scale_values_to_range(ndarray, range_min, range_max):
    ''' Returns: the array, scaled so that it has max value range_max, min value range_min.'''

    old_min = np.min(ndarray)
    old_max = np.max(ndarray)
    assert old_max > old_min, "you can't scale an array like this if it contains only one value"
    new_ndarray = ((range_max - range_min) * (ndarray - old_min) / (old_max - old_min) + range_min).astype(np.uint8)
    return new_ndarray

###################################
## visualize, visualize_predicted, visualize_sequence and visualized_sequence_predicted are / were used
## for viszalizing the output of a frame predictor.

def visualize(inputs, targets, max_n=3, store=False, rgb=False, output_dir=None):
    '''-- For 'autoencoder' models - no sequence dimension.
       :param inputs, targets: numpy-arrays of shape batch_sz x  h  x  w  x frames
       :param max_n: the maximal number of sequences to visualize '''
    if rgb:
        raise NotImplementedError("First make sure you know how to separate channels from frames "
                                  "in the last dimension, in inputs & targets. Then implement this.")

    n = min([max_n, inputs.shape[0]])
    for i in range(n):
        frame_seq = np.concatenate((inputs[i],targets[i]), axis=-1)
        assert len(frame_seq.shape) == 3
        frame_seq = frame_seq.transpose([2,0,1])
        if store:
            assert output_dir is not None
            create_gif(frame_seq, output_dir+'_'+str(i), dt=0.5)
        else:
            showAll(frame_seq)
        # first show the sequence of input frames, then the sequence of output frames.
        # make a gif of the whole sequence
        # store if required


def visualize_predicted(inputs, targets, predictions, max_n=3, store=False, rgb=False, output_dir=None):
    ''' -- For 'autoencoder' models - no sequence dimension.
      Output three gifs for each prediction: target, prediction, and a mix of both
     :param inputs, targets: numpy-arrays of shape batch_sz x  h  x  w  x frames
     :param max_n: the maximal number of sequences to visualize '''
    if rgb:
        raise NotImplementedError("First make sure you know how to separate channels from frames "
                                  "in the last dimension, in inputs & targets. Then implement this.")

    n = min([max_n, inputs.shape[0]])
    n_frames_pred = targets.shape[-1]
    assert np.all(targets.shape == predictions.shape)
    for i in range(n):
        frame_seq       = np.concatenate((inputs[i],predictions[i]), axis=-1)
        frame_seq_gt    = np.concatenate((inputs[i],targets[i]), axis=-1)
        assert len(frame_seq.shape) == 3 == len(frame_seq_gt.shape)
        # move frame dimension to front
        frame_seq       = frame_seq.transpose([2,0,1])
        frame_seq_gt    = frame_seq_gt.transpose([2,0,1])
        frame_seq_mixed =  np.zeros(shape=(2 * n_frames_pred,) + frame_seq.shape[1:])
        frame_seq_mixed[::2]  = predictions[i].transpose([2,0,1])
        frame_seq_mixed[1::2] = targets[i].transpose([2,0,1])
        if store:
            assert output_dir is not None
            create_gif(frame_seq, output_dir+'_pred_'+str(i), dt=0.5)
            create_gif(frame_seq_gt, output_dir+'_gt_'+str(i), dt=0.5)
            create_gif(frame_seq_mixed, output_dir+'_pred-gt_'+str(i), dt=0.5)
        else:
            showAll(frame_seq, title='prediction for sample '+str(i))
            showAll(frame_seq_gt, title= 'ground truth for sample' +str(i))
            showAll(frame_seq_mixed, title= 'prediction followed by gt, ' +str(i))



def visualize_sequence(inputs, targets, max_n=3, store=False, rgb=False, output_dir=None):
    ''' :param inputs, targets: numpy-arrays of shape batch_sz x  h  x  w  x frames
        :param max_n: the maximal number of sequences to visualize '''
    if rgb:
        raise NotImplementedError("First make sure you know how to separate channels from frames "
                                  "in the last dimension, in inputs & targets. Then implement this.")

    n = min([max_n, inputs.shape[0]])
    for i in range(n):
        inputs_i = inputs[i][..., -1]   # only look at last input frame at each timestep
        frame_seq = np.concatenate((inputs_i[0:3], targets[i][..., 0]), axis=0)
        assert len(frame_seq.shape) == 3 # sequence data
#        frame_seq = frame_seq.transpose([2,0,1])
        if store:
            assert output_dir is not None
            create_gif(frame_seq, output_dir+'_'+str(i), dt=0.5)
        else:
            showAll(frame_seq)
        # first show the sequence of input frames, then the sequence of output frames.
        # make a gif of the whole sequence
        # store if required


def visualize_sequence_predicted(inputs, targets, predictions, max_n, seq_lengths=None, store=False, rgb=False, output_dir=None, masks=None, kernels=None):
    ''' :param inputs, targets, predictions: numpy-arrays of shape batch_sz x max_seq_length x h  x  w  x frames
        :param max_n: the maximal number of sequences to visualize
        :param seq_lengths: either None, then each samples sequence length will be inferred. Or a list of same length as predictions.
        :param masks: if given, expected to have the same shape as predictions, except for last dimension. Num masks = size of last dimension of masks.
        :param kernels: same as masks/treated the same way. Doesnt need to have the same shape as 'predictions' of course.
        output: three gifs per sample sequence. The 'mixed' sequence contains input-prediction-input-prediction in that order,
                meaning that the * desired target frame always comes **AFTER** the predicted frame* '''

    n = min([max_n, inputs.shape[0]])
    assert seq_lengths is None or (isinstance(seq_lengths, list) and len(seq_lengths) == len(predictions)), "seq_lengths has to be None or a list of len(predictions)"
    if rgb:
        raise NotImplementedError("First make sure you know how to separate channels from frames "
                                  "in the last dimension, in inputs & targets. Then implement this.")
    else:
        for i in range(n):
            inputs_i = inputs[i][..., -1]  # only look at last input channel at each timestep
            targets_i = targets[i][..., -1]
            predictions_i = predictions[i][..., -1]
            seq_length = seq_lengths[i] if seq_lengths is not None else len(predictions_i)
            # target sequence and predicted sequence
            targ_frame_seq = targets_i[0:seq_length]
            pred_frame_seq = predictions_i[0:seq_length]
            if masks is not None:
                masks_i = masks[i]
                masks_seq = masks_i[0:seq_length]
            if kernels is not None:
                kernels_i = kernels[i]
                kernels_seq = kernels_i[0:seq_length]
            # input and predicted in turns
            mixed_frame_seq = np.zeros(shape=(2 * seq_length,) + inputs_i.shape[1:])
            mixed_frame_seq[::2] = inputs_i[0:seq_length]
            mixed_frame_seq[1::2] = predictions_i[0:seq_length]
            if store:
                assert output_dir is not None
                create_gif(targ_frame_seq, output_dir + '_target_' + str(i), dt=0.5)
                create_gif(pred_frame_seq, output_dir + '_prediction_' + str(i), dt=0.5)
                create_gif(mixed_frame_seq, output_dir + '_inputs-pred-zip_' + str(i), dt=0.5)
                if masks is not None:
                    for j in range(masks.shape[-1]):
                       create_gif(masks_seq[..., j], output_dir + '_mask-'+str(j)+'_'+str(i), dt=0.5)
                if kernels is not None:
                    for j in range(kernels.shape[-1]):
                       create_gif(kernels_seq[..., j], output_dir + '_kernel-'+str(j)+'_'+str(i), dt=0.5)
            else:
                showAll(targ_frame_seq, title='target')
                showAll(pred_frame_seq, title='prediction')
                showAll(mixed_frame_seq, title='mixed')
                # first show the sequence of input frames, then the sequence of output frames.
                # make a gif of the whole sequence
                # store if required


# delete this if there's no problem
#### for showing images: ####

#def showAll(imagelist, title=""):
#    import cv2
#    for mdx, im in enumerate(imagelist):
#        cv2.namedWindow(title+"_"+str(mdx))
#        cv2.imshow(title+"_"+str(mdx), im)
#        cv2.waitKey()
#    return


# # # # # # # # # # # # # # # # # # # #
# # # From Bachelor project code: # # #
# # # # # # # # # # # # # # # # # # # #


def cv2_resize_if_needed(img, new_w, new_h):
    import cv2
    if((not img.shape[0] == new_h) or (not img.shape[1] == new_w)):
            return cv2.resize(img, (new_w, new_h))
    else:
            return img

 # from http://stackoverflow.com/questions/6059217/cutting-one-image-into-multiple-images-using-the-python-image-library
def cropImage(l, u, w, h, img):
     bbox = (l, u, l+w, u+h)
     working_slice = img.crop(bbox)
     return working_slice

# here, imagelist: list of images, already read. Not filenames!
#       croppingMat:  a Nx4 matrix(!) with cropping info x,y,w,h.
def cropImages(imagelist, croppingMat):

    cropped_images = [None]*len(imagelist)
    for k,im in enumerate(imagelist):
        im_cropped = []
        x_cr = croppingMat[k,0]
        y_cr = croppingMat[k,1]
        w_cr = croppingMat[k,2]
        h_cr = croppingMat[k,3]
        if(-1 in [x_cr, y_cr, w_cr, h_cr]):
            cropped_images[k] = im
            continue
        try:
            im_cropped = im[y_cr: y_cr + h_cr, x_cr: x_cr + w_cr]
        except: # make this an error, otherwise, we can't preallocate to a fixed place beforehand
            print("Error: cropping info does not fit image. Returned list will have empty entries. ")
            continue
        cropped_images[k] = im_cropped

    return cropped_images


    # crop all images in imagelist with THE SAME cropping info.
    # cropping: [x_cr, y_cr, w_cr, h_cr]; -1's are allowed, ==[] ... also, but....
def cropImagesSameCrop(imagelist, cropping):
        if(cropping == []):     # no cropping info given
            #cropping = [-1,-1,-1,-1]
            return imagelist
        x_cr = cropping[0]
        y_cr = cropping[1]
        w_cr = cropping[2]
        h_cr = cropping[3]
        if(x_cr < 0 or y_cr < 0 or w_cr <= 0 or h_cr <= 0):
            # cropping info sais not to crop
            return imagelist

        cropped_images = []
        for im in imagelist:
            im_cropped = []
            try:
                im_cropped = im[y_cr: y_cr + h_cr, x_cr: x_cr + w_cr]
            except:
                print("cropping info does not fit image; skip image.")
                continue
            cropped_images.append(im_cropped)
        return cropped_images

# calculates the width and height that keep the old wi/hei ratio but fit exactly within the rectangle (destw, desth)
def scaleToLieWithin(wi, hei, destw, desth):
    width_scaling = float(destw)/float(wi)
    height_scaling = float(desth)/float(hei)
    if(width_scaling == height_scaling):
        return destw, desth
    scaling = []
    if(width_scaling <= height_scaling):
        scaling = width_scaling
    else:
        scaling = height_scaling
    newh = int(scaling*float(hei))
    neww = int(scaling*float(wi))
    return neww, newh

# return sizes, in pixels, of left, right, top and bottom border that is needed
# if one wants to fit (newWidth, newHeight)-matrix in the center of a (destWdth, destHght)-matrix.
def calcPadding(newWidth, newHeight, destWdth, destHght):
    left = int(math.floor(float(destWdth - newWidth)/2.0))
    right = int(math.ceil(float(destWdth - newWidth)/2.0))
    top = int(math.ceil(float(destHght - newHeight)/2.0))
    bottom = int(math.ceil(float(destHght - newHeight)/2.0))
    return left, right, top, bottom


# Copied from cpp code.
#* Scale the image so that in one direction, it fits to (width, height) given, then
# * pad it so that it has size (width, height). */
def scaleAndPad(img, destWdth, destHght, padCol):
    import cv2
    if(img.shape[1] == destWdth and img.shape[0] == destHght):
        return img
    img_padded = np.zeros((destHght, destWdth, 3))
    newWidth, newHeight = scaleToLieWithin(img.shape[1], img.shape[0], destWdth, destHght)
    if(newWidth==destWdth and newHeight==destHght):
        img_padded = cv2.resize(img, (destWdth, destHght))
        return img_padded
    #img_resized = np.zeros((newHeight, newWidth, 3))
    img_resized = cv2.resize(img, (newWidth, newHeight))

    # the padding:
    left, right, top, bottom = calcPadding(newWidth, newHeight, destWdth, destHght)
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padCol)

    return img_padded   # If there's a width-border, you have to search a bit for the non border values. Even if the
                        # matrix appears to consist of zeros only, have a closer look.

# similar to the function of same name here:
# https://github.com/shihenw/convolutional-pose-machines-release/blob/master/testing/src/run_benchmark.m
#
# We want to crop "boxsize" (a tuple (w,h)) around point "center" (a tuple (x,y)), so pad with pad_value as far as needed and then crop.
# (Have: image of size sz, centerpoint. Want: image of smaller size, around centerpoint.)
#
# todo: either this or the one below has an error inside!!!
def padAround(img, boxsize, center, pad_val):

    import cv2
    crop_h = boxsize[1]
    crop_w = boxsize[0]
    sz = (img.shape[1], img.shape[0])

    top = int(center[1] - np.ceil(float(crop_h)/2.0))
    bot =  int(np.ceil(float(crop_h)/2.0) + center[1])
    left =  int(center[0] - np.floor(float(crop_w)/2.0))
    right =  int(np.ceil(float(crop_w)/2.0) + center[0])

    topb = - min( 0, max(top, -crop_h)) 	# max( 0, min(top - sz[1], crop_h))
    botb = max( 0, min(bot - sz[1], crop_h))
    leftb = - min( 0, max(left, -crop_w))
    rightb =   max( 0, min(right - sz[0], crop_w))

    topc = min( max(0, top), sz[1])	#max( min(sz[1], top), 0)
    botc =  max( min(sz[1], bot), 0)	#min( max(0, bot), sz[1])
    leftc =  min( max(0, left), sz[0])
    rightc =  max( min(sz[0], right), 0)

        # I hope this is alright. It should be. (This corresponds to their matlab code; they don't have a +1 in the end, so result array goes e.g. from 1 to top -> 0 to top-1 in python.)
    img_cropped = img[topc:botc, leftc:rightc]
    img_cropped_padded = cv2.copyMakeBorder(img_cropped, topb, botb, leftb, rightb, borderType=cv2.BORDER_CONSTANT, value=pad_val)

    if not (img_cropped_padded.shape[0] == boxsize[0] and img_cropped_padded.shape[1] == boxsize[1]):
        print("Error: padAround did not create image of required size. Required size: "+str(boxsize[0])+","+str(boxsize[1])+";\n")
        print("actual size: "+str(img_cropped_padded.shape[1])+","+str(img_cropped_padded.shape[0])+".\n")
        print(", ".join([str(x) for x in [center[0], center[1], top, bot, left, right]]))
        print(", ".join([str(x) for x in [topb, botb, leftb, rightb, topc, botc, leftc, rightc]]))
        raise
    assert(img_cropped_padded.shape[0] == boxsize[0] and img_cropped_padded.shape[1] == boxsize[1])
    return img_cropped_padded

# inverse of padAround
# sz: the size img had before padAround
# center: same center point as used in padAround
# pad_val: needs to be a numpy array of right size, I think
#
## todo: Either this or padAround has and error inside!!!!!!!
def resizeIntoScaledImg(channel, sz, center, pad_val):

    import cv2
    crop_h = channel.shape[0]
    crop_w = channel.shape[1]

    top = int(center[1] - np.ceil(float(crop_h)/2.0))
    bot =  int(np.ceil(float(crop_h)/2.0) + center[1])
    left =  int(center[0] - np.floor(float(crop_w)/2.0))
    right =  int(np.ceil(float(crop_w)/2.0) + center[0])

    topb = - min( 0, max(top, -crop_h)) 	# max( 0, min(top - sz[1], crop_h))
    botb = max( 0, min(bot - sz[1], crop_h))
    leftb = - min( 0, max(left, -crop_w))
    rightb =   max( 0, min(right - sz[0], crop_w))

    # in pad Around: topc =  - min( 0, max(top, -crop_h))   # --> border that needs be added: topc
    #                botc =  max( min(sz[1], bot), 0) --> border needed: sz[1] - botc
    # (now the ..c are borders, ..b are where to cut)
    topc = min( max(0, top), sz[1])		# sz[1] -  max( min(sz[1], top), 0)
    botc = sz[1] - max( min(sz[1], bot), 0)	# min( max(0, bot), sz[1])
    leftc =   min( max(0, left), sz[0])
    rightc = sz[0] - max( min(sz[0], right), 0)

    chan_unpadded = channel[topb : crop_h - botb, leftb : crop_w - rightb]                                              # dont ask me WHY it needs a float64 when padding float32 matrices.
    chan_unpadded_uncropped = cv2.copyMakeBorder(chan_unpadded, topc, botc, leftc, rightc, borderType=cv2.BORDER_CONSTANT, value=np.float64(pad_val))

    return chan_unpadded_uncropped



# copy-paste of https://github.com/shihenw/convolutional-pose-machines-release/blob/master/testing/src/run_benchmark.m
# I believe this only works with even boxsizes !
# center: expected as !(x,y)!
def padAround_CPMs(img, boxsize_x, boxsize_y, center):
    center = (center).astype(int)
    h = img.shape[0]
    w = img.shape[1]
    pad = np.zeros((4))
    pad[0] = boxsize_y/2. - center[1]       # up            # the image is read in "headfirst"/"kopfueber"
    pad[2] = boxsize_y/2. - (h - center[1]) # down
    pad[1] = boxsize_x/2. - center[0]       # left
    pad[3] = boxsize_x/2. - (w - center[0]) # right

    if(pad[0] > 0):
        pad_up =    np.tile(np.zeros_like(img[0,:,:]), (pad[0], 1, 1)) + 128
        img_padded = np.vstack((pad_up, img))
    else:
        img_padded = img
    if pad[1] > 0:
        pad_left =  np.tile(np.zeros_like(img_padded[:,0,:]), (1, pad[1], 1)) + 128
        img_padded = np.stack((pad_left, img_padded), axis=1)
    if pad[2] > 0:
        pad_down =  np.tile(np.zeros_like(img_padded[0,:,:]), (pad[2], 1, 1)) + 128
        img_padded = np.vstack((img_padded, pad_down))
    if pad[3] > 0:
        pad_right = np.tile(np.zeros_like(img_padded[:,0,:]), (1, pad[3], 1)) + 128
        img_padded = np.stack((img_padded, pad_right), axis=1)

    center = center + np.array([ max(0, pad[1]), max(0, pad[0])])
    # cropping if needed:
    img_padded = img_padded[ center[1] - boxsize_y/2 : center[1] + boxsize_y/2 + 1 ,  center[0] - boxsize_x/2 : center[0] + boxsize_x/2 + 1, :]

    return img_padded, pad

# again copy-paste
# maps: The maps as returned directly by net.forward(); in matlab, this has dimensions  Width x Height x Channels x Num .
#       Or is Num left out already? I think it must be W x H x C already.
#       In numpy this is Num x Channels x Height x Widht.
def resizeIntoScaledImg_CPMs(maps, pad):     # map was "score"

    # why transpose?
    # maps
    num_feats = maps.size[1] - 1
    if len(maps.shape) == 3:
        channel_axis = 0
        maps = np.transpose(map, (0, 2, 1))  # map is the maps as returned directly by net.forward, so has shape ...
    else:
        raise
        #assert(len(maps.shape) == 4)
        #channel_axis = 1
        #maps = np.transpose(map, (0, 1, 3, 2))
#
#     if pad[0] < 0:
#         padup = np.stack((np.zeros((num_feats, maps.shape[1], -pad[0])), np.ones(1, maps.shape[1], -pad[0]) ), axis=0)        #  maps.shape[2] == width now; dims here are: C x W x H
#         maps =
#
#
#
#     if(pad(1) < 0)
#         padup = cat(3, zeros(-pad(1), size(score,2), np), ones(-pad(1), size(score,2), 1));
#         score = [padup; score]; % pad up
#     else
#         score(1:pad(1),:,:) = []; % crop up
#
#     if(pad(2) < 0)
#         padleft = cat(3, zeros(size(score,1), -pad(2), np), ones(size(score,1), -pad(2), 1));
#         score = [padleft score]; % pad left
#     else
#         score(:,1:pad(2),:) = []; % crop left
#     end
#
#     if(pad(3) < 0)
#         paddown = cat(3, zeros(-pad(3), size(score,2), np), ones(-pad(3), size(score,2), 1));
#         score = [score; paddown]; % pad down
#     else
#         score(end-pad(3)+1:end, :, :) = []; % crop down
#
#             if(pad(4) < 0)
#         padright = cat(3, zeros(size(score,1), -pad(4), np), ones(size(score,1), -pad(4), 1));
#         score = [score padright]; % pad right
#     else
#         score(:,end-pad(4)+1:end, :) = []; % crop right
#     end
# score = permute(score, [2 1 3]);
#
#     return map




def cutToWithinBounds(X, Y, width, height):
    if X >= width :
        X = width - 1
    else:
        if (X <= 0):
            X = 0
    if(Y >= height):
        Y = height - 1
    else:
        if(Y <= 0):
            Y = 0
    return X, Y



#left, upper: x,y values of the left upper corner of the part to cut out
def cutAndSaveImage(imfile, left, upper, width, height, outfilename):
    imge = Ie.open(imfile)
    imwidth, imheight = imge.size
    assert((imwidth >= width) & (imheight >= height))
    crop = cropImage(left, upper, width, height, imge)
    #save the slice
    crop.save(os.path.join(outfilename))



# to do: implement padding numbers with zeros. but only if needed.
def cut_images(prefix, path, first_num, last_num, step, left, upper, width, height ):

    imflist = [path+str(imnum)+".jpg" for imnum in range(first_num,last_num+1,step)]
    outflist = [path+ prefix + str(imnum)+".jpg" for imnum in range(first_num,last_num+1,step)]
    for i in range(len(imflist)):
        cutAndSaveImage(imflist[i], left, upper, width, height, outflist[i])
        print("saved cut image to file ", outflist[i])




    # overwrites image!
    # scaling_x, scaling_y: scale up points by this before drawing them on the image
def drawPointsOnImage(pointlist, image, scaling_x=1.0, scaling_y=1.0, thickness=-1):
    import cv2
    xvals = [int(float(p[0])*scaling_x) for p in pointlist]
    yvals = [int(float(p[1])*scaling_y) for p in pointlist]
    colors = utils.giveMeNColors(len(pointlist))
    for pidx in range(len(pointlist)):
        x = xvals[pidx]
        y = yvals[pidx]
        cv2.circle(image, (x, y), radius=3, color=colors[pidx], thickness=thickness)
    return image

    # input:
    #   pointmat:   (N x nPoints x 2) matrix with point x-y values
    #   imagelist:  list of length N with images
    #   indexes:    index list with those indexes for which to draw points on images
def drawPointsIndexed(pointmat, imagelist, indexes, scale_x=1.0, scale_y=1.0,thickness=-1):
    for ldx in indexes:
        drawPointsOnImage(pointmat[ldx,:,:], imagelist[ldx], scaling_x=scale_x, scaling_y=scale_y, thickness=thickness)   # lets see...
    return

# toppaths: a list of topdirectories, same length as imagefilelist
def drawPointsIndexed_fromFiles(pointmat, imagefilelist, toppaths, indexes, scale_x=1.0, scale_y=1.0):
    import cv2
    ims = []
    for l, ldx in enumerate(indexes):
        img = cv2.imread(os.path.join(toppaths[l],imagefilelist[ldx].replace("\\", "/")))
        drawPointsOnImage(pointmat[ldx,:,:], img, scaling_x=scale_x, scaling_y=scale_y)
        ims.append(img)
    return ims



# topdirfunc: a function like...  lambda imnum: getBatchTopdir(getBatchNr(imnum, batchSizes))
def imageSizesAsMat(imnames, topdirfunc):
    import cv2
    sizemat = np.zeros((len(imnames),2))
    for imdx in range(len(imnames)):
        imn = os.path.join(topdirfunc(imdx), imnames[imdx].replace("\\", "/"))
        im = cv2.imread(imn)
        sizemat[imdx, :] = [im.shape[1], im.shape[0]]
    return sizemat

# Would be faster with filemagic module, but I can't pip install that somehow:
#
# t = magic.from_file('teste.png')
#>>> t
#'PNG image data, 782 x 602, 8-bit/color RGBA, non-interlaced'
#>>> re.search('(\d+) x (\d+)', t).groups()
#('782', '602')


# returns a one channel image of size (w,h) consisting of a gaussian around (x,y) with stddev(?) sigma, with values
#  between 0 and scale.
# Yes it's a copy-paste of
# https://github.com/shihenw/convolutional-pose-machines-release/blob/master/testing/src/run_benchmark.m, bottom-most function.
def createGaussian(w, h, x, y, sigma, scale=1):
    X, Y = np.meshgrid(range(w), range(h))
    X = X - x
    Y = Y - y
    D2 = X**2 + Y**2
    Exponent = D2 / 2.0 / sigma / sigma
    gaussianMap = np.exp(-Exponent)*scale

    return gaussianMap


# you still see the original image; interpolates between gaussian and image.
def putGaussianOnImg(img,x,y,sigma):
    scale = np.max(img)
    gaussianmap = createGaussian(img.shape[1], img.shape[0], x, y, sigma, scale=scale)
    img_with_gaussian = np.expand_dims(gaussianmap, axis=3)*0.5 + img*0.5
    return img_with_gaussian

# As putGaussianOnImg, but instead of a gaussian map use given map (a h x w numpy array (= one channel image))
# img: h x w x c image (Or N x h x w x c, and then map has to be a (N x h x w) array.)
# doesn't modify img or map.
def putMapOnImg(img, map):

    assert(len(img.shape) == len(map.shape) + 1)
    img_loc = img.copy()
    map_loc = map.copy()
    max_value = np.max(img)
    map_max = np.max(map)
    map_min = np.min(map)
    map_loc -= map_min
    map_loc *= max_value / (map_max - map_min)

    #map_imgs_w = np.stack((map_loc*0.5 + img_loc[...,0]*0.5, img_loc[...,1], img_loc[...,2]), axis=new_axis_nr)
    new_axis_nr = len(map.shape)
    map_on_img = np.expand_dims(map_loc, axis=new_axis_nr)*0.5 + img_loc*0.5

    return map_on_img



#### for showing images: ####

def showAll(imagelist, title=""):
    import cv2
    for mdx, im in enumerate(imagelist):
        cv2.namedWindow(title+"_"+str(mdx))
        cv2.imshow(title+"_"+str(mdx), im)
        cv2.waitKey()
    return

def showIndexed(imlist, indexlist, title=""):
    for kdx in indexlist:
        cv2.namedWindow(title+"_idx_"+str(kdx))
        cv2.imshow(title+"_idx_"+str(kdx), imlist[kdx])
        cv2.waitKey()
    return

def showIndexedFromFilenames(imfilelist, indexlist, topdir="", title=""):
    for kdx in indexlist:
        img = cv2.imread(os.path.join(topdir, imfilelist[kdx].replace("\\", "/")))
        cv2.namedWindow(title)
        cv2.imshow(title, img)
        cv2.waitKey()
    return

# one prefix for all images
def storeAll(imagelist, path, name_prefix="", ext=".png"):
    for mdx, im in enumerate(imagelist):
        outname = name_prefix + "_" + str(mdx) + ext
        cv2.imwrite(os.path.join(path, outname), im)
    return

def storeImagesIndexed(imlist, indexlist, path, name_prefix, ext=".png"):
    for k,kdx in enumerate(indexlist):
        outname = name_prefix + "_" + str(k) + "__was_idx_" + str(kdx) + ext
        outname = outname.replace(" ","__").replace(":","_")
        cv2.imwrite(os.path.join(path, outname),imlist[kdx])    #todo: ist black.
   # also doesn't work:
        # cv2.imwrite(os.path.join(path, ".".join(outname.split(".")[:-1])+".jpg"),imlist[kdx])
    return

