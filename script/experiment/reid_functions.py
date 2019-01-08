from __future__ import print_function

import sys
import os.path
path = os.path.dirname(os.path.dirname(os.path.abspath("/home/bmsknight/triplet/person-reid-triplet-loss-baseline/tri_loss/dataset/")))
sys.path.insert(0,path)
sys.path.insert(0, '.')

import torch
from torch.autograd import Variable
import torch.optim as optim
from torch.nn.parallel import DataParallel
import PIL
import cv2
import numpy as np
import time
import os.path as osp
from tensorboardX import SummaryWriter
import numpy as np
import argparse

from tri_loss.dataset import create_dataset
from tri_loss.model.Model import Model
from tri_loss.model.TripletLoss import TripletLoss
from tri_loss.model.loss import global_loss

from tri_loss.utils.utils import time_str
from tri_loss.utils.utils import str2bool
from tri_loss.utils.utils import tight_float_str as tfs
from tri_loss.utils.utils import may_set_mode
from tri_loss.utils.utils import load_state_dict
from tri_loss.utils.utils import load_ckpt
from tri_loss.utils.utils import save_ckpt
from tri_loss.utils.utils import set_devices
from tri_loss.utils.utils import AverageMeter
from tri_loss.utils.utils import to_scalar
from tri_loss.utils.utils import ReDirectSTD
from tri_loss.utils.utils import set_seed
from tri_loss.utils.utils import adjust_lr_exp
from tri_loss.utils.utils import adjust_lr_staircase

class Config(object):
    def __init__(self):
        self.resize_h_w = (256,128)
        self.scale_im = True
        self.im_mean = [0.486, 0.459, 0.408]
        self.im_std = [0.229, 0.224, 0.225]
        self.device_id = (0,)
        self.model_weight_file = '/home/bmsknight/triplet/person-reid-triplet-loss-baseline/data/market/model_weight.pth'
        self.margin = 0.3
        
        self.last_conv_stride = 1
        self.weight_decay = 0.0005

        # Initial learning rate
        self.base_lr = 0.0002
        
def pre_process_im(cfg, im):
    """Pre-process image.
    `im` is a numpy array with shape [H, W, 3], e.g. the result of
    matplotlib.pyplot.imread(some_im_path), or
    numpy.asarray(PIL.Image.open(some_im_path))."""

    
    # Resize.
    if (cfg.resize_h_w is not None) \
        and (cfg.resize_h_w != (im.shape[0], im.shape[1])):
      im = cv2.resize(im, cfg.resize_h_w[::-1], interpolation=cv2.INTER_LINEAR)

    # scaled by 1/255.
    if cfg.scale_im:
      im = im / 255.

    # Subtract mean and scaled by std
    # im -= np.array(self.im_mean) # This causes an error:
    # Cannot cast ufunc subtract output from dtype('float64') to
    # dtype('uint8') with casting rule 'same_kind'
    if cfg.im_mean is not None:
      im = im - np.array(cfg.im_mean)
    if cfg.im_mean is not None and cfg.im_std is not None:
      im = im / np.array(cfg.im_std).astype(float)

    # May mirror image.
    

    # The original image has dims 'HWC', transform it to 'CHW'.
    im = im.transpose(2, 0, 1)

    return im

class ExtractFeature(object):
    """A function to be called in the val/test set, to extract features.
    Args:
    TVT: A callable to transfer images to specific device.
    """

    def __init__(self, model, TVT):
        self.model = model
        self.TVT = TVT

    def extract(self, ims):
        old_train_eval_model = self.model.training
        # Set eval mode.
        # Force all BN layers to use global mean and variance, also disable
        # dropout.
        self.model.eval()
        ims = Variable(self.TVT(torch.from_numpy(ims).float()))
        feat = self.model(ims)
        feat = feat.data.cpu().numpy()
        # Restore the model to its old train/eval mode.
        self.model.train(old_train_eval_model)
        return feat
    
def normalize(nparray, order=2, axis=0):
    """Normalize a N-D numpy array along the specified axis."""
    norm = np.linalg.norm(nparray, ord=order, axis=axis, keepdims=True)
    return nparray / (norm + np.finfo(np.float32).eps)


def compute_dist(array1, array2, type='euclidean'):

    if type == 'cosine':
        array1 = normalize(array1, axis=1)
        array2 = normalize(array2, axis=1)
        dist = np.matmul(array1, array2.T)
        return dist
    else:
        # shape [m1, 1]
        square1 = np.sum(np.square(array1), axis=1)[..., np.newaxis]
        # shape [1, m2]
        square2 = np.sum(np.square(array2), axis=1)[np.newaxis, ...]
        squared_dist = - 2 * np.matmul(array1, array2.T) + square1 + square2
        squared_dist[squared_dist < 0] = 0
        dist = np.sqrt(squared_dist)
        return dist
    

def extract_DB_features(ext, cfg, root='/home/bmsknight/triplet/person-reid-triplet-loss-baseline/data/ourdata/Database/Dynamic_Database/'):
    folder_list = os.listdir(root)
    
    global_feature_list = None
    name_list = []
    for name in folder_list:
        glist = []

        directory = root + name + '/'

        for filename in os.listdir(directory):
            input_image = np.asarray(PIL.Image.open(directory+filename))
            input_image = pre_process_im(cfg, input_image)
            input_image = np.reshape(input_image,(1,3,256,128))
            feature = ext.extract(input_image)
            glist.append(feature)
        glist = np.asarray(glist).reshape(-1,2048)
        if global_feature_list == None:
            global_feature_list = glist
        else:
            global_feature_list= np.vstack((global_feature_list, glist))

        name_list = name_list + ([name] * glist.shape[0])
        
    return name_list, global_feature_list


def findPerson(filepath, ext, cfg, db_name_list, db_feature_list):
    querry_image = np.asarray(PIL.Image.open(filepath))
    querry_image = pre_process_im(cfg, querry_image)
    querry_image = np.reshape(querry_image,(1,3,256,128))
    querry_feature = ext.extract(querry_image)
    
    cos_dist = compute_dist(db_feature_list, querry_feature, type='cosine')
    cos_pred = db_name_list[np.argmax(cos_dist)]
    return cos_pred
