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
    
    global_feature_list = np.asarray([])
    name_list = []
    for name in folder_list:
        glist = []

        directory = root + name + '/'

        for filename in os.listdir(directory):
            if(filename.startswith('.')):
                continue
            input_image = np.asarray(PIL.Image.open(directory+filename))
            input_image = pre_process_im(cfg, input_image)
            input_image = np.reshape(input_image,(1,3,256,128))
            feature = ext.extract(input_image)
            glist.append(feature)
        glist = np.asarray(glist).reshape(-1,2048)
        if global_feature_list.shape == (0,):
            global_feature_list = glist
        else:
            global_feature_list= np.vstack((global_feature_list, glist))

        name_list = name_list + ([name] * glist.shape[0])
    name_list = np.asarray(name_list)    
    return name_list, global_feature_list


def findPerson(filepath, ext, cfg, db_name_list, db_feature_list):
    querry_image = np.asarray(PIL.Image.open(filepath))
    querry_image = pre_process_im(cfg, querry_image)
    querry_image = np.reshape(querry_image,(1,3,256,128))
    querry_feature = ext.extract(querry_image)
    
    cos_dist = compute_dist(db_feature_list, querry_feature, type='cosine')
    cos_pred = db_name_list[np.argmax(cos_dist)]
    return cos_pred

def extract_querry_features(ext, cfg, querrypath = "/home/bmsknight/triplet/person-reid-triplet-loss-baseline/data/ourdata/Database/query3/"):
    q_list =[]
    querry_name_list = []
    for filename in sorted(os.listdir(querrypath)):
        if(filename.startswith('.')):
                continue
        querry_image = np.asarray(PIL.Image.open(querrypath+filename))
        querry_image = pre_process_im(cfg, querry_image)
        querry_image = np.reshape(querry_image,(1,3,256,128))
        querry_feature = ext.extract(querry_image)
        q_list.append(querry_feature)
        querry_name_list.append(filename[:-4].split("_"))

    querry_feature_list = np.asarray(q_list).reshape(-1,2048)
    querry_name_list = np.asarray(querry_name_list)
    return querry_name_list, querry_feature_list

def re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3):

    # The following naming, e.g. gallery_num, is different from outer scope.
    # Don't care about it.

    original_dist = np.concatenate(
      [np.concatenate([q_q_dist, q_g_dist], axis=1),
       np.concatenate([q_g_dist.T, g_g_dist], axis=1)],
      axis=0)
    original_dist = np.power(original_dist, 2).astype(np.float32)
    original_dist = np.transpose(1. * original_dist/np.max(original_dist,axis = 0))
    V = np.zeros_like(original_dist).astype(np.float32)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    query_num = q_g_dist.shape[0]
    gallery_num = q_g_dist.shape[0] + q_g_dist.shape[1]
    all_num = gallery_num

    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i,:k1+1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]
        fi = np.where(backward_k_neigh_index==i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate,:int(np.around(k1/2.))+1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,:int(np.around(k1/2.))+1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index,k_reciprocal_index))> 2./3*len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index,candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i,k_reciprocal_expansion_index])
        V[i,k_reciprocal_expansion_index] = 1.*weight/np.sum(weight)
    original_dist = original_dist[:query_num,]
    if k2 != 1:
        V_qe = np.zeros_like(V,dtype=np.float32)
        for i in range(all_num):
            V_qe[i,:] = np.mean(V[initial_rank[i,:k2],:],axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:,i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist,dtype = np.float32)


    for i in range(query_num):
        temp_min = np.zeros(shape=[1,gallery_num],dtype=np.float32)
        indNonZero = np.where(V[i,:] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0,indImages[j]] = temp_min[0,indImages[j]]+ np.minimum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])
        jaccard_dist[i] = 1-temp_min/(2.-temp_min)

    final_dist = jaccard_dist*(1-lambda_value) + original_dist*lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num,query_num:]
    return final_dist

def copy_n_frames(n,previous_frame, query_path, temp_path):
    file_list = os.listdir(query_path)
    if('.ipynb_checkpoints' in file_list):
        file_list.remove('.ipynb_checkpoints')
    total_frame_list = [int(i[:-4].split('_')[0]) for i in file_list]
    
    for k in range(len(file_list)):
        if ((total_frame_list[k] <= previous_frame+n) & (total_frame_list[k]>previous_frame)):
            os.rename(query_path + file_list[k], temp_path + file_list[k])
    return
    
def vote_for_person(reranked_list,querry_name_list,db_name_list):
    frames = list(set(querry_name_list[:,0].tolist()))
    voting_dictionary = {}
    voting_name_list = np.asarray(list(set(db_name_list)))
    no_of_candidates = voting_name_list.shape[0]
    for frame in frames:
        frame_dist_list = reranked_list[querry_name_list[:,0]== frame]
        frame_name_list = querry_name_list[querry_name_list[:,0]== frame]
        no_of_persons = frame_name_list.shape[0]
        # print(no_of_persons)
        # print(frame_name_list)
        for bb in range(min(no_of_persons,no_of_candidates)):
            min_index = np.unravel_index(np.argmin(frame_dist_list),frame_dist_list.shape)
            query_id = frame_name_list[min_index[0],1]
            person_name = db_name_list[min_index[1]]
            # print(frame,query_id,person_name)
            frame_dist_list[min_index[0],:]=1
            frame_dist_list[:,db_name_list==person_name]=1
            if (query_id not in voting_dictionary.keys()):
                voting_dictionary[query_id] = np.zeros(no_of_candidates)
            voting_dictionary[query_id] = voting_dictionary[query_id] + (voting_name_list==person_name)
            
    return voting_dictionary,voting_name_list

def find_valid_persons(voting_dictionary,voting_name_list,querry_name_list,no_of_frames):
    display_dictionary = {}
    coordinate_dictionary = {}
    centroid_dictionary = {}
    for key in voting_dictionary:
        if (np.max(voting_dictionary[key])>(no_of_frames/2)):
            person = voting_name_list[np.argmax(voting_dictionary[key])]
            display_dictionary[key] = person
            coordinates = querry_name_list[querry_name_list[:,1] == key]
            median_coordinates = np.median(coordinates[:,2:],axis=0)
            coordinate_dictionary[person] = median_coordinates
            centroid_dictionary[person] = tuple((int((median_coordinates[2] + median_coordinates[3])/2),int((median_coordinates[0] + median_coordinates[1])/2)))
    return display_dictionary, coordinate_dictionary, centroid_dictionary

            