# %%
import h5py
import sys

import tensorflow as tf
import sonnet as snt
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

import hbbgbb.plot as myplt
from hbbgbb import data
from hbbgbb import analysis

from hbbgbb.models import SimpleModel
import settings
import glob
from tqdm import tqdm

DATADIR = 'explore_output'
IMG_SIZE = 15

class Img_give:
    etamin=-0.1
    etamax= 0.1
    etabin=IMG_SIZE
    etawdt=np.divide((etamax-etamin),etabin)

    phimin=-0.1
    phimax=0.1
    phibin=IMG_SIZE
    phiwdt=np.divide((phimax-phimin),phibin)

    def give_pos(part_eta, part_phi, dim_1, dim_2):
        if np.isnan(part_eta):
            return False, 0, 0

        myeta= int(np.divide((part_eta - Img_give.etamin),Img_give.etawdt))
        myphi= int(np.divide((part_phi - Img_give.phimin), Img_give.phiwdt)) #Need to revisit this
        if myeta < 0 or myeta >= dim_1 or myphi < 0 or myphi >= dim_2:
            return False, 0, 0
        return True, myeta, myphi
    
    def give_img(pt_arr, eta_arr, phi_arr, elem, jet_img):
        filter_pos = np.vectorize(Img_give.give_pos)
        shape = jet_img.shape
        success, elem_eta, elem_phi = filter_pos(eta_arr[elem], phi_arr[elem], *shape)
        filter = success == True
        jet_img[elem_eta[filter], elem_phi[filter]] += pt_arr[elem][filter]
        return sum(success)
    
    def generate_images(pt_arr, eta_arr, phi_arr, shape, index_filter = []):
        size = len(index_filter)
        img_arr = np.zeros(shape = (size, *shape))
        counts = np.zeros(shape= size)
        for i in tqdm(range(0, size)):
            val = Img_give.give_img(pt_arr, eta_arr, phi_arr, index_filter[i], img_arr[i])
            counts[i] = val
        return img_arr, counts

def load_calo_data(tag='r10201', givecounts = False):
    """
    Reutrns calorimeter image data with corresponding labels in one-hot encoding fashion needed
    """
    path=glob.glob(f'{settings.datadir}/user.zhicaiz.309450.NNLOPS_nnlo_30_ggH125_bb_kt200.hbbTrain.e6281_s3126_{tag}_p4258.2020_ftag5dev.v0_output.h5/*.output.h5')[0]
    f=h5py.File(path, 'r')
    constit = f['fat_jet_constituents']
    c_pt = np.array(constit['pt'])
    c_dphi = np.array(constit['dphi'])
    c_deta = np.array(constit['deta'])
    f.close()
    
    

    feature_data = data.load_data(tag)
    #Apply feature data filters (mass > 500Gev, nconstit >2) on images
    #try to trust order is upheld
    img_data, counts = Img_give.generate_images(c_pt, c_deta, c_dphi, shape = (IMG_SIZE, IMG_SIZE), index_filter=feature_data.index)

    data.label(feature_data)
    label_types = ['label0', 'label1', 'label2']
    if givecounts:
        return img_data, feature_data[label_types], counts
    else:
        return img_data, feature_data[label_types]

# %% Engineering functions

def mylog(x):
    """
    Relu type log that cuts below 0
    """
    if x <= 0:
        return 0  
    return np.log(x)
mylog = np.vectorize(mylog)

def mypow(data, pow):
    if data == 0:
        return 0
    val =  np.power(data, pow)
    if np.isnan(val):
        return 0
    return val
mypow = np.vectorize(mypow)

def isnanzero(value):
    return np.isnan(value) or value == 0
isnanzero = np.vectorize(isnanzero)

# %%

class Feature_Eng:

    def double_log(data):
        """
        Uses a double rectified log with a +1 shift in between
        """
        val = mylog(data)
        val += 1
        val = mylog(val)
        maxval = np.max(data)
        val = np.divide(val, maxval)
        return val

    def single_log(data):
        """
        Uses a single rectified log
        """
        val = mylog(data)
        val = np.divide(val, np.nanmax(val))
        return val

    def root_pow(data, factor = 1/2): #current ones are 1/2 or 1/4
        """
        uses applies root power followed by restified log
        """
        val = mypow(data, factor)
        val = mylog(val)
        val = np.divide(val, np.max(val))
        return val

    def neg_pow(data, factor = -1):
        """
        Applies negative power on data
        """
        val = mypow(data, factor)
        #val = np.divide(val, np.max(val))
        return val

    current = double_log