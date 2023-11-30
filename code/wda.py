# -*- coding: utf-8 -*-
"""
Created on Wed May 26 10:56:28 2021

@author: geomet2
"""
import tensorflow as tf

import numpy as np
from skimage import io
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
import tensorflow_addons as tfa
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers, activations, initializers, constraints, Sequential
from keras import backend as K
from keras.constraints import UnitNorm, Constraint
from sklearn.decomposition import FastICA,NMF, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from sklearn.cluster import MiniBatchKMeans, KMeans, DBSCAN
import hdbscan
from joblib import Memory
from sklearn.metrics import silhouette_samples, silhouette_score
from skimage.morphology import reconstruction
import morph
import sklearn
import skimage
from skimage.morphology import disk
from pyclustering.cluster import cluster_visualizer;
from pyclustering.cluster.cure import cure;


def norm_mat(A):
    return (A - A.min())/(0.0000000001 + A.max() - A.min())
    

imgca = io.imread(r'E:/SEM-publi/data/16/16µs_Ca-Kα.tif')
img2 = np.sum(imgca, axis=-1)
img2[img2==np.min(img2)]=0
img_4ca = img2/(3*255) #*2
#img_4ca = (img_4ca - img_4ca.min())/(0.0000000001 + img_4ca.max() - img_4ca.min())


imgcu = io.imread(r'E:/SEM-publi/data/16/16µs_Cu-Kα.tif')
img2 = np.sum(imgcu, axis=-1)
img2[img2==np.min(img2)]=0
img_4cu = img2/(3*255) #*4
#img_4cu = (img_4cu - img_4cu.min())/(0.0000000001 + img_4cu.max() - img_4cu.min())

imgfe = io.imread(r'E:/SEM-publi/data/16/16µs_Fe-Kα.tif')
img2 = np.sum(imgfe, axis=-1)
img2[img2==np.min(img2)]=0
img_4fe = img2/(3*255) #*16

imgmg = io.imread(r'E:/SEM-publi/data/16/16µs_Mg-K.tif')
img2 = np.sum(imgmg, axis=-1)
img2[img2==np.min(img2)]=0
img_4mg = img2/(3*255) #*32

imgs = io.imread(r'E:/SEM-publi/data/16/16µs_S-Kα.tif')
img2 = np.sum(imgs, axis=-1)
img2[img2==np.min(img2)]=0
img_4s = img2/(3*255) #*64

imgs = io.imread(r'E:/SEM-publi/data/16/16µs_O-Kα.tif')
img2 = np.sum(imgs, axis=-1)
img2[img2==np.min(img2)]=0
#	img3 = img2*np.uint8(img2 > 8)
img_4o = img2/(3*255) #*128


img_corr = io.imread(r'E:/SEM-publi/data/16/16µs_Ch 1.tif')
img_bse = img_corr[:,:,0]
from skimage import data
from skimage import color, morphology

img_bse_b = norm_mat(img_bse)
img_bse_b = img_bse_b*np.uint8(img_bse_b>0.2)
# selem1 =  morphology.disk(1)
# res1 = morphology.white_tophat(img_bse_b, selem1)
# img_bse_b = img_bse_b - res1
# selem2 =  morphology.disk(1)
# res2 = morphology.white_tophat(img_bse_b, selem2)
# img_bse_b = img_bse_b - res2

img_4comb = np.array([img_4ca,img_4cu,img_4fe,img_4mg,img_4s,img_4o, img_bse_b])
img_4comb = np.moveaxis(img_4comb, 0, -1)
#img_4comb = np.reshape(img_4comb,[img_4comb.shape[1],img_4comb.shape[2],7])
img4_re = np.reshape(img_4comb,[img_4comb.shape[0]*img_4comb.shape[1],7])

idx = np.random.randint(786432, size=50000)

sample = img4_re
lp=sklearn.decomposition.NMF(n_components=2,
                             init='random',
                             solver='mu',
                             beta_loss=1,
                             alpha=0.4)
low_dim = lp.fit_transform(sample)
plt.figure()
plt.scatter(low_dim[:,0],low_dim[:,1],s=0.2)

# lp=sklearn.decomposition.TruncatedSVD(n_components=2,algorithm='randomized', n_iter=5000, random_state=None, tol=0.0001)

# low_dim = lp.fit_transform(sample)# def distance_matrix(A, C, tau=0):
# plt.figure()
# plt.scatter(low_dim[:,0],low_dim[:,1],s=0.2)
