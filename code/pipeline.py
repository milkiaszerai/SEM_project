# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 11:19:32 2021

@author: geomet2
"""

import numpy as np
from skimage import io
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
# from tensorflow.keras.models import Model
# import tensorflow_addons as tfa
# from keras import regularizers, activations, initializers, constraints, Sequential
# from keras import backend as K
# from keras.constraints import UnitNorm, Constraint
from sklearn.decomposition import FastICA,NMF, TruncatedSVD, PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from sklearn.cluster import MiniBatchKMeans, KMeans, DBSCAN
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
import scipy.ndimage.morphology
import plotly.io as pio
from numpy import genfromtxt

import scipy.cluster

import matplotlib.style
import matplotlib as mpl
from skimage.morphology import reconstruction
from skimage import morphology
from skimage.morphology import (square, rectangle, diamond, disk, cube,
                                octahedron, ball, octagon, star)

mpl.rcParams['grid.linewidth'] = 0.0
mpl.rcParams['figure.dpi'] = 100
mpl.rcParams['savefig.dpi'] = 600
mpl.rcParams['image.cmap'] = 'viridis'
mpl.rcParams['image.interpolation'] = 'none'
#pio.renderers.default = 'svg'
pio.renderers.default = 'browser'

k = 16

def norm_mat(A):
    return (A - A.min())/(0.0000000001 + A.max() - A.min())
    

img_corr = io.imread(r'E:/SEM-publi/data/%d/%dµs_Ch 1.tif'%(k,k))
img_bse_b = img_corr[:,:,0]
bmask = np.uint8(img_bse_b>85)
bmask = scipy.ndimage.morphology.binary_fill_holes(bmask)
img_bse_b = img_bse_b * bmask
img_bse_b = img_bse_b-np.min(img_bse_b)
img_bse_b  = img_bse_b / np.max(img_bse_b)

imgca = io.imread(r'E:/SEM-publi/data/%d/%dµs_Ca-Kα.tif'%(k,k))
img2 = np.sum(imgca, axis=-1)/(255)
img3 = img2*np.uint8(img2 > 0.016)
img_4ca = img3 * bmask
#img_4ca = (img_4ca - img_4ca.min())/(0.0000000001 + img_4ca.max() - img_4ca.min())


imgcu = io.imread(r'E:/SEM-publi/data/%d/%dµs_Cu-Kα.tif'%(k,k))
img2 = np.sum(imgcu, axis=-1)/(2*255)
img3 = img2*np.uint8(img2 > 0.016)
img_4cu = img3* bmask#*4
#img_4cu = (img_4cu - img_4cu.min())/(0.0000000001 + img_4cu.max() - img_4cu.min())

imgfe = io.imread(r'E:/SEM-publi/data/%d/%dµs_Fe-Kα.tif'%(k,k))
img2 = np.sum(imgfe, axis=-1)/(255)
img3 = img2*np.uint8(img2 > 0.016)
img_4fe = img3* bmask #*16

imgmg = io.imread(r'E:/SEM-publi/data/%d/%dµs_Mg-K.tif'%(k,k))
img2 = np.sum(imgmg, axis=-1)/(2*255)
img3 = img2*np.uint8(img2 > 0.016)
img_4mg = img3* bmask #*32

imgs = io.imread(r'E:/SEM-publi/data/%d/%dµs_S-Kα.tif'%(k,k))
img2 = np.sum(imgs, axis=-1)/(2*255)
img3 = img2*np.uint8(img2 > 0.016)
img_4s = img3* bmask #*64

imgo = io.imread(r'E:/SEM-publi/data/%d/%dµs_O-Kα.tif'%(k,k))
img2 = np.sum(imgo, axis=-1)/(255)
img3 = img2*np.uint8(img2 > 0.016)
img_4o = img3* bmask #*128
#img_bse_b = norm_mat(img_bse)

# Al, Na, K

# img_raw =np.array([img_4ca,img_4ti,img_4fe,img_4mg,img_4s,img_4o,img_4si])
# img_raw_sub = img_raw[:,3300:3900,4500:5300]
# img_raw_sub_lin = np.reshape(img_raw_sub,[7,img_raw_sub.shape[1]*img_raw_sub.shape[2]])


# img_nobse = np.dstack([img_4ca,img_4ti,img_4fe,img_4mg,img_4s,img_4o,img_4si])
# img_nobse= np.reshape(img_nobse,[img_nobse.shape[0]*img_nobse.shape[1],7])
# #img_nobse_init = img_nobse[~np.all(img_nobse == 0, axis=1)]

# #img_bse_b = img_bse_b[~np.all(img_bse_b == 0, axis=1)]
# img_bse_c = img_bse_b[:,3300:3900,4500:5300]
# img_bse_c= np.reshape(img_bse_c,[img_bse_c.shape[0]*img_bse_c.shape[1],1])

# img_4comb = np.concatenate([img_nobse, img_bse_b],axis=1)
# #img_4comb = np.moveaxis(img_4comb, 0, -1)
# #img_4comb = np.reshape(img_4comb,[img_4comb.shape[1],img_4comb.shape[2],7])
# #img4_re= np.reshape(img_4comb,[img_4comb.shape[0]*img_4comb.shape[1],8])
# img_4comb = np.float32(img_4comb)
# img4_re = img_4comb[~np.all(img_4comb == 0, axis=1)]

# transformer = PCA(n_components=1, random_state=0)
# pca_nobse = transformer.fit_transform(img_nobse)

# img4_re = np.concatenate([pca_nobse,img_bse_b],axis=1)


# kl = MiniBatchKMeans(n_clusters=11,n_init=1500,reassignment_ratio = 0.02, verbose = 2)
# bse_resh = np.reshape(img_bse_b,[img_bse_b.shape[0]*img_bse_b.shape[1]])
# pl = kl.fit(bse_resh.reshape(-1, 1))
# pll = np.reshape(pl.labels_,[img_bse_b.shape[0],img_bse_b.shape[1],1])
# plt.imshow(pll)
rr = k
se = 128/(2*k)
# morpho
pll =np.round(img_4ca)
pll = pll.astype(bool)
struc=disk(se)
mask = np.squeeze(np.float32(pll))
ife = np.float32(img_4fe)
pro = scipy.ndimage.grey_closing(np.squeeze(pll),structure=struc)
mm = np.squeeze(pro)
for i in range(rr):
    mm = morphology.remove_small_objects(mm,min_size=i,connectivity=8)
    mm = morphology.remove_small_holes(mm,area_threshold=i)
    #mm = morphology.closing(mm, selem=struc)

mm_ca = mm
plt.imshow(mm)
plt.savefig(r'Ca_%d.png'%(k))

pll =np.round(img_4cu)
pll = pll.astype(bool)
struc=disk(se)
mask = np.squeeze(np.float32(pll))
ife = np.float32(img_4fe)
pro = scipy.ndimage.grey_closing(np.squeeze(pll),structure=struc)
mm = np.squeeze(pro)
for i in range(rr):
    mm = morphology.remove_small_objects(mm,min_size=i,connectivity=8)
    mm = morphology.remove_small_holes(mm,area_threshold=i)
    #mm = morphology.closing(mm, selem=struc)
mm_cu = mm
plt.imshow(mm)
plt.savefig(r'Cu_%d.png'%(k))

pll =np.round(img_4mg)
pll = pll.astype(bool)
struc=disk(se)
mask = np.squeeze(np.float32(pll))
ife = np.float32(img_4fe)
pro = scipy.ndimage.grey_closing(np.squeeze(pll),structure=struc)
mm = np.squeeze(pro)
for i in range(rr):
    mm = morphology.remove_small_objects(mm,min_size=i,connectivity=8)
    mm = morphology.remove_small_holes(mm,area_threshold=i)
    #mm = morphology.closing(mm, selem=struc)
mm_mg = mm
plt.imshow(mm)
plt.savefig(r'Mg_%d.png'%(k))


pll =np.round(img_4fe)
pll = pll.astype(bool)
struc=disk(se)
mask = np.squeeze(np.float32(pll))
ife = np.float32(img_4fe)
pro = scipy.ndimage.grey_closing(np.squeeze(pll),structure=struc)
mm = np.squeeze(pro)
for i in range(rr):
    mm = morphology.remove_small_objects(mm,min_size=i,connectivity=8)
    mm = morphology.remove_small_holes(mm,area_threshold=i)
    #mm = morphology.closing(mm, selem=struc)
mm_fe = mm
plt.imshow(mm)
plt.savefig(r'Fe_%d.png'%(k))


pll =np.round(img_4s)
pll = pll.astype(bool)
struc=disk(se)
mask = np.squeeze(np.float32(pll))
ife = np.float32(img_4fe)
pro = scipy.ndimage.grey_closing(np.squeeze(pll),structure=struc)
mm = np.squeeze(pro)
for i in range(rr):
    mm = morphology.remove_small_objects(mm,min_size=i,connectivity=8)
    mm = morphology.remove_small_holes(mm,area_threshold=i)
    #mm = morphology.closing(mm, selem=struc)
mm_s = mm
plt.imshow(mm)
plt.savefig(r'S_%d.png'%(k))


pll =np.round(img_4o)
pll = pll.astype(bool)
struc=disk(se)
mask = np.squeeze(np.float32(pll))
ife = np.float32(img_4fe)
pro = scipy.ndimage.grey_closing(np.squeeze(pll),structure=struc)
mm = np.squeeze(pro)
for i in range(rr):
    mm = morphology.remove_small_objects(mm,min_size=i,connectivity=8)
    mm = morphology.remove_small_holes(mm,area_threshold=i)
    #mm = morphology.closing(mm, selem=struc)
mm_o = mm
plt.imshow(mm)
plt.savefig(r'O_%d.png'%(k))

m = bmask.astype(np.uint8)*(mm_o.astype(np.uint8) + mm_s.astype(np.uint8) + mm_ca.astype(np.uint8) + mm_cu.astype(np.uint8) + mm_fe.astype(np.uint8))
plt.imshow(m)
plt.savefig(r'Composite_%d.png'%(k))
# np.mask_0 = mask(mask==0)random.shuffle(img4_re)
# p_rnd = img4_re[0:5000,:]

# print('N_opt DBSCAN : ')
# print(np.unique(clusterer.labels_))



# def optimalK(data2, nrefs=3, maxClusters=15):
#     """
#     Calculates KMeans optimal K using Gap Statistic 
#     Params:
#         data: ndarry of shape (n_samples, n_features)
#         nrefs: number of sample reference datasets to create
#         maxClusters: Maximum number of clusters to test for
#     Returns: (gaps, optimalK)
#     """

#     gaps = np.zeros((len(range(1, maxClusters)),))
#     resultsdf = pd.DataFrame({'clusterCount':[], 'gap':[]})
#     for gap_index, k in enumerate(range(1, maxClusters)):# Holder for reference dispersion results

#         print(k)    
#         refDisps = np.zeros(nrefs)# For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
#         for i in range(nrefs):
#             print(nrefs)
#             np.random.shuffle(data2)
#             data = data2[0:5000,:]
#             # Create new random reference set
#             randomReference = np.random.random_sample(size=data.shape)
            
#             # Fit to it
#             km = KMeans(k)
#             km.fit(randomReference)
            
#         refDisp = km.inertia_
#         refDisps[i] = refDisp# Fit cluster to original data and create dispersion
#         km = KMeans(k)
#         km.fit(data)
        
#         origDisp = km.inertia_# Calculate gap statistic
#         gap = np.log(np.mean(refDisps)) - np.log(origDisp)# Assign this loop's gap statistic to gaps
#         gaps[gap_index] = gap
        
#         resultsdf = resultsdf.append({'clusterCount':k, 'gap':gap}, ignore_index=True)
#     return (gaps.argmax() + 1, resultsdf)
    
# score_g, df = optimalK(img4_re, nrefs=5, maxClusters=15)
# # donne 16 a priori
# plt.plot(df['clusterCount'], df['gap'], linestyle='--', marker='o', color='b');
# plt.xlabel('K');
# plt.ylabel('Gap Statistic');
# plt.title('Gap Statistic vs. K');

# np.random.shuffle(img4_re)
# data = img4_re[0:5000,:]

# # Other
# model = MiniBatchKMeans(batch_size=6000)
# # k is range of number of clusters.
# for i in range(25):
#     np.random.shuffle(img4_re)
#     data = img4_re[0:8000,:]
#     model = MiniBatchKMeans(batch_size=4096)
#     v1 = KElbowVisualizer(model, k=(5,15), timings= False)
#     v1.fit(data)        # Fit data to visualizer
#     plt.close()
#     print('N_opt Elbow : ')
#     print(v1.elbow_value_)      # Finalize and render figure
    
#     model = MiniBatchKMeans(batch_size=4096)
#     v2 = KElbowVisualizer(model, k=(5,15),metric='silhouette', timings= False)
#     v2.fit(data)        # Fit the data to the visualizer
#     plt.close()
#     print('N_opt Silhouette : ')
#     print(v2.elbow_value_)        # Finalize and render the figure
    
#     model = MiniBatchKMeans(batch_size=4096)
#     v3 = KElbowVisualizer(model, k=(5,15),metric='calinski_harabasz', timings= False)
#     v3.fit(data)        # Fit the data to the visualizer
#     plt.close()
#     print('N_opt calinski_harabasz : ')
#     print(v3.elbow_value_)       # Finalize and render the figure
# # donne 9-10-11

