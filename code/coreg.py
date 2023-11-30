## -*- coding: utf-8 -*-
"""
Created on Tue May  4 10:17:28 2021

@author: geomet2
"""
import numpy as np
from skimage import io
from skimage.morphology import square, cube # Need to add 3D example
from numpy.fft import fft2, ifft2
from numba import jit
from skimage.color import rgb2gray
import scipy as sp
import matplotlib.pylab as pl
from mpl_toolkits.mplot3d import Axes3D  # noqa
import ot
from skimage.filters.rank import mean
from skimage.morphology import disk
import ot

imgca = io.imread(r'E:/SEM-publi/data/4/4µs_Ca-Kα.tif')
img2 = np.sum(imgca, axis=-1)
img3 = img2*np.uint8(img2 > 4)
pl.imshow(img3)
img_4ca = img3/255 #*2
#img_4ca = (img_4ca - img_4ca.min())/(0.0000000001 + img_4ca.max() - img_4ca.min())


imgcu = io.imread(r'E:/SEM-publi/data/4/4µs_Cu-Kα.tif')
img2 = np.sum(imgcu, axis=-1)
img3 = img2*np.uint8(img2 > 4)
pl.imshow(img3)
img_4cu = img3/255 #*4
#img_4cu = (img_4cu - img_4cu.min())/(0.0000000001 + img_4cu.max() - img_4cu.min())

imgfe = io.imread(r'E:/SEM-publi/data/4/4µs_Fe-Kα.tif')
img2 = np.sum(imgfe, axis=-1)
img3 = img2*np.uint8(img2 > 4)
pl.imshow(img3)
img_4fe = img3/255 #*16

imgmg = io.imread(r'E:/SEM-publi/data/4/4µs_Mg-K.tif')
img2 = np.sum(imgmg, axis=-1)
img3 = img2*np.uint8(img2 > 4)
pl.imshow(img3)
img_4mg = img3/255 #*32

imgs = io.imread(r'E:/SEM-publi/data/4/4µs_S-Kα.tif')
img2 = np.sum(imgs, axis=-1)
img3 = img2*np.uint8(img2 > 4)
pl.imshow(img3)
img_4s = img3/255 #*64

imgs = io.imread(r'E:/SEM-publi/data/4/4µs_O-Kα.tif')
img2 = np.sum(imgs, axis=-1)
img3 = img2*np.uint8(img2 > 4)
pl.imshow(img3)
img_4o = img3/255 #*128

img_4comb = np.array([img_4ca,img_4cu,img_4fe,img_4mg,img_4s,img_4o])
img_4comb = (img_4comb<0.1)*0.0 + (img_4comb>0.1)*img_4comb

pl.figure()
r0 =mean(img_4o, disk(8))
r1 =mean(r0, disk(1))
r2 =mean(r1, disk(4))
r3 =mean(r2, disk(1))
r4 =mean(r3, disk(2))
r5 =mean(r3, disk(1))
rn = (1/6)*(r0+r1+r2+r3+r4+r5)
pl.imshow(rn*(rn>6))

def norm_mat(A):
    return (A - A.min())/(0.0000000001 + A.max() - A.min())

img_corr = io.imread(r'E:/SEM-publi/data/4/4µs_Ch 1.tif')
img_bse = img_corr[:,:,0]
#img_corr = rgb2gray(img_corr)
img = imgca
img2 = np.max(img,axis=-1)

    
# #@jit(nopython=True)
# def corr2(c1,c2): # Cross-correlation
#     c1r=c1.mean(axis=0)
#     c1g=c1.mean(axis=1)
#     c1b=c1.mean(axis=2)
#     c1[:,:,0] = (c1[:,:,0]-c1r.mean())/c1r.std()
#     c1[:,:,1] = (c1[:,:,1]-c1g.mean())/c1g.std()
#     c1[:,:,2] = (c1[:,:,2]-c1b.mean())/c1b.std()
#     c2r=c2.mean(axis=0)
#     c2g=c2.mean(axis=1)
#     c2b=c2.mean(axis=2)
#     c2[:,:,0] = (c2[:,:,0]-c2r.mean())/c2r.std()
#     c2[:,:,1] = (c2[:,:,1]-c2g.mean())/c2g.std()
#     c2[:,:,2] = (c2[:,:,2]-c2b.mean())/c2b.std()
#     c12r=(c1[:,:,0]*c1[:,:,0]).sum()*(c2[:,:,0]*c2[:,:,0]).sum()
#     c12g=(c1[:,:,1]*c1[:,:,1]).sum()*(c2[:,:,1]*c2[:,:,1]).sum()
#     c12b=(c1[:,:,2]*c1[:,:,2]).sum()*(c2[:,:,2]*c2[:,:,2]).sum()
#     return ((c1[:,:,0]*c2[:,:,0])/(np.sqrt(c12r))+(c1[:,:,1]*c2[:,:,1])/(np.sqrt(c12g))+(c1[:,:,2]*c2[:,:,2])/(np.sqrt(c12b)))

img4_re = np.reshape(img_4comb,[img_4comb.shape[1]*img_4comb.shape[2],6])
img_bse_vec = np.reshape(img_bse,[img_bse.shape[0]*img_bse.shape[1],1])
xt = norm_mat(img_bse_vec)
Ges = np.zeros(6)
Ges2 = np.zeros(6)


for i in range(0,6):
    xs = np.reshape(img4_re[:,i],[img4_re[:,i].shape[0],1])
    xs = norm_mat(xs)
    lambd = 1e-3
    Ges[i] = ot.wasserstein_1d(xs, xt, a=None, b=None, p=1.5)
    Ges2[i] = np.sum(np.sqrt(np.abs(xs**2-xt**2)))/xs.shape[0]
# C1 = sp.spatial.distance.cdist(img4_re, img4_re)
# C2 = sp.spatial.distance.cdist(img_corr_re, img_corr_re)

# C1 /= C1.max()
# C2 /= C2.max()

# pl.figure()
# pl.subplot(121)
# pl.imshow(C1)
# pl.subplot(122)
# pl.imshow(C2)
# pl.show()


# img3=np.zeros(imgca.shape)
# img4 = corr2(np.float32(img_4comb),np.float32(img_corr))
# img4 = (img4 - img4.min())/(0.0000000001 + img4.max() - img4.min())
#img4 = np.sin(img4)

pl.figure();pl.imshow(imgca[:,:,2]>4,origin='lower',interpolation=None)
pl.figure();pl.imshow(img4>0.85,origin='lower',interpolation=None)