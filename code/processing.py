# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 16:34:01 2021

@author: geomet2
"""
import numpy as np
import skimage as sk
from skimage import io
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.optimize as spopt
import scipy.fftpack as spfft
import scipy.ndimage as spimg
import cvxpy as cvx
import lbfgs
# L1 min

imgca = io.imread(r'E:/SEM-publi/data/4/4µs_Ca-Kα.tif')
img2 = np.sum(imgca, axis=-1)
img3 = img2*np.uint8(img2 > 4)
plt.imshow(img3)
img_4ca = img3*2

imgcu = io.imread(r'E:/SEM-publi/data/4/4µs_Cu-Kα.tif')
img2 = np.sum(imgcu, axis=-1)
img3 = img2*np.uint8(img2 > 4)
plt.imshow(img3)
img_4cu = img3*4

imgfe = io.imread(r'E:/SEM-publi/data/4/4µs_Fe-Kα.tif')
img2 = np.sum(imgfe, axis=-1)
img3 = img2*np.uint8(img2 > 4)
plt.imshow(img3)
img_4fe = img3*16

imgmg = io.imread(r'E:/SEM-publi/data/4/4µs_Mg-K.tif')
img2 = np.sum(imgmg, axis=-1)
img3 = img2*np.uint8(img2 > 4)
plt.imshow(img3)
img_4mg = img3*32

imgs = io.imread(r'E:/SEM-publi/data/4/4µs_S-Kα.tif')
img2 = np.sum(imgs, axis=-1)
img3 = img2*np.uint8(img2 > 4)
plt.imshow(img3)
img_4s = img3*64

imgs = io.imread(r'E:/SEM-publi/data/4/4µs_O-Kα.tif')
img2 = np.sum(imgs, axis=-1)
img3 = img2*np.uint8(img2 > 4)
plt.imshow(img3)
img_4s = img3*128

img_4comb = img_4ca+img_4cu+img_4fe+img_4mg+img_4s
# img_comb = io.imread(r'E:/SEM-publi/data/4/Combinee.tif')


# L1 minimization

def dct2(x):
    """Is the discrete cosine transform."""
    return spfft.dct(spfft.dct(x.T, norm='ortho', axis=0).T,
                     norm='ortho', axis=0)


def idct2(x):
    """Is the inverse discrete cosine transform."""
    return spfft.idct(spfft.idct(x.T, norm='ortho', axis=0).T,
                      norm='ortho', axis=0)


def evaluate(x, g, step):
    """An in-memory evaluation callback."""

    # we want to return two things:
    # (1) the norm squared of the residuals, sum((Ax-b).^2), and
    # (2) the gradient 2*A'(Ax-b)

    # expand x columns-first
    x2 = x.reshape((nx, ny)).T

    # Ax is just the inverse 2D dct of x2
    Ax2 = idct2(x2)

    # stack columns and extract samples
    Ax = mask.reshape(b.shape)

    # calculate the residual Ax-b and its 2-norm squared
    Axb = Ax - b
    fx = np.sum(np.power(Axb, 2))

    # project residual vector (k x 1) onto blank image (ny x nx)
    Axb2 = np.zeros(x2.shape)
    Axb2.T.flat[ri] = Axb # fill columns-first

    # A'(Ax-b) is just the 2D dct of Axb2
    AtAxb2 = 2 * dct2(Axb2)
    AtAxb = AtAxb2.T.reshape(x.shape) # stack columns

    # copy over the gradient vector
    np.copyto(g, AtAxb)

    return fx


# fractions of the scaled image to randomly sample at
sample_sizes = (0.1, 0.01)

# read original image
# Xorig = spimg.imread('escher_waterfall.jpeg')
Xorig = img_4comb
mask = np.uint8(Xorig>0)
Z= Xorig
ri = np.random.choice(Z.shape[0] * Z.shape[1], round(1*Z.shape[0] * Z.shape[1]), replace=False)
b = Z.T.astype(float)
ny, nx = Xorig.shape

# # for each sample size
# Z = [np.zeros(Xorig.shape, dtype='uint8') for s in sample_sizes]
# masks = [np.zeros(Xorig.shape, dtype='uint8') for s in sample_sizes]
# for i, s in enumerate(sample_sizes):line_search

#     # create random sampling index vector
#     k = round(nx * ny * s)
#     # random sample of indices
#     ri = np.random.choice(nx * ny, k, replace=False)
Xat2 = lbfgs.fmin_lbfgs(evaluate,x0=img_4comb,line_search='wolfe')

# transform the output back into the spatial domain
Xat = Xat2.reshape(nx, ny).T  # stack columns
Xa = idct2(Xat)
Z[:, :] = Xa.astype('uint8')
plt.figure()
plt.imshow(Z)