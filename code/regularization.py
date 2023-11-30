# -*- coding: utf-8 -*-
"""
Created on Mon May  3 08:36:13 2021

@author: geomet2
"""
# -*- coding: utf-8 -*-
"""
Undersampled img restore based on compressed sensing lasso cvx opt
"""

# hab
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.optimize as spopt
import scipy.fft as spfft
import scipy.ndimage as spimg
import scipy.special
# les packages convex
import cvxpy as cvx
import lbfgs
from sklearn.linear_model import Lasso
import numba
from numba import prange
import ot
import time

def norm_mat(A):
    return 255*(A - A.min())/(0.0000000001 + A.max() - A.min())

# Pas foirer ici max 1024 ou wavelets

Xorig = plt.imread(r'E:/SEM-publi/data/16/16µs_S-Kα.tif')
s1,s2,s = Xorig.shape
#Xmod= 0.2989 * Xorig[:,:,0] + 0.5870*Xorig[:,:,1] + 0.1140*Xorig[:,:,2]
Xmod = np.sum(Xorig, axis=-1)
Xmod = Xmod*np.uint8(Xmod > 4)

Xmod = Xmod[0:s1,0:s2]
Xorig = norm_mat(Xmod)


# dct/idct may encapsule the "right" basis already
# could try with biorth2.4

def dct2(x):
    return spfft.dct(spfft.dct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)

def idct2(x):
    return spfft.idct(spfft.idct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)


def evaluate(x, g, step=1):
    """An in-memory evaluation callback."""

    # we want to return two things:
    # (1) the norm squared of the residuals, sum((Ax-b).^2), and
    # (2) the gradient 2*A'(Ax-b)

    # expand x columns-first
    x2 = x.reshape((nx, ny)).T

    # Ax is just the inverse 2D dct of x2
    Ax2 = idct2(x2)

    # stack columns and extract samples
    Ax = Ax2.T.flat[ri].reshape(b.shape)

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

# ny,nx,nchan = Xorig.shape
ny,nx = Xorig.shape
# # fractions of the scaled image to randomly sample at
# sample_sizes = options.sampleSizes
s = 1
# for each sample size
Z = np.zeros(Xorig.shape, dtype='uint8')
masks = np.zeros(Xorig.shape, dtype='uint8')
k = round(nx * ny * s)
ri = np.random.choice(nx * ny, k, replace=False) # random sample of indices

X = Xorig[:].squeeze()
Xm = 255 * np.ones(X.shape)
Xm.T.flat[ri] = X.T.flat[ri]
mask =  np.uint8(Xmod>8)
b = X.T.flat[ri].astype(float)
Xat2 = lbfgs.fmin_lbfgs(evaluate,x0=np.zeros(nx*ny),orthantwise_c=1,
                        line_search='wolfe')
# for i in range(25):
#     Xat2 = lbfgs.fmin_lbfgs(evaluate,x0=Xat2,orthantwise_c=8,
#                         line_search='wolfe')
    
Xat = Xat2.reshape(nx, ny).T # stack columns
Xa = idct2(Xat)
Z = Xa.astype('uint8')
plt.imshow(Z)
plt.show()

# Further cleaning using Potts-like regularization

# def runIter_SW(grid, beta, n_states = 7):
#     N = grid.shape[0]
#     # sample observed edges
#     # with probability 1 - exp(-beta)
#     p = 1 - np.exp(-beta)
#     # this a N by N by 4 (top, bottom, left right) array
#     edges = np.random.choice(a=[True, False], size=(N, N, 4), p=[p, 1-p])
#     # make sure edges are only between observations of the same class
#     # check current position with one to the ...
#     #
#     edges[np.roll(a=grid, shift=1, axis=1) != grid, 0] = False
#     #
#     edges[np.roll(a=grid, shift=-1, axis=1) != grid, 1] = False
#     #
#     edges[np.roll(a=grid, shift=1, axis=0) != grid, 2] = False
#     #
#     edges[np.roll(a=grid, shift=-1, axis=0) != grid, 3] = False
#     # randomly initialize seed for cluster
#     # intial draw
#     # xy_loc = np.array([0, 4])
#     xy_loc = np.random.randint(N, size = (1, 2))[0]
#     # initialize cluster
#     xylist = [xy_loc]
#     reached_end = False
#     i = 0
#     while (not reached_end) and (i < N*N - 1):
#       # find neighbors
#       this_x = xylist[i][0]
#       this_y = xylist[i][1]
#       neighbors = np.where(edges[this_x, this_y,:])
#       if np.any(edges[this_x, this_y,:]):
#         for neighbor in np.nditer(neighbors):
#           # print(neighbor)
#           #
#           if neighbor == 3:
#             new_x = this_x + 1
#             new_xy = np.array([new_x%N, this_y])
#             if not any((new_xy == loc).all() for loc in xylist):
#               xylist.append(new_xy)
#           #
#           elif neighbor == 2:
#             new_x = this_x - 1
#             new_xy = np.array([new_x%N, this_y])
#             if not any((new_xy == loc).all() for loc in xylist):
#               xylist.append(new_xy)
#           #
#           elif neighbor == 1:
#             new_y = this_y + 1
#             new_xy = np.array([this_x, new_y%N])
#             if not any((new_xy == loc).all() for loc in xylist):
#               xylist.append(new_xy)
#           #
#           elif neighbor == 0:
#             new_y = this_y - 1
#             new_xy = np.array([this_x, new_y%N])
#             if not any((new_xy == loc).all() for loc in xylist):
#               xylist.append(new_xy)
#           # print(xylist)
#       # print("i = ", i)
#       i += 1
#       if i == len(xylist):
#         reached_end = True
#     # resulting cluster
#     # print(xylist)
#     # flip entire cluster
#     new_state = np.int64(np.random.randint(n_states))
#     while new_state == grid[this_x, this_y]:
#       new_state = np.int64(np.random.randint(n_states))
#     # all x and y indices
#     idx = [xy[0] for xy in xylist]
#     idy = [xy[1] for xy in xylist]
#     grid[idx, idy] = new_state
#     return(grid)

# init_grid = Za[7,:,:,0]
# x_denoise = np.int64(np.zeros([s1,s2]))
# for i in range(5):
#     print(i)
#     if i==0:
#         x_denoise[:,:,0] = runIter_SW(Za[7,:,:,0],2, n_states = 7)
#         x_denoise[:,:,1] = runIter_SW(Za[7,:,:,1],2, n_states = 7)
#         x_denoise[:,:,2] = runIter_SW(Za[7,:,:,2],2, n_states = 7)
#     elif i != 0:
#        x_denoise[:,:,0] = runIter_SW(x_denoise[:,:,0],3, n_states = 7)
#        x_denoise[:,:,1] = runIter_SW(x_denoise[:,:,1],3, n_states = 7)
#        x_denoise[:,:,2] = runIter_SW(x_denoise[:,:,2],3, n_states = 7)

# x_all = np.int64(x_denoise)
# plt.figure()
# plt.imshow(x_all)
# plt.show()