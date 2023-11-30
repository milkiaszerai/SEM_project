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
from keras import regularizers, activations, initializers, constraints, Sequential
from keras import backend as K
from keras.constraints import UnitNorm, Constraint
from sklearn.decomposition import FastICA,NMF, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from sklearn.cluster import MiniBatchKMeans, KMeans, DBSCAN

def norm_mat(A):
    return (A - A.min())/(0.0000000001 + A.max() - A.min())
    

imgca = io.imread(r'E:/SEM-publi/data/8/8µs_Ca-Kα.tif')
img2 = np.sum(imgca, axis=-1)
img3 = img2*np.uint8(img2 > 4)
img_4ca = img3/255 #*2
#img_4ca = (img_4ca - img_4ca.min())/(0.0000000001 + img_4ca.max() - img_4ca.min())


imgcu = io.imread(r'E:/SEM-publi/data/8/8µs_Cu-Kα.tif')
img2 = np.sum(imgcu, axis=-1)
img3 = img2*np.uint8(img2 > 4)
img_4cu = img3/255 #*4
#img_4cu = (img_4cu - img_4cu.min())/(0.0000000001 + img_4cu.max() - img_4cu.min())

imgfe = io.imread(r'E:/SEM-publi/data/8/8µs_Fe-Kα.tif')
img2 = np.sum(imgfe, axis=-1)
img3 = img2*np.uint8(img2 > 4)
img_4fe = img3/255 #*16

imgmg = io.imread(r'E:/SEM-publi/data/8/8µs_Mg-K.tif')
img2 = np.sum(imgmg, axis=-1)
img3 = img2*np.uint8(img2 > 4)
img_4mg = img3/255 #*32

imgs = io.imread(r'E:/SEM-publi/data/8/8µs_S-Kα.tif')
img2 = np.sum(imgs, axis=-1)
img3 = img2*np.uint8(img2 > 4)
img_4s = img3/255 #*64

imgs = io.imread(r'E:/SEM-publi/data/8/8µs_O-Kα.tif')
img2 = np.sum(imgs, axis=-1)
img3 = img2*np.uint8(img2 > 4)
img_4o = img3/255 #*128


img_corr = io.imread(r'E:/SEM-publi/data/8/8µs_Ch 1.tif')
img_bse = img_corr[:,:,0]
img_bse_b = norm_mat(img_bse)


img_4comb = np.array([img_4ca,img_4cu,img_4fe,img_4mg,img_4s,img_4o, img_bse_b])
img_4comb = np.moveaxis(img_4comb, 0, -1)
#img_4comb = np.reshape(img_4comb,[img_4comb.shape[1],img_4comb.shape[2],7])
img4_re = np.reshape(img_4comb,[img_4comb.shape[0]*img_4comb.shape[1],7])
img4_re = (img4_re.T/(0.00000000001+np.sum(img4_re,axis=1))).T

train_data, test_data, train_labels, test_labels = train_test_split(
    img4_re,img4_re, test_size=0.3, random_state=171)

class WeightsOrthogonalityConstraint (Constraint):
    def __init__(self, encoding_dim, weightage = 1.0, axis = 0):
        self.encoding_dim = encoding_dim
        self.weightage = weightage
        self.axis = axis
        
    def weights_orthogonality(self, w):
        if(self.axis==1):
            w = K.transpose(w)
        if(self.encoding_dim > 1):
            m = K.dot(K.transpose(w), w) - tf.eye(self.encoding_dim)
            return self.weightage * tf.math.sqrt(tf.math.reduce_sum(tf.math.square(m)))
        else:
            m = tf.math.reduce_sum(w ** 2) - 1.
            return m

    def __call__(self, w):
        return self.weights_orthogonality(w)

class UncorrelatedFeaturesConstraint (Constraint):

    def __init__(self, encoding_dim, weightage=1.0):
        self.encoding_dim = encoding_dim
        self.weightage = weightage

    def get_covariance(self, x):
        x_centered_list = []

        for i in range(self.encoding_dim):
            x_centered_list.append(x[:, i] - K.mean(x[:, i]))

        x_centered = tf.stack(x_centered_list)
        covariance = K.dot(x_centered, K.transpose(x_centered)) / \
            tf.cast(x_centered.get_shape()[0], tf.float32)

        return covariance

    # Constraint penalty
    def uncorrelated_feature(self, x):
        if(self.encoding_dim <= 1):
            return 0.0
        else:
            output = tf.math.reduce_sum(tf.math.square(
                self.covariance - tf.math.multiply(self.covariance, tf.eye(self.encoding_dim))))
            return output

    def __call__(self, x):
        self.covariance = self.get_covariance(x)
        return self.weightage * self.uncorrelated_feature(x)
    
dataset_train = tf.data.Dataset.from_tensor_slices((train_data,train_data)).shuffle(512).batch(512)
dataset_val = tf.data.Dataset.from_tensor_slices((test_data,test_data)).shuffle(512).batch(512)


opt = tfa.optimizers.LAMB()
opt2 = tf.keras.optimizers.Nadam()
# opt = tfa.optimizers.Lookahead(opt)
# opt = tfa.optimizers.MovingAverage(opt)

class Autoenc(Model):
  def __init__(self):
    super(Autoenc, self).__init__()
    self.encoder = tf.keras.Sequential([
      tf.keras.layers.Dense(7,kernel_initializer='lecun_normal', use_bias=True,
                 activation='selu'),
      tf.keras.layers.GaussianNoise(0.1),
      tf.keras.layers.Dense(8,kernel_initializer='lecun_normal', use_bias=True,
                 activation='selu'),    
      tf.keras.layers.Dense(4,kernel_initializer='lecun_normal', use_bias=True,
                 activation='selu'),
      tf.keras.layers.Dense(2,kernel_initializer='lecun_normal', use_bias=True,
                 activation='selu')
      ])

    self.decoder = tf.keras.Sequential([
      tf.keras.layers.Dense(2,kernel_initializer='lecun_normal', use_bias=False,
                 activation='selu'),
      tf.keras.layers.Dense(4,kernel_initializer='lecun_normal', use_bias=False,
                 activation='selu'),
      tf.keras.layers.Dense(8,kernel_initializer='lecun_normal', use_bias=False,
                 activation='selu'),     
      tf.keras.layers.Dense(7,kernel_initializer='lecun_normal', use_bias=False,
                 activation='selu'),
      ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

class CustomMSE(tf.keras.losses.Loss):
    def __init__(self, regularization_factor=0.5, name="custom_mse"):
        super().__init__(name=name)
        self.regularization_factor = regularization_factor

    def call(self, y_true, y_pred):
        return 500*tf.keras.losses.log_cosh(y_true, y_pred)
    
autoencoder = Autoenc()
autoencoder.compile(optimizer=opt, loss= CustomMSE())
autoencoder.build(input_shape=(None,7))
autoencoder.summary()
history = autoencoder.fit(dataset_train,
          epochs=150, 
          batch_size=512,
          validation_data=dataset_val)

rf = tf.make_ndarray(tf.make_tensor_proto(autoencoder.encoder(img4_re)))
km = MiniBatchKMeans(n_clusters=9, init='k-means++', n_init=1000,
                         init_size=50000, batch_size=4096, verbose=2, max_iter=8000,
                         reassignment_ratio = 0.01)
# km = hdbscan.HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
#     gen_min_span_tree=False, leaf_size=250,core_dist_n_jobs = 12, memory = Memory('./tmp'),
#     metric='euclidean', min_cluster_size=25000, min_samples=10, p=None)
cluster = km.fit_predict(2+rf**2)
mm = np.reshape(cluster,[768,1024])
plt.imshow(mm)