# -*- coding: utf-8 -*-
"""
Created on Wed May 26 10:56:28 2021

@author: geomet2
"""
import tensorflow as tf

import numpy as np
import shampoo
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
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer, Lambda
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import Normalizer
from sklearn.metrics import silhouette_samples, silhouette_score
import scipy.ndimage.morphology
import plotly.graph_objects as go
import plotly.io as pio
#pio.renderers.default = 'svg'
pio.renderers.default = 'browser'

def norm_mat(A):
    return (A - A.min())/(0.0000000001 + A.max() - A.min())
    

img_corr = io.imread(r'E:/SEM-publi/data/8/8µs_Ch 1.tif')
img_bse_b = img_corr[:,:,0]
bmask = np.uint8(img_bse_b>85)
bmask = scipy.ndimage.morphology.binary_fill_holes(bmask)
img_bse_b = img_bse_b * bmask
img_bse_b = img_bse_b-np.min(img_bse_b)
img_bse_b  = img_bse_b / np.max(img_bse_b)

imgca = io.imread(r'E:/SEM-publi/data/8/8µs_Ca-Kα.tif')
img2 = np.sum(imgca, axis=-1)/(255)
img3 = img2*np.uint8(img2 > 0.016)
img_4ca = img3 * bmask
#img_4ca = (img_4ca - img_4ca.min())/(0.0000000001 + img_4ca.max() - img_4ca.min())


imgcu = io.imread(r'E:/SEM-publi/data/8/8µs_Cu-Kα.tif')
img2 = np.sum(imgcu, axis=-1)/(2*255)
img3 = img2*np.uint8(img2 > 0.016)
img_4cu = img3* bmask#*4
#img_4cu = (img_4cu - img_4cu.min())/(0.0000000001 + img_4cu.max() - img_4cu.min())

imgfe = io.imread(r'E:/SEM-publi/data/8/8µs_Fe-Kα.tif')
img2 = np.sum(imgfe, axis=-1)/(255)
img3 = img2*np.uint8(img2 > 0.016)
img_4fe = img3* bmask #*16

imgmg = io.imread(r'E:/SEM-publi/data/8/8µs_Mg-K.tif')
img2 = np.sum(imgmg, axis=-1)/(2*255)
img3 = img2*np.uint8(img2 > 0.016)
img_4mg = img3* bmask #*32

imgs = io.imread(r'E:/SEM-publi/data/8/8µs_S-Kα.tif')
img2 = np.sum(imgs, axis=-1)/(2*255)
img3 = img2*np.uint8(img2 > 0.016)
img_4s = img3* bmask #*64

imgo = io.imread(r'E:/SEM-publi/data/8/8µs_O-Kα.tif')
img2 = np.sum(imgo, axis=-1)/(255)
img3 = img2*np.uint8(img2 > 0.016)
img_4o = img3* bmask #*128



#img_bse_b = norm_mat(img_bse)


img_4comb = np.array([img_4ca,img_4cu,img_4fe,img_4mg,img_4s,img_4o, img_bse_b])
img_4comb = np.moveaxis(img_4comb, 0, -1)
#img_4comb = np.reshape(img_4comb,[img_4comb.shape[1],img_4comb.shape[2],7])
img4_re= np.reshape(img_4comb,[img_4comb.shape[0]*img_4comb.shape[1],7])
#img4_re = img4_re_init[~np.all(img4_re_init == 0, axis=1)]
#img4_re = (img4_re/(0.00000000001+np.sum(img4_re,axis=0)))
#img4_re = Normalizer(norm='l1').fit_transform(img4_re)
#img4_re = img4_re + 0.0001*np.random.rand(img4_re.shape[0],img4_re.shape[1])
train_data, test_data, train_labels, test_labels = train_test_split(img4_re,img4_re, test_size=0.3, random_state=171)

class VQVAELayer(Layer):
    def __init__(self, embedding_dim, num_embeddings, commitment_cost,
                 initializer='uniform', epsilon=1e-10, **kwargs):
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.initializer = initializer
        super(VQVAELayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Add embedding weights.
        self.w = self.add_weight(name='embedding',
                                  shape=(self.embedding_dim, self.num_embeddings),
                                  initializer=self.initializer,
                                  trainable=True)

        # Finalize building.
        super(VQVAELayer, self).build(input_shape)

    def call(self, x):
        # Flatten input except for last dimension.
        flat_inputs = K.reshape(x, (-1, self.embedding_dim))

        # Calculate distances of input to embedding vectors.
        distances = (K.sum(flat_inputs**2, axis=1, keepdims=True)
                     - 2 * K.dot(flat_inputs, self.w)
                     + K.sum(self.w ** 2, axis=0, keepdims=True))

        # Retrieve encoding indices.
        encoding_indices = K.argmax(-distances, axis=1)
        encodings = K.one_hot(encoding_indices, self.num_embeddings)
        encoding_indices = K.reshape(encoding_indices, K.shape(x)[:-1])
        quantized = self.quantize(encoding_indices)

        # Metrics.
        #avg_probs = K.mean(encodings, axis=0)
        #perplexity = K.exp(- K.sum(avg_probs * K.log(avg_probs + epsilon)))
        
        return quantized

    @property
    def embeddings(self):
        return self.w

    def quantize(self, encoding_indices):
        w = K.transpose(self.embeddings.read_value())
        return tf.nn.embedding_lookup(w, encoding_indices)

def vq_vae_loss_wrapper(data_variance, commitment_cost, quantized, x_inputs):
    e_latent_loss = K.mean((K.stop_gradient(quantized) - x_inputs) ** 2)
    q_latent_loss = K.mean((quantized - K.stop_gradient(x_inputs)) ** 2)
    loss = q_latent_loss + commitment_cost * e_latent_loss
    return loss

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

    def __init__(self, encoding_dim, weightage=0.95):
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
    
dataset_train = tf.data.Dataset.from_tensor_slices(train_data).shuffle(10000).batch(1024)
dataset_val = tf.data.Dataset.from_tensor_slices(test_data).shuffle(10000).batch(1024)

opt = tfa.optimizers.LAMB()
opt_y = tfa.optimizers.Yogi()
opt2 = tf.keras.optimizers.Nadam()
opt3 = tfa.optimizers.NovoGrad()
opt4 = tfa.optimizers.Lookahead(opt2)
opt5 = tfa.optimizers.SWA(opt4)
opts = shampoo.ShampooOptimizer()

# EarlyStoppingCallback.
esc = keras.callbacks.EarlyStopping(monitor='loss', min_delta=1e-4,
                                    patience=5, verbose=0, mode='auto',
                                    baseline=None, restore_best_weights=True)

def _sort_rows(matrix, num_rows):
  """Sort matrix rows by the last column.
  Args:
      matrix: a matrix of values (row,col).
      num_rows: (int) number of sorted rows to return from the matrix.
  Returns:
      Tensor (num_rows, col) of the sorted matrix top K rows.
  """
  tmatrix = tf.transpose(a=matrix, perm=[1, 0])
  sorted_tmatrix = tf.nn.top_k(tmatrix, num_rows)[0]
  return tf.transpose(a=sorted_tmatrix, perm=[1, 0])

def sliced_wasserstein(a, b, random_sampling_count=128, random_projection_dim=3):
  """Compute the approximate sliced Wasserstein distance.
  Args:
      a: (matrix) Distribution "a" of samples (row, col).
      b: (matrix) Distribution "b" of samples (row, col).
      random_sampling_count: (int) Number of random projections to average.
      random_projection_dim: (int) Dimension of the random projection space.
  Returns:
      Float containing the approximate distance between "a" and "b".
  """
  s = tf.shape(input=a)
  means = []
  for _ in range(random_sampling_count):
    # Random projection matrix.
    proj = tf.random.normal([tf.shape(input=a)[1], random_projection_dim])
    proj *= tf.math.rsqrt(
        tf.reduce_sum(input_tensor=tf.square(proj), axis=0, keepdims=True))
    # Project both distributions and sort them.
    proj_a = tf.matmul(a, proj)
    proj_b = tf.matmul(b, proj)
    proj_a = _sort_rows(proj_a, s[0])
    proj_b = _sort_rows(proj_b, s[0])
    # Pairwise Wasserstein distance.
    wdist = tf.reduce_mean(input_tensor=tf.abs(proj_a - proj_b))
    means.append(wdist)
  return tf.reduce_mean(input_tensor=means)

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

"""
## Build the encoder
"""

latent_dim = 2
act = tf.keras.activations.selu
act2 = tf.keras.activations.softmax
init = tf.keras.initializers.Orthogonal()
l1l2 = tf.keras.regularizers.l1_l2(l1=0.0001, l2=0.0)

# VQ-VAE Hyper Parameters.
embedding_dim = 2 # Length of embedding vectors.
num_embeddings = 8 # Number of embedding vectors (high value = high bottleneck capacity).
commitment_cost = 0.25 # Controls the weighting of the loss terms.

encoder_inputs = keras.Input(shape=(7,))
x0 = layers.Dense(128, use_bias=True,
                 activation=act,
                 kernel_initializer = init)(encoder_inputs)
# x1 = tfa.layers.NoisyDense(64,use_bias=True,
#                  activation=act,
#                  activity_regularizer=UncorrelatedFeaturesConstraint(64, weightage = 1.),
#                  kernel_regularizer=l1l2)(x1)
# x1= tfa.layers.PoincareNormalize()(x1)
# x1 = tfa.layers.NoisyDense(32,use_bias=True,
#                  activation=act,
#                  activity_regularizer=UncorrelatedFeaturesConstraint(32, weightage = 1.),
#                  kernel_regularizer=l1l2)(x1)
# x1= tfa.layers.PoincareNormalize()(x1)
# x1 = layers.Dense(256,use_bias=True,
#                  activation=act,
#                  activity_regularizer=UncorrelatedFeaturesConstraint(256, weightage = 1.),
#                  kernel_regularizer=l1l2)(x1)
# x1 = layers.Dense(128,use_bias=True,
#                  activation=act,
#                  activity_regularizer=UncorrelatedFeaturesConstraint(128, weightage = 1.),
#                  kernel_regularizer=l1l2)(x1)
# x1 = layers.Dense(64,use_bias=True,
#                  activation=act,
#                  activity_regularizer=UncorrelatedFeaturesConstraint(64, weightage = 1.),
#                  kernel_regularizer=l1l2)(x1)
# x1 = layers.Dense(128,use_bias=True,
#                   activation=act,
#                   kernel_initializer = init)(x0)
# x1 = layers.BatchNormalization()(x1)
# x1 = layers.Dense(128,use_bias=True,
#                   activation=act,
#                   kernel_initializer = init)(x0)
# x1 = layers.BatchNormalization()(x1)
# x1 = layers.Dense(64,use_bias=True,
#                  activation=act,
#                  kernel_initializer = init)(x1)
# x1 = layers.BatchNormalization()(x1)
x1 = layers.Dense(64,use_bias=True,
                  activation=act,
                  kernel_initializer = init)(x0)
x1 = layers.BatchNormalization()(x1)

# x1 = tfa.layers.NoisyDense(64,use_bias=True,
#                  activation=act,
#                  activity_regularizer=UncorrelatedFeaturesConstraint(64, weightage = 1.),
#                  kernel_regularizer=l1l2)(x1)
x1 = layers.Dense(latent_dim ,use_bias=True,
                  activation=act,
                 kernel_initializer = init)(x1)
x1 = layers.BatchNormalization()(x1)

z0 = VQVAELayer(latent_dim , 8, 0.25, name="vqvae")(x1)

encoder = keras.Model(encoder_inputs, [x0, x1, z0], name="encoder")
encoder.summary()
# z_mean = layers.Dense(latent_dim, name="z_mean",activation=act)(x1)
# z_log_var = layers.Dense(latent_dim, name="z_log_var",activation=act)(x1)
# z = Sampling()([z_mean, z_log_var])


"""
## Build the decoder
"""
latent_inputs = keras.Input(shape=(latent_dim,))
x1 = layers.Dense(64, use_bias=False,
                 activation=act,
                 kernel_initializer = init)(latent_inputs)
x1 = layers.BatchNormalization()(x1)
x1 = layers.Dense(128, use_bias=False,
                  activation=act,
                  kernel_initializer = init)(x1)
x1 = layers.BatchNormalization()(x1)
# x1 = layers.Dense(64, use_bias=False,
#                   activation=act,
#                   kernel_initializer = init)(x1)
# x1 = layers.BatchNormalization()(x1)
# x1 = layers.Dense(128, use_bias=False,
#                   activation=act,
#                   kernel_initializer = init)(x1)
# x1 = layers.BatchNormalization()(x1)
# x1 = layers.Dense(128, use_bias=False,
#                   activation=act,
#                   kernel_initializer = init)(x1)
# x1 = layers.BatchNormalization()(x1)
# x1 = layers.Dense(64, use_bias=False,
#                  activation=act)(x1)
# x1 = layers.Dense(128, use_bias=False,
#                  activation=act)(x1)
# x1 = layers.Dense(256, use_bias=False,
#                  activation=act)(x1)
# x1 = tfa.layers.NoisyDense(128,use_bias=False,
#                  activation=act,
#                  activity_regularizer=UncorrelatedFeaturesConstraint(128, weightage = 1.))(x1)
# x1 = tfa.layers.NoisyDense(32,use_bias=False,
#                  activation=act,
#                  activity_regularizer=UncorrelatedFeaturesConstraint(32, weightage = 1.),
#                  kernel_regularizer=l1l2)(x1)
# x1 = tfa.layers.NoisyDense(64,use_bias=False,
#                  activation=act,
#                  activity_regularizer=UncorrelatedFeaturesConstraint(64, weightage = 1.),
#                  kernel_regularizer=l1l2)(x1)
x1 = layers.Dense(7,use_bias=False,
                 activation=act,
                 kernel_initializer = init)(x1)
decoder_outputs = x1
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            x0, x1, z0 = self.encoder(data)
            reconstruction = self.decoder(z0)
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.keras.losses.logcosh(data,reconstruction)))
            
            vari = tf.math.reduce_std(x0)**2
            #kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            #kl_loss = (1 - tf.math.sqrt(tf.abs(2*tf.exp(z_log_var)/(1+tf.square(tf.exp(z_log_var)))))*tf.exp(-0.25* tf.square(z_mean)/(1+tf.square(tf.exp(z_log_var)))))
            #kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=-1))
            kl_loss = vq_vae_loss_wrapper(vari, 0.25, z0, x1)
            total_loss = reconstruction_loss + 64*kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

autoencoder = VAE(encoder, decoder)
autoencoder.compile(optimizer=opt4)
history = autoencoder.fit(dataset_train,
          epochs=500)


km = MiniBatchKMeans(n_clusters=22, init='k-means++', n_init=15,
                     init_size=50000, batch_size=4096, verbose=0, max_iter=20000)
# km = hdbscan.HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
#     gen_min_span_tree=False, leaf_size=250,core_dist_n_jobs = 12, memory = Memory('./tmp'),
#     metric='euclidean', min_cluster_size=25000, min_samples=10, p=None)
cluster = km.fit_predict(rf)
print(silhouette_score(rf,cluster,sample_size=5000))
    
mm = np.reshape(cluster,[768,1024])
plt.imshow(mm)

# fig = go.Figure(data=go.Scatter3d(x=z_mean[:,0],y=z_mean[:,1],z=z_mean[:,2]))
# fig.show()