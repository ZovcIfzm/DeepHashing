import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense,Conv2D, MaxPool2D, Flatten
from tensorflow.keras.models import Model
from sklearn.decomposition import PCA

"""
USAGE:
This file is meant to be used as a library/module which is imported into other programs in order to test the algorithm.
You can do data processing/evaluation elsewhere, this is just where raw model code is kept.
The commented out code at the bottom just tests the functions above on dummy data
"""

"""Syntax for model layer definition:
If mode is "dense", just input a list of layer sizes where the last layer size is the hash size
If mode is "conv", input a list of alternating 3-tuple and integer where the tuple is:
(num filters, filter size, stride) and the number is max pool size
At the end append one number to the list that is the hash size
"""

EPOCH_LOSS_BLOCK_SIZE = 4
CONV_ERROR = 0.01

# Original paper: https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Liong_Deep_Hashing_for_2015_CVPR_paper.pdf

# There are M+1 layers in the network
# activation for every layer is tanh, layers are just dense
# DATA IS ALWAYS ASSUMED TO BE IN ROW VECTORS FOR THIS WHOLE PROGRAM

# W for the first layer is supposed to be initialized to the first [layer_1_size] eigenvectors of the covariance matrix
# this function just finds those eigenvectors where n_dims is the number to get
def initialize_W(data, n_dims):
    pca = PCA(n_dims).fit(data)
    return pca.components_.T


class DeepHash(Model):
    def __init__(self, layer_sizes, l2_, l3_, l1_, initial_W, mode='dense'):
        super(DeepHash, self).__init__()
        self.layer_sizes = layer_sizes
        self.l1 = l1_
        self.l2 = l2_
        self.l3 = l3_
        self.initial_W = initial_W
        self.mode = mode

    def custom_reg(self, W):
        # NOTE THAT WE DO THE REVERSE OF THE PAPER WITH OUR TRANSPOSE HERE, SINCE TENSORFLOW STORES WEIGHT MATRICES TRANSPOSED
        # W is [this_layer_nodes, next_layer_nodes] in shape
        part1 = self.l2 * tf.square(tf.norm(tf.matmul(tf.transpose(W), W) - tf.eye(W.shape[1])))
        part2 = self.l3 / 2 * tf.square(tf.norm(W))
        #print(part1+part2)
        return part1 + part2

    def custom_bias_reg(self, b):
        return self.l3 / 2 * tf.square(tf.norm(b))

    def custom_W_init(self, shape, dtype=None):
        return self.initial_W

    def build(self, input_shape):
        self.fc_layers = []
        if self.mode=='dense':
            self.fc_layers.append(Dense(self.layer_sizes[0], activation='tanh', input_shape=input_shape,
                                    kernel_regularizer=self.custom_reg, bias_regularizer=self.custom_bias_reg,
                                    bias_initializer='ones', kernel_initializer=self.custom_W_init))
        elif self.mode=='conv':
            self.fc_layers.append(Conv2D(self.layer_sizes[0][0],self.layer_sizes[0][1],self.layer_sizes[0][2],activation='tanh',padding='same'))
            self.fc_layers.append(MaxPool2D(self.layer_sizes[1]))
        for n, layer in enumerate(self.layer_sizes[1:]):
            if self.mode=='dense':
                self.fc_layers.append(Dense(layer, activation='tanh', kernel_regularizer=self.custom_reg,
                                        bias_regularizer=self.custom_bias_reg, bias_initializer='ones',
                                        kernel_initializer='identity'))
            elif self.mode=='conv':
                if n==0:
                    continue
                if n==len(self.layer_sizes)-2:
                    break
                if n%2==0:
                    self.fc_layers.append(MaxPool2D(layer))
                else:
                    self.fc_layers.append(
                        Conv2D(layer[0], layer[1], layer[2], activation='tanh',padding='same'))

        if self.mode=='conv':
            self.fc_layers.append(Flatten())
            self.fc_layers.append(Dense(self.layer_sizes[-1],activation='tanh'))


    def call(self, inputs):
        h = inputs
        for layer in self.fc_layers:
            h = layer(h)
        quantized = tf.math.sign(h)
        # NOTE WE PUT THE TRANSPOSE FIRST BECAUSE WE HAVE ROW VECTORS, NOT COLUMN VECTORS HERE
        # Trace is not affected by transposing which is why we don't need another transpose on the whole thing

        h_tilde = h-tf.expand_dims(tf.reduce_mean(h,axis=0),0)

        N = tf.cast(tf.shape(inputs)[0], tf.float32)
        J2 = tf.math.negative(self.l1 / (2 * N) * tf.linalg.trace(tf.matmul(tf.transpose(h_tilde), h_tilde)))
        print(J2)
        self.add_loss(J2)

        J1 = .5 * tf.square(tf.norm(quantized - h))/N
        print(J1)
        self.add_loss(J1)  # both are row vectors, so it's fine

        return h, quantized


# This trains a DeepHash model for the specified number of epochs given an optimizer object and training data
def train_unsupervised(model, epochs, data, optimizer, conv_error):
    old_loss = 0
    epoch_loss = 0
    print("Epoch block size: ", EPOCH_LOSS_BLOCK_SIZE)
    for i in range(epochs):
        # GradientTape creates an environment that makes it easier to find gradients.
        with tf.GradientTape() as tape:
            out = model(data)
            #print(out)
            loss = tf.reduce_sum(model.losses)

            # Finds the gradient of the loss w.r.t. the trainable variables (parameters W, c)
            grad = tape.gradient(loss, model.trainable_variables)

            #print("GRAD:")
            #print(np.mean([tf.reduce_mean(g) for g in grad]))
            # Updates W^m, c^m
            optimizer.apply_gradients(zip(grad, model.trainable_variables))
            
            # Printing loss
            if i % EPOCH_LOSS_BLOCK_SIZE == 0 and i is not 0:
                print("Epoch block", i/4, "avg. loss:", epoch_loss / EPOCH_LOSS_BLOCK_SIZE)
                epoch_loss = 0
            else:
                epoch_loss += loss

            # Stop condition
            if i > 1 and abs(loss-old_loss) < conv_error and conv_error!=0:
                print("loss flattened below convergence error of", conv_error)
                return
        old_loss = loss


# Here pos_pairs and neg_pairs are assumed to be of shape [num_pairs, 2, data_dimension]
# alpha is the coefficient for the supervised loss
def train_supervised(model, epochs, pos_pairs, neg_pairs, data, optimizer, alpha, conv_error):
    old_loss = 0
    epoch_loss = 0
    print("Epoch block size: ", EPOCH_LOSS_BLOCK_SIZE)
    for i in range(epochs):
        with tf.GradientTape() as tape:
            out = model(data)
            # this is the unsupervised loss
            loss_unsup = tf.reduce_sum(model.losses)

            # Here is the supervised loss calculation as specified in the paper
            out_pos1 = model(pos_pairs[:, 0])[0]  # we only need H here, not B, which is why we take the 0 index at the end
            out_pos2 = model(pos_pairs[:, 1])[0]
            out_neg1 = model(neg_pairs[:, 0])[0]
            out_neg2 = model(neg_pairs[:, 1])[0]
            pos_diff = out_pos1 - out_pos2
            neg_diff = out_neg1 - out_neg2
            N_pos = tf.cast(tf.shape(pos_pairs)[0], tf.float32)
            N_neg = tf.cast(tf.shape(neg_pairs)[0], tf.float32)
            sigma_W = 1 / N_pos * tf.matmul(tf.transpose(pos_diff), pos_diff)
            sigma_B = 1 / N_neg * tf.matmul(tf.transpose(neg_diff), neg_diff)
            loss_sup = loss_unsup + alpha * tf.linalg.trace(sigma_W - sigma_B)
            loss = loss_unsup + loss_sup

            # taking the gradient and applying the optimizer
            grad = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grad, model.trainable_variables))
            
            # Printing loss
            if i % EPOCH_LOSS_BLOCK_SIZE == 0 and i is not 0:
                print("Epoch block", i/4, "avg. loss:", epoch_loss / EPOCH_LOSS_BLOCK_SIZE)
                epoch_loss = 0
            else:
                epoch_loss += loss

            # Stop condition
            if i > 1 and abs(loss-old_loss) < conv_error:
                print("loss flattened below convergence error of", conv_error)
                return
        old_loss = loss

# if __name__ == '__main__':
#     d = 100
#     N = 1000
#     hash_size = 20
#     sample_data = np.random.random(size=[N,d])
#     CONV_ERROR = 0.01
#
#     model = DeepHash([64,32,20],1,.001,.001, initialize_W(sample_data,64))
#     y = model(sample_data)
#     print(y[0].shape)
#     print(y[1].shape)
#     model.summary()
#     opt = tf.keras.optimizers.Adam(.001)
#     train_unsupervised(model,50,sample_data,opt, CONV_ERROR)
#     # pos_samples, neg_samples = np.random.random(size=[N,2,d]), np.random.random(size=[N,2,d])
#     # train_supervised(model,100,pos_samples,neg_samples,sample_data,opt,.01, CONV_ERROR)
#
#     #conv version
#     imd = 32
#     chan = 3
#     N = 1000
#     hash_size = 20
#     sample_data = np.random.random(size=[N,imd, imd, chan])
#     CONV_ERROR = 0.01
#
#     model = DeepHash([(16,6,1),2,(32,4,2),2,(64,4,2),2,hash_size],1,.001,.001, None,'conv')
#     y = model(sample_data)
#     print(y[0].shape)
#     print(y[1].shape)
#     model.summary()
#     opt = tf.keras.optimizers.Adam(.001)
#     train_unsupervised(model,50,sample_data,opt, CONV_ERROR)
