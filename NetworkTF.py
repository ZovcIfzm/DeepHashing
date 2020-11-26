import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from sklearn.decomposition import PCA

#There are M+1 layers in the network
#activation for every layer is tanh, layers are just dense
#DATA IS ALWAYS ASSUMED TO BE IN ROW VECTORS FOR THIS WHOLE PROGRAM
#TODO: initializations of weight matrices as described in the paper

#loss hyperparameters
l1 = 1
l2 = 1
l3 = 1
alpha = 0 #only used for supervised loss

#W for the first layer is supposed to be initialized to the first [layer_1_size] eigenvectors of the covariance matrix
#this function just finds those eigenvectors where n_dims is the number to get
def initialize_W(data, n_dims):
    pca = PCA(n_dims).fit(data)
    return pca.components_.T



class DeepHash(Model):
    def __init__(self, layer_sizes, l2_, l3_, l1_, initial_W):
        super(DeepHash,self).__init__()
        self.hash_size = hash_size
        self.layer_sizes = layer_sizes
        self.l1 = l1_
        self.l2 = l2_
        self.l3 = l3_
        self.initial_W = initial_W

    def custom_reg(self,W):
        #NOTE THAT WE DO THE REVERSE OF THE PAPER WITH OUR TRANSPOSE HERE, SINCE TENSORFLOW STORES WEIGHT MATRICES TRANSPOSED
        #W is [this_layer_nodes, next_layer_nodes] in shape
        part1 = self.l2 * tf.square(tf.norm(tf.matmul(tf.transpose(W), W) - tf.eye(W.shape[1])))
        part2 = self.l3/2 * tf.square(tf.norm(W))
        return part1+part2

    def custom_bias_reg(self, b):
        return self.l3/2 * tf.square(tf.norm(b))

    def custom_W_init(self, shape, dtype=None):
        return self.initial_W

    def build(self, input_shape):
        self.fc_layers = []
        self.fc_layers.append(Dense(self.layer_sizes[0],activation='tanh',input_shape=input_shape,
                                    kernel_regularizer=self.custom_reg, bias_regularizer=self.custom_bias_reg,
                                    bias_initializer = 'ones', kernel_initializer=self.custom_W_init))
        for layer in self.layer_sizes[1:]:
            self.fc_layers.append(Dense(layer,activation='tanh',kernel_regularizer=self.custom_reg,
                                        bias_regularizer=self.custom_bias_reg, bias_initializer='ones',
                                        kernel_initializer='identity'))

    def call(self,inputs):
        h = inputs
        for layer in self.fc_layers:
            h = layer(h)
        quantized = tf.math.sign(h)
        self.add_loss(.5 * tf.square(tf.norm(quantized-h))) #both are row vectors, so it's fine
        #NOTE WE PUT THE TRANSPOSE FIRST BECAUSE WE HAVE ROW VECTORS, NOT COLUMN VECTORS HERE
        #Trace is not affected by transposing which is why we don't need another transpose on the whole thing
        self.add_loss(self.l1/(2*N) * tf.linalg.trace(tf.matmul(tf.transpose(h),h)))
        return h, quantized


#This trains a DeepHash model for the specified number of epochs given an optimizer object and training data
def train_unsupervised(model, epochs,data, optimizer):
    for i in range(epochs):
        with tf.GradientTape() as tape:
            out = model(data)
            loss = tf.reduce_sum(model.losses)
            grad = tape.gradient(loss,model.trainable_variables)
            optimizer.apply_gradients(zip(grad,model.trainable_variables))
            print(loss)

#Here pos_pairs and neg_pairs are assumed to be of shape [num_pairs, 2, data_dimension]
def train_supervised(model, epochs, pos_pairs, neg_pairs, data, optimizer):
    for i in range(epochs):
        with tf.GradientTape() as tape:
            out = model(data)
            #this is the unsupervised loss
            loss_unsup = tf.reduce_sum(model.losses)

            #Here is the supervised loss calculation as specified in the paper
            out_pos1 = model(pos_pairs[:,0])
            out_pos2 = model(pos_pairs[:,1])
            out_neg1 = model(neg_pairs[:,0])
            out_neg2 = model(neg_pairs[:,1])
            pos_diff = out_pos1-out_pos2
            neg_diff = out_neg1 - out_neg2
            N_pos = tf.shape(pos_pairs)[0]
            N_neg = tf.shape(neg_pairs)[0]
            sigma_W = 1/N_pos * tf.linalg.trace(tf.matmul(tf.transpose(pos_diff),pos_diff))
            sigma_B = 1 / N_neg * tf.linalg.trace(tf.matmul(tf.transpose(neg_diff), neg_diff))
            loss_sup = loss_unsup + alpha * tf.linalg.trace(sigma_W-sigma_B)
            loss = loss_unsup + loss_sup

            #taking the gradient and applying the optimizer
            grad = tape.gradient(loss,model.trainable_variables)
            optimizer.apply_gradients(zip(grad,model.trainable_variables))
            print(loss)


d = 100
N = 1000
hash_size = 20
sample_data = np.random.random(size=[N,d])

model = DeepHash([64,32,20],1,.001,.001, initialize_W(sample_data,64))
y = model(sample_data)
print(y[0].shape)
print(y[1].shape)
opt = tf.keras.optimizers.Adam(.001)
train_unsupervised(model,50,sample_data,opt)



