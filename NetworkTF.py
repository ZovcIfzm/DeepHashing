import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.losses import Loss

#There are M+1 layers in the network
#activation for every layer is tanh, layers are just dense

#TODO: initializations of weight matrices as described in the paper
#TODO: build supervised loss

#lamba values
l1 = 1
l2 = 1
l3 = 1



class DeepHash(Model):
    def __init__(self, layer_sizes, l2_, l3_, l1_):
        super(DeepHash,self).__init__()
        self.hash_size = hash_size
        self.layer_sizes = layer_sizes
        self.l1 = l1_
        self.l2 = l2_
        self.l3 = l3_

    def custom_reg(self,W):
        part1 = self.l2 * tf.square(tf.norm(tf.matmul(W, tf.transpose(W)) - tf.eye(W.shape[0])))
        part2 = self.l3/2 * tf.square(tf.norm(W))
        return part1+part2

    def custom_bias_reg(self, b):
        return self.l3/2 * tf.square(tf.norm(b))

    def build(self, input_shape):
        self.fc_layers = []
        self.fc_layers.append(Dense(self.layer_sizes[0],activation='tanh',input_shape=input_shape,
                                    kernel_regularizer=self.custom_reg, bias_regularizer=self.custom_bias_reg))
        for layer in self.layer_sizes[1:]:
            self.fc_layers.append(Dense(layer,activation='tanh',kernel_regularizer=self.custom_reg,bias_regularizer=self.custom_bias_reg))

    def call(self,inputs):
        h = inputs
        for layer in self.fc_layers:
            h = layer(h)
        quantized = tf.math.sign(h)
        self.add_loss(.5 * tf.square(tf.norm(quantized-h)))
        self.add_loss(self.l1/(2*N) * tf.linalg.trace(tf.matmul(h,tf.transpose(h))))
        return h, quantized



def train_unsupervised(model, epochs,data, optimizer):
    for i in range(epochs):
        with tf.GradientTape() as tape:
            out = model(data)
            loss = tf.reduce_sum(model.losses)
            grad = tape.gradient(loss,model.trainable_variables)
            optimizer.apply_gradients(zip(grad,model.trainable_variables))
            print(loss)


d = 100
N = 1000
hash_size = 20
sample_data = np.random.random(size=[N,d])

model = DeepHash([256,128,32],1,.001,.001)
y = model(sample_data)
print(y[0].shape)
print(y[1].shape)
opt = tf.keras.optimizers.Adam(.001)
train_unsupervised(model,5,sample_data,opt)



