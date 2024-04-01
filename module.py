# -*-coding:utf-8-*-
# @author: Yiqin Qiu
# @email: barryxxz6@gmail.com

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import backend as k
from tensorflow.keras import layers, initializers


class FFNv2(layers.Layer):
    def __init__(self, hidden_units, **kwargs):
        super(FFNv2, self).__init__(**kwargs)
        self.hidden_units = hidden_units
        self.linear1 = Dense(hidden_units * 2, use_bias=False, kernel_initializer=initializers.he_normal)
        self.leaky_relu = Activation(tf.nn.leaky_relu)
        self.linear2 = Dense(hidden_units * 2, use_bias=False, kernel_initializer=initializers.he_normal)
        self.linear3 = Dense(hidden_units, use_bias=False, kernel_initializer=initializers.he_normal)

    def call(self, inputs, **kwargs):
        x = self.leaky_relu(self.linear1(inputs)) * self.linear2(inputs)
        return self.linear3(x)


class GroupSqueezeWeighting(layers.Layer):
    def __init__(self, unit, **kwargs):
        super(GroupSqueezeWeighting, self).__init__(**kwargs)
        self.unit = unit
        self.avg_pool = GlobalAveragePooling1D()
        self.linear1 = Dense(units=unit, use_bias=False, kernel_initializer=initializers.he_normal)
        self.linear2 = Dense(units=unit, use_bias=False, kernel_initializer=initializers.he_normal)
        self.spatial_pool = Lambda(lambda tensor: k.mean(tensor, axis=-1, keepdims=True))
        self.conv1 = Conv1D(unit, 1, padding='same', groups=8, use_bias=False,
                            kernel_initializer=initializers.he_normal)
        self.conv2 = Conv1D(unit, 1, padding='same', groups=8, use_bias=False,
                            kernel_initializer=initializers.he_normal)
        self.sigmoid = Activation(tf.nn.sigmoid)

    def call(self, inputs, **kwargs):
        channel_att = inputs
        channel_att = self.avg_pool(self.conv1(channel_att))
        channel_att = tf.reshape(channel_att, (-1, 1, self.unit))
        channel_att = self.sigmoid(self.linear1(channel_att))
        out = inputs * channel_att
        spatial_att = out
        spatial_att = self.spatial_pool(self.conv2(spatial_att))
        spatial_att = self.sigmoid(self.linear2(spatial_att))
        return out * spatial_att
