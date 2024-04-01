# -*-coding:utf-8-*-
# @author: Yiqin Qiu
# @email: barryxxz6@gmail.com

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import initializers
from tensorflow.keras.models import Model
from module import FFNv2, GroupSqueezeWeighting


def residual_block(inputs, filters, stride=1, re_sample=False):
    x = LayerNormalization()(inputs)
    x = Activation(tf.nn.leaky_relu)(x)
    x = Conv1D(filters, 3, padding="same", strides=stride, kernel_initializer=initializers.he_normal)(x)
    x = LayerNormalization()(x)
    x = Activation(tf.nn.leaky_relu)(x)
    x = Conv1D(filters, 3, padding="same", kernel_initializer=initializers.he_normal)(x)
    if re_sample:
        short_cut = Conv1D(filters, 1, strides=stride, kernel_initializer=initializers.he_normal)(inputs)
        out = Add()([x, short_cut])
    else:
        out = Add()([x, inputs])
    return out


def STFF_SM(input_shape):
    lags = Input(shape=input_shape)

    """
    subframe-stitch module
    """
    int_lags = Lambda(lambda tensor: tf.reshape(tensor[:, :, 0:4], (-1, input_shape[0] * 4)))(lags)
    fra_lags = Lambda(lambda tensor: tensor[:, :, 4:])(lags)

    # subframe integration operation
    x = Embedding(145, 64)(int_lags)
    x = Conv1D(128, 1, padding='same', kernel_initializer=initializers.he_normal)(x)
    x = Activation(tf.nn.leaky_relu)(x)
    x = MaxPooling1D(2, 2)(x)
    x = Conv1D(64, 1, padding='same', kernel_initializer=initializers.he_normal)(x)
    x = Activation(tf.nn.leaky_relu)(x)
    x = MaxPooling1D(2, 2)(x)

    y = Embedding(6, 16)(fra_lags)
    y = TimeDistributed(Flatten())(y)

    x = Concatenate()([x, y])

    x = residual_block(x, filters=128, re_sample=True)
    x = residual_block(x, filters=128)
    x = residual_block(x, filters=128)
    x = GroupSqueezeWeighting(128)(x)

    x = residual_block(x, filters=256, re_sample=True)
    x = residual_block(x, filters=256)
    x = residual_block(x, filters=256)
    x = GroupSqueezeWeighting(256)(x)

    for _ in range(1):
        x = LayerNormalization()(x)

        x = LSTM(512, return_sequences=True, dropout=0.2)(x)
        x = LSTM(512, return_sequences=True, dropout=0.2)(x)

        x = LayerNormalization()(x)

        x = FFNv2(512)(x)
        x = Dropout(0.2)(x)

    features_avg = GlobalAveragePooling1D()(x)
    features_max = GlobalMaxPooling1D()(x)
    x = Concatenate()([features_avg, features_max])

    logit = Dense(2, activation='softmax')(x)

    return Model(inputs=lags, outputs=logit)
