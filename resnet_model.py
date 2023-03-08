import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing import image_dataset_from_directory
import pandas as pd
import matplotlib.pyplot as plt


def data_augmenter(X,input_shape):
    X = RandomZoom(0.1)(X)
    X = RandomFlip('horizontal',input_shape=input_shape)(X)
    X = RandomRotation(0.1)(X)
    return X

def first_layer(input_shape):
    
    input_layer = Input(shape=input_shape)
    rescale = Rescaling(1/255.)(input_layer)
    data_aug = data_augmenter(rescale, input_shape)
    
    l1 = Conv2D(filters=64,kernel_size=7,strides=(2,2))(data_aug)
    l1 = BatchNormalization(axis=3)(l1)
    l1 = Activation('relu')(l1)
    l1 = MaxPooling2D(pool_size=(2,2),strides=(2,2))(l1)
    return input_layer, l1

def conv_block(X, f, filters,s):
    
    F1,F2,F3 = filters
    X_shortcut = X
    
    X = Conv2D(filters=F1, 
               kernel_size=1, 
               padding='valid',
               strides=(s,s),
               kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-2, l2=1e-2))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    
    X = Conv2D(filters=F2, 
              kernel_size=f,
              padding='same',
              kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-2, l2=1e-2))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    
    X = Conv2D(filters=F3, 
              kernel_size=1,
              padding='valid',
              kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-2, l2=1e-2))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    
    X_shortcut = Conv2D(filters=F3, 
                       kernel_size=1,
                       padding='valid',
                       strides=(s,s),
                       kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-2, l2=1e-2))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3)(X_shortcut)
    X_shortcut= Activation('relu')(X_shortcut)

    X = Add()([X, X_shortcut])    
    return X
    
def resblocks(X, f, filters):
    
    F1,F2,F3 = filters
    X_shortcut = X
    
    X = Conv2D(filters=F1, 
               kernel_size=1, 
               padding='valid')(X)
    X = BatchNormalization(axis=3)(X)#this
    X = Activation('relu')(X)
    
    X = Conv2D(filters=F2, 
              kernel_size=f,
              padding='same')(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    
    X = Conv2D(filters=F3, 
              kernel_size=1,
              padding='valid')(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    
    X = Add()([X, X_shortcut])    
    return X

def last_layer(X,num_of_classes):
    X = AveragePooling2D((2,2))(X)
    X = Dropout(0.25)(X)
    dnn = Flatten()(X)
    output = Dense(num_of_classes,activation='softmax')(dnn)
    return output

def Model_ResNet50(input_shape):
    input_layer, X = first_layer(input_shape)
    
    X = conv_block(X, f = 3, filters = [64, 64, 256], s = 1)
    X = resblocks(X, 3, [64, 64, 256])
    X = resblocks(X, 3, [64, 64, 256])
    X = conv_block(X, f= 3, filters=[128,128,512], s=2)
    
    X = resblocks(X, 3, [128,128,512])
    X = resblocks(X, 3, [128,128,512])
    X = resblocks(X, 3, [128,128,512])
    
    X = conv_block(X, f=3, filters =[256,256,1024], s=2)
    X = resblocks(X, f=3, filters=[256,256,1024])
    X = resblocks(X, f=3, filters=[256,256,1024])
    X = resblocks(X, f=3, filters=[256,256,1024])
    X = resblocks(X, f=3, filters=[256,256,1024])
    X = resblocks(X, f=3, filters=[256,256,1024])
    
    X = conv_block(X, f=3, filters = [512,512,2048],s=2)
    X = resblocks(X, f=3, filters = [512,512,2048])
    X = resblocks(X, f=3, filters = [512,512,2048])
    
    output = last_layer(X, 6)
    model = tf.keras.Model(inputs = input_layer, outputs = output)
    return model
