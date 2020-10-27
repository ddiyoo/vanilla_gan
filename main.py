import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
def generator():
    ##
    model = tf.keras.Sequential()
    ## 25*25*128: 시퀀스 오브젝트 모델에 25*25*128개의 노드를 Dense 레이어를 통해 연결(?), bias 안씀, input 데이터의 shape이 (100,0)
    model.add(layers.Dense(25 * 25 * 128, use_bias=False,input_shape=(100,)))

    ## BatchNorm 추가
    model.add(layers.BatchNormalization())
    # print(model.variables)
    ## activation function : relu
    model.add(layers.ReLU())
    print(model.variables)
    # Reshape을 왜해주지??
    model.add(layers.Reshape((25, 25, 128)))
    # print(model.variables)
    assert model.output_shape == (None, 25, 25, 128)

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 25, 25, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv2DTranspose(64, (5,5), strides=(2,2), padding='same', use_bias=False))
    assert model.output_shape == (None, 50, 50, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    return model


g_test = generator()
# test = np.random.randint(0,225, size=100)
test = tf.random.normal([1,100])
g_test(test)
