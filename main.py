import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
def generator():
    ##
    model = tf.keras.Sequential()
    ## 25*25*128: 시퀀스 오브젝트 모델에 25*25*128개의 노드를 Dense 레이어를 통해 연결(?), bias 안씀, input 데이터의 shape이 (100,0)
    model.add(layers.Dense(25 * 25 * 128, use_bias=False,input_shape=(100,)))
    ## BatchNorm 추가
    model.add(layers.BatchNormalization())
    ## activation function : relu
    model.add(layers.ReLU())
    ## Reshape을 왜해주지??
    model.add(layers.Reshape((25, 25, 128)))
    assert model.output_shape == (None, 25, 25, 128)

    return model



