import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
# def generator():
#
#     model = tf.keras.Sequential()
#     ## 25*25*128: 시퀀스 오브젝트 모델에 25*25*128개의 노드를 Dense 레이어를 통해 연결(?), bias 안씀, input 데이터의 shape이 (100,0)
#     model.add(layers.Dense(25 * 25 * 128, use_bias=False,input_shape=(100,)))
#     model.add(layers.BatchNormalization())
#     model.add(layers.ReLU())
#
#     model.add(layers.Reshape((25, 25, 128)))
#     ## output_shape 이 25, 25, 128이 아닐경우 assertion error 출력
#     assert model.output_shape == (None, 25, 25, 128)
#
#     model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
#     assert model.output_shape == (None, 25, 25, 128)
#     model.add(layers.BatchNormalization())
#     model.add(layers.ReLU())
#
#     model.add(layers.Conv2DTranspose(64, (5,5), strides=(2,2), padding='same', use_bias=False))
#     assert model.output_shape == (None, 50, 50, 64)
#     model.add(layers.BatchNormalization())
#     model.add(layers.ReLU())
#
#     ## 최종적으로 50*50*64의 이미지가 나오도록 업샘플링링
#
#    return model

latent_depth = 100
lr = 0.0001

def generator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(25*25*128, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    assert model.output_shape == (None, 25,25,128)
    ## filter의 수, kernel(filter)의 size, padding = 'same'-> output_size를 input_size와 똑같이 함
    model.add(layers.Conv2DTranspose(128, (5,5), strides=(1,1), padding='same', use_bias=False))
    assert model.output_shape == (None, 25, 25, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv2DTranspose(64, (5,5),strides=(2,2),padding='same', use_bias=False ))
    assert model.output_shape == (None, 50, 50, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv2DTranspose(3, (5,5), strides=(2,2), padding='same', use_bias=False, activation='sigmoid'))
    assert model.output_shape == (None, 100, 100, 3)
    model.summary()
    return model


# g_test = generator()
# test = np.random.randint(0,225, size=100)
# test = tf.random.normal([1,100])
# g_test(test)


generator = generator()
z_vector = tf.random,normal([1, latent_depth])
generated_image = generator(z_vector, training=True)



def discriminator():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5,5), strides=(2,2), padding='same', input_shape=[100,100,3]))
    model.add(layers.ReLU(0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5,5), strides=(2,2), padding='same'))
    model.add(layers.ReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    model.summary()
    return model


##dcgan project and reshape layer?

discriminator = discriminator()
decision = discriminator(generated_image)

generator_optimizer = tf.keras.optimizers.Adam(lr=lr)
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


