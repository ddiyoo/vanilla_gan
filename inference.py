import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np


checkpoint_dir = './results/train_6/training_checkpoints'

latent_depth = 100
lr = 1e-4

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(25*25*128, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Reshape((25,25,128)))
    assert model.output_shape == (None, 25, 25, 128)

    model.add(layers.Conv2DTranspose(128, (5,5), strides=(1,1), padding='same',use_bias=False))
    assert model.output_shape == (None, 25, 25, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv2DTranspose(64, (5,5), strides=(2,2), padding='same', use_bias=False))
    assert model.output_shape == (None, 50, 50, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv2DTranspose(3, (5,5),strides=(2,2), padding='same', use_bias=False, activation='sigmoid'))
    assert model.output_shape == (None, 100, 100, 3)
    model.summary()
    return model

generator = make_generator_model()

generator_optimizer = tf.keras.optimizers.Adam(lr=lr)
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,generator=generator)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

def generate_and_save_images(model, latent_depth):

    plt.figure(figsize=(15,10))
    for i in range(4):

        input_ = tf.random.normal([1,latent_depth])
        images = model(input_, training=False)

        image = images[0, :, :, :]
        image = np.reshape(image, [100, 100, 3])

        plt.subplot(1, 4 ,i+1)
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        plt.title("Randomly Generated Images")

    plt.tight_layout()
    plt.savefig('Generated Images')
    plt.show()

generate_and_save_images(generator,latent_depth)