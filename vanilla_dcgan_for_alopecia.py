import random
import os
import time
# import glob
from glob import glob

import IPython.display as display

import tensorflow as tf
from tensorflow.keras import layers
import matplotlib as plt
import numpy as np
import PIL
from PIL import Image
import imageio


#model parameters
batch_size = 64
epochs = 6000
latent_depth = 100
image_shape = [100,100]
nb_channels = 3
lr = 1e-4

#fixed random seed
seed = random.seed(30)

## model save
experiment_id = "train_1"
model_save_path = os.path.join("./results",experiment_id)
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)
checkpoint_dir = os.path.join(model_save_path,'training_checkpoints')



#data_path
data_path = './datasets/alopecia/'
alopecia_severity = os.listdir(data_path)

images_path = []
# for severity in alopecia_severity:
#     images_path += glob(str(data_path + severity + '/' + '*.jpg'))

filenames = os.listdir(data_path)
test=[]
for filename in filenames:
    full_filenames = os.path.join(data_path, filename)
    full_filename = os.listdir(full_filenames)
    for full in full_filename:
        images_path = full_filenames + '/' + full
        test.append(images_path)

images_path = test








images_count =len(images_path)
print(images_path)
@tf.function
def preprocessing_data(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [image_shape[0],image_shape[1]])
    image = image / 255.0
    return image

def dataloader(paths):
    dataset = tf.data.Dataset.from_tensor_slices(paths)
    dataset = dataset.shuffle(10 * batch_size)
    dataset = dataset.map(preprocessing_data)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)
    return dataset

dataset = dataloader(images_path)
for batch in dataset.take(1):
    for img in batch:
        img_np = img.numpy()


image_list = []
for filename in images_path:
    im = Image.open(filename)
    im = im.resize((image_shape[0],image_shape[1]))
    image_list.append(im)


def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(25*25*128, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Reshape((25,25,128)))
    assert model.output_shape == (None, 25, 25, 128)

    model.add(layers.Conv2DTranspose(128, (5,5), strides=(1,1), padding='same', use_bias=False))
    assert model.output_shape == (None, 25, 25, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv2DTranspose(64, (5,5), strides=(2,2), padding='same', use_bias=False))
    assert  model.output_shape == (None, 50, 50, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv2DTranspose(3, (5,5), strides=(2,2), padding='same', use_bias=False))
    assert  model.output_shape == (None, 100,100,3)
    model.summary()
    return model

generator = make_generator_model()
noise = tf.random.normal([1, latent_depth])
generated_image = generator(noise, training=True)


def make_discriminator_model():
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

discriminator = make_discriminator_model()
decision = discriminator(generated_image)
print(decision)

generator_optimizer = tf.keras.optimizers.Adam(lr=lr)
discriminator_optimizer = tf.keras.optimizers.Adam(lr=lr)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def summary(name_data_dict,
            step=None,
            types=['mean', 'std', 'max', 'min', 'sparsity', 'histogram'],
            historgram_buckets=None,
            name='summary'):
    """Summary.
    Examples
    --------
    >>> summary({'a': data_a, 'b': data_b})
    """
    def _summary(name, data):
        if data.shape == ():
            tf.summary.scalar(name, data, step=step)
        else:
            if 'mean' in types:
                tf.summary.scalar(name + '-mean', tf.math.reduce_mean(data), step=step)
            if 'std' in types:
                tf.summary.scalar(name + '-std', tf.math.reduce_std(data), step=step)
            if 'max' in types:
                tf.summary.scalar(name + '-max', tf.math.reduce_max(data), step=step)
            if 'min' in types:
                tf.summary.scalar(name + '-min', tf.math.reduce_min(data), step=step)
            if 'sparsity' in types:
                tf.summary.scalar(name + '-sparsity', tf.math.zero_fraction(data), step=step)
            if 'histogram' in types:
                tf.summary.histogram(name, data, step=step, buckets=historgram_buckets)

    with tf.name_scope(name):
        for name, data in name_data_dict.items():
            _summary(name, data)

model_summaries_path =os.path.join(model_save_path, 'summaries', 'train')
if not os.path.exists(model_summaries_path):
    os.makedirs(model_summaries_path)

train_summary_writer = tf.summary.create_file_writer(model_summaries_path)
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

def generate_and_save_images(model, epoch):

    plt.figure(figsize=(15,10))

    for i in range(4):
        noise = tf.random.normal([1,100])
        images = model(noise, training=False)

        image = images[0, :, :, :]
        image = np.reshape(image, [100, 100, 3])

        plt.subplot(1, 4, i+1)
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        plt.title("Randomly Generated Images")

        plt.tight_layout()
        plt.savefig(os.path.join(model_save_path, 'image_at_epoch_{:02d}.png'.format(epoch)))
        plt.show()


# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([batch_size, latent_depth])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return {'Generator loss': gen_loss,
            'Discriminator loss': disc_loss}


def train(dataset, epochs):
    with train_summary_writer.as_default():
        with tf.summary.record_if(True):
            for epoch in range(epochs):
                start = time.time()

                for image_batch in dataset:
                    loss_dict = train_step(image_batch)
                summary(loss_dict, step=generator_optimizer.iterations, name='losses')
                # Save the model every 15 epochs
                if (epoch + 1) % 15 == 0:
                    checkpoint.save(file_prefix = checkpoint_prefix)
                    display.clear_output(wait=True)
                    generate_and_save_images(generator,
                                             epoch + 1)
                print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))


    # Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(generator, epochs)

train(dataset, epochs)