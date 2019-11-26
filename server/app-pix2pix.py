from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import os
import time
import datetime
import logging
from matplotlib import pyplot as plt
from IPython import display

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask import Response

# instantiate the app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'

CORS(app, resources={r'/*': {'origins': '*'}}) 
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

#Load the dataset
def loadFacadesDataset():
    _URL = 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz'
    path_to_zip = tf.keras.utils.get_file('facades.tar.gz',
                                        origin=_URL,
                                        extract=True)
    PATH = os.path.join(os.path.dirname(path_to_zip), 'facades\\')
    return PATH

#Facade img has both original facade and ground truth in the same image
def loadFacadeImg(image_file):
  image = tf.io.read_file(image_file)
  image = tf.image.decode_jpeg(image)

  w = tf.shape(image)[1]
  
  #Divide the width / 2, to seperate into two image by it's width
  w = w // 2
  real_image = image[:, :w, :]
  input_image = image[:, w:, :]

  input_image = tf.cast(input_image, tf.float32)
  real_image = tf.cast(real_image, tf.float32)

  return input_image, real_image

def showTestImg(PATH):
    print(PATH)
    inp, re = loadFacadeImg(PATH+'train\\100.jpg')
    print(inp.shape)
    # casting to int for matplotlib to show the image
    plt.figure()
    plt.imshow(inp/255.0)
    plt.figure()
    plt.imshow(re/255.0)
    plt.show()

#Input Pipeline
def load_image_train(image_file):
    i,r = loadFacadeImg(image_file)
    #TODO: Jitter & Normalize transforms on the image
    return i,r

def resize(input_image, real_image, height, width):
  input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  return input_image, real_image
def normalize(input_image, real_image):
  input_image = (input_image / 127.5) - 1
  real_image = (real_image / 127.5) - 1
  return input_image, real_image

def load_image_test(image_file):
  input_image, real_image = loadFacadeImg(image_file)
  input_image, real_image = resize(input_image, real_image,
                                   IMG_HEIGHT, IMG_WIDTH)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image

def setTrainAndTest():
    # A Dataset of strings corresponding to file names.
    train_dataset = tf.data.Dataset.list_files(PATH+'train/*.jpg')
    train_dataset = train_dataset.map(load_image_train,
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.shuffle(BUFFER_SIZE)
    train_dataset = train_dataset.batch(BATCH_SIZE)

    test_dataset = tf.data.Dataset.list_files(PATH+'test/*.jpg')
    test_dataset = test_dataset.map(load_image_test)
    test_dataset = test_dataset.batch(BATCH_SIZE)
    return train_dataset, test_dataset

#Constants
BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256
OUTPUT_CHANNELS = 3

PATH = loadFacadesDataset()
#showTestImg(PATH)
train_dataset, test_dataset = setTrainAndTest()

################# GENERATOR #############################
#Returns keras.Model
#If filters=64, returns 64 channels
#If size=4 & input=256, returns 128 kernel size (?)
def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))
    result.add(tf.keras.layers.LeakyReLU())
    return result

def upsample(filters, size, apply_dropput=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))
    result.add(tf.keras.layers.ReLU())
    return result

def Generator():
    inputs = tf.keras.layers.Input(shape=[256,256,3])
    down_stack = [
        downsample(64,4,apply_batchnorm=False)
    ]
    up_stack = [
        upsample(64,4)
    ]
    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(3, 4, 
                                            strides=2, 
                                            padding='same', 
                                            kernel_initializer=initializer,
                                            activation='tanh')
    x = inputs
    # Downsampling through the model
    for down in down_stack:
        x = down(x)

    #upsampling 
    # for up in up_stack:
    #     x = up(x)

    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x)

def Discriminator():
    inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
    tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')
    x = tf.keras.layers.concatenate([inp, tar]) # (bs, 256, 256, channels*2)

    last = tf.keras.layers.Conv2D(1, 4, strides=1)(x)
    return tf.keras.Model(inputs=[inp, tar], outputs=last)

inp, re = loadFacadeImg(PATH+'train/100.jpg')
generator = Generator()
#tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)
gen_output = generator(inp[tf.newaxis,...], training=False)
print(gen_output.shape)

discriminator = Discriminator()
disc_output = discriminator([inp[tf.newaxis,...], gen_output], training=False)
print(disc_output.shape)
#plt.imshow(disc_output[0,...,-1], vmin=-20, vmax=20, cmap='RdBu_r')
#plt.colorbar()

# plt.figure()
# plt.imshow(gen_output[0,...])
# plt.figure()
# plt.imshow(inp/255.0)
#plt.show()

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

def generate_images(model, test_input, tar):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15,15))
    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()

# for example_input, example_target in test_dataset.take(1):
#   generate_images(generator, example_input, example_target)

################### Losses ###########################################
#by default interpret y_pred as logit (0-1) value
loss_objective_f = tf.keras.losses.BinaryCrossentropy(from_logits=True)

#It is a sigmoid cross entropy loss of the generated images and an array of ones.
def generator_loss(disc_generated_output, gen_o, target):
    #Why ones_like? 
    # real output should have all [1,1,...,1] since it is true and we want
    # our generated examples to look like it, so we compare the generated output values and find out how many were fake?
    #We want to minimize this loss, so that discriminator is having a hard time detecting real from fake.
    gan_loss = loss_objective_f(tf.ones_like(disc_generated_output), disc_generated_output)
    # mean absolute error
    # l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    # total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    return gan_loss

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_objective_f(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_objective_f(tf.ones_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss

EPOCHS = 10
log_dir="logs/"

summary_writer = tf.summary.create_file_writer(
log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

@tf.function
def train_step(input_image, target, epoch):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)
        # A Discriminator detects if given image is a real or a fake.
        # Discriminators function is to learn a D model which when given the input image and a generated output image from a generator gives an output of (1 or 0) for each image. 
        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gan_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
    #Change gradients for both models based on losses
    generator_gradients = gen_tape.gradient(gan_loss,
                                          generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)
    with summary_writer.as_default():
        #tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
        tf.summary.scalar('gan_loss', gan_loss, step=epoch)
        #tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
        tf.summary.scalar('disc_loss', disc_loss, step=epoch)
    
def fit(train_ds, epochs, test_ds):
    for epoch in range(epochs):
        start = time.time()
        # for example_input, example_target in test_dataset.take(1):
        #     generate_images(generator, example_input, example_target)
        print("Epoch: ", epoch)

        # Train
        #Total dots or steps is same as the BUFFER_SIZE (400)
        for n, (input_img, target) in train_ds.enumerate():
            print('.', end='')
            train_step(input_img, target, epoch)
        print('Time per epoch {} is {} s\n'.format(epoch+1, time.time()-start))

fit(train_dataset, EPOCHS, test_dataset)
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)