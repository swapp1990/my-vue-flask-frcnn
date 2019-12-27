import tensorflow as tf
print("tf version {}".format(tf.__version__))
assert tf.test.is_gpu_available()
assert tf.test.is_built_with_cuda()

import os
import time
import sys
from matplotlib import pyplot as plt

from tensorflow import keras

BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256
EPOCHS = 150
OUTPUT_CHANNELS = 3

def loadDatasetFromUrl():
    _URL = 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz'
    path_to_zip = tf.keras.utils.get_file('facades.tar.gz', origin=_URL, extract=True)
    PATH = os.path.join(os.path.dirname(path_to_zip), 'facades/')

def load(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)
    w = tf.shape(image)[1]
    w = w//2
    real_img = image[:,:w,:]
    inp_img = image[:,w:,:]

    inp_img = tf.cast(inp_img, tf.float32)
    real_img = tf.cast(real_img, tf.float32)
    return inp_img, real_img

def testLoadImg():
    PATH = "C:\\Users\Swapinl\Documents\Datasets\\facades"
    inp, re = load(PATH + '\\train\\100.jpg')
    plt.figure()
    plt.imshow(inp/255.0)
    plt.figure()
    plt.imshow(re/255.0)
    plt.show()

def resize(i,r,h,w):
    i = tf.image.resize(i, [h,w], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    r = tf.image.resize(r, [h,w], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return i,r

def random_crop(i,r):
    #Converts a list of (x,y,3) => to (n,x,y,3). Basically converts a list into a tensor with extra dim which is the length of the list
    stacked_img = tf.stack([i,r], axis=0)
    #random crop image to a size of (given size), first value (2) is the batches of images to do random crop.
    cropped_img = tf.image.random_crop(stacked_img, size=[2,IMG_HEIGHT, IMG_WIDTH, 3])
    return cropped_img[0], cropped_img[1]

#normalize images to [-1,1]
def normalize(i,r):
    i = (i/127.5) - 1
    r = (r/127.5) - 1
    return i,r

@tf.function()
def random_jitter(i,r):
    #resize to 286x286x3
    i,r = resize(i,r,286,286)
    #random cropping to 256x256x3
    i,r = random_crop(i,r)

    if tf.random.uniform(()) > 0.5:
        #random mirroring
        i = tf.image.flip_left_right(i)
        r = tf.image.flip_left_right(r)
    return i,r

def load_img_train(img_file):
    i,r = load(img_file)
    i,r = random_jitter(i,r)
    i,r = normalize(i,r)
    return i,r

def load_img_test(image_file):
    i, r = load(image_file)
    i, r = resize(i, r, IMG_HEIGHT, IMG_WIDTH)
    i, r = normalize(i, r)
    return i, r

def downsample(filters, size, apply_batchnorm=True):
    inits = tf.random_normal_initializer(0.,0.02)
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, kernel_size=size, strides=2, padding='same', kernel_initializer=inits, use_bias=False)
    )
    result.add(tf.keras.layers.LeakyReLU())
    return result

def upsample(filters, size, apply_dropout=False):
    inits = tf.random_normal_initializer(0.,0.02)
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, kernel_size=size, strides=2, padding='same', kernel_initializer=inits, use_bias=False)
    )
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    result.add(tf.keras.layers.ReLU())
    return result

def make_generator():
    inputs = tf.keras.layers.Input(shape=[256,256,3])
    down_stack = [
        #64 filters
        downsample(64, 4, apply_batchnorm=False), #bs,256,256,3 => bs,128,128,64
        downsample(128, 4), #bs,64,64,128
        downsample(256, 4), #bs,32,32,256
        downsample(512, 4), #bs,16,16,512
        downsample(512, 4), #bs,8,8,512
        downsample(512, 4), #bs,4,4,512
        downsample(512, 4), #bs,2,2,512
        downsample(512, 4), #bs,1,1,512
    ]
    up_stack = [
        upsample(512, 4, apply_dropout=True), #bs,2,2,512
        upsample(512, 4, apply_dropout=True), #bs,4,4,512
        upsample(512, 4, apply_dropout=True), #bs,8,8,512
        upsample(512, 4), #bs,16,16,512
        upsample(256, 4), #bs,32,32,256
        upsample(128, 4), #bs,64,64,128
        upsample(64, 4), #bs,128,128,64
    ]
    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4, strides=2, padding='same', kernel_initializer=initializer, activation='tanh') #bs,256,256,3

    x = inputs
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    
    skips = reversed(skips[:-1])

    for up,skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    print(model.summary())
    return model

def make_discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)
    inp = tf.keras.layers.Input(shape=[256,256,3], name='input_image')
    target = tf.keras.layers.Input(shape=[256,256,3], name='target_image')
    x = tf.keras.layers.concatenate([inp, target]) #(bs,256,256,3*2)
    down1 = downsample(64,4,apply_batchnorm=False)(x) # (bs, 128, 128, 64)
    down2 = downsample(128, 4)(down1) #(bs,64,64,128)
    down3 = downsample(256, 4)(down2) #(bs,32,32,256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) #(bs,34,34,256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer,
                                use_bias=False)(zero_pad1) #(bs,31,31,512)
    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) #(bs,33,33,512)
    #This (30,30) patch generated from the generated and real image combination classifies a 70x70 portion of the input image. (PatchGAN)
    last = tf.keras.layers.Conv2D(1,4,strides=1, kernel_initializer=initializer)(zero_pad2) #(bs,30,30,1)
    return tf.keras.models.Model(inputs=[inp, target], outputs=last)

def generate_images(model, test_inp, target, filename):
    #print(test_inp.shape)
    prediction = model(test_inp, training=True)
    #print(prediction.shape)
    plt.figure(figsize=(15,15))
    display_list = [test_inp[0], target[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']
    for i in range(3):
        plt.subplot(1,3,i+1)
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    #plt.show()
    plt.savefig(filename)
    print("saved ", filename)

def save_image(model, test_inp, target):
    prediction = model(test_inp, training=True)
    #print(prediction.shape)
    plt.figure(figsize=(5,5))
    display_list = [test_inp[0], target[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']
    for i in range(3):
        plt.subplot(1,3,i+1)
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    
generator = make_generator()
discriminator = make_discriminator()
print(discriminator.summary())
loss_objective = tf.keras.losses.BinaryCrossentropy(from_logits=True)
gen_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
disc_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

def generator_loss(disc_out, out, tar):
    LAMBDA = 100
    #Find crossentropy loss for discriminator output and tensor of ones.
    #Basically this forces the disc_out to be 1 every time, which is because, G wants to force D to return 1 when presented with an image
    #Disc_out is tensor of (30,30) size. So the loss is given for each patch of the generated image, and if a certain patch has better loss it is not changed as much as the patch which has a bigger loss (?)
    gan_loss = loss_objective(tf.ones_like(disc_out), disc_out)
    #Mean absolute error
    #Mean loss between the generated image and the target image.
    #Gets the absolute values of direct difference between pixels corr to spatial locn in both images, and finds out a mean of the difference.
    l1_loss = tf.reduce_mean(tf.abs(tar - out))
    total_gan_loss = gan_loss + (LAMBDA * l1_loss)
    return total_gan_loss, gan_loss, l1_loss

def discriminator_loss(disc_real_out, disc_gen_out):
    real_loss = loss_objective(tf.ones_like(disc_real_out), disc_real_out)
    generated_loss = loss_objective(tf.zeros_like(disc_gen_out), disc_gen_out)
    total_loss = real_loss + generated_loss
    return total_loss

@tf.function
def train_step(inp, target, epoch, n):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(inp, training=True)
        disc_real_output = discriminator([inp, target], training=True)
        disc_generated_output = discriminator([inp, gen_output], training=True) 

        #G uses the D generated output and its own generated output and target img to be generated as the loss
        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
        if n%100 == 0:
            tf.print(gen_total_loss, disc_loss, output_stream=sys.stderr)
    
    gen_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
    disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    gen_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

def fit(train_ds, epochs, test_ds):
    for e in range(epochs):
        start = time.time()

        #for egi, egr in test_ds.take(1):
            #generate_images(generator, egi, egr)
        
        print("Epoch: ", e)
        #Train
        step = 0
        for n, (input_img,target) in train_ds.enumerate():
            print('.', end='')
            step += 1
            if(n+1)%200 == 0:
                print()
                for egi, egr in test_ds.take(1):
                    filename = "testimgs-pix2pix\exp4\plt_"+str(e)+"_"+str(step)+".png"
                    generate_images(generator, egi, egr, filename)
            train_step(input_img,target,e, n)
        print()

def init():
    PATH = "C:\\Users\Swapinl\Documents\Datasets\\facades"
    #testLoadImg()
    train_dataset = tf.data.Dataset.list_files(PATH+'\\train\*.jpg')
    #Maps with a function that passes the file names from the line above
    #The experimental value is to pick and process this images faster using parallel calls, and autotune it based on machine (?)
    train_dataset = train_dataset.map(load_img_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.shuffle(BUFFER_SIZE)
    train_dataset = train_dataset.batch(BATCH_SIZE)

    test_dataset = tf.data.Dataset.list_files(PATH+'\\test\*.jpg')
    test_dataset = test_dataset.map(load_img_test, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.batch(BATCH_SIZE)

    fit(train_dataset, EPOCHS, test_dataset)
######################################################################################
if __name__ == "__main__":
    #loadDatasetFromUrl()
    init()