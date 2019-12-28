import tensorflow as tf
print("tf version {}".format(tf.__version__))
assert tf.test.is_gpu_available()
assert tf.test.is_built_with_cuda()

import os
import time
import datetime
import sys
import logging
import statistics
from matplotlib import pyplot as plt
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask import Response
from flask_socketio import SocketIO, emit
import mpld3

from tensorflow import keras

from threads import Worker as workerCls

# instantiate the app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*", logger=True, async_mode='threading')
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
# enable CORS
CORS(app, resources={r'/*': {'origins': '*'}}) 

class Pix2Pix():
    def __init__(self):
        #Constants
        self.BUFFER_SIZE = 400 #Max number of images to train per epoch
        self.BATCH_SIZE = 1
        self.IMG_WIDTH = 256
        self.IMG_HEIGHT = 256
        self.EPOCHS = 150
        self.OUTPUT_CHANNELS = 3

        self.generator = None
        self.discriminator = None
        self.loss_objective = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.gen_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.disc_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.train_dataset = None
        self.test_dataset = None
        self.gen_lh = []
        self.mean_gen_lh = []
        self.disc_lh = []
        self.mean_disc_lh = []
        log_dir="logs/"
        self.summary_writer = tf.summary.create_file_writer(log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    def normalInit(self):
        self.makeModel()
        self.prepareData()
        self.startTraining()
    ################### Thread Methods ###################################
    def doWork(self, msg):
        print("do work pix2pix", msg)
        if msg['action'] == 'makeModel':
            self.makeModel()
        elif msg['action'] == 'prepareData':
            self.prepareData()
        elif msg['action'] == 'startTraining':
            self.startTraining()
    
    def broadcast(self, msg):
        msg["id"] = 1
        workerCls.broadcast_event(msg)

    #################### Create Model ##################################
    def makeModel(self):
        self.generator = self.make_generator()
        self.discriminator = self.make_discriminator()

    def make_generator(self):
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
        last = tf.keras.layers.Conv2DTranspose(self.OUTPUT_CHANNELS, 4, strides=2, padding='same', kernel_initializer=initializer, activation='tanh') #bs,256,256,3

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

    def make_discriminator(self):
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
        model = tf.keras.models.Model(inputs=[inp, target], outputs=last)
        print("Discriminator Model Summary ", model.summary())
        return model

    ##################### Prepare Data ##################################
    def load_img_train(self, img_file):
        i,r = load(img_file)
        i,r = random_jitter(i,r)
        i,r = normalize(i,r)
        return i,r

    def load_img_test(self, image_file):
        i, r = load(image_file)
        i, r = resize(i, r, self.IMG_HEIGHT, self.IMG_WIDTH)
        i, r = normalize(i, r)
        return i, r
    
    def prepareData(self):
        PATH = "C:\\Users\Swapinl\Documents\Datasets\\facades"
        #testLoadImg()
        self.train_dataset = tf.data.Dataset.list_files(PATH+'\\train\*.jpg')
        #Maps with a function that passes the file names from the line above
        #The experimental value is to pick and process this images faster using parallel calls, and autotune it based on machine (?)
        self.train_dataset = self.train_dataset.map(self.load_img_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.train_dataset = self.train_dataset.shuffle(self.BUFFER_SIZE)
        self.train_dataset = self.train_dataset.batch(self.BATCH_SIZE)

        self.test_dataset = tf.data.Dataset.list_files(PATH+'\\test\*.jpg')
        self.test_dataset = self.test_dataset.map(self.load_img_test, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.test_dataset = self.test_dataset.batch(self.BATCH_SIZE)

        len_train = tf.data.experimental.cardinality(self.train_dataset).numpy()
        len_test = tf.data.experimental.cardinality(self.test_dataset).numpy()
        sendLogMsg = "Prepared %d train and %d test images"%(len_train, len_test)
        print(sendLogMsg)
        self.broadcast({"log": sendLogMsg})

    ##################### Training ##################################
    
    def generator_loss(self, disc_out, out, tar):
        LAMBDA = 100
        #Find crossentropy loss for discriminator output and tensor of ones.
        #Basically this forces the disc_out to be 1 every time, which is because, G wants to force D to return 1 when presented with an image
        #Disc_out is tensor of (30,30) size. So the loss is given for each patch of the generated image, and if a certain patch has better loss it is not changed as much as the patch which has a bigger loss (?)
        gan_loss = self.loss_objective(tf.ones_like(disc_out), disc_out)
        #Mean absolute error
        #Mean loss between the generated image and the target image.
        #Gets the absolute values of direct difference between pixels corr to spatial locn in both images, and finds out a mean of the difference.
        l1_loss = tf.reduce_mean(tf.abs(tar - out))
        total_gan_loss = gan_loss + (LAMBDA * l1_loss)
        return total_gan_loss, gan_loss, l1_loss

    def discriminator_loss(self, disc_real_out, disc_gen_out):
        real_loss = self.loss_objective(tf.ones_like(disc_real_out), disc_real_out)
        generated_loss = self.loss_objective(tf.zeros_like(disc_gen_out), disc_gen_out)
        total_loss = real_loss + generated_loss
        return total_loss

    @tf.function
    def train_step(self, inp, target, epoch, n):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(inp, training=True)
            disc_real_output = self.discriminator([inp, target], training=True)
            disc_generated_output = self.discriminator([inp, gen_output], training=True) 

            #G uses the D generated output and its own generated output and target img to be generated as the loss
            gen_total_loss, gen_gan_loss, gen_l1_loss = self.generator_loss(disc_generated_output, gen_output, target)
            disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)
            # if n%100 == 0:
            #     tf.print(gen_total_loss, disc_loss, output_stream=sys.stderr)
        
        gen_gradients = gen_tape.gradient(gen_total_loss, self.generator.trainable_variables)
        disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
        self.disc_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))

        with self.summary_writer.as_default():
            tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
            tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
            tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
            tf.summary.scalar('disc_loss', disc_loss, step=epoch)
        return gen_total_loss, disc_loss

    def fit(self, train_ds, epochs, test_ds):
        const_img = None
        for egi, egr in test_ds.take(1):
            const_img = egi
        for e in range(epochs):
            start = time.time()

            #for egi, egr in test_ds.take(1):
                #generate_images(generator, egi, egr)
            
            print("Epoch: ", e)
            sendLogMsg = "Training Epoch %d/%d"%(e, self.EPOCHS)
            self.broadcast({"log": sendLogMsg, "type": "replace", "logid": "epoch"})
            #Train
            step = 0
            for n, (input_img,target) in train_ds.enumerate():
                print('.', end='')
                step += 1
                if(n+1)%200 == 0:
                    print()
                    for egi, egr in test_ds.take(1):
                        fig = self.generate_fig(self.generator, egi, egr)
                        msg = {'action': 'sendFigs', 'fig': fig}
                        self.broadcast(msg)
                    fig2 = self.generate_fig2(self.generator, const_img)
                    msg2 = {'action': 'sendFigs2', 'fig': fig2}
                    self.broadcast(msg2)
                gen_total_loss, disc_loss = self.train_step(input_img,target,e, n)
                if(n+1)%50 == 0:
                    self.calculateLossHistory(gen_total_loss, disc_loss)
                    fig = self.plotLossHistoryFig(stepGap=50)
                    msg2 = {'action': 'showGraph', 'fig': fig}
                    self.broadcast(msg2)

    def startTraining(self):
        self.fit(self.train_dataset, self.EPOCHS, self.test_dataset)

    ##################### Utils #########################################
    def calculateLossHistory(self, gen_total_loss, disc_loss):
        gen_total_loss = gen_total_loss.numpy()
        l = float("{0:.2f}".format(gen_total_loss))
        self.gen_lh.append(l)
        currMean = statistics.mean(self.gen_lh)
        self.mean_gen_lh.append(currMean)

        disc_loss = disc_loss.numpy()
        l2 = float("{0:.2f}".format(disc_loss))
        self.disc_lh.append(l2)
        self.mean_disc_lh.append(statistics.mean(self.disc_lh))

    def plotLossHistoryFig(self, stepGap=1):
        xdata = [i*stepGap for i in range(len(self.gen_lh))]
        fig = plt.figure(figsize=(8,4))
        ax1 = fig.add_subplot(121)
        ax1.plot(xdata, self.gen_lh, 'b-', xdata, self.mean_gen_lh, 'c--')
        ax2 = fig.add_subplot(122)
        ax2.plot(xdata, self.disc_lh, 'r-', xdata, self.mean_disc_lh, 'm--')
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Gen Loss')
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('Disc Loss')
        # plt.show()
        fig = mpld3.fig_to_html(fig)
        plt.close('all')
        return fig

    def generate_fig(self, model, test_inp, target):
        prediction = model(test_inp, training=True)
        my_dpi = 96
        img_size = (256*3,256)
        fig = plt.figure(figsize=(img_size[0]/my_dpi, img_size[1]/my_dpi), dpi=my_dpi)
        display_list = [test_inp[0], target[0], prediction[0]]
        title = ['Input Image', 'Ground Truth', 'Predicted Image']
        for i in range(3):
            ax = fig.add_subplot(1,3,i+1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title[i])
            ax.imshow(display_list[i] * 0.5 + 0.5, cmap='plasma')
        mp_fig = mpld3.fig_to_dict(fig)
        # plt.show()
        plt.close('all')
        return mp_fig

    def generate_fig2(self, model, test_inp):
        prediction = model(test_inp, training=True)
        my_dpi = 96
        img_size = (256*2,256)
        fig = plt.figure(figsize=(img_size[0]/my_dpi, img_size[1]/my_dpi), dpi=my_dpi)
        display_list = [test_inp[0], prediction[0]]
        title = ['Constant Input Image', 'Predicted Image']
        for i in range(2):
            ax = fig.add_subplot(1,2,i+1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title[i])
            ax.imshow(display_list[i] * 0.5 + 0.5, cmap='plasma')
        mp_fig = mpld3.fig_to_dict(fig)
        # plt.show()
        plt.close('all')
        return mp_fig

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
    cropped_img = tf.image.random_crop(stacked_img, size=[2,256, 256, 3])
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

def init():
    fit(train_dataset, EPOCHS, test_dataset)

def testInit():
    pix2pix = Pix2Pix()
    pix2pix.normalInit()
################################# Socket #############################################
@socketio.on('init')
def init(content):
    print('init')

@socketio.on('beginTraining')
def beginTraining():
    # print('train')
    pix2pix = Pix2Pix()
    thread = workerCls.Worker(0, pix2pix, socketio)
    thread.start()
    thread2 = workerCls.Worker(1, socketio=socketio)
    thread2.start()

    msg = {'id': 0, 'action': 'makeModel'}
    workerCls.broadcast_event(msg)
    msg = {'id': 0, 'action': 'prepareData'}
    workerCls.broadcast_event(msg)
    msg = {'id': 0, 'action': 'startTraining'}
    workerCls.broadcast_event(msg)

######################################################################################
if __name__ == "__main__":
    print("running socketio")
    # testInit()
    socketio.run(app)
    #loadDatasetFromUrl()
    #init()