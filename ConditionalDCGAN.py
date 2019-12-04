import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from google.colab import drive
drive.mount('/content/gdrive')
import pickle
from sklearn.preprocessing import OneHotEncoder
import datetime

path = '/content/gdrive/My Drive/DL_Project_Data/data.data'
with open(path, 'rb') as f:
  dataset = pickle.load(f)
images_temp = dataset['images']
alphabet = dataset['alphabets']
alphabet = np.reshape(np.array(alphabet), (234000, 1))
images = np.zeros((234000, 32, 32, 1))
for i in range(234000):
  images[i, :, :, 0] = images_temp[32*i:32*(i+1), :]
enc = OneHotEncoder(sparse=False)
alphabet_onehot = enc.fit_transform(alphabet)
train_data = images
train_labels = alphabet_onehot
ydim=26
zdim=100


def lkrelu(x, slope=0.01):
    return tf.maximum(slope * x, x)


def get_shape(tensor): # static shape
    return tensor.get_shape().as_list()


def batch_normalization(*args, **kwargs):
    with tf.name_scope('bn'):
        bn = tf.layers.batch_normalization(*args, **kwargs)
    return bn


def sample(X, y, length):
  max_len = len(X)
  idx =  np.random.randint(low=0, high=max_len, size=(length))
  return X[idx], y[idx]
  
# Define Discriminator
class Discriminator(object):
    def __init__(self, stddev=0.02):
        self.stddev = stddev

    def __call__(self, xs, ys, is_training, reuse=None):
        batch_dim = tf.shape(xs)[0]
        with tf.variable_scope('discriminator', initializer=tf.truncated_normal_initializer(stddev=self.stddev), reuse=reuse):
            with tf.variable_scope('conv1'):
                filters_1 = tf.get_variable('filters', [5, 5, 1, 32])
                conv_1 = tf.nn.conv2d(xs, filters_1, [1, 2, 2, 1], padding='SAME')

                # Adds y as a channel to conv_1 as described in ICGAN paper
                conv_1_concat_ys = tf.concat([conv_1, tf.tile(tf.reshape(ys, [-1, 1, 1, ys.get_shape()[-1]]),
                                                                [1, tf.shape(conv_1)[1], tf.shape(conv_1)[2], 1])], axis=3)
                a_1 = lkrelu(conv_1_concat_ys, slope=0.2)

            with tf.variable_scope('conv2'):
                filters_2 = tf.get_variable('filters', [5, 5, 32 + ydim, 32])
                conv_2 = tf.nn.conv2d(a_1, filters_2, [1, 2, 2, 1], padding='SAME')
                bn_2 = batch_normalization(conv_2,  center=False,
                            scale=False, training=is_training)
                a_2 = lkrelu(bn_2, slope=0.2)

            with tf.variable_scope('conv3'):
                filters_3 = tf.get_variable('filters', [5, 5, 32, 16])
                conv_3 = tf.nn.conv2d(a_2, filters_3, [1, 2, 2, 1], padding='SAME')
                bn_3 = batch_normalization(conv_3,  center=False,
                                            scale=False, training=is_training)
                a_3 = lkrelu(bn_3, slope=0.2)

            with tf.variable_scope('output'):
                  a_h = tf.layers.dense(tf.layers.flatten(a_3), 64)
                  a_o = tf.layers.dense(a_h, 1, activation='sigmoid')
        return a_o

# Define Generator
class Generator(object):
    def __init__(self, stddev=0.02):
        self.stddev = stddev

    def __call__(self, zs, ys, is_training):
        batch_dim = tf.shape(zs)[0]
        with tf.variable_scope('generator', initializer=tf.truncated_normal_initializer(stddev=self.stddev)):
            inputs = tf.concat([zs, ys], axis=1)
            with tf.variable_scope('volume'): 
                z_p = tf.layers.dense(inputs, 4*4*128, activation='relu')
                bn_p = tf.layers.batch_normalization(z_p, training=is_training)
                reshaped_a_p = tf.reshape(bn_p, [-1, 4, 4, 128])
            with tf.variable_scope('deconv1'): 
                deconv_1 = tf.layers.conv2d_transpose(reshaped_a_p, filters=64, kernel_size=5, strides=2, padding='same', activation='relu')
                bn_1 = tf.layers.batch_normalization(deconv_1, training=is_training)
            with tf.variable_scope('deconv2'):
                deconv_2 = tf.layers.conv2d_transpose(bn_1, filters=32, kernel_size=5, strides=2, padding='same', activation='relu')
                bn_2 = tf.layers.batch_normalization(deconv_2, training=is_training)

            with tf.variable_scope('deconv3'):
                deconv_3 = tf.layers.conv2d_transpose(bn_2, filters=16, kernel_size=5, strides=2, padding='same', activation='relu')
                bn_3 = tf.layers.batch_normalization(deconv_3, training=is_training)
                deconv_4 = tf.layers.conv2d_transpose(bn_3, filters=1, kernel_size=5, strides=1, padding='same', activation='tanh')
        return deconv_4


class CDCGAN(object):
    def __init__(self, zdim, ydim, xshape, lr=0.00005, beta1=0.5):
        self.is_training = tf.placeholder(tf.bool)

        self.zs = tf.placeholder(tf.float32, [None, zdim])
        self.g_ys = tf.placeholder(tf.float32, [None, ydim])

        self.xs = tf.placeholder(tf.float32, [None] + xshape)
        self.d_ys = tf.placeholder(tf.float32, [None, ydim])

        self.generator = Generator()
        self.discriminator = Discriminator()

        self.generator_output = self.generator(self.zs, self.g_ys, self.is_training)
        self.real_discriminator_output = self.discriminator(self.xs, self.d_ys, self.is_training)
        self.fake_discriminator_output = self.discriminator(self.generator_output, self.g_ys, self.is_training, reuse=True)

        self.generator_loss = -tf.reduce_mean(tf.log(self.fake_discriminator_output))
        self.discriminator_loss = -tf.reduce_mean(tf.log(self.real_discriminator_output) + tf.log(1.0 - self.fake_discriminator_output))

        g_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='generator')
        with tf.control_dependencies(g_update_ops):
            self.generator_train_step = tf.train.RMSPropOptimizer(lr).minimize(self.generator_loss,
                                        var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator'))
        d_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='discriminator')
        with tf.control_dependencies(d_update_ops):
            self.discriminator_train_step = tf.train.RMSPropOptimizer(lr).minimize(self.discriminator_loss,
                                            var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator'))

    def train_step(self, sess, xs, d_ys, zs, g_ys, is_training=True):
        _, dloss_curr = sess.run([self.discriminator_train_step, self.discriminator_loss],
                                    feed_dict={self.xs : xs, self.d_ys : d_ys, self.zs : zs, self.g_ys : d_ys, self.is_training : is_training})
        _, gloss_curr = sess.run([self.generator_train_step, self.generator_loss],
                                    feed_dict={self.zs : zs, self.g_ys : g_ys, self.is_training : is_training})
        return (gloss_curr, dloss_curr)

    def sample_generator(self, sess, zs, ys, is_training=True):
        return sess.run(self.generator_output, feed_dict={self.zs : zs, self.g_ys : ys, self.is_training : is_training})



batch_size = 128
epochs = 100000
draw_step = 500
model = CDCGAN(zdim, ydim, [32, 32, 1])
sess = tf.Session()
sess.run(tf.global_variables_initializer())
track_d_loss = []
track_g_loss = []
saver = tf.train.Saver()
sched = 10000
for epoch in range(epochs):
    if epoch % sched == 1 or epoch == 0:
      start = datetime.datetime.now()
    batch_xs, batch_ys = sample(train_data, train_labels, batch_size)
    gloss, dloss = model.train_step(sess, np.reshape(batch_xs, [-1, 32, 32, 1]),
                    batch_ys, np.random.uniform(-1, 1, (batch_size, zdim)), batch_ys)
    track_d_loss.append(dloss)
    track_g_loss.append(gloss)
    
    if epoch % sched == 0:
        saver.save(sess, './GAN_session.saved')
        imgs = model.sample_generator(sess, zs=np.repeat(np.random.uniform(-1, 1, (26, zdim)), 26, axis=0),
                                      ys=np.tile(np.eye(ydim), [26, 1]))
        
        fig = plt.figure()
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0.1)
        for i in range(26*26):
            fig.add_subplot(26, 26, i + 1)
            plt.imshow(imgs[i, :, :, 0], cmap='gray')
            plt.axis('off')
        plt.savefig('./iter_{}.png'.format(epoch))
        plt.show()
        plt.close()
        end = datetime.datetime.now()
        print('Epoch: {}/{},\t G loss: {:.4f}, D loss: {:.4f}\t| time: {}'.
          format(epoch, epochs, gloss, dloss, end-start))
        

    
    
