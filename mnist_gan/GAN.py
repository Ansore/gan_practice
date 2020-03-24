import tensorflow as tf
from configs import *

class GAN(object):

    def __init__(self):
        self.z = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 100], name='z')
        self.x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 28, 28, 1], name='real_x')

        self.fake_x = self.netG(self.z)

        # 判别器预训练时，判别器对真实数据的判别情况-未sigmod处理
        self.pre_logits  = self.netD(self.x, reuse      = False)
        # 判别器对真实数据的判别情况-未sigmod处理
        self.real_logits = self.netD(self.x, reuse      = True)
        # 判别器对伪造数据的判别情况-未sigmod处理
        self.fake_logits = self.netD(self.fake_x, reuse = True)

        # 预训练时判别器，判别器将真实数据判定未真的得分情况
        self.loss_pre_D = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.pre_logits, labels=tf.ones_like(self.pre_logits)))
        # 训练时，判别器将真实数据判定为真，将伪造数据判定为假的得分情况
        self.loss_D = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.real_logits, labels=tf.ones_like(self.real_logits))) + \
            tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_logits, labels=tf.zeros_like(self.fake_logits)))
        # 训练时，生成器伪造的数据，被判定为真实数据的得分情况
        self.loss_G = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_logits, labels=tf.ones_like(self.fake_logits)))

        # 获取生成器和判定器对应的变量地址，用于更新变量
        t_vars = tf.trainable_variables()
        self.g_vars = [var for var in t_vars if var.name.startswith('generator')]
        self.d_vars = [var for var in t_vars if var.name.startswith('discriminator')]

    def netG(self, z, alpha=0.01):
        with tf.variable_scope('generator') as scope:
            layer1 = tf.layers.dense(z, 4 * 4 * 512)
            layer1 = tf.reshape(layer1, [-1, 4, 4, 512])
            layer1 = tf.layers.batch_normalization(layer1, training=True)
            layer1 = tf.maximum(alpha * layer1, layer1)
            layer1 = tf.nn.dropout(layer1, keep_prob=0.8)

            layer2 = tf.layers.conv2d_transpose(layer1, 256, 4, strides= 1, padding= 'valid')
            layer2 = tf.layers.batch_normalization(layer2, training=True)
            layer2 = tf.maximum(alpha * layer2, layer2)
            layer2 = tf.nn.dropout(layer2, keep_prob=0.8)

            layer3 = tf.layers.conv2d_transpose(layer2, 128, 3, strides=2, padding='same')
            layer3 = tf.layers.batch_normalization(layer3, training=True)
            layer3 = tf.maximum(alpha * layer3, layer3)
            layer3 = tf.nn.dropout(layer3, keep_prob=0.8)

            # logits 28 x 28 x 1
            logits = tf.layers.conv2d_transpose(layer3, 1, 3, strides=2, padding='same')

            # MNIST原始像素范围0-1， 生成图片范围(-1, 1)
            # 训练时，需要把MNIST像素范围进行resize
            outputs = tf.tanh(logits)

            return outputs

    def netD(self, x, reuse = False, alpha = 0.01):
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()

            layer1 = tf.layers.conv2d(x, 128, 3, strides=2, padding='same')
            layer1 = tf.maximum(alpha * layer1, layer1)
            layer1 = tf.nn.dropout(layer1, keep_prob=0.8)

            layer2 = tf.layers.conv2d(layer1, 256, 3, strides=2, padding='same')
            layer2 = tf.layers.batch_normalization(layer2, training=True)
            layer2 = tf.maximum(alpha * layer2, layer2)
            layer2 = tf.nn.dropout(layer2, keep_prob=0.8)

            layer3 = tf.layers.conv2d(layer2, 512, 3, strides=2, padding='same')
            layer3 = tf.layers.batch_normalization(layer3, training=True)
            layer3 = tf.maximum(alpha * layer3, layer3)
            layer3 = tf.nn.dropout(layer3, keep_prob=0.8)

            flatten = tf.reshape(layer3, (-1, 4 * 4 * 512))
            f       = tf.layers.dense(flatten, 1)

            return f
