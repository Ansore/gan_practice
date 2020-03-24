import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

batch_size = 4

# 数据
X = np.random.normal(size=(1000, 2))
A = np.array([[1, 2], [-0.1, 0.5]])
b = np.array([1, 2])
X = np.dot(X, A) + b

plt.scatter(X[:, 0], X[:, 1])
plt.show()


def interate_minibatch(x, batch_size, shuffle=True):
    indices = np.arange(x.shape[0])
    if shuffle:
        np.random.shuffle(indices)

    for i in range(0, x.shape[0], batch_size):
        yield x[indices[i:i + batch_size], :]


# 封装GAN对象


class GAN(object):
    def __init__(self):
        # input output
        # 占位变量，用来保存随机产生的数
        self.z = tf.placeholder(tf.float32, shape=[None, 2], name='z')
        # 占位变量，真实的数据
        self.x = tf.placeholder(tf.float32, shape=[None, 2], name='real_x')

        # define the network
        # 生成器， 对随机变量进行加工处理，产生伪造的数据
        self.fake_x = self.netG(self.z)

        # 判别器对真是数据进行判别，反对判别结果
        # reuse=false，表示不是共享变量，需要tensorflow开辟变量地址
        self.real_logits = self.netD(self.x, reuse=False)

        # 判别器对伪造数据进行判别，返回判别结果
        # reuse=true 表示是共享变量，复用netD中的已有变量
        self.fake_logits = self.netD(self.fake_x, reuse=True)

        # define losses
        # 判别器的损失值，将真实数据的判定为真实数据，将伪造数据的判断为为找数据的得分情况
        self.loss_D = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.real_logits, labels=tf.ones_like(
                    self.real_logits))) + tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(
                            logits=self.fake_logits,
                            labels=tf.zeros_like(self.real_logits)))

        # 生成器的分数．伪造的数据，判断器判定为真实数据的得分情况
        self.loss_G = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_logits,
                                                    labels=tf.ones_like(
                                                        self.real_logits)))

        # collect variable
        t_vars = tf.trainable_variables()
        # 存放判别器中用到的变量
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

    # 生成器模型
    def netG(self, z):
        ''' 1-layer fully connected network '''

        with tf.variable_scope("generator") as scope:
            W = tf.get_variable(
                name="g_W",
                shape=[2, 2],
                initializer=tf.contrib.layers.xavier_initializer(),
                trainable=True)
            b = tf.get_variable(name="g_b",
                                shape=[2],
                                initializer=tf.zeros_initializer(),
                                trainable=True)
            return tf.matmul(z, W) + b

    def netD(self, x, reuse=False):
        ''' 3-layer fully connected network '''

        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            W1 = tf.get_variable(
                name="d_W1",
                shape=[2, 5],
                initializer=tf.contrib.layers.xavier_initializer(),
                trainable=True)
            b1 = tf.get_variable(name="d_b1",
                                 shape=[5],
                                 initializer=tf.zeros_initializer(),
                                 trainable=True)
            W2 = tf.get_variable(
                name="d_W2",
                shape=[5, 3],
                initializer=tf.contrib.layers.xavier_initializer(),
                trainable=True)
            b2 = tf.get_variable(name="d_b2",
                                 shape=[3],
                                 initializer=tf.zeros_initializer(),
                                 trainable=True)
            W3 = tf.get_variable(
                name="d_W3",
                shape=[3, 1],
                initializer=tf.contrib.layers.xavier_initializer(),
                trainable=True)
            b3 = tf.get_variable(name="d_b3",
                                 shape=[1],
                                 initializer=tf.zeros_initializer(),
                                 trainable=True)
            layer1 = tf.nn.tanh(tf.matmul(x, W1) + b1)
            layer2 = tf.nn.tanh(tf.matmul(layer1, W2) + b2)
            return tf.matmul(layer2, W3) + b3


# 开始训练
if __name__ == "__main__":
    gan = GAN()

    # 使用随机梯度下降
    d_optim = tf.train.AdamOptimizer(learning_rate=0.05).minimize(
        gan.loss_D, var_list=gan.d_vars)
    g_optim = tf.train.AdamOptimizer(learning_rate=0.01).minimize(
        gan.loss_G, var_list=gan.g_vars)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        # 将数据循环10次
        for epoch in range(10):
            avg_loss = 0.
            count = 0

            # 从真实数据中，随机抓取batch_size个值
            for x_batch in interate_minibatch(X, batch_size=batch_size):
                # generate noise z
                # 随机变量，数量为batch_size
                z_batch = np.random.normal(size=(4, 2))

                # update D network
                # 将拿到的真实数据和随机生产的数值，feed 给 sess，并bp优化一次
                loss_D, _ = sess.run([gan.loss_D, d_optim],
                                     feed_dict={
                                         gan.z: z_batch,
                                         gan.x: x_batch
                                     })

                # update G network
                loss_G, _ = sess.run([gan.loss_G, g_optim],
                                     feed_dict={
                                         gan.z: z_batch,
                                         gan.x: np.zeros(z_batch.shape)
                                     })

                avg_loss += loss_D
                count += 1

            avg_loss /= count

            # 每一个epoch都战时一次生成效果
            z = np.random.normal(size=(100, 2))
            #随机生成100个数值，0到1000  用来从真实值里面取数据
            excerpt = np.random.randint(1000, size=1000)
            fake_x, real_logits, fake_logits = sess.run(
                [gan.fake_x, gan.real_logits, gan.fake_logits],
                feed_dict={
                    gan.z: z,
                    gan.x: X[excerpt, :]
                })

            accuracy = 0.5 * (np.sum(real_logits > 0.5) / 100. +
                              np.sum(fake_logits < 0.5) / 100.)

            print('\ndiscriminator loss at epoch %d: %f' % (epoch, avg_loss))
            print('\ndiscriminator accuracy at epoch %d: %f' %
                  (epoch, accuracy))

            plt.scatter(X[:, 0], X[:, 1])
            plt.scatter(fake_x[:, 0], fake_x[:, 1])
            plt.show()
