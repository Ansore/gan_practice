from GAN import *
from configs import *
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets('data/', one_hot=True)
training = mnist.train.images
X = mnist.train.images[:, :]

# 返回真实数据
def interate_minibatch(x, batch_size, shuffe=True):
    indices = np.arange(x.shape[0])
    if shuffe:
        np.random.shuffle(indices)
    for i in range(0, x.shape[0] - 1000, batch_size):
        temp = x[indices[i:i + batch_size], :]
        temp = np.array(temp) * 2 - 1
        yield np.reshape(temp, (-1, 28, 28, 1))

def main():
    gan = GAN()

    # 预训练时的梯度优化函数
    d_pre_optim = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.4).minimize(gan.loss_pre_D, var_list=gan.d_vars)

    # 判别器的梯度优化函数
    d_optim = tf.train.AdamOptimizer(learning_rate=0.001, beta1= 0.4).minimize(gan.loss_D, var_list=gan.d_vars)

    # 预训练时的梯度优化函数
    g_optim = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.4).minimize(gan.loss_G, var_list=gan.g_vars)

    # init = tf.global_variables()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # 对判别器进行预训练, 2个epoch
        for i in range(2):
            for x_batch in interate_minibatch(X, batch_size=BATCH_SIZE):
                loss_pre_D, _ = sess.run([gan.pre_logits, d_pre_optim],
                                       feed_dict={
                                           gan.x : x_batch
                                           })
            print('discriminator: %d epoch' % (i))

        # 生成器训练5个epoch
        for epoch in range(5):
            avg_loss = 0
            count = 0

            for x_batch in interate_minibatch(X, batch_size=BATCH_SIZE):
                z_batch = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))

                loss_D, _ = sess.run([gan.loss_D, d_optim],
                                     feed_dict={
                                         gan.z : z_batch,
                                         gan.x : x_batch
                                         })
                loss_G, _ = sess.run([gan.loss_G, g_optim],
                                     feed_dict={
                                         gan.z : z_batch
                                         })
                avg_loss += loss_D
                count += 1 
                
                # print('generator: %d epoch, lodd_D: %f, loss_G: %f ' % (epoch, loss_D, loss_G))
            print('generator: %d epoch' % (epoch))

            if True:
                avg_loss /= count
                z = np.random.normal(size=(BATCH_SIZE, 100))
                excerpt = np.random.randint(100, size=BATCH_SIZE)
                needTest = np.reshape(X[excerpt, :], (-1, 28, 28, 1))
                fake_x, real_logits, fake_logits = sess.run([gan.fake_x, gan.real_logits, gan.fake_logits], feed_dict={gan.z: z, gan.x: needTest})
                print('real_logits')
                print(len(real_logits))
                print('fake_logits')
                print(len(fake_logits))
                print('\ndiscriminator loss at epoch %d: %f' % (epoch, avg_loss))
                curr_img = np.reshape(fake_x[0], (28, 28))
                plt.matshow(curr_img, cmap=plt.get_cmap('gray'))
                plt.show()
                curr_img2 = np.reshape(fake_x[10], (28, 28))
                plt.matshow(curr_img2, cmap=plt.get_cmap('gray'))
                plt.show()
                curr_img3 = np.reshape(fake_x[20], (28, 28))
                plt.matshow(curr_img3, cmap=plt.get_cmap('gray'))
                plt.show()
                curr_img4 = np.reshape(fake_x[30], (28, 28))
                plt.matshow(curr_img4, cmap=plt.get_cmap('gray'))
                plt.show()
                curr_img5 = np.reshape(fake_x[40], (28, 28))
                plt.matshow(curr_img5, cmap=plt.get_cmap('gray'))
                plt.show()

if __name__ == "__main__":
    main()
