
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist =  input_data.read_data_sets('MNIST_data',one_hot=True)

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):   # 卷积步长1*1,第一，四个参数定为1
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2_2(x):  # 池化步长2*2,第一，四个参数定为1
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

xs = tf.placeholder(tf.float32,[None,784])   # 图片28*28分辨率
ys = tf.placeholder(tf.float32,[None,10])    # 10个标签

keep_prob = tf.placeholder(tf.float32)


x_image = tf.reshape(xs,[-1,28,28,1])      # x_image又把xs reshape成了28*28*1的形状，因为是灰色图片，所以通道是1.作为训练时的input，-1代表图片数量不定

#建立卷积层
# the first
W_conv1 = weight_variable([5,5,1,32])  #第一二个参数为patch参数,第三个参数是图像通道数，第四个参数是卷积核的函数
b_conv1 = bias_variable(([32]))        #卷积结果28*28*32
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)  # 激活卷积函数
h_pool1 = max_pool_2_2(h_conv1)               # 池化  14*14*32

#the second
W_conv2 = weight_variable([5,5,32,64])   #32通道卷积，64个卷积特征
b_conv2 = bias_variable(([64]))   #14*14*64
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2 = max_pool_2_2(h_conv2)   #7*7*64


#建立全连接层
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
W_fc1 = weight_variable([7*7*64,1024])     # 第一个参数7*7*64的patch，第二个参数代表卷积个数共1024个
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)   #drop out,防止过拟合

W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)




#cost function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),
                                 reduction_indices=[1]))
train_step = tf.train.AdadeltaOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

def compute_accuracy(v_xs, v_ys):
     global prediction
     y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
     correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
     result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
     return result
#训练
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i % 50 == 0:
        print(compute_accuracy( mnist.test.images[:1000], mnist.test.labels[:1000]))