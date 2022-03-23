# 激活函数图像输出


from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import tensorflow as tf
from pylab import *


times = FontProperties(fname='/home/lcq/.local/share/fonts/Times New Roman/times.ttf', size=18) 


def show_activation(activation,y_lim=5):
    x=np.arange(-10., 10., 0.01)
    ts_x = tf.Variable(x)
    ts_y =activation(ts_x )
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        y=sess.run(ts_y)
        
    ax = gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_position(('data',0))
    ax.spines['left'].set_position(('data',0))
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    lines=plt.plot(x,y)

    plt.minorticks_on()
    # plt.xlabel('x',locals=(5,5,5))
    # plt.ylabel('f(x)',labelpad=10)

    plt.setp(lines, color='b', linewidth=2.0)
    plt.ylim(y_lim*-1-0.1,y_lim+0.1)
    plt.xlim(-10,10) 
 
    plt.show()
 
# 函数图像输出
# show_activation(tf.nn.sigmoid,y_lim=1)
# show_activation(tf.nn.softsign,y_lim=1)
# show_activation(tf.nn.tanh,y_lim=1)
show_activation(tf.nn.relu,y_lim=10)
show_activation(tf.nn.leaky_relu,y_lim=10)
# show_activation(tf.nn.softplus,y_lim=10)
# show_activation(tf.nn.elu,y_lim=10)

# 数值计算
# a = tf.constant([[1.0,2.0],[1.0,2.0],[1.0,2.0]])
# sess = tf.Session()
# print(sess.run(tf.sigmoid(a)))
