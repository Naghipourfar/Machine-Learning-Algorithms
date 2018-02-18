import sys
import tensorflow as tf

sys.setrecursionlimit(10000000)

"""
    Created by Mohsen Naghipourfar on 2/16/18.
    Email : mn7697np@gmail.com
"""
# TensorBoard Example
a = tf.constant(3, name='a')
b = tf.constant(5, name='b')
c = tf.multiply(a, b, name='c')

writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())  # Write the graph to file in specified directory
with tf.Session() as sess:
    # writer = tf.summary.FileWriter('./graphs', sess.graph)
    print(sess.run(c))
writer.close()
"""
    To run a tensorboard: (write the following commands in terminal)
    1. python filename.py
    2. tensorboard --logdir="./directoryname" --port 8080 (or whatever!)
"""
#### Neural Networks in TensorFlow

# 1.Operational Gates
# f = ax

with tf.Session() as sess:
    a = tf.Variable(4.0, dtype=tf.float32)
    x_val = 5.0
    y_val = 50.0
    x_data = tf.placeholder(dtype=tf.float32)
    y_data = tf.multiply(a, x_data)
    loss = tf.square(tf.subtract(y_data, y_val))
    init = tf.global_variables_initializer()
    sess.run(init)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train_step = optimizer.minimize(loss)
    for i in range(10):
        sess.run(train_step, feed_dict={x_data: x_val})
        a_val = sess.run(a)
        y_pred = sess.run(y_data, feed_dict={x_data: x_val})
        print(a_val, y_pred)
