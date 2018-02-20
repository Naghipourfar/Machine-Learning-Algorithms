import sys
import tensorflow as tf
import numpy as np

sys.setrecursionlimit(10000000)

"""
    Created by Mohsen Naghipourfar on 2/16/18.
    Email : mn7697np@gmail.com
"""
# # TensorBoard Example
# a = tf.constant(3, name='a')
# b = tf.constant(5, name='b')
# c = tf.multiply(a, b, name='c')
#
# # Write the graph to file in specified directory
# writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())
# with tf.Session() as sess:
#     # writer = tf.summary.FileWriter('./graphs', sess.graph)
#     print(sess.run(c))
# writer.close()
# """
#     To run a tensorboard: (write the following commands in terminal)
#     1. python filename.py
#     2. tensorboard --logdir="./directoryname" --port 8080 (or whatever!)
# """
# Neural Networks in TensorFlow

# 1. Operational Gates
# 1.1 f = ax

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
        print("{0} * {1} = {2}".format(a_val, x_val, y_pred))

# 1.2 f = ax + b

with tf.Session() as sess:
    a = tf.Variable(1.0, dtype=tf.float32)
    b = tf.Variable(1.0, dtype=tf.float32)
    x = tf.placeholder(dtype=tf.float32)
    y = a * x + b
    loss = tf.square(tf.subtract(y, 17.0))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train_step = optimizer.minimize(loss)
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(100):
        sess.run(train_step, feed_dict={x: 5.0})
        a_val = sess.run(a)
        b_val = sess.run(b)
        y_pred = sess.run(y, feed_dict={x: 5.0})
        print("{0} * {1} + {2} = {3}".format(a_val, x_val, b_val, y_pred))

# 2. Working with gates --> Sigmoid and ReLU activation function

tf.set_random_seed(5)
np.random.seed(42)

with tf.Session() as sess:
    batch_size = 50
    a1 = tf.Variable(tf.random_normal(shape=[1, 1]))
    b1 = tf.Variable(tf.random_uniform(shape=[1, 1]))
    a2 = tf.Variable(tf.random_normal(shape=[1, 1]))
    b2 = tf.Variable(tf.random_uniform(shape=[1, 1]))
    x = np.random.normal(2, 0.1, 500)
    x_data = tf.placeholder(tf.float32, shape=[None, 1])
    sigmoid_activation = tf.sigmoid(tf.add(tf.matmul(x_data, a1), b1))
    relu_activation = tf.nn.relu(tf.add(tf.matmul(x_data, a2), b2))

    sigmoid_loss = tf.reduce_mean(tf.square(tf.subtract(sigmoid_activation, 0.75)))
    relu_loss = tf.reduce_mean(tf.square(tf.subtract(relu_activation, 0.75)))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

    sigmoid_train_step = optimizer.minimize(sigmoid_loss)
    relu_train_step = optimizer.minimize(relu_loss)

    init = tf.global_variables_initializer()
    sigmoid_model_loss = []
    relu_model_loss = []

    sess.run(init)
    for _ in range(750):
        random_indices = np.random.choice(len(x), size=batch_size)
        x_vals = np.transpose(x[random_indices])
        x_vals = np.reshape(x_vals, [batch_size, 1])
        sess.run(sigmoid_train_step, feed_dict={x_data: x_vals})
        sess.run(relu_train_step, feed_dict={x_data: x_vals})
        sigmoid_model_loss.append(sess.run(sigmoid_loss, feed_dict={x_data: x_vals}))
        relu_model_loss.append(sess.run(relu_loss, feed_dict={x_data: x_vals}))
    import Graphics as g

    # g.plt.plot(sigmoid_model_loss, 'k-', label='Sigmoid Loss')
    # g.plt.plot(relu_model_loss, 'r--', label='ReLU Loss')
    # g.show()

# 3. Single-layer neural network
from tensorflow.python.framework import ops
from sklearn import datasets

ops.reset_default_graph()
iris = datasets.load_iris()

x_vals = np.array([x[0:3] for x in iris.data])
y_vals = np.array([x[3] for x in iris.data])

seed = 3
tf.set_random_seed(seed)
np.random.seed(seed)

# Split data into train/test = 80%/20%
train_indices = np.random.choice(len(x_vals), round(len(x_vals) * 0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]


# Normalize by column (min-max norm)
def normalize_cols(m):
    col_max = m.max(axis=0)
    col_min = m.min(axis=0)
    return (m - col_min) / (col_max - col_min)


x_vals_train = np.nan_to_num(normalize_cols(x_vals_train))
x_vals_test = np.nan_to_num(normalize_cols(x_vals_test))

batch_size = 50

# Initialize placeholders
x_data = tf.placeholder(shape=[None, 3], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# Create variables for both Neural Network Layers
hidden_layer_nodes = 10
A1 = tf.Variable(tf.random_normal(shape=[3, hidden_layer_nodes]))  # inputs -> hidden nodes
b1 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes]))  # one biases for each hidden node
A2 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes, 1]))  # hidden inputs -> 1 output
b2 = tf.Variable(tf.random_normal(shape=[1]))  # 1 bias for the output

# Declare model operations
hidden_output = tf.nn.relu(tf.add(tf.matmul(x_data, A1), b1))
final_output = tf.nn.relu(tf.add(tf.matmul(hidden_output, A2), b2))

# Declare loss function
loss = tf.reduce_mean(tf.square(y_target - final_output))

# Declare optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.005)
train_step = optimizer.minimize(loss)

# Initialize variables
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    # Training loop
    loss_vec = []
    test_loss = []
    for i in range(500):
        rand_index = np.random.choice(len(x_vals_train), size=batch_size, replace=False)
        rand_x = x_vals_train[rand_index]
        rand_y = np.transpose([y_vals_train[rand_index]])
        sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

        temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
        loss_vec.append(np.sqrt(temp_loss))

        test_temp_loss = sess.run(loss, feed_dict={x_data: x_vals_test, y_target: np.transpose([y_vals_test])})
        test_loss.append(np.sqrt(test_temp_loss))
        if (i + 1) % 50 == 0:
            print('Generation: ' + str(i + 1) + '. Loss = ' + str(temp_loss))

    # Plot loss (MSE) over time
    # g.plt.plot(loss_vec, 'k-', label='Train Loss')
    # g.plt.plot(test_loss, 'r--', label='Test Loss')
    # g.plt.title('Loss (MSE) per Generation')
    # g.plt.xlabel('Generation')
    # g.plt.ylabel('Loss')
    # g.plt.legend(loc='upper right')
    # g.plt.show()

# 4. Different layers in Neural Networks
ops.reset_default_graph()

# Generate 1D data
data_size = 25
data_1d = np.random.normal(size=data_size)
x_input_1d = tf.placeholder(dtype=tf.float32, shape=[data_size])


# --------Convolution--------
def conv_layer_1d(input_1d, my_filter):
    # Tensorflow's 'conv2d()' function only works with 4D arrays:
    # [batch#, width, height, channels], we have 1 batch, and
    # width = 1, but height = the length of the input, and 1 channel.
    # So next we create the 4D array by inserting dimension 1's.
    input_2d = tf.expand_dims(input_1d, 0)
    input_3d = tf.expand_dims(input_2d, 0)
    input_4d = tf.expand_dims(input_3d, 3)
    # Perform convolution with stride = 1, if we wanted to increase the stride,
    # to say '2', then strides=[1,1,2,1]
    convolution_output = tf.nn.conv2d(input_4d, filter=my_filter, strides=[1, 1, 1, 1], padding="VALID")
    # Get rid of extra dimensions
    conv_output_1d = tf.squeeze(convolution_output)
    return conv_output_1d


# 5. Multilayer Neural Networks
import requests

# url = 'https://www.umass.edu/statdata/statdata/data/lowbwt.dat'
# birth_file = requests.get(url)
# birth_data = birth_file.text.split('\r\n')[5:]
file = open("./Data/lowbwt.dat", "rt")
birth_data = list(line.rsplit(",") for line in file.readlines())
birth_data = birth_data[:-1]
birth_data = list(list(elem[:-1] if elem.__contains__('\n') else elem for elem in row) for row in birth_data)
birth_data = list(list(map(int, row)) for row in birth_data)
birth_header = ["id", "LOW", "AGE", "LWT", "RACE", "SMOKE", "PTL", "HT", "UI", "FTV", "bwt"]
batch_size = 100
# Extract y-target (birth weight)
y_vals = np.array([x[10] for x in birth_data])

# Filter for features of interest
cols_of_interest = ['LOW', 'AGE', 'LWT', 'RACE', 'SMOKE', 'PTL', 'HT', 'UI', 'FTV']
x_vals = np.array(
    [[x[ix] for ix, feature in enumerate(birth_header) if feature in cols_of_interest] for x in birth_data])

# Split data into train/test = 80%/20%
train_indices = np.random.choice(len(x_vals), round(len(x_vals) * 0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]

x_vals_train = np.nan_to_num(normalize_cols(x_vals_train))
x_vals_test = np.nan_to_num(normalize_cols(x_vals_test))


def init_weight(shape, st_dev):
    weight = tf.Variable(tf.random_normal(shape, stddev=st_dev))
    return weight


def init_bias(shape, st_dev):
    bias = tf.Variable(tf.random_normal(shape, stddev=st_dev))
    return bias


# Create a fully connected layer:
def fully_connected(input_layer, weights, biases):
    layer = tf.add(tf.matmul(input_layer, weights), biases)
    return tf.nn.relu(layer)


with tf.Session() as sess:
    # Create Placeholders
    x_data = tf.placeholder(shape=[None, 9], dtype=tf.float32)
    y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

    # --------Create the first layer (25 hidden nodes)--------
    weight_1 = init_weight(shape=[9, 25], st_dev=10.0)
    bias_1 = init_bias(shape=[25], st_dev=10.0)
    layer_1 = fully_connected(x_data, weight_1, bias_1)

    # --------Create second layer (10 hidden nodes)--------
    weight_2 = init_weight(shape=[25, 10], st_dev=10.0)
    bias_2 = init_bias(shape=[10], st_dev=10.0)
    layer_2 = fully_connected(layer_1, weight_2, bias_2)

    # --------Create third layer (3 hidden nodes)--------
    weight_3 = init_weight(shape=[10, 3], st_dev=10.0)
    bias_3 = init_bias(shape=[3], st_dev=10.0)
    layer_3 = fully_connected(layer_2, weight_3, bias_3)

    # --------Create output layer (1 output value)--------
    weight_4 = init_weight(shape=[3, 1], st_dev=10.0)
    bias_4 = init_bias(shape=[1], st_dev=10.0)
    final_output = fully_connected(layer_3, weight_4, bias_4)

    # Declare loss function (L1)
    loss = tf.reduce_mean(tf.abs(y_target - final_output))

    # Declare optimizer
    my_opt = tf.train.AdamOptimizer(0.05)
    train_step = my_opt.minimize(loss)

    # Initialize Variables
    init = tf.initialize_all_variables()
    sess.run(init)

    # Training loop
    loss_vec = []
    test_loss = []
    for i in range(1250):
        rand_index = np.random.choice(len(x_vals_train), size=batch_size)
        rand_x = x_vals_train[rand_index]
        rand_y = np.transpose([y_vals_train[rand_index]])
        sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

        temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
        loss_vec.append(temp_loss)

        test_temp_loss = sess.run(loss, feed_dict={x_data: x_vals_test, y_target: np.transpose([y_vals_test])})
        test_loss.append(test_temp_loss)
        if (i + 1) % 25 == 0:
            print('Generation: ' + str(i + 1) + '. Loss = ' + str(temp_loss))

    # Plot loss over time
    g.plt.plot(loss_vec, 'k-', label='Train Loss')
    g.plt.plot(test_loss, 'r--', label='Test Loss')
    g.plt.title('Loss per Generation')
    g.plt.xlabel('Generation')
    g.plt.ylabel('Loss')
    g.plt.legend(loc="upper right")
    g.plt.show()

    # Find the % classified correctly above/below the cutoff of 2500 g
    # >= 2500 g = 0
    # < 2500 g = 1
    actuals = np.array([x[1] for x in birth_data])
    test_actuals = actuals[test_indices]
    train_actuals = actuals[train_indices]

    test_preds = [x[0] for x in sess.run(final_output, feed_dict={x_data: x_vals_test})]
    train_preds = [x[0] for x in sess.run(final_output, feed_dict={x_data: x_vals_train})]
    test_preds = np.array([1.0 if x < 2500.0 else 0.0 for x in test_preds])
    train_preds = np.array([1.0 if x < 2500.0 else 0.0 for x in train_preds])

    # Print out accuracies
    test_acc = np.mean([x == y for x, y in zip(test_preds, test_actuals)])
    train_acc = np.mean([x == y for x, y in zip(train_preds, train_actuals)])
    print('On predicting the category of low birthweight from regression output (<2500g):')
    print('Test Accuracy: {}'.format(test_acc))
    print('Train Accuracy: {}'.format(train_acc))
