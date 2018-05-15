import gzip
import cPickle

import tensorflow as tf
import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

train_x, train_y = train_set
train_y = one_hot(train_y, 10)

valid_x, valid_y = valid_set
valid_y = one_hot(valid_y, 10)

test_x, test_y = test_set

# TODO: the neural net!!

x = tf.placeholder("float", [None, 28*28])  # samples
y_ = tf.placeholder("float", [None, 10])  # labels

W1 = tf.Variable(np.float32(np.random.rand(28*28, 10)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(10, 10)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
# h = tf.matmul(x, W1) + b1  # Try this!
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

loss = tf.reduce_sum(tf.square(y_ - y))

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

print "----------------------"
print "   Start training...  "
print "----------------------"

batch_size = 20
last_valid_data_error = 10000
valid_data_error = 0
epoch = 0
error_difference = 1
train_data_list = []
valid_data_list = []

while error_difference >= 0.001:
    epoch = epoch + 1
    for jj in xrange(len(train_x) / batch_size):

        # Train
        batch_xs = train_x[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = train_y[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    # Train error
    train_data = sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys})
    train_data_list.append(train_data)

    if epoch > 1:
        last_valid_data_error = valid_data_error

    # Valid error
    valid_data_error = sess.run(loss, feed_dict={x: valid_x, y_: valid_y})
    valid_data_list.append(valid_data_error)

    error_difference = last_valid_data_error - valid_data_error
    print "Epoch #:", epoch, "; Train error: ", train_data, "; Data error: ",\
        valid_data_error, "; Data error prev: ", last_valid_data_error,\
        "; Diferencia: ", error_difference

print "----------------------"
print "   Test result...     "
print "----------------------"

total = 0.0
error = 0.0
test_data = sess.run(y, feed_dict={x: test_x})
for b, r in zip(test_y, test_data):
    if np.argmax(b) != np.argmax(r):
        error += 1
    total += 1
fail = error / total * 100.0
print "Porcentaje de error: ", fail,"% y portenjate de exito", (100.0 - fail), "%"

plt.ylabel('Errores')
plt.xlabel('Epocas')
tr_handle, = plt.plot(train_data_list)
vl_handle, = plt.plot(valid_data_list)
plt.legend(handles=[tr_handle, vl_handle],
            labels=['Error entrenamiento', 'Error validacion'])
plt.savefig('Grafica_mnist.png')