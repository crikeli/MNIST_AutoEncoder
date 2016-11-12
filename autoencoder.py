import tensorflow as tf
import numpy as np
import mnist_data

mnist_width = 28
numberOfInputNodes = mnist_width * mnist_width
numberOfHiddenLayerNodes = 500
corruptionLevel = 0.3

# Creating a Node for Input data
X = tf.placeholder("float", [None, numberOfInputNodes], name='X')

# Creating a node that carries out corruption of input Data
mask = tf.placeholder("float", [None, numberOfInputNodes], name = 'mask')

# Creating Nodes for hidden Vars
W_init_max = 4 * np.sqrt(6. / (numberOfInputNodes + numberOfHiddenLayerNodes))
print "W_init_max", W_init_max
W_init = tf.random_uniform(shape=[numberOfInputNodes, numberOfHiddenLayerNodes], minval = -W_init_max, maxval = W_init_max)
print "W_init", W_init

W = tf.Variable(W_init, name='W')
print "W",W.get_shape()
b = tf.Variable(tf.zeros([numberOfHiddenLayerNodes]), name='b')
print "b",b.get_shape()

W_prime = tf.transpose(W)
print "W_prime",W_prime
b_prime = tf.Variable(tf.zeros([numberOfInputNodes]), name='b_prime')
print"b_prime", b_prime.get_shape()

def model(X, mask, W, b, W_prime, b_prime):
    # This is the corrupted input.
    tilde_X = mask * X
    # Hidden State
    Y = tf.nn.sigmoid(tf.matmul(tilde_X, W) + b)
    # The reconstructed input (final output).
    Z = tf.nn.sigmoid(tf.matmul(Y, W_prime) + b_prime)
    print "Z",Z
    return Z

# The output variable that always changes based on evolving X,mask,W,b,W_prime & b_prime
Z = model(X, mask, W, b, W_prime, b_prime)

# Cost Function to minimize errors.
# Minimizing the squared error
cost = tf.reduce_sum(tf.pow(X-Z, 2))
train_optimizer = tf.train.GradientDescentOptimizer(0.02).minimize(cost)

# Loading the MNIST data
mnist = mnist_data.read_data_sets("MNIST_data/", one_hot = True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.initialize_all_variables().run()

    for i in range(100):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
            input_ = trX[start:end]
            mask_np = np.random.binomial(1, 1 - corruptionLevel, input_.shape)
            sess.run(train_optimizer, feed_dict={X: input_, mask: mask_np})

        mask_np = np.random.binomial(1, 1 - corruptionLevel, teX.shape)
        print(i, sess.run(cost, feed_dict={X: teX, mask: mask_np}))
