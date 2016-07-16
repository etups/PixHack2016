import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
import numpy as np
from softmax import softmax 

# Simulate data
one_hot = {'a': [1,0,0,0,0,0,0],
           'b': [0,1,0,0,0,0,0],
           'c': [0,0,1,0,0,0,0],
           'd': [0,0,0,1,0,0,0],
           'e': [0,0,0,0,1,0,0],
           'f': [0,0,0,0,0,1,0],
           'g': [0,0,0,0,0,0,1]}

w2v = {'a': [0,0,0,0,0],
       'b': [0,0,0.1,0,0.1],
       'c': [0,0,0,0.1,0.1],
       'd': [0.1,0.1,0,0,0],
       'e': [-0.1,0,0.1,-0.1,0],
       'f': [0,-0.1,0,0,0.1],
       'g': [0,0,-0.2,0,-0.1]}

text = 'a b c a b d e g f c a b e d f f f g e b c d a b e g f d a a b c e g f c a b c d'.split(' ')

train = np.array([w2v[w] for w in text])
label = [one_hot[text[i+1]] for i in range(len(text) - 1)]
label.append([0,0,0,0,0,0,0])
label = np.array(label)

test = 'a b c d e f g a b c'.split(' ')
test = np.array([w2v[w] for w in test])

'''
input vecter: x.dot(L)
input weight: W
hidden

'''

# Parameters
training_iter = 10000
batch_size = 3
display_step = 10
dim_input = 5
dim_hidden = 2
num_steps = 10
num_words = 7

# Placehold
x = tf.placeholder('float', [None, num_steps, dim_input])
y = tf.placeholder('float', [None, num_words])

# Weights
weights = {
    'out': tf.Variable(tf.random_normal([dim_hidden, num_words]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_words]))
}

def RNN(x, weight, biases):
    # shape of input x: [batch_size, num_steps, dim_input]
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, dim_input])
    x = tf.split(0, num_steps, x)

    lstm_cell = rnn_cell.BasicLSTMCell(dim_hidden, forget_bias=1.0)

    outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)

    return tf.matmul(outputs[-1], weight['out']+biases['out'])

pred = RNN(x, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# initializing
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    step = 1
    i = 0
    while step*batch_size < training_iter:
        batch_x = train[i*batch_size*num_steps:((i+1)*batch_size*num_steps)]
        batch_x = batch_x.reshape((batch_size, num_steps, dim_input))
        batch_y = label[i*batch_size:((i+1)*batch_size)]
        if (i+1)*batch_size >= len(train):
            i = 0
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

        if step % display_step == 0:
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print "Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc)
        step += 1

    test = test.reshape((1, num_steps, dim_input))
    test_pred = sess.run(pred, feed_dict={x: test})
    print softmax(test_pred)
