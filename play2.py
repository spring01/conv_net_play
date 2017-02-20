
import numpy as np
import tensorflow as tf

def init_weight(num_in, num_out):
    max_wt = np.sqrt(6.0 / (num_in + num_out))
    return tf.Variable(tf.random_uniform([num_in, num_out], -max_wt, max_wt))

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def read_txt(filename):
    with open(filename) as data_txt:
        data = np.loadtxt(data_txt, delimiter=',')
        inp, out = data[:, :-1].astype(np.float32), data[:, -1]
        out_unique = list(np.unique(out))
        out_int = [out_unique.index(y) for y in out]
        out_oh = np.zeros((len(out), len(out_unique)), dtype=np.float32)
        out_oh[np.arange(len(out)), out_int] = 1.0
    return (inp, out_oh)

x_train, y_train_oh = read_txt('digitstrain.txt')
x_valid, y_valid_oh = read_txt('digitsvalid.txt')

with tf.device('/cpu:0'):
    num_feat = x_train.shape[1]
    num_hid = 100
    num_out = y_train_oh.shape[1]
    weight1 = init_weight(num_feat, num_hid)
    bias1 = tf.Variable(tf.zeros([num_hid]))
    weight2 = init_weight(num_hid, num_out)
    bias2 = tf.Variable(tf.zeros([num_out]))
    inp = tf.placeholder(tf.float32, [None, num_feat])
    hid = tf.nn.sigmoid(tf.matmul(inp, weight1) + bias1)
    prob = tf.nn.softmax(tf.matmul(hid, weight2) + bias2)
    out = tf.placeholder(tf.float32, [None, num_out])
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(out * tf.log(prob), 1))
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    batch_size = 500
    indices = np.arange(x_train.shape[0])
    for iter in range(200):
        err_pred = tf.not_equal(tf.argmax(prob, 1), tf.argmax(out, 1))
        err_rate = tf.reduce_mean(tf.cast(err_pred, tf.float32))
        train_err = sess.run(err_rate, feed_dict={inp: x_train, out: y_train_oh})
        valid_err = sess.run(err_rate, feed_dict={inp: x_valid, out: y_valid_oh})
        print iter, train_err, valid_err
        np.random.shuffle(indices)
        batch_x_train = batch(x_train[indices], batch_size)
        batch_y_train_oh = batch(y_train_oh[indices], batch_size)
        for batch_inp, batch_out in zip(batch_x_train, batch_y_train_oh):
            sess.run(train_step, feed_dict={inp: batch_inp, out: batch_out})






