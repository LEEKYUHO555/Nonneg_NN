import tensorflow as tf
import numpy as np
import os, re, os.path
mypath = "C:\\Users\\LEEKYUHO\\Desktop\\Code\\keras\\examples\\board\\mnist"
for root, dirs, files in os.walk(mypath):
    for file in files:
        os.remove(os.path.join(root, file))

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

EPOCH = 50
shifting_value_W1 = 0.5
shifting_value_W2 = 1.0
Q_factor = 8

#####################
## 신경망 모델 구성 ##
#####################

def shift_quantize(Weight, Q_val, maximum):         # Quantizing Func for shifted matrix
    Q_Weight = tf.scalar_mul(1/maximum, Weight)
    Q_Weight = tf.scalar_mul(Q_val, Q_Weight)
    Q_Weight = tf.round(Q_Weight)
    Q_Weight = tf.scalar_mul(1/Q_val, Q_Weight)
    Q_Weight = tf.scalar_mul(maximum, Q_Weight)
    return Q_Weight

def quantize(Weight, Q_val, maximum):               # Quantizing Func for un-shifted matrix
    sign_Weight = tf.sign(Weight)
    Q_Weight = tf.abs(Weight)
    Q_Weight = tf.scalar_mul(1 / maximum, Q_Weight)
    Q_Weight = tf.scalar_mul(Q_val, Q_Weight)
    Q_Weight = tf.round(Q_Weight)
    Q_Weight = tf.scalar_mul(1/Q_val, Q_Weight)
    Q_Weight = tf.scalar_mul(maximum, Q_Weight)
    Q_Weight = tf.multiply(sign_Weight, Q_Weight)
    return Q_Weight

with tf.variable_scope("Non_shifted"):
    X = tf.placeholder(tf.float32, [None, 784])
    Y = tf.placeholder(tf.float32, [None, 10])
    W1 = tf.get_variable("W1", shape=[784, 150], initializer=tf.contrib.layers.xavier_initializer(),
                         constraint=lambda x: tf.clip_by_value(x, -shifting_value_W1, shifting_value_W1))
    L1 = tf.nn.relu(tf.matmul(X, W1))
    W2 = tf.get_variable("W2", shape=[150, 10], initializer=tf.contrib.layers.xavier_initializer(),
                         constraint=lambda x: tf.clip_by_value(x, -shifting_value_W2, shifting_value_W2))
    model = tf.matmul(L1, W2)

    # for quantizing
    Q_W1 = quantize(W1, Q_factor - 1, shifting_value_W1)
    Q_W2 = quantize(W2, Q_factor - 1, shifting_value_W2)
    Q_L1 = tf.nn.relu(tf.matmul(X, Q_W1))
    Q_model = tf.matmul(Q_L1, Q_W2)

with tf.variable_scope("Shifted"):
    shifting_value_tensor_W1 = tf.constant(shifting_value_W1, shape=[784, 150])
    shifted_W1 = tf.add(W1, shifting_value_tensor_W1)
    shifting_value_tensor_W2 = tf.constant(shifting_value_W2, shape=[150, 10])
    shifted_W2 = tf.add(W2, shifting_value_tensor_W2)

    sum_X = tf.reduce_sum(X, 1, keepdims=True)
    Compensate_X = tf.ones([1,150])*sum_X
    shifted_H1 = tf.matmul(X, shifted_W1) - tf.scalar_mul(shifting_value_W1, Compensate_X)
    shifted_L1 = tf.nn.relu(shifted_H1)

    sum_L1 = tf.reduce_sum(shifted_L1, 1, keepdims=True)
    Compensate_L1 = tf.ones([1,10])*sum_L1
    shifted_H2 = tf.matmul(shifted_L1, shifted_W2) - tf.scalar_mul(shifting_value_W2, Compensate_L1)
    shifted_model = shifted_H2

    # for quantizing
    Q_shifted_W1 = shift_quantize(shifted_W1, Q_factor - 1, 2 * shifting_value_W1)
    Q_shifted_W2 = shift_quantize(shifted_W2, Q_factor - 1, 2 * shifting_value_W2)
    Q_shifted_H1 = tf.matmul(X, Q_shifted_W1) - tf.scalar_mul(shifting_value_W1, Compensate_X)
    Q_shifted_L1 = tf.nn.relu(Q_shifted_H1)

    Q_sum_L1 = tf.reduce_sum(Q_shifted_L1, 1, keepdims=True)
    Q_Compensate_L1 = tf.ones([1, 10]) * Q_sum_L1
    Q_shifted_H2 = tf.matmul(Q_shifted_L1, Q_shifted_W2) - tf.scalar_mul(shifting_value_W2, Q_Compensate_L1)
    Q_shifted_model = Q_shifted_H2


with tf.name_scope("power_W1") as scope:
    ref_W1 = tf.abs(W1)
    ref_sum_W1 = tf.reduce_sum(tf.matmul(X, ref_W1))

    nonneg_sum_W1 = tf.add(tf.reduce_sum(tf.matmul(X, shifted_W1)),
                           tf.reduce_sum(tf.scalar_mul(shifting_value_W1, X))) * 0.5
    power_ratio_W1 = tf.divide(nonneg_sum_W1, ref_sum_W1)

with tf.name_scope("power_W2") as scope:
    ref_W2 = tf.abs(W2)
    ref_sum_W2 = tf.reduce_sum(tf.matmul(L1, ref_W2))
    nonneg_sum_W2 = tf.add(tf.reduce_sum(tf.matmul(shifted_L1, shifted_W2)),
                           tf.reduce_sum(tf.scalar_mul(shifting_value_W2, shifted_L1))) * 0.5
    power_ratio_W2 = tf.divide(nonneg_sum_W2, ref_sum_W2)

with tf.name_scope("tensor_board") as scope:
    w1_hist = tf.summary.histogram("W1", W1)
    w2_hist = tf.summary.histogram("W2", W2)
    shifted_w1_hist = tf.summary.histogram("shifted_W1", shifted_W1)
    shifted_w2_hist = tf.summary.histogram("shifted_W2", shifted_W2)
    Q_w1_hist = tf.summary.histogram("Q_W1", Q_W1)
    Q_w2_hist = tf.summary.histogram("Q_W2", Q_W2)
    Q_shifted_w1_hist = tf.summary.histogram("Q_shifted_W1", Q_shifted_W1)
    Q_shifted_w2_hist = tf.summary.histogram("Q_shifted_W2", Q_shifted_W2)

with tf.name_scope("cost") as scope:
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))

with tf.name_scope("train") as scope:
    optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

with tf.Session() as sess:
    merged = tf.summary.merge_all()
    writer =tf.summary.FileWriter("./board/mnist", sess.graph)

#########
# 신경망 모델 학습
#########
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)

for epoch in range(EPOCH):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)

        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys})
        total_cost += cost_val

        if i==0:
            summary = sess.run(merged, feed_dict={X: batch_xs, Y: batch_ys})
            writer.add_summary(summary, epoch)

    print('Epoch:', '%04d' % (epoch + 1),
          'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))

print('최적화 완료!')

#########
# 결과 확인
#########
# Q_weight_W1 = Q_shifted_W1.eval(session=sess)
# np.savetxt(mypath + '_W1.csv', Q_weight_W1)

is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

is_correct_shifted = tf.equal(tf.argmax(shifted_model, 1), tf.argmax(Y, 1))
shifted_accuracy = tf.reduce_mean(tf.cast(is_correct_shifted, tf.float32))

Q_is_correct = tf.equal(tf.argmax(Q_model, 1), tf.argmax(Y, 1))
Q_accuracy = tf.reduce_mean(tf.cast(Q_is_correct, tf.float32))

Q_is_correct_shifted = tf.equal(tf.argmax(Q_shifted_model, 1), tf.argmax(Y, 1))
Q_shifted_accuracy = tf.reduce_mean(tf.cast(Q_is_correct_shifted, tf.float32))

accuracy_val, shifted_accuracy_val, Q_accuracy_val, Q_shifted_accuracy_val, P_ratio_W1, P_ratio_W2 = sess.run(
    [accuracy, shifted_accuracy, Q_accuracy, Q_shifted_accuracy, power_ratio_W1, power_ratio_W2],
    feed_dict={X: mnist.test.images, Y: mnist.test.labels})

print('Acc:', accuracy_val, 'Shifted_Acc:', shifted_accuracy_val, 'Q_Acc:', Q_accuracy_val,
      'Q_Shifted_Acc:', Q_shifted_accuracy_val, 'P_ratio_W1 : ', P_ratio_W1, 'P_ratio_W2 : ', P_ratio_W2)