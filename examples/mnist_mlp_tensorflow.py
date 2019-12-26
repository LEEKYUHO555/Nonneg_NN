import tensorflow as tf
import os, re, os.path
mypath = "C:\\Users\\LEEKYUHO\\Desktop\\Code\\keras\\examples\\board\mnist"
for root, dirs, files in os.walk(mypath):
    for file in files:
        os.remove(os.path.join(root, file))

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

EPOCH = 10
shifting_value_W1 = 0.5
shifting_value_W2 = 1.0

#####################
## 신경망 모델 구성 ##
#####################

# with tf.name_scope("input") as scope:
#     X = tf.placeholder(tf.float32, [None, 784])
#
# with tf.name_scope("y_") as scope:
#     Y = tf.placeholder(tf.float32, [None, 10])
#
# with tf.name_scope("weight1") as scope:
#     W1 = tf.Variable(tf.random_normal([784, 150], stddev=0.01))
#
# with tf.name_scope("layer1") as scope:
#     L1 = tf.nn.relu(tf.matmul(X, W1))
#
# with tf.name_scope("weight2") as scope:
#     W2 = tf.Variable(tf.random_normal([150, 10], stddev=0.01))
#
# with tf.name_scope("layer2") as scope:
#     model = tf.matmul(L1, W2)

with tf.variable_scope("Non_shifted"):
    X = tf.placeholder(tf.float32, [None, 784])
    Y = tf.placeholder(tf.float32, [None, 10])
    W1 = tf.get_variable("W1", shape=[784, 150], initializer=tf.contrib.layers.xavier_initializer(),
                         constraint=lambda x: tf.clip_by_value(x, -shifting_value_W1, shifting_value_W1))
    L1 = tf.nn.relu(tf.matmul(X, W1))
    W2 = tf.get_variable("W2", shape=[150, 10], initializer=tf.contrib.layers.xavier_initializer(),
                         constraint=lambda x: tf.clip_by_value(x, -shifting_value_W2, shifting_value_W2))
    model = tf.matmul(L1, W2)

with tf.variable_scope("Shifted"):
    shifting_value_tensor_W1 = tf.constant(shifting_value_W1, shape=[784, 150])
    shifted_W1 = tf.add(W1, shifting_value_tensor_W1)
    shifting_value_tensor_W2 = tf.constant(shifting_value_W2, shape=[150, 10])
    shifted_W2 = tf.add(W2, shifting_value_tensor_W2)

    sum_X = tf.reduce_sum(X, 1, keepdims=True)
    Compensate_X = tf.ones([1,150])*sum_X
    # Compensate_X = tf.transpose(tf.tile(sum_X, [150, 1]))
    shifted_H1 = tf.matmul(X, shifted_W1) - tf.scalar_mul(shifting_value_W1, Compensate_X)
    shifted_L1 = tf.nn.relu(shifted_H1)

    sum_L1 = tf.reduce_sum(shifted_L1, 1, keepdims=True)
    Compensate_L1 = tf.ones([1,10])*sum_L1
    # Compensate_L1 = tf.transpose(tf.tile(sum_L1, [10, 1]))
    shifted_H2 = tf.matmul(shifted_L1, shifted_W2) - tf.scalar_mul(shifting_value_W2, Compensate_L1)
    shifted_model = shifted_H2

with tf.name_scope("power_W1") as scope:
    ref_W1 = tf.abs(W1)
    shifting_value_tensor_W1 = tf.constant(shifting_value_W1, shape=[784,150])
    nonneg_W1 = tf.abs(tf.add(W1, shifting_value_tensor_W1))
    ref_sum_W1 = tf.reduce_sum(tf.matmul(X, ref_W1))
    nonneg_sum_W1 = tf.add(tf.reduce_sum(tf.matmul(X, nonneg_W1)), tf.reduce_sum(tf.scalar_mul(shifting_value_W1, X)))
    power_ratio_W1 = tf.divide(nonneg_sum_W1, ref_sum_W1)

with tf.name_scope("power_W2") as scope:
    ref_W2 = tf.abs(W2)
    shifting_value_tensor_W2 = tf.constant(shifting_value_W2, shape=[150,10])
    nonneg_W2 = tf.abs(tf.add(W2, shifting_value_tensor_W2))
    ref_sum_W2 = tf.reduce_sum(tf.matmul(L1, ref_W2))
    nonneg_sum_W2 = tf.add(tf.reduce_sum(tf.matmul(L1, nonneg_W2)), tf.reduce_sum(tf.scalar_mul(shifting_value_W2, L1)))
    power_ratio_W2 = tf.divide(nonneg_sum_W2, ref_sum_W2)

with tf.name_scope("tensor_board") as scope:
    w1_hist = tf.summary.histogram("weight1", W1)
    w2_hist = tf.summary.histogram("weight2", W2)

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
is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

is_correct_shifted = tf.equal(tf.argmax(shifted_model, 1), tf.argmax(Y, 1))
shifted_accuracy = tf.reduce_mean(tf.cast(is_correct_shifted, tf.float32))

accuracy_val, shifted_accuracy_val, P_ratio_W1, P_ratio_W2 = sess.run(
    [accuracy, shifted_accuracy, power_ratio_W1, power_ratio_W2], feed_dict={X: mnist.test.images, Y: mnist.test.labels})

print('Acc:', accuracy_val, 'Shifted_Acc:', shifted_accuracy_val, 'P_ratio_W1 : ', P_ratio_W1, 'P_ratio_W2 : ', P_ratio_W2)