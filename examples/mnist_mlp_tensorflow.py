import tensorflow as tf
import os, re, os.path
mypath = "C:\\Users\\LEEKYUHO\\Desktop\\Code\\keras\\examples\\board\mnist"
for root, dirs, files in os.walk(mypath):
    for file in files:
        os.remove(os.path.join(root, file))

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

shifting_value_W1 = 0.5
shifting_value_W2 = 1

#########
# 신경망 모델 구성
######
with tf.name_scope("input") as scope:
    X = tf.placeholder(tf.float32, [None, 784])

with tf.name_scope("y_") as scope:
    Y = tf.placeholder(tf.float32, [None, 10])

with tf.name_scope("weight1") as scope:
    W1 = tf.Variable(tf.random_normal([784, 150], stddev=0.01))

with tf.name_scope("layer1") as scope:
    L1 = tf.nn.relu(tf.matmul(X, W1))

with tf.name_scope("weight2") as scope:
    W2 = tf.Variable(tf.random_normal([150, 10], stddev=0.01))

with tf.name_scope("layer2") as scope:
    model = tf.matmul(L1, W2)

w1_hist = tf.summary.histogram("weight1", W1)
w2_hist = tf.summary.histogram("weight2", W2)

with tf.name_scope("cost") as scope:
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))

with tf.name_scope("power_W1") as scope:
    ref_W1 = tf.abs(W1)
    shifting_value_tensor_W1 = tf.constant(shifting_value_W1, shape=[784,150])
    nonneg_W1 = tf.abs(tf.add(W1, shifting_value_tensor_W1))
    ref_sum_W1 = tf.reduce_sum(tf.matmul(X, ref_W1))
    nonneg_sum_W1 = tf.add(tf.reduce_sum(tf.matmul(X, nonneg_W1)), tf.reduce_sum(tf.scalar_mul(0.5, X)))
    power_ratio_W1 = tf.divide(nonneg_sum_W1, ref_sum_W1)

# with tf.name_scope("power_W2") as scope:
#     ref_W2 = tf.abs(W2)
#     shifting_value_tensor_W2 = tf.constant(shifting_value_W2, shape=[150,10])
#     nonneg_W2 = tf.abs(tf.add(W1, shifting_value_tensor_W2))
#     ref_sum_W2 = tf.reduce_sum(tf.matmul(X, ref_W2))
#     nonneg_sum_W2 = tf.add(tf.reduce_sum(tf.matmul(X, nonneg_W2)), tf.reduce_sum(tf.scalar_mul(0.5, X)))
#     power_ratio_W2 = tf.divide(nonneg_sum_W2, ref_sum_W2)

with tf.name_scope("train") as scope:
    optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

with tf.Session() as sess:
    merged = tf.summary.merge_all()
    writer =tf.summary.FileWriter("./board/mnist", sess.graph)

#########
# 신경망 모델 학습
######
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)

for epoch in range(50):
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
######
is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

accuracy_val, P_ratio = sess.run([accuracy, power_ratio_W1],
                        feed_dict={X: mnist.test.images,
                                   Y: mnist.test.labels})

print('정확도:', accuracy_val, 'P_ratio : ', P_ratio)