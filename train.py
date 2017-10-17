import os
import time
import tensorflow as tf
import numpy as np
from tensorflow.core.protobuf import saver_pb2
import driving_data
import model

train_loss = 0
train_acc = 0
val_loss = 0
val_acc = 0
count_train = 0
size_train = 0
count_val = 0
size_val = 0

def accuracy_cal(xs,ys,keep_prob):
    ans = sess.run(answer,feed_dict={model.x: xs, model.y_: ys,model.keep_prob:keep_prob})
    pre = sess.run(pred,feed_dict={model.x: xs, model.y_: ys,model.keep_prob:keep_prob})
    ans = np.floor(np.abs(ans*10))
    pre = np.floor(np.abs(pre*10))
    sub = np.subtract(ans,pre)

    sub_min = sub >= -1
    sub_max = sub <= 1
    sub = tf.cast(tf.equal(sub_min,sub_max),dtype=tf.int64)

    return tf.count_nonzero(sub,dtype=tf.int32), tf.size(sub)

LOGDIR = './save'

sess = tf.InteractiveSession()

L2NormConst = 0.001

train_vars = tf.trainable_variables()
for d in ['/gpu:0','/gpu:1','/gpu:2', '/gpu:3']:
    with tf.device(d):
        loss = tf.reduce_mean(tf.square(tf.subtract(model.y_, model.y))) + tf.add_n([tf.nn.l2_loss(v) for v in train_vars]) * L2NormConst
        train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
        answer = model.y_ #answer
        pred = model.y #prediction
        
sess.run(tf.initialize_all_variables())

# create a summary to monitor cost tensor
tf.summary.scalar("loss", loss)
# merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

saver = tf.train.Saver(write_version = tf.train.SaverDef.V2)

# op to write logs to Tensorboard
logs_path = './logs'
summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

epochs = 50
batch_size = 300

start_time = time.time()

# train over the dataset about 30 times
for epoch in range(epochs):
  for i in range(int(driving_data.num_images/batch_size)):
    xs, ys = driving_data.LoadTrainBatch(batch_size)
    train_step.run(feed_dict={model.x: xs, model.y_: ys, model.keep_prob: 0.8})
    train_loss = loss.eval(feed_dict={model.x:xs, model.y_: ys, model.keep_prob: 0.8})
    if i % 10 == 0:
      xs, ys = driving_data.LoadValBatch(batch_size)
      loss_value = loss.eval(feed_dict={model.x:xs, model.y_: ys, model.keep_prob: 1.0})
      val_loss = loss_value
      print("Epoch: %d, Step: %d, Loss: %g" % (epoch, epoch * batch_size + i, loss_value))

    # write logs at every iteration
    summary = merged_summary_op.eval(feed_dict={model.x:xs, model.y_: ys, model.keep_prob: 1.0})
    summary_writer.add_summary(summary, epoch * driving_data.num_images/batch_size + i)

    if i % batch_size == 0:
      if not os.path.exists(LOGDIR):
        os.makedirs(LOGDIR)
      checkpoint_path = os.path.join(LOGDIR, "model.ckpt")
      filename = saver.save(sess, checkpoint_path)
  print("Model saved in file: %s" % filename)

for i in range(int(driving_data.num_images/batch_size)):
    xs, ys = driving_data.LoadTrainBatch(batch_size)
    count, size = accuracy_cal(xs,ys,1.0)
    count_train += count
    size_train += size
    if i % 10 == 0:
      xs, ys = driving_data.LoadValBatch(batch_size)
      count,size = accuracy_cal(xs,ys,1.0)
      count_val += count
      size_val += size

acc_train = tf.multiply(tf.divide(count_train,size_train),100)
acc_val = tf.multiply(tf.divide(count_val,size_val),100)

end_time = time.time()
running_time = end_time - start_time
print("acc_train: ",sess.run(acc_train),"acc_val: ",sess.run(acc_val),"Running_time: ",running_time)

print("Run the command line:\n" \
          "--> tensorboard --logdir=./logs " \
          "\nThen open http://0.0.0.0:6006/ into your web browser")
sess.close()
