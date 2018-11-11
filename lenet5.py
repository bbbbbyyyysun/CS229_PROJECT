import tensorflow as tf 
from tqdm import trange

import config
import utils

class_num = config.class_num

class Lenet5():

	def __init__(self, train_data, train_label, valid_data, valid_label, test_data, test_label, learning_rate=0.001):

		### param define
		self.train_data = train_data
		self.train_label = train_label
		assert (len(self.train_label) == len(self.train_data))
		assert (self.train_data[0].shape[0] == 32 and self.train_data[0].shape[1] == 32)

		self.valid_data = valid_data
		self.valid_label = valid_label
		assert (len(self.valid_label) == len(self.valid_data))
		assert (self.valid_data[0].shape[0] == 32 and self.valid_data[0].shape[1] == 32)

		self.test_data = test_data
		self.test_label = test_label
		assert (len(self.test_data) == len(self.test_label))
		assert (self.test_data[0].shape[0] == 32 and self.test_data[0].shape[1] == 32)

		### model arch
		self.x = tf.placeholder(tf.float32, shape=(None,32,32,3), name='x')
		self.y = tf.placeholder(tf.int32, shape=(None), name='y')
		self.one_hot_y = tf.one_hot(self.y, class_num, name='one_hot_y')

		weights, biases = self.get_variables()
		# conv layer 1 + pooling layer 1
		# (32,32,3) -> (28,28,6) -> (14,14,6)
		self.conv1 = tf.nn.conv2d(self.x, weights['conv1'], [1,1,1,1], padding='VALID') + biases['conv1']
		self.pool1 = tf.nn.max_pool(self.conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
		self.conv1 = tf.nn.relu(self.pool1)

		# conv layer 2 + pooling layer 2
		# (14,14,6) -> (10,10,16) -> (5,5,16)
		self.conv2 = tf.nn.conv2d(self.conv1, weights['conv2'], [1,1,1,1], padding='VALID') + biases['conv2']
		self.pool2 = tf.nn.max_pool(self.conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
		self.conv2 = tf.nn.relu(self.pool2)

		# flatten
		# (5,5,16) -> (400,1)
		self.flat = tf.contrib.layers.flatten(self.conv2)

		# fully connect layer 1
		self.fc1 = tf.matmul(self.flat, weights['fc1']) + biases['fc1']
		self.fc1 = tf.nn.relu(self.fc1)

		# fully connect layer 2
		self.fc2 = tf.matmul(self.fc1, weights['fc2']) + biases['fc2']
		self.fc2 = tf.nn.relu(self.fc2)

		# fully connect layer 3
		self.logits = tf.add(tf.matmul(self.fc2, weights['fc3']), biases['fc3'], name='logits')
		self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.one_hot_y)
		self.loss = tf.reduce_mean(self.cross_entropy)

		# optimize
		self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
		self.train_step = self.optimizer.minimize(self.loss)
		self.correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.one_hot_y, 1))
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
		self.saver = tf.train.Saver()



	def train(self, epoches=32, batch_size=32):
		assert(epoches > 0 and batch_size > 0)
		num = len(self.train_data)
		print('Training Lenet 5 Model!')

		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())

			for epoch in trange(epoches):
				self.train_data, self.train_label = utils.shuffle(self.train_data, self.train_label)

				for offset in range(0, num, batch_size):
					x_batch, y_batch = self.train_data[offset:offset+batch_size], self.train_label[offset:offset+batch_size]

					_, loss, accuracy = sess.run([self.train_step, self.loss, self.accuracy], feed_dict={self.x:x_batch, self.y:y_batch})

				print('Epoch {}: the training loss is {}, training accuracy is {}'.format(epoch+1, loss, accuracy))

				if epoch % 4 == 0 and epoch != 0:
					valid_accuracy = self.evaluate(self.valid_data, self.valid_label, batch_size)
					print('Accuracy for valid set is {}'.format(valid_accuracy))

				if epoch % 8 == 0:
					self.saver.save(sess, './model/lenet5/lenet5'.format(epoch))

			test_accuracy = self.evaluate(self.test_data, self.test_label, batch_size)
			print('Accuracy for test set is {}'.format(test_accuracy))
			print('Training done!')



	def evaluate(self, x_data, y_data, batch_size):
		num = len(x_data)
		total_accuracy = 0
		sess = tf.get_default_session()
		for offset in range(0, num, batch_size):
			x_batch, y_batch = x_data[offset:offset + batch_size], y_data[offset:offset + batch_size]
			accuracy = sess.run(self.accuracy, feed_dict={self.x:x_batch, self.y:y_batch})
			total_accuracy += (accuracy * len(x_batch))
		return total_accuracy / num



	def get_variables(self):

		weights = {
			'conv1': tf.Variable(tf.truncated_normal(shape=[5,5,3,6], mean=0, stddev=0.3)),
			'conv2': tf.Variable(tf.truncated_normal(shape=[5,5,6,16], mean=0, stddev=0.3)),
			'fc1'  : tf.Variable(tf.truncated_normal(shape=[400,120], mean=0, stddev=0.3)),
			'fc2'  : tf.Variable(tf.truncated_normal(shape=[120,84], mean=0, stddev=0.3)),
			'fc3'  : tf.Variable(tf.truncated_normal(shape=[84,class_num], mean=0, stddev=0.3))
		}

		biases = {
			'conv1': tf.get_variable(name='b_conv1', shape=[6],initializer=tf.random_normal_initializer(stddev=0.3)),
			'conv2': tf.get_variable(name='b_conv2', shape=[16],initializer=tf.random_normal_initializer(stddev=0.3)),
			'fc1'  : tf.get_variable(name='b_fc1', shape=[120],initializer=tf.random_normal_initializer(stddev=0.3)),
			'fc2'  : tf.get_variable(name='b_fc2', shape=[84],initializer=tf.random_normal_initializer(stddev=0.3)),
			'fc3'  : tf.get_variable(name='b_fc3', shape=[class_num],initializer=tf.random_normal_initializer(stddev=0.3))
		}

		return weights, biases