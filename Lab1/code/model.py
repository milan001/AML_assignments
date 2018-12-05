import tensorflow as tf
from config import cfg
import custom_ops
import pickle
import numpy as np

class Classify:
	def __init__(self, logistic=False, regularization=False, actfn='Sigmoid'):
		self.batch_size=cfg.BATCH_SIZE
		self.num_class=cfg.NUM_CLASS
		self.input_shape=cfg.INPUT_SHAPE
		self.batch_input=tf.placeholder(dtype='float32', shape=[self.batch_size]+self.input_shape)
		self.batch_output=tf.placeholder(dtype='float32', shape=[self.batch_size, self.num_class])
		self.output=self.model(logistic, actfn)
		self.reg_cons=1
		self.loss=self.compute_loss(regularization)
		self.predict=self.pred()
		self.deadnode=self.count_deadnode()

	def model(self, logistic, actfn):
		if logistic:
			output_layer=custom_ops.dense(tf.contrib.layers.flatten(self.batch_input), self.num_class, activation=custom_ops.softmax)
			return output_layer
		if actfn is 'ReLU':
			layer1 = custom_ops.dense(tf.contrib.layers.flatten(self.batch_input), 100, activation=custom_ops.relu)
			output = custom_ops.dense(layer1, self.num_class, activation=custom_ops.softmax)
			return output
		elif actfn is 'Sigmoid':
			layer1 = custom_ops.dense(tf.contrib.layers.flatten(self.batch_input), 100, activation=custom_ops.sigmoid)
			output = custom_ops.dense(layer1, self.num_class, activation=custom_ops.softmax)
			return output
		elif actfn is 'Tanh':
			layer1 = custom_ops.dense(tf.contrib.layers.flatten(self.batch_input), 100, activation=custom_ops.tanh)
			output = custom_ops.dense(layer1, self.num_class, activation=custom_ops.softmax)
			return output
		elif actfn is 'LeakyRelu':
			layer1 = custom_ops.dense(tf.contrib.layers.flatten(self.batch_input), 100, activation=custom_ops.leaky_relu)
			output = custom_ops.dense(layer1, self.num_class, activation=custom_ops.softmax)
			return output

	def compute_loss(self, regularization):
		if regularization:
			return custom_ops.sigmoid_cross_entropy(labels=self.batch_output, logits=self.output)+self.reg_cons*tf.reduce_sum(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="dense_var")[0]**2)
		return custom_ops.sigmoid_cross_entropy(labels=self.batch_output, logits=self.output)

	def pred(self):
		return tf.argmax(self.output, axis=1)

	def count_deadnode(self):
		eps=1e-5
		count=0#tf.zeros(1)
		for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="dense_var"):
			count+=tf.reduce_sum(tf.cast(tf.abs(var)<eps, dtype='float32'))
		return count

def Train_model(Dataset, ClassModel, opt):
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		Loss=np.zeros(100)
		for it in range(100):
			for batch_start in range(0, len(Dataset.Data)-cfg.BATCH_SIZE+1, cfg.BATCH_SIZE):
				batch_input=Dataset.Data[batch_start:batch_start+cfg.BATCH_SIZE]
				batch_labels=Dataset.Labels[batch_start:batch_start+cfg.BATCH_SIZE]
				batch_output=[[i==y for i in range(cfg.NUM_CLASS)] for y in batch_labels]
				sess.run(opt, {ClassModel.batch_input: batch_input, ClassModel.batch_output: batch_output})
			loss=0.0
			TestAccurecy=0.0
			TrainAccurecy=0.0
			for batch_start in range(0, len(Dataset.testData)-cfg.BATCH_SIZE+1, cfg.BATCH_SIZE):
				batch_input=Dataset.testData[batch_start:batch_start+cfg.BATCH_SIZE]
				batch_labels=Dataset.testLabels[batch_start:batch_start+cfg.BATCH_SIZE]
				batch_output=[[i==y for i in range(cfg.NUM_CLASS)] for y in batch_labels]
				pred=sess.run(ClassModel.predict, {ClassModel.batch_input: batch_input, ClassModel.batch_output: batch_output})
				TestAccurecy+=sum(pred==batch_labels)
			for batch_start in range(0, len(Dataset.Data)-cfg.BATCH_SIZE+1, cfg.BATCH_SIZE):
				batch_input=Dataset.Data[batch_start:batch_start+cfg.BATCH_SIZE]
				batch_labels=Dataset.Labels[batch_start:batch_start+cfg.BATCH_SIZE]
				batch_output=[[i==y for i in range(cfg.NUM_CLASS)] for y in batch_labels]
				ls, pred=sess.run([ClassModel.loss, ClassModel.predict], {ClassModel.batch_input: batch_input, ClassModel.batch_output: batch_output})
				loss+=ls
				TrainAccurecy+=sum(pred==batch_labels)
			print("Epoch #"+str(it)+": loss:"+str(cfg.BATCH_SIZE*loss/len(Dataset.Data))+" acc: "+str(TrainAccurecy/len(Dataset.Data))+" "+str(TestAccurecy/len(Dataset.testData)))
			Loss[it]=loss
		with open("./results/loss.pickle",'w') as f:
			pickle.dump(Loss, f)
		print("#DeadNodes: "+str(sess.run(ClassModel.deadnode)))
