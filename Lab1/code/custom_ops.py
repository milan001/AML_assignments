import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import optimizer
import numpy as np

def relu(input_tensor):
	return tf.maximum(input_tensor, 0.0)

def sigmoid(input_tensor):
	return 1.0/(1+tf.exp(-tf.maximum(input_tensor, -10)))

def tanh(input_tensor):
	return sigmoid(input_tensor/2.0)*2-1

def leaky_relu(input_tensor):
	eps=0.001
	return tf.maximum(input_tensor, -input_tensor*eps)

def softmax(input_tensor):
	output_tensor=tf.exp(tf.minimum(input_tensor, 10))
	return output_tensor/tf.reduce_sum(output_tensor, axis=1, keepdims=True)

def dense(input_tensor, output_size, activation=None):
	print(input_tensor.shape)
	weights=tf.Variable(-0.01+0.02*tf.random_normal(np.concatenate([np.array(input_tensor.shape[1:], dtype='int32'),[output_size]])), name='dense_var')
	print(weights.name)
	bais=tf.Variable(0.0)
	output=tf.matmul(input_tensor, weights)+bais
	if activation:
		output=activation(output)
	return output

def sigmoid_cross_entropy(labels, logits):
	eps=1e-7
	return tf.reduce_sum(-labels*tf.log(logits+eps)-(1-labels)*tf.log(1-logits+eps))

class custom_GD(optimizer.Optimizer):
	def __init__(self, learning_rate, name="CustomGD"):
		super(custom_GD, self).__init__(use_locking=False, name=name)
		self.learning_rate=learning_rate
		self._name=name

	def _apply_dense(self, grad, var):
		var_update=state_ops.assign_sub(var,self.learning_rate*grad)
		return control_flow_ops.group(*[var_update])

class custom_AdaGrad(optimizer.Optimizer):
	def __init__(self, learning_rate, name="CustomAdaGrad"):
		super(custom_AdaGrad, self).__init__(use_locking=False, name=name)
		self.learning_rate=learning_rate
		self._name=name
		self.eps=1e-5

	def _create_slots(self, var_list):
		for v in var_list:
			self._zeros_slot(v, "histgd", self._name)

	def _apply_dense(self, grad, var):
		histGD = self.get_slot(var, "histgd")
		histGD_t = histGD.assign(histGD + (grad**2))
		var_update=state_ops.assign_sub(var,self.learning_rate*grad/(histGD_t**0.5+self.eps))
		return control_flow_ops.group(*[var_update, histGD_t])

class custom_Adam(optimizer.Optimizer):
	def __init__(self, learning_rate, name="CustomAdam"):
		super(custom_Adam, self).__init__(use_locking=False, name=name)
		self.learning_rate=learning_rate
		self._name=name
		self.beta_1=0.9
		self.beta_2=0.999
		self.eps=1e-7

	def _create_slots(self, var_list):
		for v in var_list:
			self._zeros_slot(v, "m", self._name)
			self._zeros_slot(v, "v", self._name)
			self._zeros_slot(v, "t", self._name)

	def _apply_dense(self, grad, var):
		m = self.get_slot(var, "m")
		v = self.get_slot(var, "v")
		t = self.get_slot(var, "t")
		m_t = m.assign(self.beta_1 * m + (1-self.beta_1)*grad)
		t_t = t.assign(t+1)
		v_t = v.assign(self.beta_2 * v + (1-self.beta_2)*(grad**2))
		m_cap=m_t/(1-self.beta_1**t_t)
		v_cap=v_t/(1-self.beta_2**t_t)
		var_update=state_ops.assign_sub(var,self.learning_rate*m_cap/(v_cap**0.5+self.eps))
		return control_flow_ops.group(*[var_update, m_t, v_t])