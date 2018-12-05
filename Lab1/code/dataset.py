import pickle
import os
import numpy as np
from config import cfg
from PIL import Image

class dataset:

	def unPickle(self, file):
		with open(file, 'r') as f:
			return pickle.load(f)

	def __init__(self, standard=False):
		self.datasetDir=cfg.DATASETDIR
		train=self.unPickle(os.path.join(self.datasetDir, 'data_batch_1'))
		trainData=np.array(train['data'], dtype='float32')
		self.Labels=np.array(train['labels'])
		train=self.unPickle(os.path.join(self.datasetDir, 'data_batch_2'))
		trainData=np.concatenate([trainData,train['data']])
		self.Labels=np.concatenate([self.Labels,train['labels']])
		train=self.unPickle(os.path.join(self.datasetDir, 'data_batch_3'))
		trainData=np.concatenate([trainData,train['data']])
		self.Labels=np.concatenate([self.Labels,train['labels']])
		train=self.unPickle(os.path.join(self.datasetDir, 'data_batch_4'))
		trainData=np.concatenate([trainData,train['data']])
		self.Labels=np.concatenate([self.Labels,train['labels']])
		train=self.unPickle(os.path.join(self.datasetDir, 'data_batch_5'))
		trainData=np.concatenate([trainData,train['data']])
		self.Labels=np.concatenate([self.Labels,train['labels']])
		self.Data=np.moveaxis(trainData.ravel().reshape([32, 32, 3, 50000], order='F'), -1, 0) #.reshape([10000, 32, 32, 3])
		self.mean=np.mean(self.Data, axis=0)
		self.std=np.std(self.Data, axis=0)
		self.Data-=self.mean
		if standard:
			self.Data/=self.std
		
		test=self.unPickle(os.path.join(self.datasetDir, 'test_batch'))
		self.testData=np.moveaxis(np.array(test['data'], dtype='float32').ravel().reshape([32, 32, 3, 10000], order='F'), -1, 0)
		self.testMean=np.mean(self.testData, axis=0)
		self.testStd=np.std(self.testData, axis=0)
		self.testData-=self.testMean
		if standard:
			self.testData/=self.testStd
		self.testLabels=np.array(test['labels'])
