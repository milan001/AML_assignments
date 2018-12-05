import tensorflow as tf
import numpy as np
from config import cfg
from model import Classify, Train_model
from dataset import dataset
from PIL import Image
from custom_ops import custom_GD, custom_Adam, custom_AdaGrad

################### Task 1 ######################

print("################### Task 1 ######################")

print("#### RELU without standardization ####")
Dataset=dataset(False)
# 2 layer NN
ClassModel=Classify(False, False, actfn='ReLU')
opt = custom_GD(0.000003).minimize(ClassModel.loss)
Train_model(Dataset, ClassModel, opt)

print("#### RELU with standardization ####")
Dataset=dataset(True)
ClassModel=Classify(False, False, actfn='ReLU')
opt = custom_GD(0.00003).minimize(ClassModel.loss)
Train_model(Dataset, ClassModel, opt)

print("#### Sigmoid without standardization ####")
Dataset=dataset(False)
ClassModel=Classify(False, False, actfn='Sigmoid')
opt = custom_GD(0.00003).minimize(ClassModel.loss)
Train_model(Dataset, ClassModel, opt)

print("#### Sigmoid with standardization ####")
Dataset=dataset(True)
ClassModel=Classify(False, False, actfn='Sigmoid')
opt = custom_GD(0.0003).minimize(ClassModel.loss)
Train_model(Dataset, ClassModel, opt)

################### Task 2 ######################

print("################# Task 2 ################")

print("#### batch size=200, lr=1e-7 ####")
cfg.set_attr('BATCH_SIZE', 200)
Dataset=dataset(True)
ClassModel=Classify(True, True, actfn='Sigmoid')
opt = custom_GD(1e-7).minimize(ClassModel.loss)
Train_model(Dataset, ClassModel, opt)

print("#### batch size=200, lr=1e-6 ####")
cfg.set_attr('BATCH_SIZE', 200)
Dataset=dataset(True)
ClassModel=Classify(True, True, actfn='Sigmoid')
opt = custom_GD(1e-6).minimize(ClassModel.loss)
Train_model(Dataset, ClassModel, opt)

print("#### batch size=200, lr=1e-5 ####")
cfg.set_attr('BATCH_SIZE', 200)
Dataset=dataset(True)
ClassModel=Classify(True, True, actfn='Sigmoid')
opt = custom_GD(1e-5).minimize(ClassModel.loss)
Train_model(Dataset, ClassModel, opt)

print("#### batch size=100, lr=1e-7 ####")
cfg.set_attr('BATCH_SIZE', 100)
Dataset=dataset(True)
ClassModel=Classify(True, True, actfn='Sigmoid')
opt = custom_GD(1e-7).minimize(ClassModel.loss)
Train_model(Dataset, ClassModel, opt)

print("#### batch size=100, lr=1e-6 ####")
cfg.set_attr('BATCH_SIZE', 100)
Dataset=dataset(True)
ClassModel=Classify(True, True, actfn='Sigmoid')
opt = custom_GD(1e-6).minimize(ClassModel.loss)
Train_model(Dataset, ClassModel, opt)

print("#### batch size=100, lr=1e-5 ####")
cfg.set_attr('BATCH_SIZE', 100)
Dataset=dataset(True)
ClassModel=Classify(True, True, actfn='Sigmoid')
opt = custom_GD(1e-5).minimize(ClassModel.loss)
Train_model(Dataset, ClassModel, opt)

################### Task 3 ######################

print("################## Task 3 ###############")

print("#### two layered NN ####")
cfg.set_attr('BATCH_SIZE', 100)
Dataset=dataset(True)
ClassModel=Classify(False, False, actfn='ReLU')
opt = custom_GD(1e-5).minimize(ClassModel.loss)
Train_model(Dataset, ClassModel, opt)

################### Task 4 ######################

print("################## Task 4 ###############")

print("#### Relu ####")
cfg.set_attr('BATCH_SIZE', 100)
Dataset=dataset(True)
ClassModel=Classify(False, False, actfn='ReLU')
opt = custom_GD(1e-4).minimize(ClassModel.loss)
Train_model(Dataset, ClassModel, opt)

print("#### Sigmoid ####")
cfg.set_attr('BATCH_SIZE', 100)
Dataset=dataset(True)
ClassModel=Classify(False, False, actfn='Sigmoid')
opt = custom_GD(1e-4).minimize(ClassModel.loss)
Train_model(Dataset, ClassModel, opt)

print("#### tanh ####")
cfg.set_attr('BATCH_SIZE', 100)
Dataset=dataset(True)
ClassModel=Classify(False, False, actfn='Tanh')
opt = custom_GD(1e-4).minimize(ClassModel.loss)
Train_model(Dataset, ClassModel, opt)


print("#### LeakyRelu ####")
cfg.set_attr('BATCH_SIZE', 100)
Dataset=dataset(True)
ClassModel=Classify(False, False, actfn='LeakyRelu')
opt = custom_GD(1e-4).minimize(ClassModel.loss)
Train_model(Dataset, ClassModel, opt)

################### Task 5 ######################

print("################## Task 5 ###############")

print("#### SGD ####")
cfg.set_attr('BATCH_SIZE', 100)
Dataset=dataset(True)
ClassModel=Classify(False, False, actfn='ReLU')
opt = custom_GD(1e-4).minimize(ClassModel.loss)
Train_model(Dataset, ClassModel, opt)

print("#### Adagrad ####")
cfg.set_attr('BATCH_SIZE', 100)
Dataset=dataset(True)
ClassModel=Classify(False, False, actfn='ReLU')
opt = custom_AdaGrad(1e-2).minimize(ClassModel.loss)
Train_model(Dataset, ClassModel, opt)

print("#### Adam ####")
cfg.set_attr('BATCH_SIZE', 100)
Dataset=dataset(True)
ClassModel=Classify(False, False, actfn='ReLU')
opt = custom_Adam(1e-4).minimize(ClassModel.loss)
Train_model(Dataset, ClassModel, opt)

