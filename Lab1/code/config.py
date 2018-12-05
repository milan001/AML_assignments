from easydict import EasyDict as edict

cfg=edict()
cfg.DATASETDIR='../cifar-10-python/cifar-10-batches-py'
cfg.BATCH_SIZE=256
cfg.INPUT_SHAPE=[32, 32, 3]
cfg.NUM_CLASS=10

def set_attr(attr, val):
	cfg[attr]=val

cfg.set_attr=set_attr