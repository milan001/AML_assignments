from keras.models import Sequential
from keras.layers import SimpleRNN, Input, LSTM, Dense
import numpy as np

def predictor(block_size=20, channel_number=4):
    input_shape = (block_size, channel_number)
    model = Sequential()
    model.add(SimpleRNN(channel_number, activation=None, input_shape=input_shape))
    model.add(Dense(channel_number))
    return model

def train(model, input_train, output_train):
    model.compile(optimizer='adam',
                  loss='mse')
    model.fit(input_train, output_train,
              batch_size=32, epochs=500,
              validation_split=0.2)

def predict(model, input, extend_size, block_size):
    output_predict=[]
    for i in range(extend_size):
        output_predict.append(model.predict(np.expand_dims(input[-block_size:], axis=0)))
        input=np.append(input, output_predict[-1], axis=0)
    return input, output_predict

def dataset(block_size):
    with open('../F.txt', 'r') as f:
        Lines=[l.strip().replace('NaN', '0').split() for l in f.readlines()]
        Data=[[float(i) for i in l] for l in Lines]
        input_train=[Data[i:i+block_size] for i in range(len(Data)-block_size)]
        output_train=[Data[i+block_size] for i in range(len(Data)-block_size)]
        input_test=Data
    return np.array(input_train), np.array(output_train), np.array(input_test)

input_train, output_train, input_test=dataset(100)
mean=np.array(input_train).mean()
std=np.array(input_train).std()
input_train=(input_train-mean)/std
output_train=(output_train-mean)/std
input_test=(input_test-mean)/std
model=predictor(100, 4)
train(model, input_train, output_train)
output_appended, output_pred=predict(model, input_test, 100, 100)
output_pred=output_pred*std+mean
with open('../out_srnn.txt', 'w') as f:
    for out in output_pred:
        f.write(' '.join([str(int(note)) for note in out[0]])+'\n')