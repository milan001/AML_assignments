from keras.models import Sequential
from keras.layers import SimpleRNN, Input, LSTM
import numpy as np

def dataset(block_size=20):
    map_data=np.zeros((5000,1), dtype='float')#0.0] for i in range(5000)]
    with open('../test.txt', 'r') as f:
        Lines=f.readlines()
        Lines=[l.strip().split(',') for l in Lines]
        ids, data = [int(l[0])-1 for l in Lines], [float(l[1]) for l in Lines]
        miss_ids=[]
        for i in range(len(ids)):
            map_data[ids[i]][0]=data[i]
            miss_ids.append(ids[i])
    with open('../train.txt', 'r') as f:
        Lines=f.readlines()
        Lines=[l.strip().split(',') for l in Lines]
        ids, data = [int(l[0])-1 for l in Lines], [float(l[1]) for l in Lines]
        avail_ids=[]
        for i in range(len(ids)):
            map_data[ids[i]][0]=data[i]
            avail_ids.append(ids[i])

    input_train=[]
    output_train=[]
    input_test=[]
    output_test=[]
    last_avail=0
    for i in avail_ids:
        if i+20 in miss_ids:
            input_test.append(map_data[i:last_avail+1])
            output_test.append(map_data[i+20])
        elif i+20 in avail_ids:
            last_avail=i+20
            input_train.append(map_data[i:i+20])
            output_train.append(map_data[i+20])
    input_train, output_train =\
            np.array(input_train), np.array(output_train)
    return input_train, input_test, output_train, output_test

def rnn(block_size=20):
    input_shape = (block_size, 1)
    model = Sequential()
    model.add(SimpleRNN(1, activation=None, input_shape=input_shape))
    return model

def lstm(block_size=20):
    input_shape = (block_size, 1)
    model = Sequential()
    model.add(LSTM(1, activation=None, input_shape=input_shape))
    return model

def train(model, input_train, output_train):
    model.compile(optimizer='adam',
                  loss='mse')
    model.fit(input_train, output_train,
              batch_size=32, epochs=500,
              validation_split=0.2)

def test(model, input_test, output_test, rnn=True):
    model_output=[]
    for i in range(len(input_test)):
        if len(input_test[i])==block_size:
            model_input=np.array([input_test[i]])
        else:
            model_input=np.array([np.concatenate((input_test[i],model_output[i-block_size+len(input_test[i]):]), axis=0)])
        model_output.append(model.predict(model_input)[0])
    if rnn:
        print('Test error for RNN:')
    else:
        print('Test error for LSTM:')
    print(((np.array(model_output) - np.array(output_test))**2).mean())

if __name__=="__main__":
    block_size=20
    input_train, input_test, output_train, output_test=\
            dataset(block_size)
    mean_in=np.array(input_train).mean()
    std_in=np.array(input_train).std()
    mean_out=np.array(output_train).mean()
    std_out=np.array(output_train).std()
    input_train=(input_train-mean_in)/std_in
    output_train=(output_train-mean_out)/std_out
    input_test=(input_test-mean_in)/std_in
    output_test=(output_test-mean_out)/std_out
    model_rnn=rnn(block_size)
    model_lstm=lstm(block_size)
    train(model_rnn, input_train, output_train)
    train(model_lstm, input_train, output_train)
    test(model_rnn, input_test, output_test, True)
    test(model_lstm, input_test, output_test, False)