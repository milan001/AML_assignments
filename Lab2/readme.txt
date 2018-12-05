#################################  TASK1 ##################################
Using RNN and LSTM predict missing data-points:

run script: python task1.py

experiment results:
mean squered error(after normalization) on the test data with RNN=0.038
mean squered error(after normalization) on the test data with LSTM=0.035

#################################  TASK2 ##################################
Predict the upcoming audio:
run script: 
1. to generate file 'out_srnn.txt': python task2.py
2. to generate file 'out_srnn.wav' run in matlab: TxtToAudio('../out_srnn.txt', '../out_srnn.wav')

experiment results:
mean squered error(after normalization) on the validation data with RNN=0.096
mean squered error(after normalization) on the validation data with LSTM=0.035