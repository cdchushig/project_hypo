import numpy as np
from keras import Sequential
from keras.layers import Dense, Flatten, LSTM, Dropout, Activation, Embedding, Bidirectional
from keras.preprocessing.sequence import pad_sequences


import matplotlib.pyplot as plt


def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_' + metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])


def create_model():
    model = Sequential()
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(50))
    model.add(Dense(32, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


history = model.fit(train_dataset, epochs=10,
                    validation_data=test_dataset,
                    validation_steps=30)