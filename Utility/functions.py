from random import randint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf

# Show prediction and actual production graph for certain date
def show_graph_t(i, model, dataX, dataY, scaler):
    testY = dataY[i]
    testX = dataX[i]
    testX = np.expand_dims(testX, axis=0)
    pred = model.predict(testX)
    plt.plot(scaler.inverse_transform(pred[0]), label='Prediction')
    plt.plot(scaler.inverse_transform(testY), label='Actual Production')
    plt.legend()
    plt.show()

def data_standardization(x):
    print(x.mean())
    print(x.std())
    return (x - x.mean()) / x.std()

def min_max_scaling(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-7)

def inv_min_max_scaling(x_origin, x):
    return (x * (x_origin.max() - x_origin.min() + 1e-7)) + x_origin.min()

def scale_train_data(x_seq, y_seq, scaler_x, scaler_y):
    # If both scalers are created with option copy=False -> Inplace manipulation
    if y_seq.ndim == 1:
        y_seq = np.reshape(y_seq, (-1, 1))

    x_scaled = scaler_x.fit_transform(x_seq)
    y_scaled = scaler_y.fit_transform(y_seq)

    return x_scaled, y_scaled

# Fill Nan values by the data of the day before (24 hours before)
def fill_nan_values(dataframe):
    for x, y in np.argwhere(np.isnan(dataframe)):
        dataframe[x][y] = dataframe[x-96][y]

def fill_nan_values_hourly(dataframe):
    for x, y in np.argwhere(np.isnan(dataframe)):
        dataframe[x][y] = dataframe[x-24][y]

# Generate training data for RNN using past records (effective_days)
def generate_training_data(start_idx, end_idx, effective_days, target_hrs, stride, dataset):
    x, y = [], []
    num_effective_days_data = 4 * 24 * effective_days # four 15-minutes chunks in one hour
    if (target_hrs < 0):
        num_target_hrs_data = -target_hrs
    else:
        num_target_hrs_data = 4 * target_hrs

    energy_produced = dataset[:, -1]
    for i in range(start_idx,end_idx-num_effective_days_data-num_target_hrs_data, stride):
        x.append(dataset[i:i+num_effective_days_data])
        y.append(energy_produced[i+num_effective_days_data:i+num_effective_days_data+num_target_hrs_data])
    dataX = np.array(x)
    dataY = np.array(y)
    dataY = np.reshape(dataY, (dataX.shape[0], num_target_hrs_data, 1))
    return dataX, dataY

def generate_training_data_hourly(start_idx, end_idx, effective_days, target_hrs, stride, dataset):
    x, y = [], []
    num_effective_days_data = 24 * effective_days # four 15-minutes chunks in one hour
    if (target_hrs < 0):
        num_target_hrs_data = -target_hrs
    else:
        num_target_hrs_data = target_hrs

    energy_produced = dataset.iloc[:, -1]
    for i in range(start_idx,end_idx-num_effective_days_data-num_target_hrs_data, stride):
        x.append(dataset[i:i+num_effective_days_data])
        y.append(energy_produced[i+num_effective_days_data:i+num_effective_days_data+num_target_hrs_data])
    dataX = np.array(x)
    dataY = np.array(y)
    dataY = np.reshape(dataY, (dataX.shape[0], num_target_hrs_data, 1))
    return dataX, dataY

def gen_train_data_hour(start_idx, end_idx, seq_in_days, target_hrs, stride, dataframe):
    num_data_in_seq = 24 * seq_in_days

    if (target_hrs < 0):
        num_data_in_target_hrs = - target_hrs
    else:
        num_data_in_target_hrs = target_hrs

    energy_label = dataframe.iloc[:, -1].to_numpy()
    data_array = dataframe.to_numpy()

    # Inplace scaling
    data_array, energy_label = scale_train_data(data_array, energy_label, scaler_x, scaler_y)

    train_data_starts_at = start_idx
    train_data_ends_at = end_idx - num_data_in_seq - num_data_in_target_hrs

    data_size = train_data_ends_at - train_data_starts_at

    x = np.zeros(shape=(data_size, num_data_in_seq, data_array.shape[1]))
    y = np.zeros(shape=(data_size, num_data_in_target_hrs, 1))

    for i in range(train_data_starts_at, train_data_ends_at, stride):
        x_temp = data_array[i:i + num_data_in_seq]
        y_temp = energy_label[i + num_data_in_seq:i + num_data_in_seq + num_data_in_target_hrs]
        y_temp = np.reshape(y_temp, (num_data_in_target_hrs, 1))

        x[i - start_idx] = x_temp
        y[i - start_idx] = y_temp

    return x, y

def loop_generate_training_data(start_idx, end_idx, effective_days, target_hrs, stride, dataset):
    x, y = [], []
    num_effective_days_data = 4 * 24 * effective_days # four 15-minutes chunks in one hour
    if (target_hrs < 0):
        num_target_hrs_data = -target_hrs
    else:
        num_target_hrs_data = 4 * target_hrs

    for i in range(start_idx,end_idx-num_effective_days_data-num_target_hrs_data, stride):
        x.append(dataset[i:i+num_effective_days_data])
        y.append(dataset[i+num_effective_days_data:i+num_effective_days_data+num_target_hrs_data])
    dataX = np.array(x)
    dataY = np.array(y)
    return dataX, dataY

# Validation split in model.fit is inappropriate for time series
def split_data(dataX, dataY, size_percent):
    # Generate random number between 0 and length of dataX (call it x)
    # return Validation dataset starts at X and ends at X+size
    size = int(dataX.shape[0] * size_percent)
    x = randint(0, dataX.shape[0] - size)
    trainX = np.concatenate((dataX[:x],dataX[x+size:]), axis=0)
    trainY = np.concatenate((dataY[:x], dataY[x+size:]), axis=0)
    valX = dataX[x:x+size]
    valY = dataY[x:x+size]
    return trainX, trainY, valX, valY

# Custom loss defined on the competition website
def custom_loss(y_true, y_pred):
    loss = K.abs(y_pred - y_true)
    loss = K.batch_dot(loss , y_true, axes = 1)
    temp = 1/K.sum(y_true, axis=1)
    loss = K.batch_dot(loss, temp, axes = 1)
    loss = K.flatten(loss)
    return loss

def repeat_prediction(x_test, pred_number, model):
    x_len = len(x_test)
    number_of_pred = 96//pred_number
    day_prediction = []
    for i in range(number_of_pred):
        x = np.reshape(x_test,(1, x_test.shape[0], x_test.shape[1]))
        day_prediction.append(model.predict(x))
        x_test = np.concatenate((x_test, np.reshape(day_prediction[-1], (pred_number, x_test.shape[-1]))), axis=0)
        x_test = x_test[-x_len:]
    res = day_prediction[0][0][:,-1]
    for i in range(1, len(day_prediction)):
        res = np.concatenate((res, day_prediction[i][0][:,-1]),axis=0)
    return res

def avg_pred(ens, reg):
    avg = []
    reg = np.expand_dims(reg, axis = 1)
    for i in range(24):
        if(ens[i][0] < 1):
            avg.append(0)
        else:
            avg.append((ens[i][0] + reg[i][0]) /2)
    return avg