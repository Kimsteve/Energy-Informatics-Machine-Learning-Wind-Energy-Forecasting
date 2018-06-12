from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from pandas import to_datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot
import numpy

# date-time parsing function for loading the dataset
def parser2(x):
    return to_datetime(str(x), format='%Y%m%d %H:%M')

# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
	df = DataFrame(data)
	columns = [df.shift(i) for i in range(1, lag+1)]
	columns.append(df)
	df = concat(columns, axis=1)
	df.fillna(0, inplace=True)
	return df

# scale train and test data to [-1, 1]
def scale(train, test):
	# fit scaler
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaler = scaler.fit(train)
	# transform train
	train = train.reshape(train.shape[0], train.shape[1])
	train_scaled = scaler.transform(train)
	# transform test
	test = test.reshape(test.shape[0], test.shape[1])
	test_scaled = scaler.transform(test)
	return scaler, train_scaled, test_scaled

# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
	new_row = [x for x in X] + [value]
	array = numpy.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]

# fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, neurons):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        print("model.fit: {}".format(i))
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
        model.reset_states()
    return model

# make a one-step forecast
def forecast_lstm(model, batch_size, X):
	X = X.reshape(1, 1, len(X))
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]


if __name__ == "__main__":
	# load dataset
	series_train = read_csv('TrainData.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser2)
	series_test = read_csv('Solution.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser2)

	# transform data to be stationary
	raw_values_train = series_train.values
	raw_values_test = series_test.values

	# transform data to be supervised learning
	supervised_train = timeseries_to_supervised(raw_values_train, 1)
	supervised_test = timeseries_to_supervised(raw_values_test, 1)
	train = supervised_train.values
	test = supervised_test.values

	# prepare data
	scaler, train_scaled, test_scaled = scale(train, test)

	# train model
	lstm_model = fit_lstm(train_scaled, 1, 1, 4)

	# forecast the entire training dataset to build up state for forecasting
	train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
	lstm_model.predict(train_reshaped, batch_size=1)

	# walk-forward validation on the test data
	predictions = list()
	for i in range(len(test_scaled)):
		X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
		yhat = forecast_lstm(lstm_model, 1, X)
		yhat = invert_scale(scaler, X, yhat)
		predictions.append(yhat)
	# report performance
	rmse = sqrt(mean_squared_error(raw_values_test, predictions))
	print('%d) Test RMSE: %.3f' % (1, rmse))

	# prediction presentation 
	series_prediction = series_test.copy()

	lm_model_data = numpy.arange(1, 721, 1.0)
	null_model_data = numpy.ones(720)
	#    (Intercept)      TIMESTAMP2 
	#0.3902306481555 0.0001239384317 
	lm_model_data = 0.3902306481555 + (0.0001239384317 *lm_model_data)


	series_prediction = series_test.copy()
	lm_prediction = series_test.copy()
	null_model_predictions = series_test.copy()

	for i in range(0,len(predictions)):
	    series_prediction.values[i] = predictions[i]
	    lm_prediction.values[i] = lm_model_data[i]
	    null_model_predictions.values[i] = null_model_data[i]

	pyplot.rcParams["figure.figsize"] = (13,7)

	pyplot.plot(series_train, 'b-')
	pyplot.plot(series_test, 'b-')
	pyplot.plot(series_prediction, 'r-')
	pyplot.plot(lm_prediction, 'g-')
	pyplot.plot(null_model_predictions, 'm-')

	pyplot.xlim((735167,735205))
	pyplot.savefig('RNN_LM_NULL.png')
	#pyplot.show()







	null_model_data = null_model_data * numpy.mean(series_train)
