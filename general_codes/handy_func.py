# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 01:07:33 2020

""" 

from numpy import std
from statistics import mean
from pandas import DataFrame, concat
from sklearn.preprocessing import MinMaxScaler
# --------------------------------------------------------------------------------------------- 
# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg
# --------------------------------------------------------------------------------------------- 
# MinMax Scaling (Normalization) 
def min_max(data):
	data = data.reshape(-1,1)
	scaler = MinMaxScaler(feature_range=(-1,1))
	return scaler.fit_transform(data)
# --------------------------------------------------------------------------------------------- 
# inverting scaling (Scaled) 
def un_min_max(inv_data, data):
    data = data.reshape(-1,1)
    inv_data = inv_data.reshape(-1,1)
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled = scaler.fit_transform(data)
    return scaler.inverse_transform(inv_data)
# --------------------------------------------------------------------------------------------- 
def Standardize(data):
    return (data - mean(data))/std(data)
# --------------------------------------------------------------------------------------------- 
def deStandardize(inv_data, data):
    return mean(data) + (inv_data * std(data))
# --------------------------------------------------------------------------------------------- 
# To forecast future values
def insert_end(Xin, timestep, new_input):
	for i in range(timestep-1):
		Xin[:, i, :] = Xin[:, i+1, :]
	Xin[:, timestep-1, :] = new_input
	return Xin

