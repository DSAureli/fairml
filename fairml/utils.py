from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

# import dictionary with perturbation strategies.
from .perturbation_strategies import perturbation_strategy_dictionary


def mse(y, y_hat):	# mean squared error (mean of the squares of the errors)
	""" function to calculate mse between two numpy vectors """

	y = np.array(y)	# transform to numpy array object
	y_hat = np.array(y_hat)	# transform to numpy array object

	y_hat = np.reshape(y_hat, (y_hat.shape[0],))	# flatten to 1D array
	y = np.reshape(y, (y.shape[0],))	# flatten to 1D array

	diff = y - y_hat	# array of the differences (diff[i] = y[i] - y_hat[i])
	diff_squared = np.square(diff)	# array of the squares of the differences
	mse = np.mean(diff_squared)	# mean of the squares of the errors

	return mse


def accuracy(y, y_hat):
	""" function to calculate accuracy of y_hat given y"""
	y = np.array(y)	# transform to numpy array object
	y_hat = np.array(y_hat)	# transform to numpy array object

	y = y.astype(int)	# cast array elements to int (truncation)
	y_hat = y_hat.astype(int)	# cast array elements to int (truncation)

	y_hat = np.reshape(y_hat, (y_hat.shape[0],))	# flatten to 1D array
	y = np.reshape(y, (y.shape[0],))	# flatten to 1D array

	equal = (y == y_hat)	# array of booleans where equal[i] = y[i] == y_hat[i]
	accuracy = np.sum(equal) / y.shape[0]	# np.sum(equal) = count of True in equal
											# y.shape[0] = rows in y

	return accuracy


def replace_column_of_matrix(X, col_num, random_sample, ptb_strategy):	# replace the values in the column_number-th column in X with a fixed value
	"""
	Arguments: data matrix, n X k
	random sample: row of data matrix, 1 X k
	column number: 0 <-> k-1

	replace all elements of X[column number] X
	with random_sample[column_number]
	"""

	# need to implement random permutation.
	# need to implement perturbation strategy as a function
	# need a distance metrics file.
	# this probably does not work right now, I need to go through to fix.
	if col_num >= random_sample.shape[0]:
		raise ValueError("column {} entered. Column # should be"
						 "less than {}".format(col_num,
											   random_sample.shape[0]))

	if (ptb_strategy == "random-shuffle"):
		col = X[:, col_num]
		np.random.shuffle(col)
		X[:, col_num] = col
		return X
	
	# select the specific perturbation function chosen
	# obtain value from that function
	val_chosen = perturbation_strategy_dictionary[ptb_strategy](X,
																col_num,
																random_sample)
	constant_array = np.repeat(val_chosen, X.shape[0])
	X[:, col_num] = constant_array

	return X


def detect_feature_sign(predict_function, X, col_num):

	normal_output = predict_function(X)	# with predict_function = clf [example.py:32], this is an array of 0 and 1 (?)
	column_range = X[:, col_num].max() - X[:, col_num].min()	# range of values in the col_num-th column of X

	X[:, col_num] = X[:, col_num] + np.repeat(column_range, X.shape[0])	# add range to all values in the col_num-th column
	new_output = predict_function(X)

	diff = new_output - normal_output	# diff[i,j] = new_output[i,j] - normal_output[i,j]
	total_diff = np.mean(diff)	# mean over the flattened array

	if total_diff >= 0:
		return 1
	else:
		return -1


def main():
	pass


if __name__ == '__main__':
	main()
