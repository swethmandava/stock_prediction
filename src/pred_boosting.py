import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from parse import *
import matplotlib.pyplot as plt
import sys

#Skeleton code for xgboost.
def train(dtrain, dvalid, evallist, params, epochs, early_stopping_iterations):
	bst = xgb.train(
		params, dtrain, epochs, evallist, 
		early_stopping_rounds=early_stopping_iterations, verbose_eval=False)

	pred = bst.predict(dvalid)
	error = (pred - valid_y)
	error = np.multiply(error, error)
	avg_error = np.sum(error) * 1.0 / error.size
	return  bst, np.sum(error) * 1.0/error.size

#Perform cross validation for best set of parameters
def cross_validate_train(train_x, valid_x, train_y, valid_y):
	dtrain = xgb.DMatrix(train_x, label=train_y)
	dvalid = xgb.DMatrix(valid_x, label=valid_y)

	params = {}
	learning_rates = [0.001, 0.01, 0.05, 0.1]
	gammas = [0.001, 0.01, 0.1] #Loss to split further
	max_depths = [1, 2, 5, 10] #Maximum depth of tree
	subsamples = [0.5, 0.75, 1.0] #percentage of batch to use in each epoch
	lambdas = [0, 0.001, 0.01, 0.1, 1, 10] #L2 regularization
	alphas = [0, 0.001, 0.01, 0.1, 1, 10] #L1 regularization
	params['silent'] = 1
	epochs = 10000
	early_stopping_iterations = 10

	evallist = [(dvalid, 'eval'), (dtrain, 'train')]
	best_error = np.inf
	best_model = None
	best_params = dict(params)

	for learning_rate in learning_rates:
		for gamma in gammas:
			for max_depth in max_depths:
				for subsample in subsamples:
					for lambda_i in lambdas:
						for alpha in alphas:

							params['eta'] = learning_rate
							params['gamma'] = gamma 
							params['max_depth'] = max_depth 
							params['subsample'] = subsample
							params['lambda'] = lambda_i
							params['alpha'] = alpha

							bst, avg_error = train(dtrain, dvalid, evallist,
								params, epochs, early_stopping_iterations)
							
							print "Parameters are: ", params
							print "Average error is: ", avg_error, "\n"
							if avg_error < best_error:
								best_params = params
								best_error = avg_error
								best_model = bst
	best_model.save_model('../stock_data/boosting_results/model_0.model')

	# xgb.plot_importance(best_model) #plots histogram showing importance of features
	# plt.show()
	# xgb.plot_tree(bst, num_trees=5) #plots 2 trees

	#There are a ton more parameters we could experiment with here
	# xgboost.readthedocs.io/en/latest/parameter.html
	return bst, best_params

def incremental_train(x, y, valid_x, valid_y, model, params, model_name):
	dupdate = xgb.DMatrix(x, label=y)
	dvalid = xgb.DMatrix(valid_x, label=valid_y)
	evallist = [(dvalid, 'eval'), (dupdate, 'train')]
	model = xgb.train(params, dupdate, 1, xgb_model=model_name)
	return model

if __name__ == '__main__':
	filename = 'stock_data/Stocks/aple.us.txt'
	data_x, data_y, stream, _ = get_data(filename, initial_size=300)
	valid_split_ratio = 0.1

	#Splits in a chronological manner
	[num_samples, num_features] = data_x.shape
	index_split = int((1-valid_split_ratio) * num_samples)
	train_x = data_x[:index_split, :]
	train_y = data_y[:index_split]
	valid_x = data_x[index_split:, :]
	valid_y = data_y[index_split:]

	#Trains Prior Model
	# model, params = cross_validate_train(train_x, valid_x, train_y, valid_y)
	params = {'silent': 1, 'subsample': 1.0, 'eta': 0.1, 'alpha': 10, 'max_depth': 10, 'gamma': 0.1, 'lambda': 10}
	model = xgb.Booster({'nthread':4})
	model.load_model("../stock_data/boosting_results/model_0.model")
	if model is None:
		print "Model incorrect"
		sys.exit()

	time_series_y = []
	time_series_pred_y = []
	
	#Streaming Data:
	print params
	params["eta"] = 2 * params["eta"] #Give more weightage to new data
	day = 0
	model_name = '../stock_data/boosting_results/model_'+str(day)+".model"
	try:
		while True:
			new_x,new_y = stream.next()

			x = valid_x[[0], :]
			y = valid_y[[0]]

			valid_x = np.roll(valid_x, -1, axis=0)
			valid_y = np.roll(valid_y, -1, axis=0)
			valid_x[-1, :] = new_x
			valid_y[-1] = new_y

			dtest = xgb.DMatrix(x)
			y_pred = model.predict(dtest)
			time_series_y.append(y)
			time_series_pred_y.append(y_pred)
			model = incremental_train(x, y, valid_x, valid_y, model, params, model_name)

			day = day + 1
			model_name = '../stock_data/boosting_results/model_'+str(day)+".model"
			model.save_model(model_name)
			prev_x = x
			prev_y = y
	except StopIteration:
		pass
	np.save("../stock_data/boosting_results/time_series_y", time_series_y)
	np.save('../stock_data/boosting_results/time_series_pred_y', time_series_pred_y)