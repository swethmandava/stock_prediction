import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from parse import *

#Skeleton code for xgboost.
#Perform cross validation for best set of parameters

def train(train_x, valid_x, train_y, valid_y):
	dtrain = xgb.DMatrix(train_x, label=train_y)
	dvalid = xgb.DMatrix(valid_x, label=valid_y)

	params = {}
	params['eta'] = 0.1 #learning rate
	params['gamma'] = 0.01 #Loss to split further
	params['max_depth'] = 5 #Maximum depth of tree
	params['subsample'] = 0.5 #percentage of batch to use in each epoch
	params['lambda'] = 0.01 #L2 regularization
	params['alpha'] = 0 #L1 regularization
	params['silent'] = 1
	epochs = 1000

	#There are a ton more parameters we could experiment with here
	# xgboost.readthedocs.io/en/latest/parameter.html

	evallist = [(dvalid, 'eval'), (dtrain, 'train')]
	bst = xgb.train(params, dtrain, epochs, evallist, early_stopping_rounds=20,
		verbose_eval=10)
	bst.save_model('../stock_data/boosting_results/best_model.model')


	pred = bst.predict(dvalid)
	# df_pred = pd.DataFrame(columns=['data', 'predict', 'target', 'square_error'])
	# df_pred['data'] = valid_x
	# df_pred['predict'] = pred
	# df_pred['target'] = valid_y
	error = (pred - valid_y)
	error = np.multiply(error, error)
	# df_pred['square_error'] = error
	print "Average Validation Error is", np.sum(error)*1.0/error.size

	# bst.plot_importance(bst) #plots histogram showing importance of features
	# bst.plot_tree(bst, num_trees=5) #plots 2 trees

if __name__ == '__main__':
	filename = 'stock_data/Stocks/aple.us.txt'
	data_x, data_y, _ = get_dataset(filename)
	train_x, valid_x, train_y, valid_y= train_test_split(
		data_x, data_y, test_size=0.1, random_state=2017)
	train(train_x, valid_x, train_y, valid_y)