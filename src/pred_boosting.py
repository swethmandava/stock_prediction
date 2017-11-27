import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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
	epochs = 100

	#There are a ton more parameters we could experiment with here
	# xgboost.readthedocs.io/en/latest/parameter.html

	evallist = [(dtest, 'eval'), (dtrain, 'train')]
	bst = xgb.train(param, dtrain, epochs, evallist, early_stopping_rounds=20,
		verbose_eval=10)
	bst.save_model('../stock_data/boosting_results/best_model.model')


	pred = bst.predict(dvalid)
	df_pred = pd.DataFrame(columns=['data', 'predict', 'target'])
	df_pred['data'] = valid_x
	df_pred['predict'] = pred
	df_pred['target'] = valid_y

	df_errors = df_pred.loc[df_pred['predict'] != df_pred['target']]
	df_errors.to_csv("../stock_data/boosting_results/errors.csv", index=False)

	bst.plot_importance(bst) #plots histogram showing importance of features
	bst.plot_tree(bst, num_trees=5) #plots 2 trees

if __name__ == '__main__':
	train_x, valid_x, train_y, valid_y= train_test_split(
		data_x, data_y, test_size=0.1, random_state=2017)
	train(train_x, valid_x, train_y, valid_y)