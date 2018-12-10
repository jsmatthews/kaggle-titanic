import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from inspect_data import inspect_data as inspect
from plot_learning_curve import plot_learning_curve
from plot_validation_curve import plot_validation_curve
from feature_engineering import process_data
from scale_data import scale_data

#################
### Load Data ###
#################

X = pd.read_csv('./train.csv')
X_test = pd.read_csv('./test.csv')
prediction_passenger_id = X_test['PassengerId'].values

print(f'Shape of X: {X.shape}')
print(f'Shape of X_test: {X_test.shape}')

# Initial data inspection
# inspect(X)

##############################################
### Feature Engineering: Prepare TEST data ###
##############################################

X_test = process_data(X_test)

###############################################
### Feature Engineering: Prepare TRAIN data ###
###############################################

X = process_data(X)
y = X['Survived']
X = X[X.loc[:, X.columns != 'Survived'].columns]

###################
### Train Model ###
###################

kf = KFold(n_splits=11)
split_data = kf.split(X)

final_predictions = pd.DataFrame(np.zeros((len(X_test.index), 11)))

count = 0
training_set_accuracy = 0

for train, validation in split_data:
	# print('train: %s, validation: %s' % (X.loc[train, :].head(), X.loc[validation, :].head()))
	X_train = X.iloc[train]
	Y_train = y.iloc[train]

	X_cv = X.iloc[validation]
	Y_cv = y.iloc[validation]

	result = scale_data(X_train, X_cv, Y_train, X_test)
	X_train = result[0]
	X_cv = result[1]
	X_test_new = result[2]

	# LogisticRegressor
	logreg = LogisticRegression(C=1)
	logreg.fit(X_train, Y_train)
	predictions = logreg.predict(X_cv)
	final_predictions[count] = logreg.predict(X_test_new)

	new_training_set_accuracy = (predictions == Y_cv).mean()
	print(f'Training Set Accuracy: {new_training_set_accuracy:.3f}')
	
	if(new_training_set_accuracy > training_set_accuracy):
		best_model = pd.DataFrame({ 'PassengerId': prediction_passenger_id, 'Survived': final_predictions[count]})
		best_model.to_csv("./best_model.csv", index=False)
		training_set_accuracy = new_training_set_accuracy

	count += 1
	print(final_predictions.head(10))

final_predictions['mean'] = round(final_predictions.mean(axis=1)).astype(int)

################################
### Generate submission file ###
################################

submission = pd.DataFrame({ 'PassengerId': prediction_passenger_id, 'Survived': final_predictions[9]})
submission.to_csv("./submission.csv", index=False)
