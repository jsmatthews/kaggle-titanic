import numpy as np
import pandas as pd
from scipy.stats import boxcox
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# def poly_data(X_train, X_test, Y_train):
# 	# Apply Box-Cox transformation
# 	X_train_transformed = X_train.copy()
# 	X_train_transformed['Fare'] = boxcox(X_train_transformed['Fare'] + 1)[0]
# 	X_test_transformed = X_test.copy()
# 	X_test_transformed['Fare'] = boxcox(X_test_transformed['Fare'] + 1)[0]

# 	# Rescale data
# 	scaler = MinMaxScaler()
# 	X_train_transformed_scaled = scaler.fit_transform(X_train_transformed)
# 	X_test_transformed_scaled = scaler.transform(X_test_transformed)

# 	# Get polynomial features
# 	poly = PolynomialFeatures(degree=2).fit(X_train_transformed)
# 	X_train_poly = poly.transform(X_train_transformed_scaled)
# 	X_test_poly = poly.transform(X_test_transformed_scaled)

def scale_data(X_train, X_cv, Y_train, X_test):

	# Apply Box-Cox transformation
	X_train_transformed = X_train.copy()
	X_train_transformed['Fare'] = boxcox(X_train_transformed['Fare'] + 1)[0]
	X_cv_transformed = X_cv.copy()
	X_cv_transformed['Fare'] = boxcox(X_cv_transformed['Fare'] + 1)[0]
	X_test_transformed = X_test.copy()
	X_test_transformed['Fare'] = boxcox(X_test_transformed['Fare'] + 1)[0]

	# Rescale data
	scaler = MinMaxScaler()
	X_train_transformed_scaled = scaler.fit_transform(X_train_transformed)
	X_cv_transformed_scaled = scaler.transform(X_cv_transformed)
	X_test_transformed_scaled = scaler.transform(X_test_transformed)

	# Get polynomial features
	poly = PolynomialFeatures(degree=2).fit(X_train_transformed)
	X_train_poly = poly.transform(X_train_transformed_scaled)
	X_cv_poly = poly.transform(X_cv_transformed_scaled)
	X_test_poly = poly.transform(X_test_transformed_scaled)

	# Select features using chi-squared test
	## Get score using original model
	logreg = LogisticRegression(C=1)
	logreg.fit(X_train, Y_train)
	scores = cross_val_score(logreg, X_train, Y_train, cv=10)
	# print('CV accuracy (original): %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
	highest_score = np.mean(scores)
	k_features_highest_score = X_train_poly.shape[1]
	std = 0

	## Get score using models with feature selection
	for i in range(1, X_train_poly.shape[1]+1, 1):
		# Select i features
		select = SelectKBest(score_func=chi2, k=i)
		select.fit(X_train_poly, Y_train)
		X_train_poly_selected = select.transform(X_train_poly)

		# Model with i features selected
		logreg.fit(X_train_poly_selected, Y_train)
		scores = cross_val_score(logreg, X_train_poly_selected, Y_train, cv=10)
		# print('CV accuracy (number of features = %i): %.3f +/- %.3f' % (i, np.mean(scores), np.std(scores)))
		
		# Save results if best score
		if np.mean(scores) > highest_score:
			highest_score = np.mean(scores)
			std = np.std(scores)
			k_features_highest_score = i
		elif np.mean(scores) == highest_score:
			if np.std(scores) < std:
				highest_score = np.mean(scores)
				std = np.std(scores)
				k_features_highest_score = i
			
	# Print the number of features
	print(f'Number of features when highest score: {k_features_highest_score}')

	# Select features
	select = SelectKBest(score_func=chi2, k=k_features_highest_score)
	select.fit(X_train_poly, Y_train)

	X_train_poly_selected = select.transform(X_train_poly)
	X_cv_poly_selected = select.transform(X_cv_poly)
	X_test_poly_selected = select.transform(X_test_poly)

	return (X_train_poly_selected, X_cv_poly_selected, X_test_poly_selected)