import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve

# Plot validation curve
def plot_validation_curve(estimator, title, X, y, param_name, param_range, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
	train_scores, test_scores = validation_curve(estimator, X, y, param_name, param_range, cv)
	train_mean = np.mean(train_scores, axis=1)
	train_std = np.std(train_scores, axis=1)
	test_mean = np.mean(test_scores, axis=1)
	test_std = np.std(test_scores, axis=1)
	plt.plot(param_range, train_mean, color='r', marker='o', markersize=5, label='Training score')
	plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha=0.15, color='r')
	plt.plot(param_range, test_mean, color='g', linestyle='--', marker='s', markersize=5, label='Validation score')
	plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, alpha=0.15, color='g')
	plt.grid() 
	plt.xscale('log')
	plt.legend(loc='best') 
	plt.xlabel('Parameter') 
	plt.ylabel('Score') 
	plt.ylim(ylim)
	return plt