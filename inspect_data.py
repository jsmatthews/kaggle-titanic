import numpy as np
import pandas as pd

def inspect_data(data):
	num_features = len(data.columns)
	print(f'Number of features: {num_features}')

	for feature in range(0, num_features):
		column = data.iloc[:, feature]
		print(f'Feature: {column.name}')
		print(f'Shape = {column.size}')
		print(f'Missing Values = {column.isnull().sum()}')
		print(f'Data type = {column.dtype}')
		if(column.dtype != object):
			print(f'Median: {column.median()}')
		print('\n')