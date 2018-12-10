import pandas as pd
import numpy as np

# Initial data inspection
# inspect.inspect_data(X)

# Observations:
# - PassengerId: Drop
# - Pclass 
# - Name: Drop
# - Sex: Categorise
# - Age has null values which can be replaced
# - Sibsp (Sibling / Spouse)
# - Parch (Parent / Children)
# - Ticket: Drop
# - Fare
# - Cabin has too many null values and can be dropped
# - Embarked: Categorise and drop the people who did not embark (embarked = null)

def process_data(X):
	# PassengerId
	X.drop(['PassengerId'], inplace=True, axis=1)

	# Sex - Categorise Sex into Male / Female / Child
	def get_person(passenger):
		age, sex = passenger
		if age < 16:
			return 'child'
		else: 
			return sex
    
	X['Person'] = X[['Age','Sex']].apply(get_person,axis=1)
	X['Person'] = pd.Categorical(X['Person'])
	X.drop(['Sex'], inplace=True, axis=1)

	# Title - Create new title feature from the name
	X['Title']=0
	X['Title'] = X['Name'].str.extract('([A-Za-z]+)\.', expand=False)

	means = X.groupby('Title')['Age'].mean()
	map_means = means.to_dict()

	idx_nan_age = X.loc[np.isnan(X['Age'])].index
	X.loc[idx_nan_age,'Age'].loc[idx_nan_age] = X['Title'].loc[idx_nan_age].map(map_means)

	# Keep track of the modified null age values
	X['Imputed'] = 0
	X.at[idx_nan_age.values, 'Imputed'] = 1

	# Replace missing Age values with the estimated age from the title
	X['Age'].fillna(X['Title'].map(map_means), inplace=True)
	X['Age'].fillna(X.Age.median(), inplace=True)
	
	# Segment Age into 3 categories
	X['Age'] = pd.cut(X['Age'], bins=[0, 12, 50, 200], labels=['Child','Adult','Elder'])

	# Title - Parse the titles
	X['Title'] = X['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
	X['Title'] = X['Title'].replace('Mlle', 'Miss')
	X['Title'] = X['Title'].replace('Ms', 'Miss')
	X['Title'] = X['Title'].replace('Mme', 'Mrs')

	X['Title'] = pd.Categorical(X['Title'])

	# Embarked
	X['Embarked'] = X['Embarked'].fillna('S')
	X['Embarked'] = pd.Categorical(X['Embarked'])

	# Create an IsAlone feature for people without families
	X['FamilySize'] = X['SibSp'] + X['Parch']
	X['IsAlone'] = 0
	X['IsAlone'] = X['FamilySize'].loc[X['FamilySize'] == 1] = 1
	X = X.drop(['SibSp','Parch'], axis=1)

	# Replace missing Fare values with the fare median
	fare_median = X['Fare'].dropna().median()
	X['Fare'].fillna(fare_median, inplace=True)

	# Drop: Name, Cabin, Ticket
	X.drop(['Name'], inplace=True, axis=1)
	X.drop(['Ticket'], inplace=True, axis=1)
	X.drop(['Cabin'], inplace=True, axis=1)

	X.groupby(['Person', 'Pclass'])['Age']

	# Transform categorical data into dummies
	X = pd.get_dummies(X)

	return X