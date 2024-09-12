import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("housing.csv")
data.dropna(inplace=True)  # Drop rows with missing values

from sklearn.model_selection import train_test_split

# Separate features and target variable
x = data.drop(['median_house_value'], axis=1)
y = data['median_house_value']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
train_data = x_train.join(y_train)

# Apply log transformation to reduce skewness
train_data['total_rooms'] = np.log(train_data['total_rooms'] + 1)
train_data['total_bedrooms'] = np.log(train_data['total_bedrooms'] + 1)
train_data['population'] = np.log(train_data['population'] + 1)
train_data['households'] = np.log(train_data['households'] + 1)

# One-hot encode the 'ocean_proximity' categorical feature
train_data = train_data.join(pd.get_dummies(train_data.ocean_proximity)).drop(["ocean_proximity"], axis=1)
train_columns = train_data.columns

# Plot the data
plt.figure(figsize=(15, 8))
sns.scatterplot(x="latitude", y="longitude", data=train_data, hue="median_house_value", palette="coolwarm")
#plt.show()

# Create new features
train_data['bedroom_ratio'] = train_data['total_bedrooms'] / train_data['total_rooms']
train_data['household_rooms'] = train_data['total_rooms'] / train_data['households']

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Standardize the features
scaler = StandardScaler()
x_train, y_train = train_data.drop(['median_house_value'], axis=1), train_data['median_house_value']
x_train_s = scaler.fit_transform(x_train)

# Train a linear regression model
regression = LinearRegression()
regression.fit(x_train_s, y_train)

# Prepare the test data
test_data = x_test.join(y_test)
test_data['total_rooms'] = np.log(test_data['total_rooms'] + 1)
test_data['total_bedrooms'] = np.log(test_data['total_bedrooms'] + 1)
test_data['population'] = np.log(test_data['population'] + 1)
test_data['households'] = np.log(test_data['households'] + 1)

# One-hot encode the 'ocean_proximity' categorical feature in test data
test_data = test_data.join(pd.get_dummies(test_data.ocean_proximity)).drop(["ocean_proximity"], axis=1)
test_data.reindex(columns=train_columns, fill_value=0)  # Reindex columns to match train data

# Create new features in test data
test_data['bedroom_ratio'] = test_data['total_bedrooms'] / test_data['total_rooms']
test_data['household_rooms'] = test_data['total_rooms'] / test_data['households']

# Standardize the test features
x_test, y_test = test_data.drop(['median_house_value'], axis=1), test_data['median_house_value']
x_test_s = scaler.transform(x_test)

# Evaluate the linear regression model
print("Linear Regression score: ", regression.score(x_test_s, y_test))

from sklearn.ensemble import RandomForestRegressor

# Train a random forest regressor
forest = RandomForestRegressor()
forest.fit(x_train_s, y_train)
print("Random Forest score: ", forest.score(x_test_s, y_test))

from sklearn.model_selection import GridSearchCV

# Perform grid search to find the best hyperparameters for the random forest
forest = RandomForestRegressor()
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': [2, 4],
    'min_samples_split': [None, 4, 8]
}
grid_search = GridSearchCV(forest, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(x_train_s, y_train)

# Evaluate the best random forest model
best_forest = grid_search.best_estimator_
print("Improved Random Forest score: ", best_forest.score(x_test_s, y_test))