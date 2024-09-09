import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("housing.csv")
data.dropna(inplace=True)


from sklearn.model_selection import train_test_split

x = data.drop(['median_house_value'], axis=1)
y = data['median_house_value']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
train_data = x_train.join(y_train)

train_data['total_rooms'] = np.log(train_data['total_rooms'])
train_data['total_bedrooms'] = np.log(train_data['total_bedrooms'])
train_data['population'] = np.log(train_data['population'])
train_data['households'] = np.log(train_data['households'])

train_data = train_data.join(pd.get_dummies(train_data.ocean_proximity)).drop(["ocean_proximity"], axis=1)

##plt.figure(figsize=(15, 8))
#sns.scatterplot(x="latitude", y="longitude", data=train_data, hue="median_house_value", palette="coolwarm")
#plt.show()

train_data['bedroom_ratio'] = train_data['total_bedrooms'] / train_data['total_rooms']
train_data['household_rooms'] = train_data['total_rooms'] / train_data['households']

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train, y_train = train_data.drop(['median_house_value'], axis=1), train_data['median_house_value']
x_train_s = scaler.fit_transform(x_train)

regression = LinearRegression()
regression.fit(x_train, y_train)

test_data = x_test.join(y_test)

test_data['total_rooms'] = np.log(test_data['total_rooms'])
test_data['total_bedrooms'] = np.log(test_data['total_bedrooms'])
test_data['population'] = np.log(test_data['population'])
test_data['households'] = np.log(test_data['households'])

test_data = test_data.join(pd.get_dummies(test_data.ocean_proximity)).drop(["ocean_proximity"], axis=1)

test_data['bedroom_ratio'] = test_data['total_bedrooms'] / test_data['total_rooms']
test_data['household_rooms'] = test_data['total_rooms'] / test_data['households']

x_test, y_test = test_data.drop(['median_house_value'], axis=1), test_data['median_house_value']
x_test_s = scaler.fit_transform(x_train)
regression.score(x_test_s, y_test)