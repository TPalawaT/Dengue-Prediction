'''
In this code, we will be predicting dengue in San Jose, Peurto Rico and Iquitos, Peru.
We will be using 2 univariate methods, 2 wrapper methods and PCA for feature selection.
The features we obtain will be used in Linear Regression.
To check the efficiency of our model we will use Mean Square Error.
'''


import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_absolute_error

#Reading CSV Files
df_features = pd.read_csv('dengue_features_train.csv')
df_labels = pd.read_csv('dengue_labels_train.csv')

print(df_features.isnull().sum())
print(df_features.shape)
df_features = df_features.dropna()
print(df_features.shape)

df_test = pd.read_csv('dengue_features_test.csv')
print(df_features.head())
df_features.drop(df_features.columns[np.r_[0,3]], axis=1, inplace=True)
print(df_features.head())

#print(type(df_features['week_start_date'][1]))
#print(len(df_labels['total_cases']))

#adding total number of cases in a week and sorting it in descending order to remove outliers.
#Removed 10% of the top values which comes down to 145 values. 
temp = pd.merge(df_features, df_labels, how='left')
temp.sort_values(['total_cases'], ascending=False, inplace=True)
temp = temp.iloc[145:]
temp.drop(columns = ["city"], inplace=True)
temp.to_csv('features.csv', index=False)
print(temp.head())

features = np.array(temp.iloc[:,:22])
labels = np.array(temp.iloc[:,22])

print(features[0])

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)

ridge = Ridge(alpha=0.0005)

ridge.fit(features_train,labels_train)
print("Coefficients:",ridge.coef_)
print("Intercept:", ridge.intercept_)

col = temp.columns

x = pd.DataFrame(features_train, columns=col[:-1])
y = pd.DataFrame(features_test, columns=col[:-1])
labels_pred = ridge.predict(features_test)

print(mean_absolute_error(labels_test, labels_pred))
print(temp.corr()['year']['year'])

print(np.shape(features_train[0:]))

plt.scatter(x['weekofyear'], labels_train, color='b', label='Train')
plt.scatter(y['weekofyear'], labels_test, color='r', label='Test')
plt.plot(features_train, ridge.predict(features_train))
#plt.plot(features_test, ridge.predict(features_test))
plt.xlabel('Week')
plt.ylabel('Cases')
plt.show()
'''
plt.matshow(temp.corr())
plt.xticks(range(len(temp.columns)), temp.columns)
plt.yticks(range(len(temp.columns)), temp.columns)
plt.colorbar()
plt.show()
'''