# %%
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import linear_model
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)

# %% initalizing dataframes
df = pd.read_csv('res/AmesHousing.csv')
df.drop(columns=['PID', 'Order'], inplace=True)
df['MS SubClass'] = df['MS SubClass'].apply(str)
df['Overall Cond'] = df['Overall Cond'].astype(str)
df['Yr Sold'] = df['Yr Sold'].astype(str)
df['Mo Sold'] = df['Mo Sold'].astype(str)
df_num = df.select_dtypes(exclude='object').copy()
df_cat = df.select_dtypes(include='object').copy()

# Exercise 1
# %% Descriptive/summary statistics for all continuous variables
df_num.describe()

# %% Descriptive/summary statistics for all factor variables
df_cat.describe()

# %% Missing values
missing_val = df.isnull().sum()
missing_val_per = (df.isnull().sum() / len(df)) * 100

# Exercise 2
# %% Fill in missing values with values given in the exercise
# df_cat.fillna('100', inplace=True)
# df_num = df_num.apply(lambda x: x.fillna(x.mean()))
#
# df.update(df_cat)
# df.update(df_num)
#
# # delete to spare memory consumption, can be removed if df's are needed in the future.
# del df_cat
# del df_num

# %% Handle missing values
# According to the documentation on the ames housing data data set some NaN values are replaced with None or 0,
# depending on the data type of the column.
df = df.dropna(axis=0, subset=['Electrical', 'Garage Area'])

basement_num_cols = ['BsmtFin SF 1', 'BsmtFin SF 2', 'Bsmt Unf SF', 'Total Bsmt SF', 'Bsmt Full Bath', 'Bsmt Half Bath']
df[basement_num_cols] = df[basement_num_cols].fillna(0)

basement_cat_cols = ['Bsmt Qual', 'Bsmt Cond', 'Bsmt Exposure', 'BsmtFin Type 1', 'BsmtFin Type 2']
df[basement_cat_cols] = df[basement_cat_cols].fillna('None')

df["Mas Vnr Type"] = df["Mas Vnr Type"].fillna("None")
df["Mas Vnr Area"] = df["Mas Vnr Area"].fillna(0)

garage_cat_cols = ['Garage Type', 'Garage Finish', 'Garage Qual', 'Garage Cond']
df[garage_cat_cols] = df[garage_cat_cols].fillna('None')
df['Garage Yr Blt'] = df['Garage Yr Blt'].fillna(0)

df['Fireplace Qu'] = df['Fireplace Qu'].fillna('None')

df['Lot Frontage'] = df['Lot Frontage'].fillna(df['Lot Frontage'].mean())

df = df.drop(columns=['Fence', 'Alley', 'Misc Feature', 'Pool QC'])
# Excercise 3
# %% Descriptive summary of SalePrice
dfY = df['SalePrice']
dfY.describe()

# %% Visualize distribution of SalePrice in a histogram
box_plt = plt.boxplot(dfY)
whiskers_data = [item.get_ydata() for item in box_plt['whiskers']]
min_box, max_box = whiskers_data[0].min(), whiskers_data[1].max()
plt.show()

# %% Delete outliers
# TODO: Better outlier handling
df = df[((df['SalePrice'] >= min_box) & (df['SalePrice'] <= max_box))]

# %% Price distribution grouped per neighbourhood
plt.figure(figsize=(10, 20))
sns.boxplot(data=df, x='SalePrice', y='Neighborhood')
plt.tight_layout()
plt.show()

# %% Price distribution grouped per housing style
plt.figure(figsize=(10, 12))
sns.boxplot(data=df, x='SalePrice', y='House Style')
plt.tight_layout()
plt.show()

# Data cleaning and selecting
# %% Select relevant columns from dataset (creating subset)
cm = df.corr()
sale_price_corr = cm['SalePrice']
fig, ax = plt.subplots(figsize=(15, 15))
sns.heatmap(ax=ax, data=cm, xticklabels=True, yticklabels=True)
plt.tight_layout()
plt.show()

# %% drop columns that are NOT needed -0.1 > 0.1
# columns that are NOT needed -0.1 > 0.1 :
df = df.drop(columns=['Pool Area', 'Mo Sold', '3Ssn Porch', 'BsmtFin SF 2', 'Misc Val', 'Yr Sold', 'Bsmt Half Bath',
                      'Low Qual Fin SF'])

# Excercise 4
# %% prepare dataframes
df_num = df.select_dtypes(exclude='object').copy()
df_cat = df.select_dtypes(include='object').copy()
df_cat = pd.get_dummies(df_cat, drop_first=True)
sub_df = pd.concat([df_num, df_cat], axis=1)
df_X = sub_df.drop(columns=['SalePrice'])
df_Y = sub_df['SalePrice']
df_X_train, df_X_test, df_y_train, df_y_test = train_test_split(df_X, df_Y, test_size=0.20, random_state=23)
df_X_train.sort_index(inplace=True)
df_X_test.sort_index(inplace=True)
df_y_train.sort_index(inplace=True)
df_y_test.sort_index(inplace=True)

# %% generate model
lowest = (sys.maxsize, 0)
scores = []
for i in range(1, 100):
    knn = KNeighborsRegressor(n_neighbors=i)
    knn.fit(df_X_train, df_y_train)
    actual_y = np.array(df_y_test)
    predicted_y = knn.predict(df_X_test)
    scores.append(mean_squared_log_error(actual_y, predicted_y, squared=False))
scores = pd.Series(scores)

# %% generate Lasso
lowest = (sys.maxsize, 0)
scores = []
for i in np.arange(0, 5, 0.1):
    lassoAlgo = linear_model.Lasso(alpha=i)
    lassoAlgo.fit(df_X_train, df_y_train)
    actual_y = np.array(df_y_test)
    predicted_y = lassoAlgo.predict(df_X_test)
    scores.append([i, mean_squared_log_error(actual_y, predicted_y, squared=False)])
scores = pd.DataFrame(scores)
