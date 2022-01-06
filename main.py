# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_log_error

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)

# %% initalizing dataframes
df = pd.read_csv('res/AmesHousing.csv')
df['MS SubClass'] = df['MS SubClass'].apply(str)
df_num = df.select_dtypes(exclude='object').copy()
df_cat = df.select_dtypes(include='object').copy()

# Exercise 1
# %% Descriptive/summary statistics for all continuous variables
df_num.describe()

# %% Descriptive/summary statistics for all factor variables
df_cat.describe()

# %% Missing values
df.isna().sum()

# Exercise 2
# %% Fill in missing values with values given in the exercise
df_cat.fillna('100', inplace=True)

for col in df_num.columns:
    med = df_num[col].median()
    df_num[col].fillna(med, inplace=True)

df.update(df_cat)
df.update(df_num)

# delete to spare memory consumption, can be removed if df's are needed in the future.
del df_cat
del df_num

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
# for col_name in df.columns:
#     if df[col_name].dtype == object:
#         le = LabelEncoder()
#         df[col_name] = le.fit_transform(df[col_name]).astype(int)

cm = df.corr()
sale_price_corr = cm['SalePrice']
fig, ax = plt.subplots(figsize=(15, 15))
sns.heatmap(ax=ax, data=cm, xticklabels=True, yticklabels=True)
plt.tight_layout()
plt.show()

# %% drop columns that are NOT needed -0.1 > 0.1
# columns that are NOT needed -0.1 > 0.1 :
# PID: unique key column
# Pool Area: insufficient correlation with sale price
# Condition 1: " "
# Condition 2: " "
# Roof Matl: " "
# Pool Area: " "
# Land Slope: " "
# Street: " "
# Pool QC: " "
# Mo Sold: " "
# 3Ssn Porch: " "
# BsmtFin SF 2: " "
# Misc Val: " "
# Yr Sold: " "
# Order: " "
# Utilities: " "
# Land Contour: " "
# Low Qual Fin SF: " "
# BsmtFin Type 1: " "
# Sale Type: " "
# Bldg Type: " "
# Lot Config: " "
# Misc Feature: " "
# MS SubClass: " "
# Alley: " "
# Heating: " "
# Bsmt Half Bath: will be added to Half Bath

sub_df = df.drop(columns=['Pool Area', 'Condition 1', 'Condition 2', 'Roof Matl', 'Pool Area', 'Land Slope',
                          'Street', 'Pool QC', 'Mo Sold', '3Ssn Porch', 'BsmtFin SF 2', 'Misc Val', 'Yr Sold', 'Order',
                          'Utilities', 'Land Contour', 'Low Qual Fin SF', 'BsmtFin Type 1',
                          'Sale Type', 'Bldg Type', 'Lot Config', 'Misc Feature', 'MS SubClass', 'Alley', 'Heating'])

fig, ax = plt.subplots(figsize=(15, 15))
sns.heatmap(ax=ax, data=sub_df.corr(), xticklabels=True, yticklabels=True)
plt.tight_layout()
plt.show()

# %% drop columns that MIGHT NOT be needed 0.1 > 0.2 & -0.2 > -0.1 :
# Columns that MIGHT NOT be needed 0.1 > 0.2 & -0.2 > -0.1 :
# Bsmt Unf SF: insufficient correlation with sale price
# House Style: " "
# Bsmt Cond: " "
# Bedroom AbvGr: " "
# BsmtFin Type 2: " "
# Exterior 1st: " "
# Exterior 2nd: " "
# Exter Cond: " "
# Functional: " "
# Screen Porch: " "
# Overall Cond: " "
# Kitchen AbvGr: " "
# Enclosed Porch: " "
# Mas Vnr Type: " "
# MS Zoning: " "
# Fence: " "

sub_df.drop(columns=['Bsmt Unf SF', 'House Style', 'Bsmt Cond', 'Bedroom AbvGr', 'BsmtFin Type 2', 'Exterior 1st',
                     'Exterior 2nd', 'Exter Cond', 'Functional', 'Screen Porch', 'Overall Cond', 'Kitchen AbvGr',
                     'Enclosed Porch', 'Mas Vnr Type', 'MS Zoning', 'Fence'], inplace=True)

fig, ax = plt.subplots(figsize=(15, 15))
sns.heatmap(ax=ax, data=sub_df.corr(), xticklabels=True, yticklabels=True)
plt.tight_layout()
plt.show()

# %% drop columns that should have no influence on price
# PID
# Garage Cond -> Garage Qual has equal correlation
# Garage Cars -> Garage Area has equal correlation and better significance
# Bsmt Half bath -> will be added to half bath
# Bsmt Full Bath -> will be added to full bath
# Fireplace Qu -> Fireplaces has higher significance

sub_df['Full Bath'] = sub_df['Full Bath'] + sub_df['Bsmt Full Bath']
sub_df['Half Bath'] = sub_df['Half Bath'] + sub_df['Bsmt Half Bath']

sub_df.drop(columns=['PID', 'Garage Cond', 'Garage Cars', 'Bsmt Full Bath', 'Bsmt Half Bath',
                     'Fireplace Qu'],
            inplace=True)

fig, ax = plt.subplots(figsize=(15, 15))
sns.heatmap(ax=ax, data=sub_df.corr(), xticklabels=True, yticklabels=True)
plt.tight_layout()
plt.show()
# Excercise 4
# %% prepare dataframes
df_num = df.select_dtypes(exclude='object').copy()
df_cat = df.select_dtypes(include='object').copy()
df_cat = pd.get_dummies(df_cat, drop_first=True)
sub_df = pd.concat([df_num, df_cat], axis=1)
df_X = sub_df.drop(columns=['SalePrice', 'PID'])
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
    # acc = mean_squared_error(df_y_test.values, yPredict, squared=False)
    # # acc = sum([abs(yPredict[i] - yActual[i]) for i in range(len(yActual))]) / len(yActual)
    # if acc < lowest[0]:
    #     lowest = (acc, i)
    # print(acc)
# print(f'Final Accuracy: \nAverage Difference: {lowest[0]}\nK-value: {lowest[1]}')
scores = pd.Series(scores)

# %%
