# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import sys

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)

# %% initalizing dataframes
df = pd.read_csv('res/AmesHousing.csv')
dfCatigorical = pd.DataFrame()
dfNumeric = pd.DataFrame()

# Exercise 1
# %% Seperate df into classes of each of the variables
for col in df.columns:
    if df[col].dtype == object:
        dfCatigorical[col] = df[col]
    else:
        dfNumeric[col] = df[col]

# %% Descriptive/summary statistics for all continuous variables
dfNumeric.describe()

# %% Descriptive/summary statistics for all factor variables
dfCatigorical.describe()

# %% Missing values
df.isna().sum()

# Exercise 2
# %% Fill in missing values with values given in the exercise
dfCatigorical.fillna('100', inplace=True)

for col in dfNumeric.columns:
    med = dfNumeric[col].median()
    dfNumeric[col].fillna(med, inplace=True)

df.update(dfCatigorical)
df.update(dfNumeric)

# delete to spare memory consumption, can be removed if df's are needed in the future.
del dfCatigorical
del dfNumeric

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
for col_name in df.columns:
    if df[col_name].dtype == object:
        le = LabelEncoder()
        df[col_name] = le.fit_transform(df[col_name]).astype(int)

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
# %%
dfX = sub_df
dfY = sub_df['SalePrice']
dfX = dfX.drop('SalePrice', 1)
dfXTrain, dfXTest, dfYTrain, dfYTest = train_test_split(dfX, dfY, test_size=0.20, random_state=23)
dfXTest.sort_index(inplace=True)
dfXTrain.sort_index(inplace=True)
dfYTest.sort_index(inplace=True)
dfYTrain.sort_index(inplace=True)

lowest = (sys.maxsize, 0)
for i in range(1, 100):
    nnAlgo = KNeighborsClassifier(n_neighbors=i)
    nnAlgo.fit(dfXTrain, dfYTrain)
    yPredict = nnAlgo.predict(dfXTest)
    yActual = dfYTest.tolist()
    acc = mean_squared_error(yActual, yPredict, squared=False)
    # acc = sum([abs(yPredict[i] - yActual[i]) for i in range(len(yActual))]) / len(yActual)
    if acc < lowest[0]:
        lowest = (acc, i)
    print(acc)
print(f'Final Accuracy: \nAverage Difference: {lowest[0]}\nK-value: {lowest[1]}')

# %%
