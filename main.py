# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

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

# %% Exercise 2
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
plt.hist(dfY, bins=20)
plt.show()
# %% Price distribution grouped per neighbourhood
neighbourhoods = df['Neighborhood'].unique()
for val in neighbourhoods:
    fig, ax = plt.subplots()
    dfNbh = df[df['Neighborhood'] == val]['SalePrice']
    ax.hist(dfNbh, bins=15)
    ax.set_title(val)
    plt.show()
# %% Price distribution grouped per housing style
styles = df['House Style'].unique()
for val in styles:
    fig, ax = plt.subplots()
    dfStyle = df[df['House Style'] == val]['SalePrice']
    ax.hist(dfStyle, bins=15)
    ax.set_title(val)
    plt.show()
# %% Correlation matrix
for col_name in df.columns:
    if df[col_name].dtype == object:
        le = LabelEncoder()
        df[col_name] = le.fit_transform(df[col_name]).astype(int)

fig, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(ax=ax, data=df.corr(), xticklabels=True, yticklabels=True)
plt.show()
