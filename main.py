#%%
import pandas as pd
import seaborn as sns

#%% Exercise 1
df = pd.read_csv('res/AmesHousing.csv')
dfCatigorical = pd.DataFrame()
dfNumeric = pd.DataFrame()
# %% 
for col in df.columns:
    if df[col].dtype == object:
        dfCatigorical[col] = df[col]
    else:
        dfNumeric[col] = df[col]

# %%
dfNumeric.describe()
# %%
dfCatigorical.describe()
