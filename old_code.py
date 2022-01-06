# for col_name in df.columns:
#     if df[col_name].dtype == object:
#         le = LabelEncoder()
#         df[col_name] = le.fit_transform(df[col_name]).astype(int)

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