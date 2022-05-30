#!/usr/bin/env python
# coding: utf-8

# # Crime Rate Analysis

# In[1]:


folder = "/home/alexbui/workspace/HandbookForDatascience/notebooks/data/crime_rate"


# In[2]:


import pandas as pd


# ## 1. Data loading & preparation
# 
# - Property_crime = burglary + larcency + motor_vehicle_theft
# - We will remove these three columns and recalculate total_crime & overall crime rate

# In[3]:


cr_data = pd.read_csv(folder + "/estimated_crimes.csv")


# In[ ]:


cr_data = cr_data[cr_data['year'] <= 2019]
cr_data.rename(columns={'state_name':'state name'}, inplace=True)
cr_data.drop(['burglary', 'larceny', 'motor_vehicle_theft'], axis=1, inplace=True)


# In[ ]:


cr_data['total crime'] = cr_data['violent_crime'] + cr_data['homicide']                             + cr_data['robbery'] + cr_data['aggravated_assault'] + cr_data['property_crime']
cr_data['overall crime rate'] = cr_data['total crime'] / cr_data['population']


# In[ ]:


cr_data


# In[ ]:


demography = pd.read_csv(folder + "/demography.csv")
demography.columns = [c.lower() for c in demography.columns]


# We will remove all columns with absolute values as we want to estimate crime rate.

# In[ ]:


def remove_absolute_columns(df, keeps):
    rm_cols = [c for c in df.columns if not c in keeps and not 'percent' in c]
    df.drop(rm_cols, axis=1, inplace=True)

def clean_column_name(df):
    cols = df.columns
    new_cols = []
    for c in cols:
        c = c.replace("percent!!percentage of","")            .replace("percent of", "")            .replace("percent in ","")            .replace("percent","")
        if c[0] == ' ':
            c = c[1:]
        new_cols.append(c)
    df.columns = new_cols


# In[ ]:


keeps = ['year', 'state name', 'total population']
remove_absolute_columns(demography, keeps)


# In[ ]:


clean_column_name(demography)


# In[ ]:


demography


# ***Load economic data***

# In[ ]:


economics = pd.read_csv(folder + "/economics.csv")
economics.columns = [c.lower() for c in economics.columns]
remove_absolute_columns(economics, ['state name', 'year', 'population 16 years and over'])
economics.drop(['percent of civilian labor force','percent!!income and benefits !!less than $10,000',            'percent!!income and benefits !!$10,000 to $14,999','percent!!income and benefits !!$15,000 to $24,999',            'percent!!income and benefits !!$25,000 to $34,999','percent!!income and benefits !!$35,000 to $49,999',            'percent!!income and benefits !!$50,000 to $74,999','percent!!income and benefits !!$75,000 to $99,999',            'percent!!income and benefits !!$100,000 to $149,999','percent!!income and benefits !!$150,000 to $199,999',            'percent!!income and benefits !!$200,000 or more'], axis=1, inplace=True)
clean_column_name(economics)


# In[ ]:


eco_new_names = {
'income and benefits total households' : 'total inc & bnf',
'income and benefits less than $10,000' : 'inc & bnf lt 10k',
'income and benefits $10,000 to $14,999' : 'inc & bnf 10k-14.9K',
'income and benefits $15,000 to $24,999' : 'inc & bnf 15k-24.9K',
'income and benefits $25,000 to $34,999' : 'inc & bnf 25k-34.9K',
'income and benefits $35,000 to $49,999' : 'inc & bnf 35k-49.9K',
'income and benefits $50,000 to $74,999' : 'inc & bnf 50k-74.9K',
'income and benefits $75,000 to $99,999' : 'inc & bnf 75k-99.9K',
'income and benefits $100,000 to $149,999' : 'inc & bnf 100k-149.9K',
'income and benefits $150,000 to $199,999' : 'inc & bnf 150k-199.9K',
'income and benefits $200,000 or more' : 'inc & bnf 200k or more',
'income and benefits with earnings' : 'inc & bnf w/ earnings',
'income and benefits with social security' : 'inc & bnf w/ social security',
'income and benefits with retirement income' : 'inc & bnf w/ retirement inc',
'!!income and benefits !!with supplemental security income' : 'inc & bnf w/ supplemental security inc',
'!!income and benefits !!with cash public assistance income' : 'inc & bnf w/ cash public assistance inc',
'!!income and benefits !!with food stamp/snap benefits in the past 12 months' : 'inc & bnf w/ food stamp/snap benefits',
'!!income and benefits !!families' : 'inc & bnf families',
'!!income and benefits !!nonfamily households' : 'inc & bnf nonfamily households',
'!!health insurance coverage!!civilian noninstitutionalized population':'civilian noninstitutionalized elibible for h_ins',
'!!health insurance coverage!!civilian noninstitutionalized population under 18 years':'civilian noninstitutionalized u18 elibible for h_ins',
'!!health insurance coverage!!no health insurance coverage':'no h_ins cvrge',
'!!health insurance coverage!!civilian noninstitutionalized population 18 to 64 years':'civilian noninstitutionalized 18-64 eligible for h_ins',
'!!health insurance coverage!!in labor force':'in lbr eligible for h_ins',
'!!health insurance coverage!!in labor force!!employed':'empl eligible for h_ins',
'!!health insurance coverage!!in labor force!!employed!!with health insurance coverage':'empl w/ h_ins',
'!!health insurance coverage!!in labor force!!employed!!with health insurance coverage!!with private health insurance':'empl w/ private h_ins',
'!!health insurance coverage!!in labor force!!employed!!with health insurance coverage!!with public coverage':'empl w/ public h_ins',
'!!health insurance coverage!!in labor force!!employed!!no health insurance coverage':'empl w/o h_ins cvrge',
'!!health insurance coverage!!in labor force!!unemployed':'unempl elibible for h_ins',
'!!health insurance coverage!!in labor force!!unemployed!!with health insurance coverage':'unempl w/ h_ins cvrge',
'!!health insurance coverage!!in labor force!!unemployed!!with health insurance coverage!!with private health insurance':'unempl w/ private h_ins',
'!!health insurance coverage!!in labor force!!unemployed!!with health insurance coverage!!with public coverage':'unempl w/ public h_ins',
'!!health insurance coverage!!in labor force!!unemployed!!no health insurance coverage':'unempl w/o h_ins',
'!!health insurance coverage!!not in labor force':'not in lbr eligible for h_ins',
'!!health insurance coverage!!not in labor force!!with health insurance coverage':'not in lbr w/ h_ins',
'!!health insurance coverage!!not in labor force!!with health insurance coverage!!with private health insurance':'not in lbr w/ private h_ins',
'!!health insurance coverage!!not in labor force!!with health insurance coverage!!with public coverage':'not in lbr w/ public h_ins',
'!!health insurance coverage!!not in labor force!!no health insurance coverage':'not in lbr w/o h_ins',
'families and people whose income in the past 12 months is below the poverty level!!all families!!with related children under 18 years': 'fm_pp inc(12m) lt pvt lv (all families & rel children u18)',
'families and people whose income in the past 12 months is below the poverty level!!all families!!with related children under 18 years!!with related children under 5 years only': 'fm_pp inc(12m) lt pvt lv (all families & rel children u5)',
'families and people whose income in the past 12 months is below the poverty level!!married couple families': 'fm_pp inc(12m) lt pvt lv (married)',
'families and people whose income in the past 12 months is below the poverty level!!married couple families!!with related children under 18 years': 'fm_pp inc(12m) lt pvt lv (married & rel children u18)',
'families and people whose income in the past 12 months is below the poverty level!!married couple families!!with related children under 18 years!!with related children under 5 years only': 'fm_pp inc(12m) lt pvt lv (married & rel children u5)',
'families and people whose income in the past 12 months is below the poverty level!!families with female householder, no husband present': 'fm_pp inc(12m) lt pvt lv (no husband)',
'families and people whose income in the past 12 months is below the poverty level!!families with female householder, no husband present!!with related children under 18 years': 'fm_pp inc(12m) lt pvt lv (no husband & rel children u18)',
'families and people whose income in the past 12 months is below the poverty level!!families with female householder, no husband present!!with related children under 18 years!!with related children under 5 years only': 'fm_pp inc(12m) lt pvt lv (no husband & rel children u5)',
'families and people whose income in the past 12 months is below the poverty level!!under 18 years!!related children under 18 years': 'fm_pp inc(12m) lt pvt lv (rel children u18)',
'families and people whose income in the past 12 months is below the poverty level!!under 18 years!!related children under 18 years!!related children under 5 years': 'fm_pp inc(12m) lt pvt lv (rel children u5)',
'families and people whose income in the past 12 months is below the poverty level!!under 18 years!!related children under 18 years!!related children 5 to 17 years': 'fm_pp inc(12m) lt pvt lv (rel children 5-17)',
'families and people whose income in the past 12 months is below the poverty level!!18 years and over': 'fm_pp inc(12m) lt pvt lv (rel o18)',
'families and people whose income in the past 12 months is below the poverty level!!18 years and over!!65 years and over': 'fm_pp inc(12m) lt pvt lv (rel o65)',
'families and people whose income in the past 12 months is below the poverty level!!people in families': 'fm_pp inc(12m) lt pvt lv (rel)',
'families and people whose income in the past 12 months is below the poverty level!!unrelated individuals 15 years and over': 'fm_pp inc(12m) lt pvt lv (unrel o15)'
}
economics.rename(columns=eco_new_names, inplace=True)


# ***Load social characteristic data***

# In[ ]:


social = pd.read_csv(folder + "/social_characteristics.csv")
social.columns = [c.lower() for c in social.columns]
clean_column_name(social)


# In[ ]:


social


# ***Merge all together***

# In[ ]:


merge_data = cr_data.set_index(['year', 'state name'])                     .join(demography.set_index(['year', 'state name']), how='left')                     .join(economics.set_index(['year', 'state name']), how='left')                     .join(social.set_index(['year', 'state name']), how='left')


# In[ ]:


merged_columns = merge_data.columns
sc = set([])
for c in merged_columns:
    indices = merge_data[merge_data[c].isnull()].index
    merge_data.drop(indices, axis=0, inplace=True)


# ***1. We look at the data correlation***
# 
# Correlation matrix presents the association of attributes. We can filter out columns which are lowly correlated with the target. Nevertheless, correlation is not causation [1]. For example, as you can see population is highly correlated with all types of crimes. However, can you state that a high crime rate is caused by a large population size?
# 
# [1] The book of why, Judea Pearl

# In[ ]:


data_corr = merge_data.corr()
data_corr


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


merge_data_cp = merge_data.copy()
merge_data_cp['total crime per 1k population'] = merge_data['total crime'] / merge_data['population'] * 1000


# In[ ]:


fig, ax = plt.subplots(figsize=(5, 5))
sns.histplot(data=merge_data_cp, x='total crime per 1k population', ax=ax)
ax.set_title('Total Crime')


# ***2. Filter columns w/ threshold***

# In[ ]:


crime_names = ["violent_crime","homicide","robbery","aggravated_assault","property_crime"]


# In[ ]:


columns = data_corr[(data_corr['total crime'] >= 0.1) | (data_corr['total crime'] <= -0.1)]['total crime']


# ## 2. Crime Rate Analysis with all US data

# ### 2.1. Prepare training data
# 
# Using data from previous year to predict the next year crime rate

# In[ ]:


def get_train_test_data(selected_data):
    train_data = selected_data[(selected_data['year'] >= 2010) & (selected_data['year'] < 2018)]                             .sort_values(by=['state name', 'year'])                             .drop(['state name', 'year'], axis=1)
    test_data = selected_data[(selected_data['year'] >= 2018) & (selected_data['year'] < 2019)]                             .sort_values(by=['state name', 'year'])                             .drop(['state name', 'year'], axis=1)
    return train_data, test_data


# In[ ]:


def get_train_test_label(target_label, lb='overall crime rate'):
    train_label = target_label[target_label['year'] < 2019].sort_values(by=['state name', 'year'])[lb]
    test_label = target_label[target_label['year'] == 2019].sort_values(by=['state name', 'year'])[lb]
    return train_label, test_label


# In[ ]:


def extract_label(merge_data, lb='overall crime rate'):
    target_label = merge_data.reset_index()[['state name', 'year', lb]]
    target_label = target_label[target_label['year'] > 2010].sort_values(by=['state name', 'year']) 
    target_label[lb] = target_label[lb] * 1000
    return target_label


# In[ ]:


target = 'overall crime rate'
skip_columns = ['total crime', 'overall crime rate']
selected_features = []
for i, v in zip(columns.index, columns):
    if i in crime_names or i in skip_columns:
        continue
    selected_features.append(i)
selected_data = merge_data[selected_features]


# In[ ]:


selected_data = selected_data.reset_index()
target_label = extract_label(merge_data)
train_data, test_data = get_train_test_data(selected_data)
train_label, test_label = get_train_test_label(target_label)


# ### 2.2. Build a model to predict crime rate of USA

# In[ ]:


import shap
shap.initjs()
from xgboost import XGBRegressor


# In[ ]:


from sklearn.metrics import mean_absolute_error as mae


# In[ ]:


def init_model(train_data, train_label):
    model = XGBRegressor().fit(train_data, train_label)
    return model


# In[ ]:


model1 = init_model(train_data, train_label)
pred = model1.predict(test_data)
acc = mae(test_label, pred)
acc


# ### 2.3 Model Interpretation

# ***Initialize explainer***

# In[ ]:


explainer1 = shap.Explainer(model1)
sh_values = explainer1(test_data)


# ***Plotting global summarization***

# In[ ]:


shap.plots.bar(sh_values, max_display=10)


# In[ ]:


shap.plots.beeswarm(sh_values, max_display=10)


# In[ ]:


# you can printout a hierachical tree of feature interaction
# shap.plots.bar(sh_values, max_display=30, clustering=shap.utils.hclust(test_data, test_label))


# ***Explanation:***
# 1. % of white people. Higher, less crime rate.
# 2. % families & people whose income in the past 12 months < poverty level. The higher, the more crime rate. 
# 3. % of employed in civilian labor force. The higher, the less crime rate.
# 4. natural resources, construction, & maintenance occupations. Higher, more crime rate.
# 5. % Unemployed in civilian labor force. Lower, less crime rate.
# Similar for other factors. You should also check additional resources for more detailed information about each factor.

# ## 3. Selecting 5 states w/ different profiles and analyzing them 
# 
# You don't need to build separate models for each state as we only has 9 records for each one of them. Instead, using 1 big model that fits all states will be better. States with similar properties will have the same pattern. Therefore, if you really want to make separate models for them, each model should be trained with a cluster of data instead of individual state data.
# 
# You only need to select 5 states for task 4.
# 
# To select 5 states, you can either clasify them into clusters or using feature similarity methods.
# 
# Let's use K-mean for simplicity

# In[ ]:


import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


# In[ ]:


aggregated_data = merge_data.groupby(by='state_abbr').mean()


# In[ ]:


scaled_data = MinMaxScaler().fit_transform(aggregated_data)

pca = PCA(n_components=2)
scaled_data = pca.fit_transform(scaled_data)
scaled_data = pd.DataFrame(scaled_data, index=aggregated_data.index, columns=['x1', 'x2'])


# In[ ]:


cluster_engine = KMeans(n_clusters=5)
cluster_engine.fit(scaled_data)
clusters = cluster_engine.predict(scaled_data)


# In[ ]:


df_clusters = pd.DataFrame(np.concatenate([scaled_data, clusters[:, np.newaxis]], axis=1), index=aggregated_data.index, columns=['x1','x2','cluster'])
df_clusters['cluster'] = df_clusters['cluster'].astype(np.int32) + 1
df_clusters = df_clusters.sort_values(by='cluster', axis=0)
df_clusters


# In[ ]:


fig, ax = plt.subplots(figsize=(10,6))
sns.scatterplot(x='x1', y='x2', palette='Set1', hue='cluster', data=df_clusters, ax=ax)
plt.show()


# ***State selection:***
# Based on the clustering result, we can select 5 states as follows:
# 1. Nevada
# 2. Florida
# 3. Alaska
# 4. Ohio
# 5. Louisiana
# 
# K-mean result may be different at each execution time due to its initialization state.

# In[ ]:


states = target_label[(target_label['year'] == 2019)]['state name'].tolist()
state_index = {}
for i, s in enumerate(states):
    state_index[s] = i
state_index


# In[ ]:


shap.plots.waterfall(sh_values[27])


# In[ ]:


shap.plots.waterfall(sh_values[8])


# In[ ]:


shap.plots.waterfall(sh_values[1])


# In[ ]:


shap.plots.waterfall(sh_values[34])


# In[ ]:


shap.plots.waterfall(sh_values[17])


# ## 4. Crime Analysize for 2 Specific Types of Crimes Homicide and Property Crime
# 
# As mentioned above, correlation is not causation. Therefore, as of now, we can't say which causes high crime rates or low crime rates. Therefore, let's select two types of crimes based on its natural motive for crime: homicide and property_crime.

# ### 4.1 Homicide

# In[ ]:


h_crime_names = set(["total crime","overall crime rate","homicide","violent_crime","robbery","aggravated_assault","property_crime","burglary","larceny","motor_vehicle_theft"])
columns = data_corr[(data_corr['homicide'] >= 0.1) | (data_corr['homicide'] <= -0.1)]['homicide']

h_selected_features = []
for i, v in zip(columns.index, columns):
    if i in h_crime_names:
        continue
    h_selected_features.append(i)


# In[ ]:


merge_data_2 = merge_data.copy()
merge_data_2['homicide_rate'] = merge_data_2['homicide'] / merge_data_2['population']
h_selected_data = merge_data[h_selected_features]
h_selected_data = h_selected_data.reset_index()
h_train_data, h_test_data = get_train_test_data(h_selected_data)
h_target_label = extract_label(merge_data_2, 'homicide_rate')
h_train_label, h_test_label = get_train_test_label(h_target_label, lb='homicide_rate')


# In[ ]:


h_model = init_model(h_train_data, h_train_label)


# In[ ]:


h_pred = h_model.predict(h_test_data)
acc = mae(h_test_label, h_pred)
acc


# In[ ]:


h_explainer = shap.Explainer(h_model)
sh_h_values = h_explainer(h_test_data)


# In[ ]:


shap.plots.bar(sh_h_values, max_display=20)


# In[ ]:


shap.plots.beeswarm(sh_h_values, max_display=20)


# ***Explanation:***
# 1. % of black or african american. The higher, the more homicide crime rate. It's a sensitive factor as we don't know Whether black people are victims or murderers. Please refer to this figure: https://www.statista.com/statistics/251877/murder-victims-in-the-us-by-race-ethnicity-and-gender/.
#     As pointed here, non-white people are died from homicide cases more than white.
# 2. Labor force. The higher, the less. 
# 3. % of white. The higher, the less. Corresponding to 1st factor.

# ### 4.2 Property Crime

# In[ ]:


p_crime_names = set(["total crime","overall crime rate","violent_crime","homicide","robbery","aggravated_assault","burglary","larceny","motor_vehicle_theft"])
columns = data_corr[(data_corr['property_crime'] >= 0.1) | (data_corr['property_crime'] <= -0.1)]['property_crime']

p_selected_features = []
for i, v in zip(columns.index, columns):
    if i in p_crime_names:
        continue
    p_selected_features.append(i)


# In[ ]:


merge_data_3 = merge_data.copy()
merge_data_3['property_crime_rate'] = merge_data_2['property_crime'] / merge_data_2['population']
p_selected_data = merge_data[p_selected_features]
p_selected_data = p_selected_data.reset_index()
p_train_data, p_test_data = get_train_test_data(p_selected_data)
p_target_label = extract_label(merge_data_3, 'property_crime_rate')
p_train_label, p_test_label = get_train_test_label(p_target_label, lb='property_crime_rate')


# In[ ]:


p_model = init_model(p_train_data, p_train_label)
p_pred = p_model.predict(p_test_data)
acc = mae(p_test_label, p_pred)
acc


# In[ ]:


p_explainer = shap.Explainer(p_model)
sh_p_values = p_explainer(p_test_data)


# In[ ]:


shap.plots.bar(sh_p_values, max_display=20)


# In[ ]:


shap.plots.beeswarm(sh_p_values, max_display=20)


# ***Explanation:***
# 1. Family w/ related children under 5. The higher, the more property crime rate. Surprisingly. Poverty level ~ crime rate
# 2. Percent of unemployed in labor force. It's a natural factor. The higher, the more property crime rate.
# 3. Similar to 2nd factor.
# 4. inc & benifit from 50 -> 74.9K. Higher, less crime rate. When people have enough money for food and living, they tend not to commit property crime.
# 5. % White. Guess: white people are usually rich. The higher, the less property crime rate. Property crimes usually commited by non-white people.
