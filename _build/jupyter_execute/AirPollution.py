#!/usr/bin/env python
# coding: utf-8

# # Seoul Air Quality Level Prediction

# ## 1. Seoul Air Quality Dataset
# 
# This dataset is collected from @seoul_air, including Seoul air quality data from 2008 to 2018. Air quality is impacted by many factors such as traffic volume, neighboring area AQ situations, weather, seasonal information, and other economic activities. Many works have addressed the relationship between AQ level and other factors via numerous modeling approaches. For instance, during the Chuseok holidays, the AQI tends to get better, while it is serious during weekdays, especially with foggy weather conditions or in the yellow dust season. You can refer to [3-5] for more information on how researchers used this dataset in their works.

# | Column | Description |
# |---------|---------|
# | Datetime | Timestamp |
# | District | District code 0-25 (Code 0 represents the average value of all 25 districts in Seoul). Other districts are identified from 1 to 25.  The order of district codes is 0 - 평균, 1 - 종로구, 2 - 중구, 3 - 용산구, 4 - 성동구, 5 - 광진구, 6 - 동대문구, 7 - 중랑구, 8 - 성북구, 9 - 강북구, 10 - 도봉구, 11 - 노원구, 12 - 은평구, 13 - 서대문구, 14 - 마포구, 15 - 양천구, 16 - 강서구, 17 - 구로구, 18 - 금천구, 19 - 영등포구, 20 - 동작구, 21 - 관악구, 22 - 서초구, 23 - 강남구, 24 - 송파구, 25 - 강동구 |
# | PM10_CONC | PM10 concentration (µg/m3) |
# | PM2_5_CONC | PM2.5 concentration (µg/m3) |
# |O3         | Ozone concentration (µg/m3) |
# | NO2 | NO2 concentration (µg/m3) |
# | CO | CO concentration (µg/m3) |
# | SO2 | SO2 concentration (µg/m3) |
# | PM10_AQI | PM10 AQI Index according to US Standard AQI Index |
# | PM2_5_AQI | PM2.5 AQI Index according to US Standard AQI Index |

# ## 2. Additional Data Sources
# <figure>
# <img src="./_images/ml_system.png" alt="ml_system" width="80%" height="80%">
# <figcaption>Image Source From https://proceedings.neurips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf</figcaption>
# </figure>
# 
# As the figure shows, we spend most of the time on data collection, cleansing, and pre-processing. Only a small amount of time is for machine learning model development. To increase the accuracy of models, we must try to find additional data to verify our hypotheses.
# 
# ### 2.1 Weather Data
# 
# <figure>
# <img width="500px" src="./_images/seoul_weather.png" alt="ml_system" width="80%" height="80%">
# <figcaption>Seoul Weather from worldweatheronline.com</figcaption>
# </figure>
# 
# Many researches have pointed out that air quality level relates to weather conditions. For instance, AQ levels get better after a heavy rain, or it usually gets worse during the winter season. For more information, please check out reference papers.
# 
# ### 2.2 Holiday Information
# 
# <figure>
# <img width="500px" src="./_images/holiday.png" alt="ml_system" width="80%" height="80%">
# <figcaption>Seoul Holidays from timeanddata.com</figcaption>
# </figure>
# 
# Similar to weather data, we can collect holiday information from websites like timeanddata.com.

# ## 3. Data Pre-processing

# In[1]:


import pandas as pd
import numpy as np


# ### 3.1 Data loading
# 
# As the original Seoul AQ dataset contains 25 information of 25 districts, it's too large for this example. Therefore, we only work with the overall AQ dataset only. In short, we extract city-level air quality data from 2014 -> 2018 from the original dataset.

# In[2]:


path = "/home/alexbui/workspace/HandbookForDatascience/notebooks/"


# In[3]:


seoul_air = pd.read_csv(path + 'data/seoul_air_avg.csv')
seoul_air.drop(["PM10_AQI", "PM2_5_AQI"], axis=1, inplace=True)
seoul_air.columns = [c.lower() for c in seoul_air.columns]


# In[4]:


seoul_air


# ***Load weather data***

# In[5]:


weather = pd.read_csv(path + "data/weather_forecasts.csv")
weather = weather[weather['datetime'] <= "2018-06-18 11:00:00"]
weather


# ### 3.2 Check missing values

# In[6]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[7]:


for c in seoul_air.columns:
    print(c, seoul_air[c].isnull().sum())


# In[8]:


for c in weather.columns:
    print(c, weather[c].isnull().sum())


# ### 3.3 Check outlier values

# In[9]:


def check_outliners(seoul_air, c):
    col = seoul_air.loc[:,c]
    abs_skew = col.skew()
    mean_v = col.mean()
    median_v = col.median()
    q3 = np.nanpercentile(col, 75)
    q1 = np.nanpercentile(col, 25)
    iqr = (q3 - q1) * 1.5
    ceiling = iqr + q3
    # floor = q1 - iqr 
    # col[(col > ceiling) | (col < floor)]
    print("num of outlier", c, col[col > ceiling].count())
    if abs_skew > 1:
        col[col > ceiling] = median_v
    else:
        col[col > ceiling] = mean_v    


# In[10]:


for c in ["temperature(C)",	"feel_like(C)",	"wind_speed(km/h)",	"wind_gust(km/h)", "cloud(%)", "humidity(%)", "rain(mm)", "pressure"]:
    check_outliners(weather, c)


# ## 3.4 Merge Air Data & Weather Data
# 
# We have to check which datetime data is missing and interpolate it. The simplest way is to filling it with near by neighbors or average values of near by neighbors.

# In[11]:


air_weather = pd.merge(weather, seoul_air, on='datetime', how='outer')
air_weather[air_weather['pm10_conc'].isnull()]


# In[12]:


air_weather2 = air_weather.interpolate(method='linear')
air_weather2[air_weather['pm10_conc'].isnull()]


# ### 3.5 Plotting

# ***Plot correlation to first understand feature interactions***

# In[13]:


corr = seoul_air.iloc[:,1:7].corr()
fix, ax = plt.subplots(figsize=(10,10))
sns.heatmap(corr)
plt.show()


# ***Align 1h to check correlation with previous hour***

# In[14]:


def concat_dataframe(df, timeshift=1):
    df1 = df.iloc[:-timeshift,:]
    df1.columns = [c + "_m%i" % timeshift for c in df1.columns]
    df2 = df.iloc[timeshift:,:].reset_index().drop(["index"],axis=1)
    return pd.concat([df1, df2], axis=1)


# In[15]:


def plot_corr(df):
    align_corr = df.corr()
    plt.subplots(figsize=(10,10))
    sns.heatmap(align_corr)
    plt.show()
    return align_corr


# In[16]:


align1 = concat_dataframe(seoul_air.iloc[:,1:7], 1)


# In[17]:


plot_corr(align1)


# ***Align 4h to check correlation with 4 hours ago***

# In[18]:


align4 = concat_dataframe(seoul_air.iloc[:,1:7], 4)
plot_corr(align4)


# ***Plot weather & air quality together***

# In[19]:


plot_corr(air_weather2)


# In[20]:


air_weather4 = concat_dataframe(air_weather2, 4)


# In[21]:


plot_corr(air_weather4)


# ### 3.6 Training, Testing Split

# In[22]:


target = ['pm2_5_conc', 'pm10_conc']


# In[23]:


def build_dataset(timeshift=1):
    drp_columns = ['datetime', 'datetime_m%i'%timeshift, 'weather_m%i'%timeshift, 'wind_direction_m%i'%timeshift, 'weather', 'wind_direction']
    dataset1 = concat_dataframe(air_weather2, timeshift)
    training1 = dataset1[dataset1['datetime'] <= "2016-12-31 23:00:00"]
    training1.drop(drp_columns, axis=1, inplace=True)
    testing1 = dataset1[(dataset1['datetime'] > "2016-12-31 23:00:00") & (dataset1['datetime'] <= "2017-12-31 23:00:00")]
    testing1.drop(drp_columns, axis=1, inplace=True)
    X1_train, y1_train = training1.drop(target, axis=1), training1['pm2_5_conc']
    X1_test, y1_test = testing1.drop(target, axis=1), testing1['pm2_5_conc']
    return X1_train, y1_train, X1_test, y1_test


# ***Create training dataset to predict time ahead: 1h, 4h, 8h, 12h, 16h, 24h***

# In[24]:


X1_train, y1_train, X1_test, y1_test = build_dataset(1)
X4_train, y4_train, X4_test, y4_test = build_dataset(4)
X8_train, y8_train, X8_test, y8_test = build_dataset(8)
X12_train, y12_train, X12_test, y12_test = build_dataset(12)
X16_train, y16_train, X16_test, y16_test = build_dataset(16)
X20_train, y20_train, X20_test, y20_test = build_dataset(20)
X24_train, y24_train, X24_test, y24_test = build_dataset(24)


# ## 4. Model Construction

# In[25]:


import xgboost as xgb
from sklearn.metrics import mean_absolute_error


# ***Create simple XGBoost model for corresponding dataset***

# In[26]:


def plot_pred(pred, label):
    p1_df = pd.DataFrame({'pred': pred, 'label': label, 'time': list(range(len(pred)))})
    fg, ax = plt.subplots(figsize=(10,10))
    sns.lineplot(data=p1_df, x='time', y='pred', label="pred")
    sns.lineplot(data=p1_df, x='time', y='label', label="label")
    plt.xlabel("Time")
    plt.ylabel("PM2_5 Concentration")
    plt.show()


# In[27]:


model1 = xgb.XGBRegressor().fit(X1_train, y1_train)
pred1 = model1.predict(X1_test)
mean_absolute_error(pred1, y1_test)


# In[28]:


plot_pred(pred1, y1_test)


# In[29]:


model4 = xgb.XGBRegressor().fit(X4_train, y4_train)
pred4 = model4.predict(X4_test)
mean_absolute_error(pred4, y4_test)


# In[30]:


plot_pred(pred4, y4_test)


# In[31]:


model8 = xgb.XGBRegressor().fit(X8_train, y8_train)
pred8 = model8.predict(X8_test)
mean_absolute_error(pred8, y8_test)


# In[32]:


plot_pred(pred8, y8_test)


# In[33]:


model12 = xgb.XGBRegressor().fit(X12_train, y12_train)
pred12 = model8.predict(X12_test)
mean_absolute_error(pred12, y12_test)


# In[34]:


plot_pred(pred12, y12_test)


# In[35]:


model24 = xgb.XGBRegressor().fit(X24_train, y24_train)
pred24 = model24.predict(X24_test)
mean_absolute_error(pred24, y24_test)


# In[36]:


plot_pred(pred24, y24_test)


# ## 5. Explain the Results
