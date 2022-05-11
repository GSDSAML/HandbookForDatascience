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
# As the original Seoul AQ dataset contains 25 information of 25 districts, it's too large for this example. Therefore, we only work with the overall AQ dataset only. In short, we extract city-level air quality data from the original dataset.

# In[2]:


seoul_air = pd.read_csv('/home/alexbui/workspace/HandbookForDatascience/notebooks/data/seoul_air_avg.csv')


# In[3]:


seoul_air


# ### 3.2 Check missing values

# In[4]:


for c in seoul_air.columns:
    print(c, seoul_air[c].isnull().sum())


# In[5]:


import seaborn as sns
import matplotlib.pyplot as plt


# ### 3.3 Check outlier values

# In[6]:


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


# In[7]:


for c in seoul_air.columns[1:7]:
    check_outliners(seoul_air, c)


# ### 3.4 Plotting

# ***Plot correlation to first understand feature interactions***

# In[8]:


corr = seoul_air.iloc[:,1:7].corr()
fix, ax = plt.subplots(figsize=(10,10))
sns.heatmap(corr)
plt.show()


# ***Align 1h to check correlation with previous hour***

# In[9]:


align0 = seoul_air.iloc[:-1,1:7]
align0.columns = [c + "_m1" for c in align0.columns]
align1 = seoul_air.iloc[1:,1:7]
align = pd.concat([align1, align0], axis=1)


# In[10]:


align_corr = align.corr()
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(align_corr)
plt.show()


# ***Align 4h to check correlation with 4 hours ago***

# In[11]:


align04 = seoul_air.iloc[:-4,1:7]
align04.columns = [c + "_m4" for c in align04.columns]
align14 = seoul_air.iloc[4:,1:7]
align4 = pd.concat([align14, align04], axis=1)


# In[12]:


align_corr4 = align4.corr()
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(align_corr4)
plt.show()


# ## 4. Model Construction

# ## 5. Explain the Results
