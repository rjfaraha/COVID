#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 13:41:17 2021

@author: rfarahani
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
sns.set()
from sklearn.feature_selection import f_regression, mutual_info_regression
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from joblib import dump, load
from sklearn.inspection import permutation_importance
#import warnings
#warnings.filterwarnings('ignore')

#List of countries
list_country = ['Algeria', 'Angola', 'Argentina', 'Australia', 'Austria', 'Bahrain', 'Bangladesh',\
                'Belgium', 'Benin', 'Botswana', 'Brazil', 'Bulgaria', 'Burkina Faso', 'Burundi',\
                'Cameroon', 'Canada', 'Central African Republic', 'Chad', 'Chile', 'China', 'Colombia'\
                , 'Croatia', 'Czech Republic', 'Denmark', 'Djibouti', 'Egypt', 'El Salvador'\
                , 'Eritrea', 'Estonia', 'Ethiopia', 'Finland', 'France', 'Gabon', 'Gambia', 'Germany'\
                , 'Ghana', 'Greece', 'Guinea', 'Hong Kong', 'Hungary', 'India', 'Indonesia', 'Iran'\
                , 'Iraq', 'Ireland', 'Italy', 'Japan', 'Jordan', 'Kenya', 'Kuwait', 'Latvia', 'Lebanon'\
                , 'Lesotho', 'Liberia', 'Libya', 'Lithuania', 'Luxembourg', 'Madagascar', 'Malawi'\
                , 'Malaysia', 'Mali', 'Mauritania', 'Mauritius', 'Mexico', 'Morocco', 'Mozambique'\
                , 'Namibia', 'Netherlands', 'New Zealand', 'Niger', 'Nigeria', 'Norway', 'Oman'\
                , 'Pakistan', 'Palestine', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Qatar'\
                , 'Romania', 'Russia', 'Rwanda', 'Saudi Arabia', 'Senegal', 'Serbia', 'Seychelles'\
                , 'Sierra Leone', 'Singapore', 'Slovenia', 'Somalia', 'South Sudan', 'Spain', 'Sudan'\
                , 'Sweden', 'Switzerland', 'Syria', 'Taiwan', 'Tanzania', 'Thailand', 'Togo', 'Trinidad and Tobago', \
                'Tunisia', 'Turkey', 'Uganda', 'United Arab Emirates', 'United States', 'Uruguay', 'Venezuela'\
                , 'Vietnam', 'Yemen', 'Zambia', 'Zimbabwe']
    
#Live data from OXCGRT
#Vaccine information
VC =  pd.read_csv("https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_vaccines_full.csv")


#Read cultural metrics for countries
CUL = pd.read_excel("/Users/rfarahani/Documents/TDI/COVID/6-dimensions-for-website-2015-08-16_modified.xls")

#Goverment policies
NPI = pd.read_csv("https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest_combined.csv")
NPI = NPI.drop(['C1_combined','C2_combined','C3_combined','C4_combined','C5_combined','C6_combined',\
          'C7_combined','C8_combined','E1_combined', 'E2_combined','H1_combined','H2_combined',\
          'H3_combined','H6_combined','H7_combined','H8_combined','CountryCode','RegionName',\
          'RegionCode', 'Jurisdiction'], axis=1)
#Change date column to date
NPI[['str_date']] = NPI[['Date']].applymap(str).applymap(lambda s: "{}/{}/{}".format(s[4:6],s[6:], s[0:4]))
VC[['str_date']] = VC[['Date']].applymap(str).applymap(lambda s: "{}/{}/{}".format(s[4:6],s[6:], s[0:4]))
VC= VC[['CountryName','Date','V2_Vaccine Availability (summary)',
       'V2_General 16-19 yrs', 'V2_General 20-24 yrs', 'V2_General 25-29 yrs',
       'V2_General 30-34 yrs', 'V2_General 35-39 yrs', 'V2_General 40-44 yrs',
       'V2_General 45-49 yrs', 'V2_General 50-54 yrs', 'V2_General 55-59 yrs',
       'V2_General 60-64 yrs', 'V2_General 65-69 yrs', 'V2_General 70-74 yrs',
       'V2_General 75-79 yrs', 'V2_General 80+ yrs','str_date'
       
     ]]

#Not considering vaccine data since it is not complete
df = pd.DataFrame()
for country in list_country: 
    print(country)
    NPI_Country = pd.DataFrame()
    merged = pd.DataFrame()
    NPI_Country = NPI[NPI['CountryName'] == country]
    #Get daily confirmed
    NPI_Country['daily_ConfirmedCases'] = NPI_Country['ConfirmedCases'].diff()
    #7 day smoothing
    #NPI_Country_av = NPI_Country.apply (lambda a: a.rolling(window=7).mean())
    NPI_Country['smooth_daily'] = NPI_Country.iloc[:,31].rolling(window=7).mean()
    #Cumulative using the smoothed data
    NPI_Country['smooth_cum'] = NPI_Country['smooth_daily'].cumsum()
    #Cumulative confirmed Cases 14 days before
    NPI_Country['smooth_cum_14_days_before'] = NPI_Country['smooth_cum'].shift(periods=13)
    NPI_Country['smooth_cum_14_days_before'] =NPI_Country['smooth_cum_14_days_before'].replace(0,1)
    #NPI_Country['smooth_cum_30_days_before'] = NPI_Country['smooth_cum'].shift(periods=29)
    #NPI_Country['smooth_cum_30_days_before'] =NPI_Country['smooth_cum_30_days_before'].replace(0,1)
    NPI_Country['CGI'] = NPI_Country['smooth_cum']/NPI_Country['smooth_cum_14_days_before'] - 1
    NPI_Country['CGI_14days_later'] = NPI_Country['CGI'].shift(periods=-13)
    #NPI_Country['CGI'] = NPI_Country['smooth_cum']/NPI_Country['smooth_cum_30_days_before'] - 1
    #NPI_Country['CGI_30days_later'] = NPI_Country['CGI'].shift(periods=-29)
    #NPI_Country.reset_index(inplace=True)
    NPI_Country = NPI_Country.dropna()
    #start from 04/01/2020
    NPI_Country = NPI_Country [NPI_Country['str_date']>= '04/01/2020']
    merged = NPI_Country.merge(CUL, left_on = 'CountryName', right_on = 'country')
    
   
    df = pd.concat([df, merged])
    
#Consider data where smooth_daily is at least 10   
df = df[df['smooth_daily']>=10]
df_with_date = df.drop(columns=['StringencyIndexForDisplay',\
                     'StringencyLegacyIndexForDisplay','GovernmentResponseIndexForDisplay'\
                     ,'ContainmentHealthIndexForDisplay','EconomicSupportIndexForDisplay',\
                     'daily_ConfirmedCases','smooth_cum','smooth_cum_14_days_before','ConfirmedCases'\
                     ,'ConfirmedDeaths','ctr', 'country'])
df = df.drop(columns=['CountryName',\
                      'str_date','StringencyIndexForDisplay',\
                     'StringencyLegacyIndexForDisplay','GovernmentResponseIndexForDisplay'\
                     ,'ContainmentHealthIndexForDisplay','EconomicSupportIndexForDisplay',\
                     'daily_ConfirmedCases','smooth_cum','smooth_cum_14_days_before','ConfirmedCases'\
                     ,'ConfirmedDeaths','Date','ctr', 'country'])
   
#X and y for the model
#Fill NAN WITH ZERO
df = df.fillna(0)
#Y = df['CGI_14days_later']
#X = df.drop(columns =['CGI_14days_later'])
#Y = df['CGI_30days_later']
#X = df.drop(columns =['CGI_30days_later'])
df_with_date = df_with_date.fillna(0)
#Sort them based on Date
df_sorted = df_with_date.sort_values(by='Date', ascending=True)
#Y_sorted = df_sorted['CGI_30days_later']
#X_sorted = df_sorted.drop(columns =['CGI_30days_later','str_date'])
Y_sorted = df_sorted['CGI_14days_later']
X_sorted = df_sorted.drop(columns =['CGI_14days_later','str_date','Date','CountryName'])

#Mutual information
mi = mutual_info_regression(X_sorted, Y_sorted)
mi /= np.max(mi)

mi_df = pd.DataFrame()
mi_df['features'] = X_sorted.columns
mi_df ['mi'] = mi
mi_df.plot.bar(x='features',y='mi',figsize=(20,10))

#Split data to train and test
X_train2, X_test2, Y_train2, Y_test2 = train_test_split(X_sorted, Y_sorted, test_size=0.33, shuffle=False)

#RandomForest Model
param_grid_random_forest = {
    #'regressor__max_depth': [15,20,25],
    'regressor__max_depth': [15,20,25],
    #'regressor__min_samples_leaf': [5, 10, 15],
    'regressor__min_samples_leaf': [80, 100, 150],
    'regressor__n_estimators': [20,50, 100, 200]
   }

pipeline = Pipeline ([('scaling', StandardScaler()),('regressor',RandomForestRegressor(random_state=42))])
tscv = TimeSeriesSplit(n_splits=3)
grid2 = GridSearchCV(pipeline, param_grid_random_forest,cv = tscv, n_jobs = -1, verbose = 2)

#Fit the model
grid2.fit(X_train2,Y_train2)

dump(grid2, 'rainforest_capstone_timeseries_split_all_countries_14days.joblib') 
print ("score_test = %3.2f" %(grid2.score(X_test2,Y_test2)))
print ("score_train = %3.2f" %(grid2.score(X_train2,Y_train2)))
print (grid2.best_params_)

#Most important features
imp_fea_df2 = pd.DataFrame()
imp_fea_df2['features'] = X_sorted.columns
imp_fea_df2['feature_imp'] = grid2.best_estimator_._final_estimator.feature_importances_
imp_fea_df2_sorted = imp_fea_df2.sort_values(by=['feature_imp'], ascending=False)
imp_fea_df2_sorted

#Most important features using permutation importance
result = permutation_importance(
    grid2, X_train2, Y_train2, n_repeats=10, random_state=42, n_jobs=2)
forest_importances = pd.Series(result.importances_mean)
per_df = pd.DataFrame()
per_df['features'] = X_sorted.columns
per_df['feature_per'] = forest_importances
per_df_sorted = per_df.sort_values(by=['feature_per'], ascending=False)
per_df_sorted



