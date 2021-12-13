#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 16:19:52 2021

@author: rfarahani
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from joblib import dump, load
#from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression



st.title('COVID Infection Growth Prediction')
#st.write("This is a sentence")
#st.sidebar.title("Selector")
st.markdown('<style>body{background-color: lightblue;}</style>',unsafe_allow_html=True)
st.markdown('<style>.ReactVirtualized__Grid__innerScrollContainer div[class^="row"], .ReactVirtualized__Grid__innerScrollContainer div[class^="data row"]{ background:lgreen; } </style>', unsafe_allow_html=True)

    
country_select = st.sidebar.selectbox('Select a Country',('Algeria', 'Angola', 'Argentina', 'Australia', 'Austria', 'Bahrain', 'Bangladesh',\
                'Belgium', 'Benin', 'Botswana', 'Brazil', 'Bulgaria', 'Burkina Faso', 'Burundi',\
                'Cameroon', 'Chile', 'Colombia'\
                , 'Croatia', 'Czech Republic', 'Denmark', 'Djibouti', 'Egypt', 'El Salvador'\
                , 'Eritrea', 'Estonia', 'Ethiopia', 'Finland', 'France', 'Gabon', 'Gambia', 'Germany'\
                , 'Ghana', 'Guinea', 'Hungary', 'India', 'Indonesia', 'Iran'\
                , 'Iraq', 'Ireland', 'Italy', 'Japan', 'Jordan', 'Kenya', 'Latvia', 'Lebanon'\
                , 'Lesotho', 'Libya', 'Lithuania', 'Luxembourg', 'Madagascar', 'Malawi'\
                , 'Malaysia', 'Mali', 'Mauritania', 'Mauritius', 'Mexico', 'Morocco', 'Mozambique'\
                , 'Namibia', 'Netherlands', 'New Zealand', 'Niger', 'Nigeria', 'Norway', 'Oman'\
                , 'Pakistan', 'Palestine', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Qatar'\
                , 'Russia', 'Rwanda', 'Saudi Arabia', 'Serbia', 'Seychelles'\
                , 'Singapore', 'Slovenia', 'Somalia', 'South Sudan', 'Spain'\
                , 'Sweden', 'Switzerland', 'Syria', 'Thailand', 'Trinidad and Tobago',\
                'Turkey', 'Uganda', 'United Arab Emirates', 'Uruguay', 'Venezuela'\
                ))
year_select = st.sidebar.selectbox('Year',("2021","2020"))
month_select = st.sidebar.selectbox('Month',("01","02","03","04","05","06","07",\
                                    "08","09","10","11","12"))
day_select = st.sidebar.selectbox('Day',("01","02","03","04","05","06","07",\
                                    "08","09","10","11","12","13","14","15",\
                                        "16","17","18","19","20","21","22",\
                                            "23","24","25","26","27","28",\
                                                "29","30"))
                                                          
school_select = st.sidebar.selectbox('School Closing',('no measures',\
                                                       'recommend closing','require closing partially',\
                                                           'require closing all '))
travel_select = st.sidebar.selectbox('International travel control',('No restrictions','Screening arrivals'\
                                                                    ,'Quarantine arrivals','Ban arrivals, partially'\
                                                                        ,'Border Closer'))

work_select = st.sidebar.selectbox('Workplace Closing',('no measures',\
                                                       'recommend closing','require closing partially',\
                                                           'require closing all'))

                                                           
   
event_select = st.sidebar.selectbox('Cancel public events',('no measures',\
'recommend','require'))

transport_select = st.sidebar.selectbox('Close Public Transport',('no measures',\
'recommend closing' , 'require closing'))
    
gathering_select = st.sidebar.selectbox('Restrictions on Gatherings',('no restrictions'\
'restrictions on very large gatherings, 1000 people',\
'restrictions on gatherings between 101-1000 people',\
'restrictions on gatherings between 11-100 people',\
'restrictions on gatherings of 10 people or less'))
    
stay_select = st.sidebar.selectbox('Stay at Home',('no measures',\
'recommend','require with exceptions','require, minimal exceptions'))
    
internal_select = st.sidebar.selectbox('Restrictions on Internal Movement',\
                                       ('no measures','recommend','require'))
    
income_select = st.sidebar.selectbox('Income Support',('none',\
                                                       'government is replacing less than 50% of lost salary',\
                                                        'government is replacing 50% or more of lost salary'))

debt_select = st.sidebar.selectbox('Debt/Contract Relief',('none',\
'narrow relief','broad debt/contract relief'))
    
#facial_select = st.sidebar.selectbox('Facial_covering',('No policy',' Recommended', ' Required some shared areas',\
                                                       # 'Required  all shared areas','Required everywhere')) 


df_sorted = pd.read_csv("df_sorted.csv")
grid2 = load('rainforest_capstone_timeseries_split_all_countries_14days.joblib')





country_name = country_select
#Find correlation to the predicted CGI
df_sorted_country = df_sorted[df_sorted['CountryName'] == country_name]
df_sorted_country = df_sorted_country.drop(columns=['Unnamed: 0', 'CountryName', 'Date',
       'StringencyIndex', 'StringencyLegacyIndex', 'GovernmentResponseIndex',
       'ContainmentHealthIndex', 'EconomicSupportIndex', 'str_date',
        'pdi', 'idv', 'mas', 'uai',
       'ltowvs', 'ivr','H1_combined_numeric','H2_combined_numeric',\
           'H3_combined_numeric','H6_combined_numeric','H7_combined_numeric',\
               'H8_combined_numeric'])

X = df_sorted_country.drop(columns=['CGI_14days_later'])
Y = df_sorted_country['CGI_14days_later']
#Mutual information
mi = mutual_info_regression(X, Y)
mi /= np.max(mi)
mi_df = pd.DataFrame()
mi_df['Features'] = X.columns
mi_df[mi_df['Features'] == 'C1_combined_numeric'] = 'School Closing'
mi_df[mi_df['Features'] == 'C2_combined_numeric'] = 'Workplace Closing'
mi_df[mi_df['Features'] == 'C3_combined_numeric'] = 'Cancel Public Events'
mi_df[mi_df['Features'] == 'C4_combined_numeric'] = 'Restrictions on Gatherings'
mi_df[mi_df['Features'] == 'C5_combined_numeric'] = 'Close Public Transport'
mi_df[mi_df['Features'] == 'C6_combined_numeric'] = 'Stay at Home Order'
mi_df[mi_df['Features'] == 'C7_combined_numeric'] = 'Restrictions on Internal Movement'
mi_df[mi_df['Features'] == 'C8_combined_numeric'] = 'International Travel Controls'
mi_df[mi_df['Features'] == 'E1_combined_numeric'] = 'Income Support'
mi_df[mi_df['Features'] == 'E2_combined_numeric'] = 'Debt/Contract Relief'


mi_df ['mi'] = mi
mi_df.sort_values(by='mi', ascending = False, inplace = True)
mi_df = mi_df[mi_df['mi']>0.25]
mi_df = mi_df[(mi_df['Features'] != 'CGI') & (mi_df['Features'] != 'smooth_daily')]
mi_df = mi_df.reset_index()



#st.write("Effective Interventions " + country_select ,mi_df['features'])
def Intervention(url):
     st.markdown(f'<p style="font-size:20px;border-radius:2%;">{url}</p>', unsafe_allow_html=True)


Intervention("- This application predicts the future daily COVID-19 growth across\
             14 days using Random Forest Model. We combined the Oxford COVID-19 Government Response Tracker data set,\
    Hofstede cultural dimensions, and daily reported COVID-19 infection\
        case numbers for 114 countries to train this model.")
st.markdown("You can see the code [here](https://github.com/rjfaraha/COVID)")

Intervention("- Use the menu at left to select a country, date and intervention metrics.\
             Below you will see a table that shows possible effective interventions for the selected country.\
                 Please note that the interventions show their effectiveness when the current \
                     growth is high. Two plots will appear below. The first one is a \
                    bar chart that shows current infection growth as well as 14-days growth\
                       prediction. The second plot shows total confirmed  infected cases versus\
                           total confirmed deaths over the period of pandemic for the\
                               selected country.")


Intervention("- Possible Effective Interventions:  " + country_select)

if mi_df.shape[0]>0:
    st.dataframe(mi_df['Features'])
else:
    def header(url):
     st.markdown(f'<p style="background-color:#0066cc;color:#33ff33;font-size:24px;border-radius:2%;">{url}</p>', unsafe_allow_html=True)
    #header('No Effective Interventions found')
    st.write('No Effective Interventions found')

#Confirmed Cases and Deaths
latest = pd.read_csv("latest_combined.csv")
latest_country = latest[latest["CountryName"] == country_name]
latest_country = latest_country[["ConfirmedCases", "ConfirmedDeaths"]]


date = int(year_select + month_select + day_select)
#date = 20211001
df_sorted_country = df_sorted [df_sorted['CountryName'] == country_name]
df_max_country = df_sorted_country[df_sorted_country.Date == date]
if df_max_country.shape[0] == 0:
      #st.write('No Data Available for this Date')
      st.markdown("""<style>.big-font {font-size:30px;background-color: white; color:Blue; !important;}</style>""", unsafe_allow_html=True)

      st.markdown('<p class="big-font">No Data Available for this Date</p>', unsafe_allow_html=True)
else:
    row_1 = df_max_country
    row_1 = row_1.drop(columns=['CountryName', 'Date','str_date','CGI_14days_later',\
                            'Unnamed: 0'])

    

    if school_select == 'no measures':
        row_1['C1_combined_numeric'] = 0
    elif school_select == 'recommend closing':
        row_1['C1_combined_numeric'] = 1
    elif school_select == 'require closing partially':
        row_1['C1_combined_numeric'] = 2
    elif school_select == 'require closing all':
     row_1['C1_combined_numeric'] = 3

    
        
    if work_select == 'no measures':
        row_1['C2_combined_numeric'] = 0
    elif work_select == 'recommend closing':
        row_1['C2_combined_numeric'] = 1
    elif work_select == 'require closing partially':
        row_1['C2_combined_numeric'] = 2
    elif work_select == 'require closing all':
        row_1['C2_combined_numeric'] = 3
        
    
                                                  
    #if facial_select == 'No policy':
    #    row_1['H6_combined_numeric'] = 0
    #elif facial_select == 'Recommended':
     #   row_1['H6_combined_numeric'] = 1
    #elif facial_select == 'Required some shared areas':
    #    row_1['H6_combined_numeric'] = 2
    #elif facial_select == 'Required  all shared areas':
     #   row_1['H6_combined_numeric'] = 3
    #elif facial_select == 'Required everywhere':
     #   row_1['H6_combined_numeric'] = 4
     
        
    if travel_select == 'No restrictions':
        row_1['C8_combined_numeric'] = 0
    elif  travel_select == 'Screening arrivals':
        row_1['C8_combined_numeric'] = 1
    elif  travel_select == 'Quarantine arrivals':
        row_1['C8_combined_numeric'] = 2
    elif  travel_select == 'Ban arrivals, partially':
        row_1['C8_combined_numeric'] = 3
    elif  travel_select == 'Border Closer':
        row_1['C8_combined_numeric'] = 4
    
    
    if transport_select == 'no measures':
        row_1['C5_combined_numeric'] = 0
    elif  travel_select == 'recommend closing':
        row_1['C5_combined_numeric'] = 1
    elif  travel_select == 'require closing':
        row_1['C5_combined_numeric'] = 2
        
    
    if event_select == 'no measures':
        row_1['C3_combined_numeric'] = 0
    elif  event_select == 'recommend':
        row_1['C3_combined_numeric'] = 1
    elif  event_select == 'require':
        row_1['C3_combined_numeric'] = 2
        
        
    if gathering_select == 'no restrictions':
        row_1['C4_combined_numeric'] = 0
    elif  gathering_select == 'restrictions on very large gatherings, 1000 people':
        row_1['C4_combined_numeric'] = 1
    elif  gathering_select == 'estrictions on gatherings between 101-1000 people':
        row_1['C4_combined_numeric'] = 2
    elif  gathering_select == 'restrictions on gatherings between 11-100 people':
        row_1['C4_combined_numeric'] = 3
    elif  gathering_select == 'restrictions on gatherings of 10 people or less':
        row_1['C4_combined_numeric'] = 4
     
    
    if stay_select == 'no measures':
        row_1['C6_combined_numeric'] = 0
    elif  stay_select == 'recommend':
        row_1['C6_combined_numeric'] = 1
    elif  stay_select == 'require with exceptions':
        row_1['C6_combined_numeric'] = 2
    elif  stay_select == 'require, minimal exceptions':
        row_1['C6_combined_numeric'] = 3
        
    
    if internal_select == 'no measures':
        row_1['C7_combined_numeric'] = 0
    elif  internal_select == 'recommend':
        row_1['C7_combined_numeric'] = 1
    elif  internal_select == 'require':
        row_1['C7_combined_numeric'] = 2    
        
    
    if income_select == 'none':
        row_1['E1_combined_numeric'] = 0
    elif  income_select == 'government is replacing less than 50% of lost salary':
        row_1['E1_combined_numeric'] = 1
    elif  income_select == 'government is replacing 50% or more of lost salary':
        row_1['E1_combined_numeric'] = 2    
        
    
    if debt_select == 'none':
        row_1['E2_combined_numeric'] = 0
    elif  debt_select == 'narrow relief':
        row_1['E2_combined_numeric'] = 1
    elif  debt_select == 'broad debt/contract relief':
        row_1['E2_combined_numeric'] = 2 
        
    row_1.to_csv("row1.csv")
    
    predict = grid2.predict(row_1)
    
    current_value = (df_max_country['CGI'].array)[0] *100
    predict_value =  predict[0]*100
    
    df_plot = pd.DataFrame({'Time': ['Current Growth', '14-days Growth Prediction'], 'Growth %': [current_value, predict_value]})
    
    df_plot.plot.bar(x='Time',y='Growth %',figsize=(10,5))
    
    
    
    fig = px.bar(df_plot, x='Time',y="Growth %")
    fig.update_traces(marker_color='green')
    fig.update_layout(    
        font_size = 14,
        font_color="Black",
    )
    st.plotly_chart(fig)
    
    
    df_plot = latest_country
    
    df_plot.plot.scatter(x='ConfirmedDeaths',y='ConfirmedCases',figsize=(10,5))
    
    
    
    fig1 = px.scatter(df_plot, x='ConfirmedDeaths',y="ConfirmedCases")

    st.write(fig1)
    
    
