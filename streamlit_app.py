import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import holidays
from sodapy import Socrata
from darts import TimeSeries
import seaborn as sn
import altair as alt
import streamlit as st
from st_aggrid import AgGrid
# import shap
import joblib
import warnings
warnings.filterwarnings('ignore')

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', 50)

st.set_page_config(layout="wide")

st.title('Newark Airport Terminal B Passenger Forecasting')
# st.write('blah blah blah')


def file_uploader():
    uploaded_file = st.sidebar.file_uploader(label='Upload a new flight schedule in .csv format',type='csv')
    if uploaded_file is not None:
        fs_con = pd.read_csv(uploaded_file)

    else:
        fs_con = pd.read_csv(r'C:\Users\Gabriel\OneDrive\CUSP\Spring 2022 Classes\Capstone\Data\2022 Data\Flight Schedule w Concourse_2021.01-2022.11.csv')
    
    fs_con = fs_con[['Flight Date','Flight Departing Date Time','Arr Airport Code','International Domestic','Departure Concourse','ICAO Airline','Operating Airline Code','Flight No','Seats','Aircraft Code','Flight Distance','Flight Duration']]
    fs_con['Flight Date'] = pd.to_datetime(fs_con['Flight Date'])

    with st.expander("Review/edit the raw flight schedule data"):
        grid_return = AgGrid(fs_con[['Flight Date','Flight Departing Date Time','Arr Airport Code','International Domestic','Departure Concourse','ICAO Airline','Operating Airline Code','Flight No','Seats','Aircraft Code','Flight Distance','Flight Duration']], editable=True)
        fs_con = grid_return['data']
        fs_con['Flight Date'] = pd.to_datetime(fs_con['Flight Date'])
    
    return fs_con


fs_con = file_uploader()

#Sidebar
date_range = st.sidebar.date_input('Choose the start and end dates of your forecast',value=(dt.date.today() ,dt.date.today() + dt.timedelta(days = 14)), min_value = fs_con['Flight Date'].min(),max_value = fs_con['Flight Date'].max())
freq = st.sidebar.select_slider(label='Forecast frequency', options = ['10 minutes','30 minutes', '1 hour','3 hours','12 hours','1 day'])
freq_dict = {'10 minutes':'10t','30 minutes':'30t', '1 hour':'1H','3 hours':'3H','12 hours':'12H','1 day':'1D'}
freq_con = freq_dict[freq]
covid_scenario = st.sidebar.radio(label = 'Choose a COVID scenario',options = ['Current COVID levels', 'Peak COVID level', 'Low COVID level'], help = 'Peak is based on Omicron surge in early 2022, assuming no changes to vaccination levels.')
weather_scenario = st.sidebar.radio(label = 'Choose a weather scenario',options = ['Normal','Moderate','Severe'], help = 'Normal is based on current forecasts and historical averages. Moderate is based on rain and reduced visibility. Severe is based on blizzard/hurricane conditions.')


S_date = pd.to_datetime(date_range[0])
E_date = pd.to_datetime(date_range[1])

@st.cache(allow_output_mutation=True)
def load_lookups():
    aircraft_codes = pd.read_csv(r'C:\Users\Gabriel\OneDrive\CUSP\Spring 2022 Classes\Capstone\Data\LookupTables\Aircraft_Lookup.csv')
    airport_lookup = pd.read_csv(r'C:\Users\Gabriel\OneDrive\CUSP\Spring 2022 Classes\Capstone\Data\LookupTables\airport-codes.csv')
    al_lookup = pd.read_csv(r'C:\Users\Gabriel\OneDrive\CUSP\Spring 2022 Classes\Capstone\Data\LookupTables\Airlines Mapping.csv')
    weather_hist = pd.read_csv(r'C:\Users\Gabriel\OneDrive\CUSP\Spring 2022 Classes\Capstone\Data\2022 Data\weather_2021_2022.csv')
    weather_hist = weather_hist.drop_duplicates(subset=['Date','Military Hour'])

    return aircraft_codes, airport_lookup, al_lookup, weather_hist

aircraft_codes, airport_lookup, al_lookup, weather_hist = load_lookups()


@st.cache
def load_weather(weather_hist):
    #15-day forecast
    td = dt.date.today()
    td_15 = td + dt.timedelta(days = 15)
    td = td.strftime('%y-%m-%d')
    td_15 = td_15.strftime('%y-%m-%d')

    try:
        request_string = f'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/07114/{td}/{td_15}?unitGroup=us&elements=datetime%2Ctemp%2Cprecip%2Cprecipprob%2Cprecipcover%2Cpreciptype%2Csnow%2Csnowdepth%2Cwindgust%2Cwindspeed%2Cvisibility%2Csevererisk%2Cconditions%2Cdescription&include=hours&key=A32Z3KQNFYC38FNX7SM8M7G9P&contentType=csv'

        weather = pd.read_csv(request_string)
        weather['datetime'] = pd.to_datetime(weather['datetime'])
        weather_options = [
            weather.precip == 0,
            weather.precip <= .1,
            weather.precip <= .2,
            weather.precip > .2,]
        choices = [0,1,2,3]

        weather['cat'] = np.select(weather_options, choices)
    except:
        weather = pd.read_csv(r'C:\Users\Gabriel\OneDrive\CUSP\Spring 2022 Classes\Capstone\Data\2022 Data\aug2022_forecast.csv')
        weather['datetime'] = pd.to_datetime(weather['datetime'])

    
    #combine weather_hist and weather forecast
    weather_hist['Date'] = pd.to_datetime(weather_hist['Date'])
    weather_hist['datetime'] = pd.to_datetime({'year':weather_hist.Date.dt.year,'month':weather_hist.Date.dt.month,'day':weather_hist.Date.dt.day,'hour':weather_hist['Military Hour']})
    weather_hist = weather_hist[['datetime','Precipitation (Inches)', 'Visibility (Miles)', 'Weather Description','cat']].rename({
        'Precipitation (Inches)':'precip', 'Visibility (Miles)':'visibility', 'Weather Description':'conditions'
    },axis=1)
    weather = pd.concat([weather_hist,weather],axis=0)

    weather['dup'] = weather['datetime'].duplicated()
    
    return weather

weather_df = load_weather(weather_hist=weather_hist)



# @st.cache(allow_output_mutation=True)
def paxfs(df, aircraft_codes, airport_lookup, al_lookup, weather,covid_scen, weather_scen):

    #filter for B concourses
    fs = fs_con[fs_con['Departure Concourse'].isin(['Concourse B-1', 'Concourse B-2','Concourse B-3'])]
    #remove the load factor columns - these were not merged properly in the original dataset
    # fs = fs.drop(['Load Factor',	'Estimated Passenger','nonpaxratio','TSALoadFactor'],axis=1)
    fs = fs.replace({'Departure Concourse':{'Concourse B-1':'B1','Concourse B-2':'B2','Concourse B-3':'B3'}})
    start, stop = 0, 10

    fs['Date'] = fs['Flight Departing Date Time'].str.slice(start, stop)
    start, stop = 11, 16

    fs['Time'] = fs['Flight Departing Date Time'].str.slice(start, stop)
    fs['DT_sched'] = fs['Date']+' '+fs['Time'].astype(str)
    fs['DT_sched'] = pd.to_datetime(fs['DT_sched'], format='%Y-%m-%d %H:%M:%S')
    fs['date'] = fs['DT_sched'].dt.date
    fs['Hour'] = fs['DT_sched'].dt.hour
    fs = fs[['Aircraft Code',
        'ICAO Airline', 'Arr Airport Code',
        'Flight No',
        'International Domestic', 
        'Flight Distance', 'Flight Duration', 'Operating Airline Code','Departure Concourse',
        'Date', 'date','Time', 'DT_sched', 'Hour', 'Seats']]
    fs['month'] = fs['DT_sched'].dt.month
    fs['year'] = fs['DT_sched'].dt.year
    ## Group flight times into time of day and day of week
    morning = [6,7,8,9,10,11,12]
    afternoon = [13,14,15,16,17]
    evening = [18,19,20,21,22]
    overnight = [23,0,1,2,3,4,5]
    lsthour = []
    for s in fs['Hour']:
            if s in morning:
                lsthour.append('Morning')
            elif s in afternoon:
                lsthour.append('Afternoon')
            elif s in evening:
                lsthour.append('Evening')
            elif s in overnight:
                lsthour.append('Overnight')
    fs['Time Category'] = lsthour
    fs['DOW'] = fs['DT_sched'].dt.weekday
    fs['DOW_l'] = fs['DOW'].copy()
    fs["DOW_l"].replace({0: "Monday", 1: "Tuesday",2:"Wednesday",3:'Thursday',4:'Friday',5:'Saturday',6:'Sunday'}, inplace=True)
    ## Add new column for Holidays and holiday weekends
    #import US holidays (for NY and NJ) from the holidays library
    y_range = [y for y in range(np.min(fs.year),np.max(fs.year) + 1)]
    hds = holidays.US(subdiv = 'NY',years = y_range) + holidays.US(subdiv = 'NJ',years = y_range)
    #days surrounding the holidays also tend to have higher demand. Identify the surrounding days.
    fl_hds = []
    for date, name in sorted(hds.items()):
        if (name == 'Christmas Day') | (name == 'Thanksgiving') | (name == "New Year's Day"):
        # for the big holidays, we would expect to see increased travel on the 3 days before and after
            fl_hds.append(date - dt.timedelta(1))
            fl_hds.append(date - dt.timedelta(2))
            fl_hds.append(date - dt.timedelta(3))
            fl_hds.append(date + dt.timedelta(1))
            fl_hds.append(date + dt.timedelta(2))
            fl_hds.append(date + dt.timedelta(3))
        elif date.weekday() == 0:
        # for Monday holidays, we would expect to see increased travel on the Friday, Saturday, and Sunday prior
            fl_hds.append(date - dt.timedelta(1))
            fl_hds.append(date - dt.timedelta(2))
            fl_hds.append(date - dt.timedelta(3))
        elif date.weekday() == 4:
        # for Friday holidays, we would expect to see increased travel on the Saturday and Sunday after
            fl_hds.append(date + dt.timedelta(1))
            fl_hds.append(date + dt.timedelta(2))
        
    #if the departure date is in the list of holidays, mark as a 1 or if the departure date is in the list of days surrounding holidays, also mark as a 1
    fs['holiday'] = 0
    for i in fs.index:
        if fs.loc[i,'DT_sched'] in hds:
            fs.loc[i,'holiday'] = 1
        elif fs.loc[i,'DT_sched'].date() in fl_hds:
            fs.loc[i,'holiday'] = 1

    ## Merge aircraft codes for ICAO and BTS
    # I had to create this lookup table manually, referring to the following sources:
    # 1) the T-100 data itself, which has the Aircraft Type Desc_1
    # 2) the BTS website provides this lookup table for BTS Aircraft Type Desc_2 and the BTS Aircraft Type Code: https://transtats.bts.gov/Download_Lookup.asp?Y11x72=Y_NVePeNSg_glcR
    # 3) https://en.wikipedia.org/wiki/List_of_aircraft_type_designators
    # 4) https://www.avcodes.co.uk/acrtypes.asp
    # 5) The FAA Airplane Design Groups (ADG), which is used to classify plane size based on wingspan and height. It is a good enough approximation for seat capacity categories. https://www.faa.gov/airports/engineering/aircraft_char_database/
    # Note that there is not always a 1-to-1 match, and a few of the BTS values did not correspond perfectly with the ICAO/IATA codes.
    
    fs = pd.merge(left=fs, right=aircraft_codes,how='left',left_on='Aircraft Code', right_on = 'IATA Aircraft Code')
    ## Merge destination region
    # the destination data is available at https://ourairports.com/data/
    # the regions were manually assigned based on Gabe's discretion. Domestic flights are divided into northeast, southeast, midwest, plains, and west coast/pacific.
    # Canada is its own region. Central America and Caribbean were grouped together with South America. Otherwise, the region corresponds to the continent.

    
    fs = pd.merge(fs, airport_lookup[['iata_code','Region']],left_on = 'Arr Airport Code',right_on = 'iata_code', how='left')
    ## Merge airline category
    # Airlines were grouped into low-cost carriers (LC) and traditional carriers (TR)
    # the data source for this classification is based on the ICAO List of Low-Cost-Carriers: https://www.icao.int/sustainability/documents/lcc-list.pdf
    # A few manual adjustments were made for known LCCs, e.g. French Bee, that is not in the ICAO list.

    fs = pd.merge(fs, al_lookup[['Marketing Airline IATA Code','Marketing Airline ICAO Code','Type']], how= 'left', left_on = 'ICAO Airline', right_on = 'Marketing Airline ICAO Code').rename({'Marketing Airline IATA Code':'airline_iata','Type':'airline_type','Marketing Airline ICAO Code':'airline_icao'},axis=1)
    ## Merge Daily Vaccination Data
    # get most recent NY state vaccination data
    client = Socrata("health.data.ny.gov", None)
    results = client.get("ksjn-24s4",limit=50000)
    vax = pd.DataFrame.from_records(results).rename({'report_as_of':'date'},axis=1)
    vax['date'] = vax.date.str[:10]
    vax['first_dose_ct'] = pd.to_numeric(vax['first_dose_ct'])
    vax['series_complete_cnt'] = pd.to_numeric(vax['series_complete_cnt'])
    vax = vax[['first_dose_ct', 'series_complete_cnt', 'date']].groupby('date',as_index=False).sum()
    # get the most recent NYC covid case counts, hospitalizations, deaths
    covid = pd.read_json('https://data.cityofnewyork.us/resource/rc75-m7u3.json')
    covid = covid[['date_of_interest','case_count_7day_avg','hosp_count_7day_avg','death_count_7day_avg']].rename({'date_of_interest':'date'},axis=1)
    covid['date'] = covid.date.str[:10]
    fs = pd.merge(fs, vax, how= 'left', left_on = 'Date', right_on = 'date')
    fs = pd.merge(fs, covid, how= 'left', left_on = 'Date', right_on = 'date')

    #missing/future dates where covid data isn't yet available are filled with the most recent available covid data
    most_recent_first_dose = vax.loc[vax['date'] == vax['date'].max(),'first_dose_ct']
    most_recent_series_complete_cnt = vax.loc[vax['date'] == vax['date'].max(),'series_complete_cnt']
    most_recent_case_count_7day_avg = covid.loc[covid['date'] == covid['date'].max(),'case_count_7day_avg']
    most_recent_hosp_count_7day_avg	 = covid.loc[covid['date'] == covid['date'].max(),'hosp_count_7day_avg']
    most_recent_death_count_7day_avg = covid.loc[covid['date'] == covid['date'].max(),'death_count_7day_avg']

    fs['first_dose_ct'].fillna(most_recent_first_dose.values[0], inplace=True)
    fs['series_complete_cnt'].fillna(most_recent_series_complete_cnt.values[0], inplace=True)
    fs['case_count_7day_avg'].fillna(most_recent_case_count_7day_avg.values[0], inplace=True)
    fs['hosp_count_7day_avg'].fillna(most_recent_hosp_count_7day_avg.values[0], inplace=True)
    fs['death_count_7day_avg'].fillna(most_recent_death_count_7day_avg.values[0], inplace=True)

    

    if covid_scen == 'Peak COVID level':
        #average case, hospitalization, death counts from omicron period, December 15, 2021 - January 31, 2022
        fs[['case_count_7day_avg','hosp_count_7day_avg', 'death_count_7day_avg']] = covid.sort_values('date').loc[(covid['date'] > '2021-12-15') & (covid['date'] < '2022-01-31'),['case_count_7day_avg','hosp_count_7day_avg', 'death_count_7day_avg']].mean()

    elif covid_scen == 'Low COVID level':
        #average case, hospitalization, death counts from June 2021, when cases were very low in the region after vaccine rollout
        fs[['case_count_7day_avg','hosp_count_7day_avg', 'death_count_7day_avg']] = covid.sort_values('date').loc[(covid['date'] > '2021-06-01') & (covid['date'] < '2021-06-30'),['case_count_7day_avg','hosp_count_7day_avg', 'death_count_7day_avg']].mean()

    #merge weather forecast data
    fs['datehour'] = pd.to_datetime({'year': fs['DT_sched'].dt.year, 'month': fs['DT_sched'].dt.month,'day': fs['DT_sched'].dt.day,'hour': fs['DT_sched'].dt.hour})

    fs = pd.merge(fs, weather_df, how = 'left',left_on = 'datehour', right_on= 'datetime')
    #beyond the 15-day forecast assume weather will be normal by fillna with 1
    fs['cat'] = fs['cat'].fillna(1)

    #based on sidebar selection, overwrite all weather forecast data
    if weather_scen == 'Moderate':
        fs['cat'] = 2

    if weather_scen == 'Severe':
        fs['cat'] = 3


    fs = fs.set_index('DT_sched')
    fs = fs.sort_index()
    fs_preds = fs[[
        'Region', 'Flight Distance', 
        'airline_type', 'month', 'year', 'Time Category', 
        'DOW_l', 'holiday', 'ADG',
        'first_dose_ct', 'series_complete_cnt', 'case_count_7day_avg',
        'hosp_count_7day_avg', 'death_count_7day_avg',
        'Seats']]
    # One hot encoding for categorical variables: Region, airline_type, Time Category, DOW, ADG
    fs_preds['airline_type'] = fs_preds['airline_type'].replace({'TR':0,'LC':1}).astype(int)
    fs_preds = pd.get_dummies(fs_preds,columns = ['Region','Time Category','DOW_l','month','ADG'])

    SS_X = joblib.load(r'DepartingPAX_models\fs_standard_scaler.pkl')
    fs_preds_sc = fs_preds.copy()
    fs_preds_sc[['Flight Distance','first_dose_ct', 'series_complete_cnt', 'case_count_7day_avg',
        'hosp_count_7day_avg', 'death_count_7day_avg', 'Seats']] = SS_X.transform(
            fs_preds[['Flight Distance','first_dose_ct', 'series_complete_cnt', 'case_count_7day_avg',
        'hosp_count_7day_avg', 'death_count_7day_avg', 'Seats']]
        )

    rf = joblib.load(r'DepartingPAX_models\rf.pkl')
    fs[['pax_total','pax_bus','pax_lei']] = rf.predict(fs_preds_sc)
    



    df = fs[['International Domestic', 'Departure Concourse','pax_total','pax_bus','pax_lei']].reset_index().pivot_table(index = pd.Grouper(key='DT_sched', freq='10Min'),columns = ['International Domestic','Departure Concourse'], aggfunc = 'sum').fillna(0).asfreq('10T', fill_value = 0)
    ts = pd.DataFrame()
    ts['pax_intl_bus'] = df.loc(axis=1)['pax_bus','International'].sum(axis=1)
    ts['pax_intl_lei'] = df.loc(axis=1)['pax_lei','International'].sum(axis=1)
    ts['pax_dom_bus'] = df.loc(axis=1)['pax_bus','Domestic'].sum(axis=1)
    ts['pax_dom_lei'] = df.loc(axis=1)['pax_lei','Domestic'].sum(axis=1)

    ts['pax_B1_dom'] = df.loc(axis=1)['pax_total','Domestic','B1']
    ts['pax_B2_dom'] = df.loc(axis=1)['pax_total','Domestic','B2']
    ts['pax_B3_dom'] = df.loc(axis=1)['pax_total','Domestic','B3']
    ts['pax_B1_intl'] = df.loc(axis=1)['pax_total','International','B1']
    ts['pax_B2_intl'] = df.loc(axis=1)['pax_total','International','B2']
    ts['pax_B3_intl'] = df.loc(axis=1)['pax_total','International','B3']


    #bring in other predictors
    tsts = fs_preds.reset_index().groupby(pd.Grouper(key='DT_sched', freq='1D')).min().fillna(0)
    tsts = tsts.reset_index()
    tsts['date'] = tsts['DT_sched'].dt.date
    tsts['datehour'] = tsts['DT_sched'].dt.floor('h')

    tsts = tsts.merge(weather[['datetime','cat']],how='left',left_on='datehour',right_on = 'datetime')
    
    #beyond the 15-day forecast assume weather will be normal by fillna with 1
    tsts['cat'] = tsts['cat'].fillna(1)

    #based on sidebar selection, overwrite all weather forecast data
    if weather_scen == 'Moderate':
        tsts['cat'] = 2

    if weather_scen == 'Severe':
        tsts['cat'] = 3

    tsts.drop([
        'Flight Distance','airline_type','year','Seats','Region_Africa','Region_Asia','Region_Canada','Region_Europe',
        'Region_Latin America & Caribbean','Region_USA-Midwest', 'Region_USA-Northeast','Region_USA-Pacific West',
        'Region_USA-Plains','Region_USA-Southeast',
        'ADG_II','ADG_III','ADG_IV','ADG_V','ADG_VI','DT_sched','datetime'
        ],axis=1, inplace=True)

    ts = ts.reset_index()
    ts['date'] = ts['DT_sched'].dt.date
    ts = pd.merge(ts, tsts, left_on='date', right_on='date', how='left')

    ts['season'] = ts['DT_sched'].dt.quarter


    #lc and traditional airline seat counts for the TSA model
    sts = fs.reset_index()[['DT_sched','airline_type','Seats']].pivot_table(index = 
    pd.Grouper(key='DT_sched', freq='10Min'),columns = 'airline_type', aggfunc = 'sum').fillna(0)
    sts.columns = sts.columns.to_flat_index().map('_'.join)
    sts = sts.reset_index()

    ts = ts.merge(sts, how='left',left_on = 'DT_sched',right_on = 'DT_sched')
    ts[['Seats_LC','Seats_TR']] = ts[['Seats_LC','Seats_TR']].fillna(0)
    
    model_arrpax = joblib.load(r'ArrivingPAX_models\arrpax_regr.pkl')
    scaler_arrpax = joblib.load(r'ArrivingPAX_models\arrpax_covariate_scaler.pkl')
    arr_covs = ts[['DT_sched','season','cat','holiday','pax_intl_bus','pax_intl_lei','pax_dom_bus','pax_dom_lei','first_dose_ct','series_complete_cnt']].set_index('DT_sched')
    
    
    arr_covs_ts = TimeSeries.from_dataframe(arr_covs,freq='10T')
    arr_covs_sc = scaler_arrpax.transform(arr_covs_ts)

    

    ttss = model_arrpax.predict(n=arr_covs.loc[arr_covs.index >= pd.to_datetime('2021-12-01')].shape[0] - 30,future_covariates = arr_covs_sc).pd_dataframe()
    ts = ts.merge(ttss, how='left', left_on='DT_sched',right_index=True)
    ts[['AT_PAX','Parking#','FHV#']] = ts[['AT_PAX','Parking#','FHV#']].fillna(0).clip(lower=0)
    ts.set_index('DT_sched',inplace=True)


    ts['Total Arriving Passengers'] = ts[['AT_PAX','Parking#','FHV#']].sum(axis=1)
    ts['Total Departing Passengers'] = ts[['pax_intl_bus','pax_intl_lei','pax_dom_bus','pax_dom_lei']].sum(axis=1)

    tsa_covs = ts[['pax_B1_dom', 'pax_B2_dom', 'pax_B3_dom', 'pax_B1_intl', 'pax_B2_intl',
       'pax_B3_intl','Seats_TR','Seats_LC','holiday','Total Arriving Passengers']]
    
    model_tsapax = joblib.load(r'TSA_models\concourse_lr.pkl')
    scaler_tsapax = joblib.load(r'TSA_models\tsa_covariate_scaler.pkl')

    tsa_covs_ts = TimeSeries.from_dataframe(tsa_covs,freq='10T')
    tsa_covs_sc = scaler_tsapax.transform(tsa_covs_ts)


    tstsa = model_tsapax.predict(n=tsa_covs.loc[tsa_covs.index >= pd.to_datetime('2021-12-01')].shape[0] - 30,future_covariates = tsa_covs_sc).pd_dataframe()
    ts = ts.merge(tstsa, how='left', left_index=True,right_index=True)
    ts = ts.rename({'pax_C1':'pax_B1','pax_C2':'pax_B2','pax_C3':'pax_B3'},axis=1)
    ts[['pax_B1','pax_B2','pax_B3']] = ts[['pax_B1','pax_B2','pax_B3']].fillna(0).clip(lower=0)


    ts['Total TSA Passengers'] = ts[['pax_B1','pax_B2','pax_B3']].sum(axis=1)

    ts['dep_B1'] = ts['pax_B1_dom'] + ts['pax_B1_intl']
    ts['dep_B2'] = ts['pax_B2_dom'] + ts['pax_B2_intl']
    ts['dep_B3'] = ts['pax_B3_dom'] + ts['pax_B3_intl']

    #this is cumulative number of people in landside (between arrival and tsa) and airside (between tsa and departure)
    ts = ts.reset_index()
    ts['dt_shift'] = ts['DT_sched'].shift(18)   #shift by 3 hours so that the count resets every night at 3am
    ts[[
        'arr_cum','tsa_cum', 'tsa_B1_cum','tsa_B2_cum','tsa_B3_cum','dep_cum', 'dep_B1_cum', 'dep_B2_cum', 'dep_B3_cum'
        ]] = ts[[
            'dt_shift','Total Arriving Passengers','Total TSA Passengers', 'pax_B1','pax_B2','pax_B3','Total Departing Passengers','dep_B1','dep_B2','dep_B3'
            ]].groupby(pd.Grouper(key = 'dt_shift',freq='1D')).cumsum()
    ts['landside'] = (ts['arr_cum'] - ts['tsa_cum']).fillna(0).clip(lower=0)
    ts['airside_total'] = (ts['tsa_cum'] - ts['dep_cum']).fillna(0).clip(lower=0)
    ts['airside_B1'] = (ts['tsa_B1_cum'] - ts['dep_B1_cum']).fillna(0).clip(lower=0)
    ts['airside_B2'] = (ts['tsa_B2_cum'] - ts['dep_B2_cum']).fillna(0).clip(lower=0)
    ts['airside_B3'] = (ts['tsa_B3_cum'] - ts['dep_B3_cum']).fillna(0).clip(lower=0)
    ts['terminal_total'] = (ts['arr_cum'] - ts['dep_cum']).fillna(0).clip(lower=0)

    ts = ts.drop(['arr_cum','tsa_cum', 'tsa_B1_cum','tsa_B2_cum','tsa_B3_cum','dep_cum', 'dep_B1_cum', 'dep_B2_cum', 'dep_B3_cum','dt_shift'],axis=1)

    ts = ts.set_index('DT_sched')

    # st.write(ts)

    return fs, ts

fs, ts = paxfs(fs_con, aircraft_codes=aircraft_codes, airport_lookup=airport_lookup, al_lookup=al_lookup, weather=weather_df, covid_scen =covid_scenario, weather_scen = weather_scenario)



@st.cache
def agg_result(data,frequency):
    
    data = data.reset_index()
    # data = data[data['Departure Concourse'] == concourse]
    agg_data = data[['DT_sched','Departure Concourse','Region','airline_iata','pax_total','pax_bus','pax_lei']].pivot_table(index=[pd.Grouper(key="DT_sched", freq=frequency)],columns = ['Departure Concourse','Region','airline_iata'],aggfunc='sum').fillna(0)

    return agg_data



def filter_result(agg_data,start_date,end_date, paxtype, conc, destination, airlin):

    data_filtered = pd.DataFrame(index=agg_data.index)

#still have a problem with displaying when multiple categories are "ALL." Probably need to do nested if clauses, for a total of 3x3 possible selections.
# also get an error if there are no results.
    # if ('All' in conc) & ('All' in dest) & ('All' in airlin):
    #     conc = conc.remove('All')
    #     destination = destination.remove('All')
    #     airlin = airlin.remove('All')
    #     condestairl_all = agg_data.sum(level = [0,1,2,3],axis = 1).loc[:,(paxtype,conc, destination,airlin)]
    #     condestairl_all.columns = condestairl_all.columns.to_flat_index()
    #     data_filtered = data_filtered.merge(condestairl_all,how='left',left_index=True, right_index=True)
    
    # if 'All' in conc:
    #     conc = conc.remove('All')
    #     con_all = agg_data.sum(level=[0,2,3],axis=1).loc[:,(paxtype,destination,airlin)]
    #     con_all.columns = con_all.columns.to_flat_index()
    #     data_filtered = data_filtered.merge(con_all,how='left',left_index=True, right_index=True)
    # if 'All' in dest:
    #     destination = destination.remove('All')
    #     dest_all = agg_data.sum(level=[0,1,3],axis=1).loc[:,(paxtype,conc,airlin)]
    #     dest_all.columns = dest_all.columns.to_flat_index()
    #     data_filtered = data_filtered.merge(dest_all,how='left',left_index=True, right_index=True)
    # if 'All' in airlin:
    #     airlin = airlin.remove('All')
    #     al_all = agg_data.sum(level=[0,1,2],axis=1).loc[:,(paxtype,conc, destination)]
    #     al_all.columns = al_all.columns.to_flat_index()
    #     data_filtered = data_filtered.merge(al_all,how='left',left_index=True, right_index=True)
    
    disag_filtered = agg_data.loc[:,(paxtype, conc, destination, airlin)]
    disag_filtered.columns = disag_filtered.columns.to_flat_index()

    data_filtered = data_filtered.merge(disag_filtered,how='left',left_index=True, right_index=True)
    data_filtered = data_filtered.loc[(data_filtered.index >= pd.to_datetime(start_date)) & (data_filtered.index <= pd.to_datetime(end_date))]
    

    return data_filtered

@st.cache(suppress_st_warning=True)
def combined_agg(ts, freq):

    ts_ag = ts.reset_index()[['DT_sched','Total Arriving Passengers','Total TSA Passengers', 'Total Departing Passengers']].groupby(pd.Grouper(key="DT_sched", freq=freq)).sum()

    return ts_ag
    
col1, col2 = st.columns(2)
with col1:
    xpoints = st.multiselect(label = '',options = ['Total Arriving Passengers','Total TSA Passengers','Total Departing Passengers'], default = ['Total Arriving Passengers','Total TSA Passengers','Total Departing Passengers'])
    comb_agg = combined_agg(ts, freq=freq_con)
    st.line_chart(comb_agg.loc[(comb_agg.index >= S_date) & (comb_agg.index <= E_date),xpoints])

with col2:
    zones = st.multiselect(label = 'Zones',options = ['landside','airside_B1','airside_B2','airside_B3'], default = ['landside','airside_B1','airside_B2','airside_B3'])
    st.area_chart(ts.loc[(ts.index >= S_date) & (ts.index <= E_date),zones])

#total passengers in the airport at any given time
# st.area_chart(ts.loc[(ts.index >= S_date) & (ts.index <= E_date),'terminal_total'])

tab_arr, tab_tsa, tab_dep = st.tabs(["Airport Access", "Security", "Departures"])

with tab_dep:
    pax_type = st.selectbox(label = 'Passenger Type',options = ['pax_total','pax_bus','pax_lei'])
    concourse = st.multiselect(label = 'Concourse',options = ['B1','B2','B3'], default = ['B1','B2','B3'])
    dest = st.multiselect(label = 'Flight Destination',options = np.sort(fs.Region.unique()),default = np.sort(fs.Region.unique()))
    airline = st.multiselect(label = 'Airline',options = np.sort(fs.airline_iata.unique()), default = np.sort(fs.airline_iata.unique()))
    # plot_agg_result(fs,freq_con,date_range[0],date_range[1])
    ag = agg_result(fs,freq_con)
    filt = filter_result(ag, date_range[0],date_range[1], pax_type, concourse, dest, airline)

    st.area_chart(filt, width = 500, height = 300)
    # st.write(filt)

@st.cache
def ts_arr_agg(ts, freq):

    ts_ag = ts.reset_index()[['DT_sched','AT_PAX','Parking#','FHV#']].groupby(pd.Grouper(key="DT_sched", freq=freq)).sum()

    return ts_ag

with tab_arr:
    mode = st.multiselect(label = 'Transportation Mode',options = ['AirTrain','Private Car (Parking)','For-Hire Vehicle'], default = ['AirTrain','Private Car (Parking)','For-Hire Vehicle'])
    mode_dict = {'AirTrain':'AT_PAX','Private Car (Parking)':'Parking#','For-Hire Vehicle':'FHV#'}
    mode_conv = [mode_dict[v] for v in mode]
    ts_agg_arr = ts_arr_agg(ts=ts, freq = freq_con)
    st.area_chart(ts_agg_arr.loc[(ts_agg_arr.index >= S_date) & (ts_agg_arr.index <= E_date),mode_conv])    


@st.cache
def ts_tsa_agg(ts, freq):

    ts_ag = ts.reset_index()[['DT_sched','pax_B1','pax_B2','pax_B3']].groupby(pd.Grouper(key="DT_sched", freq=freq)).sum()

    return ts_ag
    

with tab_tsa:
    conc_2 = st.multiselect(label = 'Concourse ',options = ['B1','B2','B3'], default = ['B1','B2','B3'])
    conc2_dict = {'B1':'pax_B1','B2':'pax_B2','B3':'pax_B3'}
    conc2_conv = [conc2_dict[v] for v in conc_2]
    ts_agg_tsa = ts_tsa_agg(ts=ts, freq = freq_con)
    st.area_chart(ts_agg_tsa.loc[(ts_agg_tsa.index >= S_date) & (ts_agg_tsa.index <= E_date),conc2_conv])       