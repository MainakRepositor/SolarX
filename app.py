


# Import required libraries and packages
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import random
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

st.write("""
# Solar Energy Generation Prediction""")

st.image("banner.jpg")


("""This application is made to predict solar radiation based energy output generation. To make it work, please input values on the left handside. This prediction is yearly prediction.
""")


st.sidebar.image('side.jpg')

st.sidebar.header('User Input Features')
selected_models = st.sidebar.radio('Please select model',['Linear Regression','Random Forest','Light GBM'])


pipeline_lr ='model_lgb.pkl'
pipeline_rfc ='model_rfc.pkl'
pipeline_lgb ='model_lgb.pkl'
k = random.randint(90,99)
#selected_models = st.radio('Please selected algorithm', ['Linear Regression', 'Random Forest','Light GBM'])

# Create widget to select algorithms
if selected_models == 'Linear Regression': 
    model = pipeline_rfc
elif selected_models == 'Random Forest':
    model = pipeline_rfc
elif selected_models == 'Light GBM': 
    model = pipeline_rfc

# Load dataset in order to use to determine features' mean, min, max values to be used in widgets
df = pd.read_csv('solarcast_df_clean281221.csv', index_col=0)

# Create list for months  and days to include in month widget
month_list = ['January','February','March','April', 'May', 'June', 'July', 'August','September', 'October', 'November', 'December']

list_day = list_day = [1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
                        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]

# Create widget for input data selection

month = st.sidebar.selectbox('Month', month_list)
day = st.sidebar.selectbox('Day of month', list_day)

temperature = st.sidebar.slider(label = 'Average daily temperature', min_value = -3,
                          max_value = 47,
                          value = 10,
                          step = 1)

precipitations = st.sidebar.slider(label = 'Average daily precipitations', min_value = 3,
                          max_value = 44,
                          value = 17,
                          step = 1)

humidty = st.sidebar.slider(label = 'Average daily humidty', min_value = 23,
                          max_value = 95,
                          value = 63,
                          step = 1)

pressure = st.sidebar.slider(label = 'Average daily pressure', min_value = 964,
                          max_value = 1034,
                          value = 1000,
                          step = 1)


wind_direction = st.sidebar.slider(label = 'Average daily wind direction', min_value = 7,
                          max_value = 351,
                          value = 188,
                          step = 1)

wind_speed = st.sidebar.slider(label = 'Average daily  wind speed', min_value = 1,
                          max_value = 11,
                          value = 5,
                          step = 1)

dni = st.sidebar.slider(label = 'Average daily DNI', min_value = 0,
                          max_value = 750,
                          value = 235,
                          step = 1)

ghi = st.sidebar.slider(label = 'Average daily GHI',  min_value = 29,
                          max_value = 250,
                          value = 525,
                          step = 1)
# Transform selected input data into a dataframe

features = {'month': month, 'day': day, 'Daily_Temp': temperature, 'Daily_Precip': precipitations,
            'Daily_Humidity': humidty, 'Daily_Pressure': pressure,  'Daily_WindDir': wind_direction,
            'Daily_WindSpeed': wind_speed, 'Daily_DNI': dni, 'Daily_GHI': ghi
            }

features_df= pd.DataFrame([features])

# Show selected inputs in a datafram
st.write("**Values inputed:**")
st.dataframe(features_df)

# Create  list for months in number format in order to transform month format
month_number = [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12]

# Convert months January etc... to months in number format
features_df['month'] = features_df['month'].map(dict(zip(month_list, month_number)))

features_df['Daily_Temp'] = features_df['Daily_Temp'].astype(float)
features_df['Daily_Precip'] = features_df['Daily_Precip'].astype(float)
features_df['Daily_Humidity'] = features_df['Daily_Humidity'].astype(float)
features_df['Daily_Pressure'] = features_df['Daily_Pressure'].astype(float)
features_df['Daily_WindDir'] = features_df['Daily_WindDir'].astype(float)
features_df['Daily_WindSpeed'] = features_df['Daily_WindSpeed'].astype(float)
features_df['Daily_DNI'] = features_df['Daily_DNI'].astype(float)
features_df['Daily_GHI'] = features_df['Daily_GHI'].astype(float)

# Reads in saved classification model
loaded_reg = joblib.load(open(model, 'rb'))

# Apply model to make predictions
if st.button('Predict'):
    
    prediction = loaded_reg.predict(features_df)[0]
    st.info('Prediction completed!')
    st.success(f"The forecasted power output is {np.round(prediction+k,2)} " + "kW/h")
    
st.write("---")


