import numpy as np
import pandas as pd
import pickle
import bz2file as bz2
import streamlit as st
import plotly.express as px



def decompress_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = pickle.load(data)
    return data

infile=open('columns.pkl','rb')
columns=pickle.load(infile)
infile.close()
model=decompress_pickle('model.pbz2')
mx=decompress_pickle('vectorizer.pbz2')
df=decompress_pickle('DataFrame.pbz2')
mean_df=df.groupby(['company','year'])['Price'].mean().reset_index()

input_dict=dict()
for i in columns:
    input_dict[i]=[0]


st.title('Car_price_Prediction')
import streamlit as st
year = st.slider('How old are you?', 1995, 2022, 2000)
kms_driven = st.number_input('Insert a kms_driven',min_value=0,help='Eg 10000 kms')
Company = options = st.multiselect(
    'What are your favorite colors',
    ['hyundai', 'mahindra', 'ford', 'maruti', 'skoda', 'audi', 'toyota',
     'renault', 'honda', 'datsun', 'mitsubishi', 'tata', 'volkswagen',
     'chevrolet', 'mini', 'bmw', 'nissan', 'others', 'mercedes'])
Fuel_type = st.radio(
    "What\'s your favorite movie genre",
    ('Petrol', 'LPG', 'Diesel'))

if st.button('Submit'):

    input_dict['year']=int(year)
    input_dict['kms_driven']=int(kms_driven)


    Fuel_str='fuel_type_'+Fuel_type
    input_dict[Fuel_str]=[1]

    resell_values=list()
    for i  in Company:
        input_dict['company_'+i]=[1]
        input_pd=pd.DataFrame.from_dict(input_dict)
        input_scaled=mx.transform(input_pd)
        resell_value=model.predict(input_scaled).round()
        resell_values.append([i,resell_value])
        input_dict['company_' + i] = [0]

    plot_df=pd.DataFrame(columns=['company','year','Price'])
    for i in resell_values:
        if(i[1]<10000):
            resell_value="Not Sellable"
        st.metric(label="RESELL VALUE for brand " + i[0], value=resell_value)
        temp_df = mean_df[mean_df['company'] == i[0]]
        plot_df=pd.concat([plot_df, temp_df])

        #st.dataframe(plot_df)
    fig = px.line(plot_df, x="year", y="Price", color='company')
    st.plotly_chart(fig)
    print(Company)

