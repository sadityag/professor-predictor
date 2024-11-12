import streamlit as st
import numpy
import pandas as pd
import sklearn.linear_model as fit

#@st.cache_data
#function that gets input and output dataframes


df = pd.DataFrame({
  'year': [2000, 2001, 2002, 2003,2004],
  'GDP': [69, 420, 421, 0,1],
  "Chiefs Winning Percentage":[20,40,60,23,100],
  'Total Faculty': [69, 420, 429, 0,1],
  "New Faculty":[1,2,6,9,10]
})

df=pd.read_csv('../data-collection/data_interpolated.csv')

option_inputs = st.selectbox(
    'Which input would you like to investigate?',
     df.keys())


option_outputs = st.selectbox(
    'Which output would you like to investigate?',
     #df.keys()[1]
    ["faculty"])

'You chose to study the relationship between ', option_inputs, ' and ', option_outputs
st.divider()
st.divider()

first_chart=st.line_chart(df,x='year',y=[option_inputs,option_outputs])
st.divider()
st.scatter_chart(df,x=option_inputs,y=option_outputs)


analysis_techiques=["linear Regression"]

def LinReg(df, key_in,key_out):
    xvals=df[key_in]
    yvals=df[key_out]
    reg=fit.LinearRegression(xvals,yvals)
    return 
