import streamlit as st
import numpy
import pandas as pd
import sklearn.linear_model as fit

#@st.cache_data
#function that gets input and output dataframes

df = pd.DataFrame({
  'year': [2000, 2001, 2002, 2003,2004],
  'GDP': [69, 420, 421, 0,1],
  "Chiefs Winning Percentage":[1,6,7,2,1],
  'Total Faculty': [69, 420, 429, 0,1],
  "New Faculty":[1,2,6,9,10]
})


option_Inputs = st.selectbox(
    'Which input would you like to investigate?',
     df.keys()[1:3])

option_Outputs = st.selectbox(
    'Which output would you like to investigate?',
     df.keys()[3:5])

'You chose to study the relationship between ', option_Inputs, ' and ', option_Outputs
st.divider()

first_chart=st.line_chart(df,x='year',y=[option_Inputs,option_Outputs])
st.scatter_chart(df,x=option_Inputs,y=option_Outputs)

anal_tech=["linear Regression"]

def LinReg(df, key_in,key_out):
    xvals=df[key_in]
    yvals=df[key_out]
    reg=fit.LinearRegression(xvals,yvals)
    return 
