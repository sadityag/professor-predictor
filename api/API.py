import streamlit as st
import numpy
import pandas as pd
import sklearn.linear_model as fit

from importlib.machinery import SourceFileLoader

analysis = SourceFileLoader("analysis_line-regression_class", "../data-analysis/analysis-framework/analysis_line-regression_class.py").load_module()

df=pd.read_csv('../data-collection/data_interpolated.csv')
input_df=df.drop(columns=["Number of Faculty Positions"])
output_df=df["Number of Faculty Positions"]


option_inputs = st.selectbox(
    'Which input would you like to investigate?',
     input_df.keys())


option_outputs = "Number of Faculty Positions"
'You chose to study the relationship between ', option_inputs, ' and ', option_outputs
st.divider()
if option_inputs!="Year":
  first_chart=st.line_chart(df,x='Year',y=[option_inputs])
  st.divider()
  second_chart=st.line_chart(df,x='Year',y=[option_outputs])
  st.divider()
  st.scatter_chart(df,x=option_inputs,y=option_outputs)
  st.divider()
else:
  first_chart=st.line_chart(df,x='Year',y=[option_outputs])
  st.divider()


analysis_techiques=["Time Series Regression"]
analysis_choice = st.selectbox(
    'Which type of analysis would you like to perform?',
     analysis_techiques)

if analysis_choice=="Time Series Regression":
  out_dict=analysis.TimeSeriesRegression().linear_regression(input_df[option_inputs],output_df)
  "This model predicts that there will be "+str(int(out_dict['prediction']))+" faculty positions in the year "+str(int(max(input_df[option_inputs])+1))+"."
