# Import necessary libraries
import streamlit as st
import numpy
import pandas as pd
import sklearn.linear_model as fit
import plotly.express as px
import math
from importlib.machinery import SourceFileLoader

# Load external analysis module
import analysis_classes as analysis

# Utility function: Round a number to a given number of significant figures
def round_sigfigs(value, sig_figs):
    """
    Rounds a given value to the specified number of significant figures.
    Returns NaN for non-numeric inputs.
    """
    try:
        # Ensure the value is a float
        value = float(value)
        if value == 0:
            return 0  # Zero remains zero
        # Compute significant figure rounding
        return round(value, sig_figs - int(math.floor(math.log10(abs(value))) + 1))
    except (ValueError, TypeError):
        # Return NaN if the input is not numeric
        return float('nan')


# Load data
df = pd.read_csv('data_interpolated.csv')

# Define metadata dictionaries for inputs and outputs
year_dict = dict(
    display='Year',
    display_lower='year',
    display_upper_lower='Year',
    units='',
    type=int,
    key='year'
)

NSF_dict = dict(
    display='Number of NSF Awards',
    display_lower='number of NSF awards',
    display_upper_lower='Number of NSF awards',
    units='',
    type=int,
    key='NSF_awards'
)

# Additional input metadata dictionaries
Infl_dict = dict(display='Inflation Rate', display_lower='inflation rate', display_upper_lower='Inflation rate',
                 units='(%)', type=float, key='inflation_rate')
Fed_dict = dict(display='Federal Budget', display_lower='federal budget', display_upper_lower='Federal budget',
                units='(Billions of Dollars)', type=float, key='Fed_Budget')
PA_dict = dict(display='PA Budget Difference', display_lower='PA budget difference',
               display_upper_lower='PA budget difference', units='(Millions of Dollars)', type=int, key='PA_Budget_diff')
GDP_dict = dict(display='Gross Domestic Product (GDP)', display_lower='gross domestic product',
                display_upper_lower='Gross domestic product', units='(Billions of Dollars)', type=float, key='GDP')
CPI_dict = dict(display='Consumer Price Index (CPI)', display_lower='consumer price index',
                display_upper_lower='Consumer price index', units='', type=float, key='CPI_inflation')
LaborBS_dict = dict(display='Labor BS', display_lower='labor BS', display_upper_lower='Labor BS',
                    units='(?)', type=float, key='Labor_BS')
LaborC_dict = dict(display='Labor Cond', display_lower='labor cond', display_upper_lower='Labor cond',
                   units='(?)', type=float, key='Labor_cond')
UnemployBS_dict = dict(display='Unemployment BS', display_lower='unemployment BS',
                       display_upper_lower='Unemployment BS', units='(?)', type=float, key='Unemploy_BS')
Unemploy_dict = dict(display='Unemployment', display_lower='unemployment',
                     display_upper_lower='Unemployment', units='(?)', type=float, key='Unemploy')

# Output metadata dictionary
fac_dict = dict(
    display='Number of Faculty Positions',
    display_lower='number of faculty positions',
    display_upper_lower='Number of faculty positions',
    units='',
    type=int,
    key='faculty'
)

# Organize input and output metadata
inputs = dict(
    year=year_dict,
    NSF=NSF_dict,
    Infl=Infl_dict,
    fed=Fed_dict,
    PA=PA_dict,
    GDP=GDP_dict,
    CPI=CPI_dict,
    LaborBS=LaborBS_dict,
    LaborC=LaborC_dict,
    UnemplyBS=UnemployBS_dict,
    Unemploy=Unemploy_dict
)
outputs = dict(fac=fac_dict)

# Utility function to retrieve input metadata by display name
def get_input_by_display(display_value, inputs_dict):
    """
    Retrieves metadata for a given input based on its display name.
    """
    for key, details in inputs_dict.items():
        if details['display'] == display_value:
            return details
    return None  # Return None if not found


# Display options for input selection
option_inputs = st.selectbox(
    'Which input would you like to investigate?',
    [details['display'] for details in inputs.values()]
)
selected_input = get_input_by_display(option_inputs, inputs)

# Output selection (currently fixed)
option_outputs = "Number of Faculty Positions"
selected_output = get_input_by_display(option_outputs, outputs)

# Display the selected relationship
st.write('You chose to study the relationship between ', selected_input['display_lower'], ' and ',
         selected_output['display_lower'])
st.divider()

# Plotting logic
if selected_input['display'] != "Year":
    # Plot the selected input and output over time
    fig = px.line(df, x='year', y=selected_input['key'],
                  title=f"{selected_input['display_upper_lower']} over time",
                  labels={'year': 'Year', selected_input['key']: f"{selected_input['display']} {selected_input['units']}"})
    st.plotly_chart(fig)
    st.divider()

    # Scatter plot: Input vs Output
    fig = px.scatter(df, x=selected_input['key'], y=selected_output['key'],
                     title=f"{selected_output['display_upper_lower']} vs. {selected_input['display_lower']}",
                     labels={selected_input['key']: f"{selected_input['display']} {selected_input['units']}",
                             selected_output['key']: selected_output['display']})
    st.plotly_chart(fig)
    st.divider()

else:
    # Special case: Year as the input
    fig = px.line(df, x="year", y=selected_output['key'],
                  title="Number of faculty positions over time",
                  labels={'year': 'Year', selected_output['key']: selected_output['display']})
    st.plotly_chart(fig)
    st.divider()


# Analysis section: Perform predictive analysis based on user-selected input and output

#renames analysis statistics to presentable strings
display_stats_mapping = {
            'lag': 'Optimal lag value',
            'prediction' : 'Predicted number of faculty positions next year',
            'r2'        : '$R^2$ value',
            'rmse'      : 'Root mean square error (RMSE)',
            'mae'       : 'Mean absolute error (MAE)',
            'aic'       : 'Akaike Information Criterion (AIC)',
            'std'       : 'Prediction standard deviation',
            'cv_score'  : 'Mean validation score',
            'cv_error'  : 'Standard deviation of validation scores'
}


if selected_input['display'] != "Year":
    # Define available analysis techniques for non-year inputs
    analysis_techniques = [
        "Locally Weighted Scatterplot Smoothing (LOWESS)",
        "Autoregressive Integrated Moving Average (ARIMA)",
        "Polynomial Regression",
        "Linear Regression",
        "Gaussian Process Regression"
    ]

    # User selects the analysis method
    analysis_choice = st.selectbox(
        'Which type of analysis would you like to perform?',
        analysis_techniques
    )

    # Initialize the predictive regression class
    tsr = analysis.PredictiveRegression()

    # Perform the selected analysis
    if analysis_choice == 'Locally Weighted Scatterplot Smoothing (LOWESS)':
        results = tsr.lowess_regression(df[selected_input['key']], df[selected_output['key']])
    elif analysis_choice == "Autoregressive Integrated Moving Average (ARIMA)":
        results = tsr.arima_regression(df[selected_input['key']], df[selected_output['key']])
    elif analysis_choice == "Polynomial Regression":
        results = tsr.polynomial_regression(df[selected_input['key']], df[selected_output['key']])
    elif analysis_choice == "Linear Regression":
        results = tsr.linear_regression(df[selected_input['key']], df[selected_output['key']])
    elif analysis_choice == "Gaussian Process Regression":
        results = tsr.gaussian_process_regression(df[selected_input['key']], df[selected_output['key']])

    # Prepare data for plotting the results
    plot_data = results['plot_data'].sort_values(by='X')
    plot_data_long = pd.melt(
        plot_data, id_vars=['X'], value_vars=['Y_data', 'Y_pred'],
        var_name='Line', value_name='Y'
    )

    # Generate a line plot for data and predictions
    fig = px.line(
        plot_data_long, x='X', y='Y', color='Line',
        title=f"{selected_output['display_upper_lower']} vs. {selected_input['display_lower']}",
        labels={
            'X': f"{selected_input['display']} {selected_input['units']}",
            'Y': selected_output['display'],
            'Line': ''
        }
    )

    # Update the legend names for the plot
    display_name_mapping = {
        'Y_data': 'Data',
        'Y_pred': 'Model Prediction'
    }
    fig.for_each_trace(lambda t: t.update(name=display_name_mapping[t.name]))
    st.plotly_chart(fig)
    st.divider()

    # Residual plot
    plot_data['Residual'] = plot_data['Y_data'] - plot_data['Y_pred']
    fig = px.line(
        plot_data, x='X', y='Residual',
        title=f"Residual {selected_output['display_lower']} vs. {selected_input['display_lower']}",
        labels={
            'X': f"{selected_input['display']} {selected_input['units']}",
            'Residual': f"Residual {selected_output['display']}",
            'Line': ''
        }
    )
    st.plotly_chart(fig)
    st.divider()

    # Display summary statistics
    st.subheader("Summary Statistics")
    del results['plot_data']  # Exclude plot data from summary statistics
    display_results = {
        display_stats_mapping.get(old_key, old_key): value
        for old_key, value in results.items()
    }

    # Display rounded statistics
    for key in display_results.keys():
        if not math.isnan(round_sigfigs(display_results[key], 3)):
            st.markdown(f"{key}: {round_sigfigs(display_results[key], 3)}")

else:
    # Special case: Time series analysis for "Year" input
    analysis_techniques = ["Time Series Regression"]
    analysis_choice = st.selectbox(
        'Which type of analysis would you like to perform?',
        analysis_techniques
    )

    # Initialize the predictive regression class
    tsr = analysis.PredictiveRegression()

    # Perform time series regression
    if analysis_choice == 'Time Series Regression':
        results = tsr.time_series_regression(df[selected_input['key']], df[selected_output['key']])

    # Prepare data for plotting the results
    plot_data = results['plot_data'].sort_values(by='time')
    plot_data_long = pd.melt(
        plot_data, id_vars=['time'], value_vars=['Y_data', 'Y_pred','trend'],
        var_name='Line', value_name='Y'
    )

    # Generate a line plot for data and predictions
    fig = px.line(
        plot_data_long, x='time', y='Y', color='Line',
        title=f"{selected_output['display_upper_lower']} over time",
        labels={
            'time': f"{selected_input['display']} {selected_input['units']}",
            'Y': selected_output['display'],
            'Line': ''
        }
    )

    # Update the legend names for the plot
    display_name_mapping = {
        'Y_data': 'Data',
        'Y_pred': 'Model Prediction',
        'trend' : 'Trend',
    }
    fig.for_each_trace(lambda t: t.update(name=display_name_mapping[t.name]))
    st.plotly_chart(fig)
    st.divider()

    # Cycle plot
    fig = px.line(
        plot_data, x='time', y='cycle',
        title=f"Cycle of {selected_output['display_lower']}",
        labels={
            'time': f"{selected_input['display']} {selected_input['units']}",
            'cycle': f"{selected_output['display']} - Trend",
            'Line': ''
        }
    )
    st.plotly_chart(fig)
    st.divider()
    # Display summary statistics
    st.subheader("Summary Statistics")
    del results['plot_data']  # Exclude plot data from summary statistics
    display_results = {
        display_stats_mapping.get(old_key, old_key): value
        for old_key, value in results.items()
    }

    # Display rounded statistics
    for key in display_results.keys():
        if not math.isnan(round_sigfigs(display_results[key], 3)):
            st.markdown(f"{key}: {round_sigfigs(display_results[key], 3)}")


