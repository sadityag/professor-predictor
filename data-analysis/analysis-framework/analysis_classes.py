import numpy as np
import pandas as pd
from scipy import signal
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

class TimeSeriesRegression:
    def clean_series(self, X, Y):
        """
        Aligns two time series based on the first valid index.
        Inputs:
        X           : yearly time series data
        Y           : yearly time series data
        Outputs:
        tuple       : (X_aligned, Y_aligned)
        X_aligned   : time series data aligned based on first valid index
        Y_aligned   : time series data aligned based on first valid index
        """
        # Ensure we're working with pandas Series
        if not isinstance(X, pd.Series):
            X = pd.Series(X)
        if not isinstance(Y, pd.Series):
            Y = pd.Series(Y)
            
        # Get the overlapping time period
        first_index = max(X.first_valid_index(), Y.first_valid_index())
        last_index = min(X.last_valid_index(), Y.last_valid_index())
        
        return X[first_index:last_index], Y[first_index:last_index]

    def align_with_lag(self, X, Y, lag):
        """
        Aligns two time series based on a given lag value.
        Inputs:
        X    : yearly time series data
        Y    : yearly time series data
        lag  : integer lag value to shift X backwards
        Outputs:
        tuple: (X_aligned, Y_aligned) where X is shifted back by lag periods
        """
        # Ensure we're working with pandas Series
        if not isinstance(X, pd.Series):
            X = pd.Series(X)
        if not isinstance(Y, pd.Series):
            Y = pd.Series(Y)
        
        # Get Y's time range
        y_start = Y.first_valid_index()
        y_end = Y.last_valid_index()
        
        # Select X values from (y_start - lag) to (y_end - lag)
        X_lagged = X.loc[y_start - lag:y_end - lag]
        
        # Ensure the indices align properly
        X_aligned = X_lagged.reset_index(drop=True)
        Y_aligned = Y.loc[y_start:y_end].reset_index(drop=True)
        
        return X_aligned, Y_aligned

    def max_lag(self, X, Y, max_lag_years=10):
        """
        Finds the lag that maximizes correlation between two time series.
        Inputs:
        X             : yearly time series data
        Y             : yearly time series data
        max_lag_years : optional integer that specifies 
                        maximum number of years to check for lag
        Outputs:
        tuple           : (optimal_lag, max_correlation)
        optimal_lag     : maximizes the correlation of the 2 series
        max_correlation : the value of the correlation
        """
        # Get the clean, aligned series first
        X_clean, Y_clean = self.clean_series(X, Y)
        
        # Convert to numpy arrays for correlation calculation
        X_arr = np.array(X_clean)
        Y_arr = np.array(Y_clean)
        
        # Convert to standard normal
        X_norm = (X_arr - np.mean(X_arr)) / np.std(X_arr)
        Y_norm = (Y_arr - np.mean(Y_arr)) / np.std(Y_arr)
        
        # Calculate cross-correlation
        correlations = signal.correlate(Y_norm, X_norm, mode='full')
        lags = signal.correlation_lags(len(X_norm), len(Y_norm))
        
        # Only consider positive lags up to max_lag_years
        valid_indices = (lags >= 0) & (lags <= max_lag_years)
        valid_correlations = correlations[valid_indices]
        valid_lags = lags[valid_indices]
        
        # Find the lag with maximum correlation
        max_corr_index = np.argmax(valid_correlations)
        optimal_lag = valid_lags[max_corr_index]
        max_correlation = valid_correlations[max_corr_index] / len(X_arr)  # Normalize by series length
        
        return optimal_lag, max_correlation
    
    def linear_regression(self, X, Y):
        """
        Finds the lag that maximizes correlation between two time series.
        Inputs:
        X             : yearly time series data, regression parameter
        Y             : yearly time series data, target
        max_lag_years : optional integer that specifies 
                        maximum number of years to check for lag
        Outputs:
        dict           : {
            'best lag': optimal lag value,
            'prediction for next year': predicted value for Y in the next period,    
            'dataset correlation': correlation between lagged series,
            'R value': R-squared value of the regression
        }
        """
        # Find optimal lag
        best_lag, correlation = self.max_lag(X, Y)
        
        # Align series based on optimal lag
        X_aligned, Y_aligned = self.align_with_lag(X, Y, best_lag)
        
        # Perform linear regression
        X_with_const = add_constant(X_aligned)
        model = OLS(Y_aligned, X_with_const).fit()
        
        # Calculate prediction for next year
        last_X = X.iloc[-1]  # Use last available X value
        next_year_pred = model.params[0] + model.params[1] * last_X
        
        # Calculate dataset correlation
        dataset_corr = stats.pearsonr(X_aligned, Y_aligned)[0]
        
        return {
            "lag": best_lag,
            "prediction": next_year_pred,
            "correlation": dataset_corr,
            "R": model.rsquared
        }