# Standard libraries
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Union, List
import itertools
import warnings

# Scikit-learn imports
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.exceptions import ConvergenceWarning

# Stats models
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
# Non-parametric regression
from statsmodels.nonparametric.smoothers_lowess import lowess

# SciPy
from scipy import signal
from scipy.stats import norm
from scipy.fft import fft, fftfreq, ifft


# Suppress specific warnings (optional)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)

class PredictiveRegression:
    def _calculate_metrics(self, y_true, y_pred, n_params=0):
        """
        Calculates standardized error metrics for model evaluation.
        Inputs:
        y_true    : actual values
        y_pred    : predicted values
        n_params  : number of model parameters for AIC calculation
        
        Outputs:
        dict      : dictionary containing standard error metrics
            - r2   : R-squared value
            - rmse : Root Mean Square Error
            - mae  : Mean Absolute Error
            - aic  : Akaike Information Criterion
        """
        n = len(y_true)
        if n < 2:
            raise ValueError("Need at least 2 points to calculate metrics")
        
        # Calculate residuals
        residuals = y_true - y_pred
        
        # Calculate RSS and MSE
        rss = np.sum(residuals ** 2)
        mse = rss / n
        
        # R-squared
        tss = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (rss / tss) if tss != 0 else 0
        
        # RMSE
        rmse = np.sqrt(mse)
        
        # MAE
        mae = np.mean(np.abs(residuals))
        
        # AIC (assuming Gaussian errors)
        # AIC = n * ln(RSS/n) + 2k, where k is number of parameters
        aic = n * np.log(rss/n) + 2 * n_params
        
        return {
            "r2": r2,
            "rmse": rmse,
            "mae": mae,
            "aic": aic
        }

    def _calculate_non_parametric_aic(self, y_true, y_pred, effective_params):
        """
        Calculates pseudo-AIC for non-parametric models.
        Inputs:
        y_true           : actual values
        y_pred           : predicted values
        effective_params : effective degrees of freedom for the model
        
        Outputs:
        float    : pseudo-AIC value
        """
        n = len(y_true)
        residuals = y_true - y_pred
        rss = np.sum(residuals ** 2)
        
        # Pseudo-AIC for non-parametric models
        # Uses trace of smoothing matrix as effective parameters
        return n * np.log(rss/n) + 2 * effective_params
    
    def clean_series(self, X, Y):
        """
        Aligns two time series based on shared valid data points.
        
        Inputs:
        X           : yearly time series data
        Y           : yearly time series data
        
        Outputs:
        tuple       : (X_aligned, Y_aligned)
        """
        if not isinstance(X, pd.Series):
            X = pd.Series(X)
        if not isinstance(Y, pd.Series):
            Y = pd.Series(Y)
        
        # Get overlapping indices where both series have valid data
        valid_indices = X.notna() & Y.notna()
        
        # Return aligned series with only valid data points
        return X[valid_indices], Y[valid_indices]

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
        if not isinstance(X, pd.Series):
            X = pd.Series(X)
        if not isinstance(Y, pd.Series):
            Y = pd.Series(Y)
        
        # Create pairs of indices for alignment
        Y_indices = Y[Y.notna()].index
        X_indices = X[X.notna()].index
        
        # Calculate lagged indices
        Y_start, Y_end = Y_indices.min(), Y_indices.max()
        X_start, X_end = X_indices.min() + lag, X_indices.max() + lag
        
        # Find overlapping period
        start_idx = max(Y_start, X_start)
        end_idx = min(Y_end, X_end)
        
        # If no overlap after lag, return empty series
        if start_idx > end_idx:
            return pd.Series(), pd.Series()
        
        # Align the series
        Y_aligned = Y.loc[start_idx:end_idx]
        X_aligned = X.loc[start_idx-lag:end_idx-lag]
        
        # Get only points where both series have valid data
        valid_indices = X_aligned.notna() & Y_aligned.notna()
        return X_aligned[valid_indices], Y_aligned[valid_indices]

    def max_lag(self, X, Y, max_lag_years=6):
        """
        Finds the lag that maximizes correlation between two time series.
        
        Inputs:
        X             : yearly time series data
        Y             : yearly time series data
        max_lag_years : maximum number of years to check for lag
        
        Outputs:
        tuple           : (optimal_lag, max_correlation, metrics)
        optimal_lag     : lag that maximizes correlation
        max_correlation : correlation value at optimal lag
        metrics        : evaluation metrics at optimal lag
        """
        if max_lag_years < 0:
            raise ValueError("max_lag_years must be non-negative")
        
        # Initial data cleaning to get valid data points
        X_clean, Y_clean = self.clean_series(X, Y)
        
        if len(X_clean) < 2 or len(Y_clean) < 2:
            raise ValueError("Insufficient valid data points in series")
        
        # Convert to numpy arrays for computation
        X_arr = np.array(X_clean)
        Y_arr = np.array(Y_clean)
        
        # Standardize series
        X_norm = (X_arr - np.mean(X_arr)) / (np.std(X_arr) + 1e-10)
        Y_norm = (Y_arr - np.mean(Y_arr)) / (np.std(Y_arr) + 1e-10)
        
        # Try different lags and compute correlations
        correlations = []
        valid_lags = []
        
        for lag in range(max_lag_years + 1):
            X_lagged, Y_aligned = self.align_with_lag(X_clean, Y_clean, lag)
            if len(X_lagged) >= 2:  # Need at least 2 points for correlation
                corr = np.corrcoef(X_lagged, Y_aligned)[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
                    valid_lags.append(lag)
        
        if not correlations:
            # If no valid correlations found, return lag 0
            X_aligned, Y_aligned = self.clean_series(X, Y)
            return 0, 0, self._calculate_metrics(Y_aligned, X_aligned, n_params=1)
        
        # Find optimal lag
        best_idx = np.argmax(np.abs(correlations))
        optimal_lag = valid_lags[best_idx]
        max_correlation = correlations[best_idx]
        
        # Calculate metrics at optimal lag
        X_lagged, Y_aligned = self.align_with_lag(X, Y, optimal_lag)
        metrics = self._calculate_metrics(Y_aligned, X_lagged, n_params=1)
        
        return optimal_lag, max_correlation, metrics 

    def _do_cross_validation(self, X, Y, model_func, params, k_folds=5):
        """
        Helper function to perform cross validation for any model
        Inputs:
        X           : aligned X data for validation
        Y           : aligned Y data for validation
        model_func  : model fitting function
        params      : dictionary of model parameters
        k_folds     : number of folds for CV
        
        Outputs:
        tuple       : (cv_score, cv_error)
        cv_score    : mean R2 score across folds
        cv_error    : standard deviation of R2 scores across folds
        """
        # Initialize time series cross-validation
        tscv = TimeSeriesSplit(n_splits=k_folds, test_size=max(2, len(X) // (k_folds + 1)))
        val_scores = []
        
        try:
            for train_idx, val_idx in tscv.split(X):
                try:
                    # Split data into training and validation sets
                    X_train = X.iloc[train_idx]
                    Y_train = Y.iloc[train_idx]
                    X_val = X.iloc[val_idx]
                    Y_val = Y.iloc[val_idx]
                    
                    # Convert to numpy arrays if needed
                    if isinstance(X_train, pd.Series):
                        X_train = X_train.values
                    if isinstance(Y_train, pd.Series):
                        Y_train = Y_train.values
                    if isinstance(X_val, pd.Series):
                        X_val = X_val.values
                    if isinstance(Y_val, pd.Series):
                        Y_val = Y_val.values
                    
                    # Reshape arrays
                    X_train = X_train.reshape(-1, 1)
                    Y_train = Y_train.reshape(-1)
                    X_val = X_val.reshape(-1, 1)
                    Y_val = Y_val.reshape(-1)
                    
                    # Fit model
                    if 'X' in params and 'Y' in params:
                        model = model_func(**params)
                    else:
                        model = model_func(X=X_train, Y=Y_train)
                    
                    # Get predictions
                    if hasattr(model, 'predict'):
                        Y_pred = model.predict(X_val)
                    elif callable(model):
                        Y_pred = model(X_val)
                    else:
                        Y_pred = model.forecast(steps=len(val_idx), exog=X_val)
                    
                    # Handle array-like predictions
                    Y_pred = np.array(Y_pred).reshape(-1)
                    Y_val = np.array(Y_val).reshape(-1)
                    
                    # Calculate R2 score for this fold
                    fold_score = r2_score(Y_val, Y_pred)
                    if not np.isnan(fold_score):
                        val_scores.append(fold_score)
                        
                except Exception as e:
                    print(f"Warning: Error in fold: {str(e)}")
                    continue
            
            if not val_scores:
                return np.nan, np.nan
            
            # Calculate mean and standard deviation of scores
            cv_score = np.mean(val_scores)
            cv_error = np.std(val_scores)
            
            return cv_score, cv_error
        
        except Exception as e:
            print(f"Error in cross-validation: {str(e)}")
            return np.nan, np.nan
    
    def time_series_regression(self, X, Y, method='linear', do_cv=True, k_folds=5):
        """
        Performs time series regression with trend and cyclical components.
        Cross validation is performed only on the trend component.
        
        Args:
            X (pd.DataFrame): Years data
            Y (pd.Series): Time series data
            method (str): Regression method ('linear', 'polynomial', 'lowess', 'arima', 'gaussian_process')
            do_cv (bool): Whether to perform cross-validation
            k_folds (int): Number of folds for cross-validation
        """
        # Convert X to yearly format and handle errors
        try:
            X = pd.Series(X) if not isinstance(X, pd.Series) else X
            Y = pd.Series(Y) if not isinstance(Y, pd.Series) else Y
        except Exception as e:
            raise ValueError(f"Could not convert input data to pandas Series: {str(e)}")
        
        # Clean and align the data
        X_clean, Y_clean = self.clean_series(X, Y)
        
        if len(X_clean) < 2 * k_folds:
            raise ValueError(f"Insufficient data points ({len(X_clean)}) for {k_folds}-fold cross-validation")
        
        # Initialize storage for cross-validation
        cv_scores = []
        
        if do_cv:
            tscv = TimeSeriesSplit(n_splits=k_folds)
            for train_idx, test_idx in tscv.split(X_clean):
                # Split data
                X_train = X_clean.iloc[train_idx]
                Y_train = Y_clean.iloc[train_idx]
                X_test = X_clean.iloc[test_idx]
                Y_test = Y_clean.iloc[test_idx]
                
                if len(X_train) < 4:  # Minimum required for meaningful analysis
                    continue
                    
                # Get trend
                try:
                    trend_results = self.trend_regression(X_train, Y_train, X_test, method)
                    # Calculate fold score on trend only
                    score = r2_score(Y_test, trend_results['prediction'])
                    if not np.isnan(score):
                        cv_scores.append(score)
                except Exception as e:
                    print(f"Warning: Error in fold: {str(e)}")
                    continue
        
        # Fit on full dataset and predict through 2025
        future_years = pd.Series(np.arange(float(X_clean.iloc[-1]) + 1, 2026),
                            index=range(len(X_clean), len(X_clean) + int(2025 - X_clean.iloc[-1])))
        
        # Get final trend
        trend_results = self.trend_regression(X_clean, Y_clean, future_years, method)
        
        # Get cyclical component for the complete series
        Y_detrended = Y_clean - trend_results['fitted_values']
        cycle_results = self.cyclicality_analysis(Y_detrended, X_clean, future_years)
        
        # Combine final predictions
        Y_pred = trend_results['fitted_values'] + cycle_results['fitted_values']
        final_predictions = trend_results['prediction'] + cycle_results['prediction']
        
        # Create plot data with aligned indices
        all_times = pd.concat([X_clean, future_years])
        plot_data = pd.DataFrame({
            'time': all_times,
            'Y_data': pd.concat([Y_clean, pd.Series(index=future_years.index, dtype=float)]),
            'trend': pd.concat([trend_results['fitted_values'], trend_results['prediction']]),
            'cycle': pd.concat([cycle_results['fitted_values'], cycle_results['prediction']]),
            'Y_pred': pd.concat([Y_pred, final_predictions])
        })
        
        # Calculate metrics for complete model
        metrics = self._calculate_metrics(Y_clean, Y_pred)
        
        # Calculate effective parameters based on method
        if method == 'linear':
            effective_params = 2
        elif method == 'polynomial':
            effective_params = 3
        elif method == 'arima':
            effective_params = 3
        else:
            effective_params = int(len(X_clean) * 0.1)  # 10% of data points for non-parametric methods
        
        results = {
            'prediction': float(final_predictions.iloc[-1]),
            'r2': metrics['r2'],
            'rmse': metrics['rmse'],
            'mae': metrics['mae'],
            'aic': self._calculate_non_parametric_aic(Y_clean, Y_pred, effective_params),
            'plot_data': plot_data
        }
        
        if do_cv and cv_scores:
            results.update({
                'cv_score': np.mean(cv_scores),
                'cv_error': np.std(cv_scores)
            })
        
        return results

    def trend_regression(self, X_train, Y_train, X_pred, method='linear'):
        """
        Performs trend regression using specified method.
        
        Args:
            X_train (pd.Series): Training years
            Y_train (pd.Series): Training data
            X_pred (pd.Series): Prediction years
            method (str): Regression method
        """
        # Scale X values with error handling
        scaler = StandardScaler()
        try:
            X_train_scaled = scaler.fit_transform(X_train.values.reshape(-1, 1))
            X_pred_scaled = scaler.transform(X_pred.values.reshape(-1, 1))
        except Exception as e:
            raise ValueError(f"Error scaling data: {str(e)}")
        
        if method == 'linear':
            X_const = add_constant(X_train_scaled)
            model = OLS(Y_train, X_const).fit()
            fitted_values = model.predict(X_const)
            X_pred_const = add_constant(X_pred_scaled)
            predictions = model.predict(X_pred_const)
        
        elif method == 'polynomial':
            poly = PolynomialFeatures(degree=2)
            X_poly_train = poly.fit_transform(X_train_scaled)
            X_poly_pred = poly.transform(X_pred_scaled)
            model = OLS(Y_train, X_poly_train).fit()
            fitted_values = model.predict(X_poly_train)
            predictions = model.predict(X_poly_pred)
        
        elif method == 'lowess':
            lowess_fit = lowess(Y_train, X_train_scaled.ravel(), frac=0.3)
            fitted_values = np.interp(X_train_scaled.ravel(), lowess_fit[:, 0], lowess_fit[:, 1])
            predictions = np.interp(X_pred_scaled.ravel(), lowess_fit[:, 0], lowess_fit[:, 1])
        
        elif method == 'gaussian_process':
            kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
            model = GaussianProcessRegressor(kernel=kernel, random_state=42)
            model.fit(X_train_scaled, Y_train)
            fitted_values = model.predict(X_train_scaled)
            predictions = model.predict(X_pred_scaled)
        
        elif method == 'arima':
            # Determine ARIMA order based on data length
            if len(X_train) < 10:
                order = (1,0,0)
            else:
                order = (1,1,1)
            
            try:
                model = ARIMA(Y_train, order=order, exog=X_train_scaled)
                fitted_model = model.fit()
                fitted_values = fitted_model.fittedvalues
                predictions = fitted_model.forecast(steps=len(X_pred), exog=X_pred_scaled)
            except Exception as e:
                print(f"ARIMA failed, falling back to linear regression: {str(e)}")
                X_const = add_constant(X_train_scaled)
                model = OLS(Y_train, X_const).fit()
                fitted_values = model.predict(X_const)
                X_pred_const = add_constant(X_pred_scaled)
                predictions = model.predict(X_pred_const)
        
        return {
            'fitted_values': pd.Series(fitted_values, index=X_train.index),
            'prediction': pd.Series(predictions, index=X_pred.index)
        }

    def cyclicality_analysis(self, Y_detrended, X_train, X_pred):
        """
        Analyzes cyclical component using FFT.
        
        Args:
            Y_detrended (pd.Series): Detrended time series data
            X_train (pd.Series): Training years
            X_pred (pd.Series): Prediction years
        """
        # Convert to numpy array for FFT
        Y_values = Y_detrended.values.astype(float)
        
        # Perform FFT
        fft_result = fft(Y_values)
        freqs = fftfreq(len(Y_values))
        
        # Get dominant frequencies (excluding DC component)
        power_spectrum = np.abs(fft_result) ** 2
        
        # Limit frequency analysis to meaningful periods
        min_period = 2  # minimum 2-year cycle
        max_period = len(Y_values) // 2  # maximum half the length of data
        valid_freq_mask = (np.abs(freqs) >= 1/max_period) & (np.abs(freqs) <= 1/min_period)
        valid_freq_mask[0] = False  # exclude DC component
        
        # Find top frequencies
        n_freqs = min(3, sum(valid_freq_mask) // 2)  # up to 3 frequencies, but no more than half of valid ones
        sorted_freq_idx = np.argsort(power_spectrum * valid_freq_mask)[-n_freqs:]
        
        dom_freqs = freqs[sorted_freq_idx]
        dom_amplitudes = np.abs(fft_result[sorted_freq_idx]) / len(Y_values)
        dom_phases = np.angle(fft_result[sorted_freq_idx])
        
        # Reconstruct signal using dominant frequencies
        def reconstruct_signal(t):
            signal = np.zeros_like(t, dtype=float)
            for freq, amp, phase in zip(dom_freqs, dom_amplitudes, dom_phases):
                signal += 2 * amp * np.cos(2 * np.pi * freq * t + phase)
            return signal
        
        # Generate time points
        t_train = np.arange(len(Y_values))
        t_pred = np.arange(len(Y_values), len(Y_values) + len(X_pred))
        
        # Generate fitted values and predictions
        fitted_values = reconstruct_signal(t_train)
        predictions = reconstruct_signal(t_pred)
        
        return {
            'fitted_values': pd.Series(fitted_values, index=X_train.index),
            'prediction': pd.Series(predictions, index=X_pred.index)
        }

    def linear_regression(self, X, Y, do_cv=True, k_folds=5):
        """
        Performs linear regression with optional cross validation.
        """
        # Find optimal lag and align series
        best_lag, _, _ = self.max_lag(X, Y)
        X_aligned, Y_aligned = self.align_with_lag(X, Y, best_lag)
        
        # Fit model
        X_const = add_constant(X_aligned)
        model = OLS(Y_aligned, X_const).fit()
        
        # Make prediction
        current_X = X[2024 if 2024 in X.index else X.index.max()]
        # Use iloc to avoid the deprecation warning
        next_year_pred = model.params.iloc[0] + model.params.iloc[1] * current_X
        
        # Create plot data
        plot_data = pd.DataFrame({
            'X': X_aligned,
            'Y_data': Y_aligned,
            'Y_pred': model.fittedvalues
        })
        
        # Calculate standardized metrics
        metrics = self._calculate_metrics(Y_aligned, model.fittedvalues, n_params=2)
        
        results = {
            "lag": best_lag,
            "prediction": next_year_pred,
            "r2": metrics["r2"],
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "aic": metrics["aic"],
            "plot_data": plot_data
        }
        
        if do_cv:
            def ols_model(X, Y):
                # Ensure proper dimensions for training data
                X = X.reshape(-1)  # Flatten X
                X_const = add_constant(X)  # Add constant term
                model = OLS(Y, X_const).fit()
                
                # Return a prediction function that handles new data properly
                def predict(X_new):
                    X_new = X_new.reshape(-1)  # Flatten new X
                    X_new_const = add_constant(X_new)  # Add constant term
                    return model.predict(X_new_const)
                
                return predict
            
            cv_score, cv_error = self._do_cross_validation(
                X_aligned, 
                Y_aligned,
                ols_model,
                {},
                k_folds
            )
            results.update({
                'cv_score': cv_score,
                'cv_error': cv_error
            })
        
        return results

    def polynomial_regression(self, X, Y, degree=2, do_cv=True, k_folds=5):
        """
        Performs polynomial regression.
        Inputs:
        X       : yearly time series data 
        Y       : yearly time series data
        degree  : polynomial degree
        do_cv   : whether to perform cross validation
        k_folds : number of folds for CV
        
        Outputs:
        dict    : {
            'lag'        : optimal lag value,
            'prediction' : predicted value for next period,
            'r2'        : R-squared value,
            'rmse'      : root mean square error,
            'mae'       : mean absolute error,
            'aic'       : Akaike Information Criterion,
            'plot_data' : pandas DataFrame with X, Y_pred, Y_data columns,
            'cv_score'  : mean validation score (if do_cv=True),
            'cv_error'  : std of validation scores (if do_cv=True)
        }
        """
        # Find optimal lag and align series
        best_lag, _, _ = self.max_lag(X, Y)
        X_aligned, Y_aligned = self.align_with_lag(X, Y, best_lag)
        
        # Drop NaN values before fitting
        valid_mask = ~np.isnan(X_aligned) & ~np.isnan(Y_aligned)
        X_clean = X_aligned[valid_mask].values if hasattr(X_aligned, 'values') else X_aligned[valid_mask]
        Y_clean = Y_aligned[valid_mask].values if hasattr(Y_aligned, 'values') else Y_aligned[valid_mask]
        
        # Prepare polynomial features
        X_array = X_clean.reshape(-1, 1)
        poly = PolynomialFeatures(degree)
        X_poly = poly.fit_transform(X_array)
        
        # Fit model
        model = OLS(Y_clean, X_poly).fit()
        
        # Make prediction for next year
        current_X = X[2024 if 2024 in X.index else X.index.max()]
        if np.isnan(current_X):
            next_year_pred = np.nan
        else:
            next_year_pred = model.predict(poly.transform([[current_X]]))[0]
        
        # Get predictions for plotting
        X_plot = X_array
        Y_pred = model.predict(poly.transform(X_plot))
        
        # Create plot data
        plot_data = pd.DataFrame({
            'X': X_aligned[valid_mask],
            'Y_data': Y_aligned[valid_mask],
            'Y_pred': Y_pred
        })
        
        # Calculate metrics
        metrics = self._calculate_metrics(Y_clean, Y_pred, n_params=degree + 1)
        
        results = {
            "lag": best_lag,
            "prediction": next_year_pred,
            "r2": metrics["r2"],
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "aic": metrics["aic"],
            "plot_data": plot_data
        }
        
        if do_cv:
            def poly_model(X, Y):
                # Ensure X and Y are numpy arrays
                X = np.asarray(X).reshape(-1)
                Y = np.asarray(Y).reshape(-1)
                
                # Remove NaN values
                valid_mask = ~np.isnan(X) & ~np.isnan(Y)
                X_valid = X[valid_mask].reshape(-1, 1)
                Y_valid = Y[valid_mask]
                
                # Fit polynomial features
                poly_cv = PolynomialFeatures(degree)
                X_poly_cv = poly_cv.fit_transform(X_valid)
                model_cv = OLS(Y_valid, X_poly_cv).fit()
                
                def predict(X_new):
                    X_new = np.asarray(X_new).reshape(-1)
                    predictions = np.full(X_new.shape, np.nan)
                    valid_mask = ~np.isnan(X_new)
                    
                    if np.any(valid_mask):
                        X_valid = X_new[valid_mask].reshape(-1, 1)
                        X_poly_pred = poly_cv.transform(X_valid)
                        predictions[valid_mask] = model_cv.predict(X_poly_pred)
                    
                    return predictions
                
                return predict
            
            cv_score, cv_error = self._do_cross_validation(
                X_aligned,
                Y_aligned,
                poly_model,
                {},
                k_folds
            )
            results.update({
                'cv_score': cv_score,
                'cv_error': cv_error
            })
        
        return results
    
    def arima_regression(self, X, Y, max_p=3, max_d=2, max_q=3, do_cv=True, k_folds=5):
        """
        Performs ARIMA regression with simplified implementation.
        """
        # Find optimal lag and align series
        best_lag, _, _ = self.max_lag(X, Y)
        X_aligned, Y_aligned = self.align_with_lag(X, Y, best_lag)
        
        # Convert to numpy arrays if needed
        X_arr = X_aligned.values.reshape(-1, 1) if hasattr(X_aligned, 'values') else X_aligned.reshape(-1, 1)
        Y_arr = Y_aligned.values if hasattr(Y_aligned, 'values') else Y_aligned
        
        # Determine minimum differencing order
        d_min = 0 if adfuller(Y_arr)[1] < 0.05 else 1
        
        # Find best ARIMA model
        best_aic = np.inf
        best_model = None
        best_params = None
        
        # Grid search for parameters
        for p, d, q in itertools.product(
            range(max_p + 1),
            range(d_min, max_d + 1),
            range(max_q + 1)
        ):
            try:
                model = ARIMA(Y_arr, exog=X_arr, order=(p, d, q)).fit()
                if model.aic < best_aic:
                    best_aic = model.aic
                    best_model = model
                    best_params = (p, d, q)
            except:
                continue
                
        if best_model is None:
            raise ValueError("Could not fit any ARIMA model with given parameters")
        
        # Get fitted values
        Y_pred = best_model.fittedvalues
        
        # Make prediction for next period
        current_X = X_arr[-1]
        try:
            next_year_pred = best_model.forecast(steps=1, exog=np.array([[current_X]]))[0]
        except:
            next_year_pred = Y_pred[-1]  # Fallback to last fitted value
        
        # Create plot data
        plot_data = pd.DataFrame({
            'X': X_aligned,
            'Y_data': Y_aligned,
            'Y_pred': Y_pred
        })
        
        # Calculate metrics
        n_params = sum(best_params) + 1
        metrics = self._calculate_metrics(Y_aligned, Y_pred, n_params=n_params)
        
        results = {
            "lag": best_lag,
            "prediction": float(next_year_pred),
            "r2": metrics["r2"],
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "aic": metrics["aic"],
            "params": best_params,
            "plot_data": plot_data
        }
        
        if do_cv:
            def arima_model(X, Y):
                # Fit ARIMA model with best parameters
                X = X.reshape(-1, 1)
                model = ARIMA(Y, exog=X, order=best_params).fit()
                
                # Return simple prediction function
                return lambda x: model.forecast(steps=len(x), exog=x.reshape(-1, 1))
            
            cv_score, cv_error = self._do_cross_validation(
                X_aligned,
                Y_aligned,
                arima_model,
                {},
                k_folds
            )
            results.update({
                'cv_score': cv_score,
                'cv_error': cv_error
            })
        
        return results

    def lowess_regression(self, X, Y, frac=0.3, do_cv=True, k_folds=5):
        """
        Performs LOWESS regression with proper vector handling.
        """
        # Find optimal lag and align series
        best_lag, _, _ = self.max_lag(X, Y)
        X_aligned, Y_aligned = self.align_with_lag(X, Y, best_lag)
        
        # Convert to numpy arrays if needed
        X_arr = X_aligned.values if hasattr(X_aligned, 'values') else X_aligned
        Y_arr = Y_aligned.values if hasattr(Y_aligned, 'values') else Y_aligned
        
        # Fit LOWESS model
        smoothed = lowess(
            Y_arr,
            X_arr,
            frac=frac,
            return_sorted=True
        )
        
        # Make prediction
        if isinstance(X, pd.Series):
            current_X = X[2024 if 2024 in X.index else X.index.max()]
        else:
            current_X = X[-1]
        next_year_pred = np.interp(current_X, smoothed[:, 0], smoothed[:, 1])
        
        # Create plot data and interpolate fitted values
        X_sorted_idx = np.argsort(X_arr)
        Y_pred = np.interp(X_arr, smoothed[:, 0], smoothed[:, 1])
        
        plot_data = pd.DataFrame({
            'X': X_aligned,
            'Y_data': Y_aligned,
            'Y_pred': Y_pred
        })
        
        # Calculate effective parameters (degrees of freedom)
        n = len(X_arr)
        effective_params = max(1, int(frac * n))
        
        # Calculate metrics
        metrics = self._calculate_metrics(Y_aligned, Y_pred, n_params=effective_params)
        
        results = {
            "lag": best_lag,
            "prediction": float(next_year_pred),
            "r2": metrics["r2"],
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "aic": self._calculate_non_parametric_aic(Y_aligned, Y_pred, effective_params),
            "plot_data": plot_data
        }
        
        if do_cv:
            def lowess_model(X, Y):
                # Ensure X and Y are 1D arrays
                X = X.ravel()
                Y = Y.ravel()
                
                # Fit LOWESS on training data
                smoothed_train = lowess(Y, X, frac=frac, return_sorted=True)
                
                # Return prediction function that handles vectors properly
                def predict(X_new):
                    X_new = X_new.ravel()  # Ensure input is 1D
                    return np.interp(X_new, smoothed_train[:, 0], smoothed_train[:, 1])
                
                return predict
            
            cv_score, cv_error = self._do_cross_validation(
                X_aligned,
                Y_aligned,
                lowess_model,
                {},
                k_folds
            )
            results.update({
                'cv_score': cv_score,
                'cv_error': cv_error
            })
        
        return results

    def gaussian_process_regression(self, X, Y, length_scale=1.0, do_cv=True, k_folds=5):
        """
        Performs Gaussian Process Regression.
        Inputs:
        X            : yearly time series data
        Y            : yearly time series data
        length_scale : RBF kernel length scale parameter
        do_cv        : whether to perform cross validation
        k_folds      : number of folds for CV
        
        Outputs:
        dict    : {
            'lag'        : optimal lag value,
            'prediction' : predicted value for next period,
            'r2'        : R-squared value,
            'rmse'      : root mean square error,
            'mae'       : mean absolute error,
            'aic'       : AIC value,
            'std'       : prediction standard deviation,
            'plot_data' : pandas DataFrame with X, Y_pred, Y_data columns,
            'cv_score'  : mean validation score (if do_cv=True),
            'cv_error'  : std of validation scores (if do_cv=True)
        }
        """
        # Find optimal lag and align series
        best_lag, _, _ = self.max_lag(X, Y)
        X_aligned, Y_aligned = self.align_with_lag(X, Y, best_lag)
        
        # Drop NaN values before fitting
        valid_mask = ~np.isnan(X_aligned) & ~np.isnan(Y_aligned)
        X_clean = X_aligned[valid_mask].values if hasattr(X_aligned, 'values') else X_aligned[valid_mask]
        Y_clean = Y_aligned[valid_mask].values if hasattr(Y_aligned, 'values') else Y_aligned[valid_mask]
        
        # Reshape data
        X_fit = X_clean.reshape(-1, 1)
        Y_fit = Y_clean.reshape(-1)
        
        # Scale the data
        def robust_scale(data):
            median = np.median(data)
            iqr = np.percentile(data, 75) - np.percentile(data, 25)
            if iqr == 0:
                iqr = np.std(data)
            if iqr == 0:
                iqr = 1.0
            rescaled = (data - median) / (iqr + 1e-8)
            return rescaled, median, iqr
        
        X_scaled, X_median, X_iqr = robust_scale(X_fit)
        Y_scaled, Y_median, Y_iqr = robust_scale(Y_fit)
        
        # Define kernel and fit model
        kernel = RBF(length_scale=length_scale) + WhiteKernel(noise_level=0.1)
        gpr = GaussianProcessRegressor(
            kernel=kernel,
            random_state=42,
            n_restarts_optimizer=5,
            normalize_y=False
        )
        gpr.fit(X_scaled, Y_scaled)
        
        def predict_scaled(X_new, scaler_params):
            X_new = np.asarray(X_new).reshape(-1)
            if np.all(np.isnan(X_new)):
                return np.array([np.nan]), np.array([np.nan])
                
            X_median, X_iqr = scaler_params['X']
            Y_median, Y_iqr = scaler_params['Y']
            
            X_valid = ~np.isnan(X_new)
            X_new_valid = X_new[X_valid]
            X_scaled_new = (X_new_valid - X_median) / (X_iqr + 1e-8)
            
            Y_scaled_pred, Y_scaled_std = gpr.predict(
                X_scaled_new.reshape(-1, 1), 
                return_std=True
            )
            
            full_pred = np.full(X_new.shape, np.nan)
            full_std = np.full(X_new.shape, np.nan)
            
            full_pred[X_valid] = Y_scaled_pred * (Y_iqr + 1e-8) + Y_median
            full_std[X_valid] = Y_scaled_std * (Y_iqr + 1e-8)
            
            return full_pred, full_std
        
        # Make predictions
        scaler_params = {'X': (X_median, X_iqr), 'Y': (Y_median, Y_iqr)}
        current_X = X[2024 if 2024 in X.index else X.index.max()]
        next_year_pred, next_year_std = predict_scaled(np.array([current_X]), scaler_params)
        Y_pred, Y_std = predict_scaled(X_aligned.values, scaler_params)
        
        # Create plot data
        plot_data = pd.DataFrame({
            'X': X_aligned,
            'Y_data': Y_aligned,
            'Y_pred': Y_pred,
            'Y_std': Y_std
        })
        
        # Calculate metrics using only non-NaN values
        valid_metrics = ~np.isnan(Y_pred) & ~np.isnan(Y_aligned)
        metrics = self._calculate_metrics(
            Y_aligned[valid_metrics], 
            Y_pred[valid_metrics], 
            n_params=len(gpr.kernel_.theta)
        )
        
        results = {
            "lag": best_lag,
            "prediction": float(next_year_pred[0]),
            "r2": metrics["r2"],
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "aic": metrics["aic"],
            "std": float(next_year_std[0]),
            "plot_data": plot_data
        }
        
        if do_cv:
            def cv_gpr_model(X, Y):
                # Ensure X and Y are numpy arrays
                X = np.asarray(X).reshape(-1)
                Y = np.asarray(Y).reshape(-1)
                
                # Remove NaN values
                valid_mask = ~np.isnan(X) & ~np.isnan(Y)
                X_valid = X[valid_mask].reshape(-1, 1)
                Y_valid = Y[valid_mask]
                
                # Scale the data
                X_scaled, X_median, X_iqr = robust_scale(X_valid)
                Y_scaled, Y_median, Y_iqr = robust_scale(Y_valid)
                
                cv_gpr = GaussianProcessRegressor(
                    kernel=kernel.clone_with_theta(kernel.theta),
                    random_state=42,
                    normalize_y=False
                )
                cv_gpr.fit(X_scaled.reshape(-1, 1), Y_scaled)
                
                def predict(X_new):
                    X_new = np.asarray(X_new).reshape(-1)
                    predictions = np.full(X_new.shape, np.nan)
                    valid_mask = ~np.isnan(X_new)
                    
                    if np.any(valid_mask):
                        X_valid = X_new[valid_mask]
                        X_scaled = (X_valid - X_median) / (X_iqr + 1e-8)
                        Y_scaled_pred = cv_gpr.predict(X_scaled.reshape(-1, 1))
                        predictions[valid_mask] = Y_scaled_pred * (Y_iqr + 1e-8) + Y_median
                    
                    return predictions
                
                return predict
            
            cv_score, cv_error = self._do_cross_validation(
                X_aligned,
                Y_aligned,
                cv_gpr_model,
                {},
                k_folds
            )
            results.update({
                'cv_score': cv_score,
                'cv_error': cv_error
            })
        
        return results