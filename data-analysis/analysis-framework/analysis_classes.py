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


# Suppress specific warnings (optional)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)

class TimeSeriesRegression:
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
        Aligns two time series based on the first valid index.
        Inputs:
        X           : yearly time series data
        Y           : yearly time series data
        Outputs:
        tuple       : (X_aligned, Y_aligned)
        X_aligned   : time series data aligned based on first valid index
        Y_aligned   : time series data aligned based on first valid index
        """
        if not isinstance(X, pd.Series):
            X = pd.Series(X)
        if not isinstance(Y, pd.Series):
            Y = pd.Series(Y)
            
        first_index = max(X.first_valid_index(), Y.first_valid_index())
        last_index = min(X.last_valid_index(), Y.last_valid_index())
        
        # Add check for valid index range
        if first_index >= last_index:
            raise ValueError("Insufficient overlapping data between X and Y series")
        
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
        if not isinstance(X, pd.Series):
            X = pd.Series(X)
        if not isinstance(Y, pd.Series):
            Y = pd.Series(Y)
        
        y_start = Y.first_valid_index()
        y_end = Y.last_valid_index()
        
        if y_start is None or y_end is None:
            raise ValueError("Y series contains no valid data")
        
        x_start = y_start - lag
        x_end = y_end - lag
        
        if x_start < X.first_valid_index() or x_end > X.last_valid_index():
            raise ValueError(f"Insufficient X data for lag {lag}")
        
        X_lagged = X.loc[x_start:x_end]
        Y_aligned = Y.loc[y_start:y_end]
        
        # Reset indices while preserving data integrity
        X_aligned = X_lagged.reset_index(drop=True)
        Y_aligned = Y_aligned.reset_index(drop=True)
        
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
        tuple           : (optimal_lag, max_correlation, metrics)
        optimal_lag     : maximizes the correlation of the 2 series
        max_correlation : the value of the correlation at optimal lag
        metrics        : dictionary of evaluation metrics at optimal lag
        """
        if max_lag_years < 0:
            raise ValueError("max_lag_years must be non-negative")
        
        # Initial data cleaning
        X_clean, Y_clean = self.clean_series(X, Y)
        
        # Convert to numpy arrays for efficient computation
        X_arr = np.array(X_clean)
        Y_arr = np.array(Y_clean)
        
        # Standardize series
        X_norm = (X_arr - np.mean(X_arr)) / (np.std(X_arr) + 1e-10)
        Y_norm = (Y_arr - np.mean(Y_arr)) / (np.std(Y_arr) + 1e-10)
        
        # Calculate cross-correlation
        correlations = signal.correlate(Y_norm, X_norm, mode='full')
        lags = signal.correlation_lags(len(X_norm), len(Y_norm))
        
        # Filter valid lags
        valid_indices = (lags >= 0) & (lags <= max_lag_years)
        valid_correlations = correlations[valid_indices]
        valid_lags = lags[valid_indices]
        
        if len(valid_correlations) == 0:
            raise ValueError("No valid lags found within specified range")
        
        # Find optimal lag
        max_corr_idx = np.argmax(valid_correlations)
        optimal_lag = valid_lags[max_corr_idx]
        max_correlation = valid_correlations[max_corr_idx] / len(X_arr)
        
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
        Performs polynomial regression with proper cross-validation handling.
        """
        # Find optimal lag and align series
        best_lag, _, _ = self.max_lag(X, Y)
        X_aligned, Y_aligned = self.align_with_lag(X, Y, best_lag)
        
        # Prepare polynomial features
        X_array = X_aligned.values.reshape(-1, 1) if hasattr(X_aligned, 'values') else X_aligned.reshape(-1, 1)
        poly = PolynomialFeatures(degree)
        X_poly = poly.fit_transform(X_array)
        
        # Fit model
        model = OLS(Y_aligned, X_poly).fit()
        
        # Make prediction
        current_X = X[2024 if 2024 in X.index else X.index.max()]
        next_year_pred = model.predict(poly.transform([[current_X]]))[0]
        
        # Create plot data
        plot_data = pd.DataFrame({
            'X': X_aligned,
            'Y_data': Y_aligned,
            'Y_pred': model.fittedvalues
        })
        
        # Calculate standardized metrics
        metrics = self._calculate_metrics(Y_aligned, model.fittedvalues, n_params=degree + 1)
        
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
                # Handle both pandas Series and numpy array inputs
                X = X.reshape(-1, 1)
                poly_cv = PolynomialFeatures(degree)
                X_poly_cv = poly_cv.fit_transform(X)
                model_cv = OLS(Y, X_poly_cv).fit()
                
                def predict(X_new):
                    X_new = X_new.reshape(-1, 1)
                    return model_cv.predict(poly_cv.transform(X_new))
                
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
        Performs Gaussian Process Regression with RBF kernel and improved stability.
        """
        # Find optimal lag and align series first
        best_lag, _, _ = self.max_lag(X, Y)
        X_aligned, Y_aligned = self.align_with_lag(X, Y, best_lag)
        
        # Input validation
        if len(X_aligned) < 3:
            raise ValueError("Need at least 3 data points for GPR")
        
        # Prepare data
        X_fit = X_aligned.values.reshape(-1, 1)
        Y_fit = Y_aligned.values.reshape(-1, 1)
        
        # Robust scaling function
        def robust_scale(data):
            median = np.median(data)
            iqr = np.percentile(data, 75) - np.percentile(data, 25)
            if iqr == 0:
                iqr = np.std(data)
            if iqr == 0:
                iqr = 1.0
            scaled = (data - median) / (iqr + 1e-8)
            return scaled, median, iqr
        
        # Scale the data
        X_scaled, X_median, X_iqr = robust_scale(X_fit)
        Y_scaled, Y_median, Y_iqr = robust_scale(Y_fit)
        
        # Optimize length scale based on data characteristics
        avg_X_dist = np.median(np.abs(np.diff(np.sort(X_scaled.ravel()))))
        length_scale = max(avg_X_dist, 1e-3)
        
        # Define kernel
        kernel = RBF(
            length_scale=length_scale,
            length_scale_bounds=(1e-3, 1e3)
        ) + WhiteKernel(
            noise_level=0.1,
            noise_level_bounds=(1e-5, 1.0)
        )
        
        # Initialize GPR
        gpr = GaussianProcessRegressor(
            kernel=kernel,
            random_state=42,
            n_restarts_optimizer=5,
            normalize_y=False,
            alpha=1e-10
        )
        
        # Fit model
        try:
            gpr.fit(X_scaled, Y_scaled.ravel())
        except Exception as e:
            print(f"Warning: Initial fit failed, trying with increased regularization: {str(e)}")
            gpr.alpha = 1e-6
            gpr.fit(X_scaled, Y_scaled.ravel())
        
        # Prediction function that handles scaling
        def predict_scaled(X_new, scaler_params):
            X_median, X_iqr = scaler_params['X']
            Y_median, Y_iqr = scaler_params['Y']
            
            X_scaled_new = (X_new - X_median) / (X_iqr + 1e-8)
            Y_scaled_pred, Y_scaled_std = gpr.predict(X_scaled_new.reshape(-1, 1), return_std=True)
            
            # Inverse transform predictions
            Y_pred = Y_scaled_pred * (Y_iqr + 1e-8) + Y_median
            Y_std = Y_scaled_std * (Y_iqr + 1e-8)
            
            return Y_pred, Y_std
        
        # Current prediction
        current_X = X[2024 if 2024 in X.index else X.index.max()]
        scaler_params = {
            'X': (X_median, X_iqr),
            'Y': (Y_median, Y_iqr)
        }
        next_year_pred, next_year_std = predict_scaled(np.array([[current_X]]), scaler_params)
        
        # Get predictions for all points
        Y_pred, Y_std = predict_scaled(X_fit, scaler_params)
        
        # Create plot data
        plot_data = pd.DataFrame({
            'X': X_aligned,
            'Y_data': Y_aligned,
            'Y_pred': Y_pred,
            'Y_std': Y_std
        })
        
        # Calculate metrics
        metrics = self._calculate_metrics(Y_aligned, Y_pred, n_params=len(gpr.kernel_.theta))
        
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
                # Ensure proper dimensions and scale data
                X = X.reshape(-1, 1)
                Y = Y.reshape(-1, 1)
                
                # Scale training data
                X_scaled, X_median, X_iqr = robust_scale(X)
                Y_scaled, Y_median, Y_iqr = robust_scale(Y)
                
                # Create new GPR instance
                cv_gpr = GaussianProcessRegressor(
                    kernel=kernel.clone_with_theta(kernel.theta),
                    random_state=42,
                    n_restarts_optimizer=5,
                    normalize_y=False,
                    alpha=1e-10
                )
                
                # Fit model
                try:
                    cv_gpr.fit(X_scaled, Y_scaled.ravel())
                except Exception:
                    cv_gpr.alpha = 1e-6
                    cv_gpr.fit(X_scaled, Y_scaled.ravel())
                
                # Return prediction function
                def predict(X_new):
                    X_new = X_new.reshape(-1, 1)
                    X_scaled_new = (X_new - X_median) / (X_iqr + 1e-8)
                    Y_scaled_pred = cv_gpr.predict(X_scaled_new)
                    return Y_scaled_pred * (Y_iqr + 1e-8) + Y_median
                
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