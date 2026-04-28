"""
clp_transformers.py — Custom sklearn transformers for CLP/USD model.

# The pkl was re-saved from the notebook with __module__='clp_transformers'.
# app_CLPUSD_logret.py injects proxy modules so this file covers any residual
# module name the pkl might reference (__main__, transformers, clp_transformers).
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class LogReturnTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X_arr = X.values.astype(float) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            log_X = np.log(X_arr)
        result = np.full_like(log_X, np.nan)
        result[1:] = log_X[1:] - log_X[:-1]
        return result
    def get_feature_names_out(self, feature_names_in=None):
        return None if feature_names_in is None else np.array([f'logret_{n}' for n in feature_names_in])


class MonthlyDiffTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X_arr = X.values.astype(float) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
        result = np.full_like(X_arr, np.nan)
        result[1:] = X_arr[1:] - X_arr[:-1]
        return result
    def get_feature_names_out(self, feature_names_in=None):
        return None if feature_names_in is None else np.array([f'diff_{n}' for n in feature_names_in])


class ForwardFillTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        return X.ffill().bfill() if isinstance(X, pd.DataFrame) else pd.DataFrame(X).ffill().bfill().values


class SimpleImputerModel(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.means_ = X.mean() if isinstance(X, pd.DataFrame) else np.mean(X, axis=0)
        return self
    def transform(self, X):
        return X.fillna(self.means_) if isinstance(X, pd.DataFrame) else np.where(np.isnan(X), self.means_, X)


class TargetPreprocessorLogReturn(BaseEstimator, TransformerMixin):
    def fit(self, y, X=None): return self
    def transform(self, y, **kwargs):
        y_clean = y.ffill().bfill() if isinstance(y, pd.Series) else pd.Series(y).ffill().bfill()
        log_y = np.log(y_clean.values.astype(float))
        result = np.full_like(log_y, np.nan)
        result[1:] = log_y[1:] - log_y[:-1]
        return result.ravel()
    def fit_transform(self, y, X=None):
        return self.fit(y, X).transform(y)
    def inverse_transform(self, logret_preds, y_prev):
        return np.array(y_prev).ravel() * np.exp(np.array(logret_preds).ravel())