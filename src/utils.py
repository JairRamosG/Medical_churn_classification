"""Módulo de utilidades para la ingeniería de características y el encoding de variables"""

from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineering(BaseEstimator, TransformerMixin):
    '''Crea nuevas características a partir de las existentes del dataset'''
    def __init__(self):
        pass

    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        X['frecuencia_visitas'] = X['Visits_Last_Year'] / 12
        
        visits_no_zero = X['Visits_Last_Year'].replace(0, 1)
        X['ratio_citas'] = X['Missed_Appointments'] / visits_no_zero
        
        X['costoxvisita'] = X['Avg_Out_Of_Pocket_Cost'] / visits_no_zero
        
        return X
    
class SafeFrequencyEncoder(BaseEstimator, TransformerMixin):
    '''Encoding de frecuencia para variables categóricas'''
    def __init__(self):
        self.freq_maps_ = {}
    
    def fit(self, X, y=None):
        for col in X.columns:
            self.freq_maps_[col] = X[col].value_counts(normalize=True).to_dict()
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        for col in X.columns:
            X_copy[col] = X[col].map(self.freq_maps_.get(col, {})).fillna(0)
        return X_copy