"""
Módulo de:
    Utilidades para la ingeniería de características y el encoding de variables
    Guardar matriz de confusión
    Guardad medidas de desempeño
"""

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import (accuracy_score, recall_score, balanced_accuracy_score, precision_score, f1_score, matthews_corrcoef)

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

def save_confusion_matrix(y_test, y_pred, ruta_img):
    """
    Genera y muestra una matriz de confusión personalizada
    con la disposición [[TP, FN], [FP, TN]].
    """

    cm = confusion_matrix(y_test, y_pred)

    TN, FP = cm[0, 0], cm[0, 1]
    FN, TP = cm[1, 0], cm[1, 1]

    cm_custom = np.array([
        [TP, FN],
        [FP, TN]
    ])

    cm_df = pd.DataFrame(
        cm_custom,
        index=['Pred Positivo', 'Pred Negativo'],
        columns=['Real Positivo', 'Real Negativo']
    )

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_custom,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Real +', 'Real -'],
        yticklabels=['Pred +', 'Pred -']
    )

    plt.xlabel('Clase real')
    plt.ylabel('Clase predicha')
    plt.title('Matriz de Confusión (TP FN / FP TN)')
    plt.tight_layout()
    plt.savefig(ruta_img)

def save_medidas_biclase(y_test, y_pred, ruta_resultados):
    '''
    Calculas las medidas de desempeño solicitadas en las instrucciones
    regresa una tabla con las medidas
    '''

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0 

    metrics = {

        'Accuracy': accuracy_score(y_test, y_pred),
        'Error Rate': 1 - accuracy_score(y_test, y_pred),
        'Recall (Sensitivity)': recall_score(y_test, y_pred),
        'Specificity': specificity,
        'Balanced Accuracy': balanced_accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'MCC': matthews_corrcoef(y_test, y_pred)
    }

    resultados =  pd.DataFrame(list(metrics.items()), columns = ['Medida', 'Valor'])
    resultados.to_csv(ruta_resultados)