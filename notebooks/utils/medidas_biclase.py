from sklearn.metrics import (accuracy_score, recall_score, balanced_accuracy_score, precision_score, f1_score, matthews_corrcoef)
import pandas as pd
from sklearn.metrics import confusion_matrix

def calcula_medidas_biclase(y_test, y_pred):
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

    return pd.DataFrame(list(metrics.items()), columns = ['Medida', 'Valor'])