import streamlit as st
import panda as pd
import pickle


def cargar_modelo(file_name):
    '''
    Docstring para cargar modelo desde un archivo .pkl
    Args: 
        file_name (str): nombre del modelo .pkl
    '''

    with open(str(file_name), 'rb') as file:
        modelo = pickle.load(file)

file_name = 'EXP_01.pkl'
modelo = cargar_modelo(file_name)

def predecir_churn(modelo, datos, umbral = 0.5):
    '''
    Realiza la predicción del churn de un cliente usando el modelo pkl cargado
    Args:
        modelo (pkl): pesos del modelo entrenado
        datos (dict): datos usados para obtener predicción
        umbral (float): valor para umbral en la toma de desición de la clasificación 
    '''

    df_cliente = pd.DataFrame(datos)
    # Predicción Churn
    pred = modelo.predict(df_cliente)
    # Probabilidad clase positiva
    pred_prob_si = modelo.predict_proba(df_cliente)[::, 1]
    # Probabilidad ambas clases
    pred_prob_base = modelo.predict_proba(df_cliente)

    # Volver a filtrar el resultado a traves del umbral
    if pred_prob_si >= umbral:
        pred = 1
        probabilidad = pred_prob_si[0]
    else:
        pred = 0
        probabilidad = 1 - pred_prob_si[0]
    
    return pred, probabilidad, pred_prob_base



