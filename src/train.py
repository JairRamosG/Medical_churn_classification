import yaml
from pathlib import Path
import os
import logging
import pandas as pd
import numpy as np
import random

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

import joblib
import json

from src.utils import FeatureEngineering, SafeFrequencyEncoder

import argparse

def train_model(config_path):
    """
    Función principal que entrena el modelo.

    Args:
        config_path (srt): Ruta de un archivo de configuración .yaml
    """
    # Cargar el archivo de configuración
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Raiz
    BASE_DIR = Path(__file__).resolve().parent.parent
    # Rutas archivo datos, rutas modelo , emtadata y logs.
    DATA_FILE = Path(os.getenv("DATA_FILE", BASE_DIR / "data" / "raw" / "patient_churn_dataset.csv")) 
    MODEL_DIR = Path(os.getenv("MODEL_DIR", BASE_DIR / "models"))
    METADATA_DIR = Path(os.getenv("METADATA_DIR", BASE_DIR / "metadata"))
    LOGS_DIR = Path(os.getenv("LOGS_DIR", BASE_DIR / "logs"))


    MODEL_DIR.mkdir(parents=True, exist_ok= True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok= True)

    # Configurar el archivo para loggins
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.info)

    handler = logging.FileHandler(LOGS_DIR / config['log_file']) ## usar un nombre desde yaml
    handler.setFormatter('%(asctime)s - %(levelname)s - %(message)s')
    logger.addHandler(handler)

    # Semillas
    seed = config['seed']
    random.seed(seed)
    np.random.seed(seed)

    # Cargar los datos
    logger.info("Iniciando el entrenamiento del modelo")
    if not DATA_FILE.exists():
        logger.error(f"Archivo de datos no encontrado en: {DATA_FILE}")
        raise FileNotFoundError(f'No existe: {DATA_FILE}')
    data = pd.read_csv(DATA_FILE)
    logger.info(f"Datos cargados correctamente desde {DATA_FILE}")

    # Validación que no esté vacio
    if data.empty:
        logger.warning(f"El archivo de datos está vacío")
        raise ValueError(f"Dataset vacío")

    # Validar que están todas las columnas esperadas
    required_cols = [
        'Churned', 
        'PatientID', 'Age', 'Gender', 'State', 'Specialty', 'Insurance_Type',
        'Visits_Last_Year', 'Missed_Appointments', 'Avg_Out_Of_Pocket_Cost',
        'Referrals_Made', 'Billing_Issues', 'Portal_Usage', 'Last_Interaction_Date'
    ]
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        logger.error(f"Faltan columnas requeridas en el proceso: {missing_cols}")
        raise ValueError(f"Faltan columnas: {missing_cols}")

    logger.info(f"Todas las columnas requeridas están presentes")

    # ------------------------------------------------------------------------------------
    target_variable = config['target_variable']
    X = data.drop(columns = [target_variable])
    y = data[target_variable]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

    real_ordinal_cols = ['Referrals_Made']
    real_nominal_cols = ['Billing_Issues', 'Portal_Usage'] 

    # Numericas
    numerical_cols = data.select_dtypes(include = ['int64', 'float64']).columns.tolist()
    numerical_cols = [col for col in numerical_cols if col not in real_nominal_cols and col != 'PatientID']  

    # Categoricas Ordinales 
    ordinal_cols = real_ordinal_cols

    # Categoricas Nominales
    nominal_cols = [col for col in data.columns if col not in numerical_cols and col not in ordinal_cols and col != 'PatientID' and col != 'Last_Interaction_Date']




    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None)
    args = parser.parse_args()
    
    BASE_DIR = Path(__file__).resolve().parent.parent
    config_path = Path(args.config) if args.config else BASE_DIR / "config" / "model_config.yaml"
    train_model(str(config_path))