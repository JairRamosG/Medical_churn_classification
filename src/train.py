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

from utils import FeatureEngineering, SafeFrequencyEncoder

import argparse

import sys


def train_model(config_path):
    """
    Función principal que entrena el modelo.

    Args:
        config_path (srt): Ruta de un archivo de configuración .yaml
    """
    # CARGAR EL ARCHIVO DE CONFIGURACIÓN
    with open(config_path) as f:
        config = yaml.safe_load(f)

    columnas_config = config.get('columnas', {})
    
    # Listas de columnas desde el config
    IGNORAR_COLS = columnas_config.get('ignorar', [])
    NUMERICAS_DEFINIDAS = columnas_config.get('numericas', [])
    ORDINALES_DEFINIDAS = columnas_config.get('ordinales', [])
    NOM_OHE_DROP_DEFINIDAS = columnas_config.get('nominales_ohe_drop', [])
    NOM_OHE_DEFINIDAS = columnas_config.get('nominales_ohe', [])
    NOM_FREQ_DEFINIDAS = columnas_config.get('nominales_frecuencia', [])
    
    # Raiz
    BASE_DIR = Path(__file__).resolve().parent.parent
    # Rutas archivo datos, rutas modelo , emtadata y logs.
    DATA_FILE = Path(os.getenv("DATA_FILE", BASE_DIR / "data" / "raw" / "patient_churn_dataset.csv")) 
    MODEL_DIR = Path(os.getenv("MODEL_DIR", BASE_DIR / "models"))
    METADATA_DIR = Path(os.getenv("METADATA_DIR", BASE_DIR / "metadata"))
    LOGS_DIR = Path(os.getenv("LOGS_DIR", BASE_DIR / "logs"))

    sys.path.insert(0, str(BASE_DIR / "src"))


    MODEL_DIR.mkdir(parents=True, exist_ok= True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok= True)

    # CONFIGURAR EL ARCHIVO PARA LOS LOGGINS
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(LOGS_DIR / config['log_file'])
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

    # ESTABLECERLE UNAS SEMILLAS
    seed = config['seed']
    random.seed(seed)
    np.random.seed(seed)

    # CARGAR LOS DATOS PARA TRABAJAR
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
    required_cols = (
        [config['target_variable']] + 
        IGNORAR_COLS + 
        NUMERICAS_DEFINIDAS + 
        ORDINALES_DEFINIDAS + 
        NOM_OHE_DROP_DEFINIDAS + 
        NOM_OHE_DEFINIDAS + 
        NOM_FREQ_DEFINIDAS
    )
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        logger.error(f"Faltan columnas requeridas en el proceso: {missing_cols}")
        raise ValueError(f"Faltan columnas: {missing_cols}")

    logger.info(f"Todas las columnas requeridas están presentes")

    # SEPARAR EL TARGET Y CREAR LOS SPLITS
    target_variable = config['target_variable']
    X = data.drop(columns = [target_variable])
    y = data[target_variable]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config.get('test_size', 0.2),
        random_state=seed,
        stratify=y)
    logger.info("Datos divididos en train/test")

    # IDENTIFICACIÓN DE COLUMNAS NUMÉRICAS, ORDINALES Y NOMINALES
    numerical_cols = NUMERICAS_DEFINIDAS.copy()  
    
    # 2. ORDINALES: Usamos las definidas en el config
    ordinal_cols = ORDINALES_DEFINIDAS.copy()
    
    # 3. NOMINALES: Tenemos 3 grupos definidos en el config
    nominal_ohe_drop_cols = NOM_OHE_DROP_DEFINIDAS.copy()
    nominal_ohe_cols = NOM_OHE_DEFINIDAS.copy()
    freq_cols = NOM_FREQ_DEFINIDAS.copy()

    # IGNORADAS
    ignored_cols = IGNORAR_COLS.copy()
    
    logger.info("Columnas desde el archivo de configuración:")
    logger.info(f"Numéricas: {len(numerical_cols)}")
    logger.info(f"Ordinales: {len(ordinal_cols)}")
    logger.info(f"Nominales OHE: {len(nominal_ohe_cols)}")
    logger.info(f"Nominales OHE DROP: {len(nominal_ohe_drop_cols)}")
    logger.info(f"Nominales Frecuency: {len(freq_cols)}")
    logger.info(f"Ignoradas: {len(ignored_cols)}")
    
    # CONSTRUIR EL PREPROCESSOR CON COLUMNTRANSFORMER
    pipeline = build_full_pipeline(numerical_cols, ordinal_cols, nominal_ohe_drop_cols, nominal_ohe_cols, freq_cols)
    logger.info('sigue parametrizar una función para el pipeline')

    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None)
    args = parser.parse_args()
    
    BASE_DIR = Path(__file__).resolve().parent.parent
    config_path = Path(args.config) if args.config else BASE_DIR / "config" / "model_config.yaml"
    train_model(str(config_path))