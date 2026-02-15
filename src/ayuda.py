# src/train.py
import yaml
from pathlib import Path
import os
import logging
import pandas as pd
import numpy as np
import random
import sys
import joblib
import json
import argparse

from sklearn.model_selection import train_test_split

# Importar las funciones de pipelines.py
from pipelines import build_full_pipeline

def train_model(config_path):
    """
    Función principal que entrena el modelo usando la configuración.
    """
    # 1. CARGAR CONFIGURACIÓN
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # 2. CONFIGURAR DIRECTORIOS
    BASE_DIR = Path(__file__).resolve().parent.parent
    
    # Rutas desde config o variables de entorno
    DATA_FILE = Path(os.getenv("DATA_FILE", BASE_DIR / config['paths']['data_file']))
    MODEL_DIR = Path(os.getenv("MODEL_DIR", BASE_DIR / config['paths']['models_dir']))
    METADATA_DIR = Path(os.getenv("METADATA_DIR", BASE_DIR / config['paths']['metadata_dir']))
    LOGS_DIR = Path(os.getenv("LOGS_DIR", BASE_DIR / config['paths']['logs_dir']))
    
    # Crear directorios si no existen
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 3. CONFIGURAR LOGGING
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    log_file = LOGS_DIR / config['logging']['file']
    handler = logging.FileHandler(log_file)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    
    # También agregar handler para consola
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)
    
    # 4. ESTABLECER SEMILLAS
    seed = config['seed']
    random.seed(seed)
    np.random.seed(seed)
    
    logger.info("="*50)
    logger.info("INICIANDO ENTRENAMIENTO DEL MODELO")
    logger.info(f"Configuración: {config_path}")
    logger.info(f"Semilla: {seed}")
    
    # 5. CARGAR DATOS
    logger.info(f"Cargando datos desde: {DATA_FILE}")
    if not DATA_FILE.exists():
        logger.error(f"Archivo no encontrado: {DATA_FILE}")
        raise FileNotFoundError(f'No existe: {DATA_FILE}')
    
    data = pd.read_csv(DATA_FILE)
    logger.info(f"Datos cargados: {data.shape[0]} filas, {data.shape[1]} columnas")
    
    if data.empty:
        logger.error("Dataset vacío")
        raise ValueError("Dataset vacío")
    
    # 6. VALIDAR COLUMNAS
    target_variable = config['target_variable']
    columnas_config = config.get('columnas', {})
    
    # Reunir todas las columnas requeridas
    required_cols = [target_variable]
    for col_list in columnas_config.values():
        if isinstance(col_list, list):
            required_cols.extend(col_list)
    
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        logger.error(f"Faltan columnas: {missing_cols}")
        raise ValueError(f"Faltan columnas: {missing_cols}")
    
    logger.info("Todas las columnas requeridas están presentes")
    
    # 7. SEPARAR TARGET Y CREAR SPLITS
    X = data.drop(columns=[target_variable])
    y = data[target_variable]
    
    split_config = config.get('data_split', {})
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=split_config.get('test_size', 0.2),
        random_state=seed,
        stratify=y if split_config.get('stratify', True) else None
    )
    
    logger.info(f"Train set: {X_train.shape[0]} muestras")
    logger.info(f"Test set: {X_test.shape[0]} muestras")
    
    # 8. CONSTRUIR PIPELINE COMPLETO (AQUÍ SE USA pipelines.py)
    logger.info("Construyendo pipeline con configuración...")
    try:
        pipeline = build_full_pipeline(config,  )
        logger.info("Pipeline construido exitosamente")
    except Exception as e:
        logger.error(f"Error construyendo pipeline: {str(e)}")
        raise
    
    # 9. ENTRENAR MODELO
    logger.info("Iniciando entrenamiento...")
    try:
        pipeline.fit(X_train, y_train)
        logger.info("Entrenamiento completado")
    except Exception as e:
        logger.error(f"Error durante entrenamiento: {str(e)}")
        raise
    
    # 10. EVALUAR
    logger.info("Evaluando modelo...")
    train_score = pipeline.score(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)
    
    logger.info(f"Train accuracy: {train_score:.4f}")
    logger.info(f"Test accuracy: {test_score:.4f}")
    
    # 11. GUARDAR MODELO Y MÉTRICAS
    model_filename = config['paths']['model_filename'].format(seed=seed, test_score=f"{test_score:.4f}")
    model_path = MODEL_DIR / model_filename
    
    logger.info(f"Guardando modelo en: {model_path}")
    joblib.dump(pipeline, model_path)
    
    # Guardar métricas
    metrics = {
        'train_accuracy': train_score,
        'test_accuracy': test_score,
        'seed': seed,
        'config_used': config_path,
        'model_file': str(model_path)
    }
    
    metrics_path = METADATA_DIR / f"metrics_{seed}.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Métricas guardadas en: {metrics_path}")
    logger.info("="*50)
    logger.info("ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
    
    return pipeline, metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None, help='Ruta al archivo de configuración YAML')
    args = parser.parse_args()
    
    BASE_DIR = Path(__file__).resolve().parent.parent
    config_path = Path(args.config) if args.config else BASE_DIR / "config" / "model_config.yaml"
    
    train_model(str(config_path))