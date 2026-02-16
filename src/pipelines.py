from imblearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from utils import FeatureEngineering, SafeFrequencyEncoder

def build_preprocessor(columnas_config, preprocessor_config):
    """
    Construye el ColumnTransformer con con el archivo de configuración
    Args:
        columnas_config (dict): Diccionario de la lista de columnas
        preprocessor_config (dict): Configuración del preprocessador
    """
    
    # Extraer las columnas
    ordinal_cols = columnas_config.get('ordinales', [])
    nominal_ohe_drop_cols = columnas_config.get('nominales_ohe_drop', [])
    nominal_ohe_cols = columnas_config.get('nominales_ohe', [])
    freq_cols = columnas_config.get('nominales_frecuencia', [])

    ## Pipeline para datos ordinales
    ordinal_categories = preprocessor_config.get('ordinal_categories', [[0, 1, 2, 3]])
    ord_pipeline = Pipeline([
        ('ord_encoder', OrdinalEncoder(categories=ordinal_categories))
    ])

    ## Pipeline para datos nominales OHE con drop
    ohe_drop_config = preprocessor_config.get('onehot_drop', {})
    ohe_drop_pipeline = Pipeline([
        ('nom_ohe_dropfirst', OneHotEncoder(drop = ohe_drop_config.get('drop', 'first'),
                                             handle_unknown= ohe_drop_config.get('handle_unknown', 'ignore')))
    ])

    ## Pipeline para datos nominales OHE
    ohe_config = preprocessor_config.get('onehot', {})
    ohe_pipeline = Pipeline([
        ('nom_ohe', OneHotEncoder(handle_unknown=ohe_config.get('handle_unknown', 'ignore')))
    ])

    ## Pipeline para los nominales con Frecuency
    #freq_config = preprocessor_config.get('frequency', {})
    frecuency_pipeline = Pipeline([
        ('nom_frec', SafeFrequencyEncoder())
    ])

    ## ColumnTransformer
    preprocesor = ColumnTransformer(
        transformers = [
            ('ord_encoder', ord_pipeline, ordinal_cols),
            ('nom_ohe_dropfirst', ohe_drop_pipeline, nominal_ohe_drop_cols),
            ('nom_ohe', ohe_pipeline, nominal_ohe_cols ),
            ('nom_frec', frecuency_pipeline, freq_cols) 
        ],
        remainder= preprocessor_config.get('remainder', 'drop')
    )

    return preprocesor

def build_model(models_config, seed):
    """
    Construye el modelo de stacking con el archivo de configuración
    Args:
        models_config (dict): Configuración de los modelos utilizados
        seed (int): Semilla aleatoria

    """

    base_models_config = models_config.get('base_models', {})
    base_models = [
    ('logistic', LogisticRegression(random_state=seed, 
                                    **base_models_config.get('logistic', {'max_iter': 1000}))),
    ('rforest', RandomForestClassifier(random_state=seed,
                                    **base_models_config.get('rforest', {}))),
    ('xgboost', XGBClassifier(random_state=seed,
                              **base_models_config.get('xgboost', {}))),
    ('knn', KNeighborsClassifier(
        **base_models_config.get('knn', {})
    )),
    ('svc', SVC(random_state = seed, 
                **base_models_config.get('svc', {}))),
    ('gnb', GaussianNB(
        **base_models_config.get('gnb', {})
    )),
    ('dtree', DecisionTreeClassifier(random_state=seed,
                                     ** base_models_config.get('dtree', {})))
    ]

    blender_model_config = models_config.get('blender', {})
    blender = LogisticRegression(random_state = seed, **blender_model_config)

    stacking_model_config = models_config.get('stacking', {})
    return StackingClassifier(
        estimators=base_models,
        final_estimator=blender,
        **stacking_model_config)

def build_full_pipeline(config, seed):
    """
     Construye el pipeline completo usando todo el archivo de configuración.
     
     Args
        config (dict): Archivo de configuración completo de YAML    
        seed (int): Semilla aleatoria
    """

    # Extraer las secciónes de la configuración
    try:    
        columnas_config = config.get('columnas', {})
        preprocessor_config = config.get('preprocessing', {})
        feature_engineering_config = config.get('feature_engineering', {})
        smote_config = config.get('smote', {})
        models_config = config.get('models', {})
    except Exception as e:
        print(str(e))
    

    # Construcción de los componentes    
    preprocessor = build_preprocessor(columnas_config, preprocessor_config)
    model = build_model(models_config, seed)

    pipeline = Pipeline([
    ('feature_engineering', FeatureEngineering(**feature_engineering_config)),
    ('preprocessor', preprocessor),
    ('smote', SMOTE(**smote_config)),
    ('model', model)
    ])
    
    return pipeline