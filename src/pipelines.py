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

def build_preprocessor(numerical_cols, ordinal_cols, nominal_ohe_drop_cols, nominal_ohe_cols, freq_cols):
    """Construye el ColumnTransformer con pipelines para cada tipo de dato."""
    
    ## Pipeline para datos ordinales
    ordinal_categories = [[0, 1, 2, 3]]
    ord_pipeline = Pipeline([
        ('ord_encoder', OrdinalEncoder(categories=ordinal_categories))
    ])

    ## Pipeline para datos nominales
    ohe_drop_pipeline = Pipeline([
        ('nom_ohe_dropfirst', OneHotEncoder(drop = 'first', handle_unknown='ignore'))
    ])

    ohe_pipeline = Pipeline([
        ('nom_ohe', OneHotEncoder(handle_unknown='ignore'))
    ])

    frecuency_pipeline = Pipeline([
        ('nom_frec', SafeFrequencyEncoder())
    ])

    ## ColumnTransformer
    preprocesor = ColumnTransformer(
        transformers = [
            ('ord_encoder', ord_pipeline, ordinal_cols),
            ('nom_ohe_dropfirst', ohe_drop_pipeline, ['Gender', 'Billing_Issues', 'Portal_Usage']),
            ('nom_ohe', ohe_pipeline, ['Insurance_Type'] ),
            ('nom_frec', frecuency_pipeline, ['State', 'Specialty']) 
        ],
        remainder='drop'
    )

    return preprocesor

def build_model(seed):
    """Construye el modelo de stacking."""

    base_models = [
    ('logistic', LogisticRegression(random_state=seed, max_iter=1000)),
    ('rforest', RandomForestClassifier(random_state=seed)),
    ('xgboost', XGBClassifier(random_state=seed, use_label_encoder=False, eval_metric='logloss')),
    ('knn', KNeighborsClassifier()),
    ('svc', SVC(random_state = seed)),
    ('gnb', GaussianNB()),
    ('dtree', DecisionTreeClassifier(random_state=seed))
    ]
    blender = LogisticRegression(random_state = seed)

    return StackingClassifier(
        estimators=base_models,
        final_estimator=blender)

def build_full_pipeline(numerical_cols, ordinal_cols, nominal_ohe_drop_cols, nominal_ohe_cols, freq_cols, seed):
     """Construye el pipeline completo: FE + Preprocessor + SMOTE + Modelo."""
    
    preprocessor = build_preprocessor(numerical_cols, ordinal_cols, nominal_ohe_drop_cols, nominal_ohe_cols, freq_cols)
    model = build_model(seed)

    pipeline = Pipeline([
    ('feature_engineering', FeatureEngineering()),
    ('preprocessor', preprocessor),
    ('smote', SMOTE(sampling_strategy='auto', random_state=seed)),
    ('model', model)
])
    pass


## Creo que tengo que hardcodear los parametros desde el archivo de config