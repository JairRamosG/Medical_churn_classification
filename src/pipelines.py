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

def build_preprocessor(ordinal_cols, ):
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
