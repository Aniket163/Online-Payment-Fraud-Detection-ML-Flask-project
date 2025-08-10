import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score
from imblearn.over_sampling import SMOTE
import joblib
import requests

DATA_DIR = 'data'
os.makedirs(DATA_DIR, exist_ok=True)
CSV_PATH = os.path.join(DATA_DIR, 'transactions.csv')

# Public raw GitHub CSV (used as fallback). If this URL stops working, replace with local CSV.
GITHUB_CSV_URL = 'https://raw.githubusercontent.com/jay01varma/Online-Payment-Fraud-Detection/master/dataset.csv'

def download_if_needed():
    if not os.path.exists(CSV_PATH):
        print('transactions.csv not found locally — attempting to download from GitHub...')
        try:
            r = requests.get(GITHUB_CSV_URL, timeout=30)
            r.raise_for_status()
            with open(CSV_PATH, 'wb') as f:
                f.write(r.content)
            print('Downloaded dataset to', CSV_PATH)
        except Exception as e:
            print('Failed to download dataset:', e)
            print('Please place your dataset CSV at', CSV_PATH)
            raise SystemExit(1)

def load_data():
    download_if_needed()
    df = pd.read_csv(CSV_PATH)
    print('Loaded CSV with shape', df.shape)
    return df

def preprocess(df):
    # Drop identifiers that leak info or are not useful directly
    df = df.copy()
    cols_drop = ['nameOrig','nameDest']
    for c in cols_drop:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    # If 'isFraud' not present, try 'isfraud' lowercase
    if 'isFraud' not in df.columns and 'isfraud' in df.columns:
        df.rename(columns={'isfraud':'isFraud'}, inplace=True)

    # Basic handling for missing columns
    if 'isFraud' not in df.columns:
        raise ValueError('Expected column isFraud not found.')

    # Treat 'type' as categorical, amount numeric
    cat_features = []
    if 'type' in df.columns:
        cat_features.append('type')
    num_features = [c for c in df.columns if c not in cat_features + ['isFraud','isFlaggedFraud','step']]
    # we will drop step and isFlaggedFraud for training
    if 'step' in df.columns:
        df.drop(columns=['step'], inplace=True)
    if 'isFlaggedFraud' in df.columns:
        df.drop(columns=['isFlaggedFraud'], inplace=True)

    X = df.drop(columns=['isFraud'])
    y = df['isFraud']

    # Column transformer
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object','category']).columns.tolist()

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), categorical_cols)
    ])

    X_trans = preprocessor.fit_transform(X)
    print('Transformed shape:', X_trans.shape)
    return X_trans, y, preprocessor

def train_and_save(X, y, preprocessor):
    # handle imbalance with SMOTE
    print('Original fraud ratio:', np.mean(y))
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    print('Resampled shape:', X_res.shape, 'Fraud ratio now:', np.mean(y_res))

    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)

    models = {
        'rf': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        'xgb': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1)
    }
    best_model = None
    best_score = -1
    for name, model in models.items():
        print('Training', name)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        f1 = f1_score(y_test, preds, pos_label=1)
        print(name, 'F1:', f1)
        if f1 > best_score:
            best_score = f1
            best_model = model
    print('Best model:', best_model.__class__.__name__, 'score', best_score)

    # Save model and preprocessor
    joblib.dump(best_model, 'model.joblib')
    joblib.dump(preprocessor, 'preprocessor.joblib')
    print('Saved model.joblib and preprocessor.joblib')

if __name__ == '__main__':
    df = load_data()
    X, y, pre = preprocess(df)
    train_and_save(X, y, pre)
