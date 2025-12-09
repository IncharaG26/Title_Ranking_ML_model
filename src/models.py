# src/models.py
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

def get_default_models(random_state=42):
    models = {
        "xgb": XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=random_state, tree_method='auto', verbosity=0),
        "rf": RandomForestRegressor(n_estimators=300, random_state=random_state, n_jobs=-1),
        "ridge": Ridge(alpha=1.0, random_state=random_state),
        "svr": SVR(C=1.0, kernel='rbf', gamma='scale')
    }
    # prefer xgb + rf for final runs
    return models
