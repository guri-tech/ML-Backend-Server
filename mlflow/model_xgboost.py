from train_mlflow import compute_model_mlflow
from pre_processing import pre_processing
from xgboost import XGBClassifier

x, y, numeric_features, categorical_features = pre_processing()

xgb = XGBClassifier()
compute_model_mlflow(xgb, x, y, numeric_features, categorical_features)
