from train_mlflow import compute_model_mlflow
from pre_processing import pre_processing
from sklearn.ensemble import RandomForestClassifier

x, y, numeric_features, categorical_features = pre_processing()

rfc = RandomForestClassifier()
compute_model_mlflow(rfc, x, y, numeric_features, categorical_features)
