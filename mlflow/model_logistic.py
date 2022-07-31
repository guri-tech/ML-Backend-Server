from sklearn.linear_model import LogisticRegression
from train_mlflow import compute_model_mlflow
from pre_processing import pre_processing

x, y, numeric_features, categorical_features = pre_processing()

rfc = LogisticRegression(max_iter=1000, solver='saga', n_jobs=-1)
compute_model_mlflow(rfc, x, y, numeric_features, categorical_features)
