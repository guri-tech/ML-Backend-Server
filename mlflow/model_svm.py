from train_mlflow import compute_model_mlflow
from pre_processing import pre_processing
from sklearn.svm import SVC
from grid_train_mlflow import grid_compute_model

x, y, numeric_features, categorical_features = pre_processing()

params = {"kernel": ("linear", "rbf"), "C": [1, 10]}

sgc = SVC()
# compute_model_mlflow(sgc, x, y, numeric_features, categorical_features)
grid_compute_model(sgc, params, x, y, numeric_features, categorical_features)
