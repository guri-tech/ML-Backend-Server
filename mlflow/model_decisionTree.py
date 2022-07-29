from train_mlflow import compute_model_mlflow
from pre_processing import pre_processing
from sklearn.tree import DecisionTreeClassifier

x, y, numeric_features, categorical_features = pre_processing()

dtc = DecisionTreeClassifier()
compute_model_mlflow(dtc, x, y, numeric_features, categorical_features)
