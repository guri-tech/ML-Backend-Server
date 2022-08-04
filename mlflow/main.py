from train_mlflow import compute_model_mlflow
from pre_processing import pre_processing

x, y, numeric_features, categorical_features = pre_processing()

# from xgboost import XGBClassifier
# params = {'booster':'gbtree', 'max_depth':6, 'learning_rate':0.1, 'n_estimators':100, 'n_jobs':-1}
# model = XGBClassifier(**params)

# from sklearn.tree import DecisionTreeClassifier
# params = {'max_depth':10, 'min_samples_leaf':4}
# model = DecisionTreeClassifier(**params)

from sklearn.linear_model import LogisticRegression
params = {'max_iter':100, 'solver':'saga', 'n_jobs':-1}
model = LogisticRegression(**params)

# from sklearn.ensemble import RandomForestClassifier
# model = RandomForestClassifier()

compute_model_mlflow(model, params, x, y, numeric_features, categorical_features)
