import sys, os
sys.path.append(os.pardir)
import pandas as pd
import mlflow
import set_mlflow
from train_code.del_columns import deleteColumns
from db.select_db import get_db

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV

from pprint import pprint

def fetch_logged_data(run_id):
    client = mlflow.tracking.MlflowClient()
    data = client.get_run(run_id).data
    tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in client.list_artifacts(run_id, "model")]
    return data.params, data.metrics, tags, artifacts


def grid_compute_model(model, params, x, y, numeric_features, categorical_features):
    with mlflow.start_run() as run:

        # 수치형 변수: SimpleImputer 결측치는 중앙값으로 대치 & 정규화
        numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
        # 범주형 변수: 원핫인코딩
        categorical_transformer = OneHotEncoder(handle_unknown="ignore")

        ### 원핫인코딩, 결측치 대치, 정규화
        ct = ColumnTransformer([
                ('numerical', numeric_transformer, numeric_features),
                ('categorical', categorical_transformer, categorical_features),
            ])

        x_data = ct.fit_transform(x)
        y_data = y.values.reshape(-1,)

        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, stratify=y_data)

        mlflow.sklearn.autolog(serialization_format='pickle')

        model = GridSearchCV(model, params)
        model.fit(x_train, y_train)

        # show data logged in the parent run
        print("========== parent run ==========")
        for key, data in fetch_logged_data(run.info.run_id).items():
            print("\n---------- logged {} ----------".format(key))
            pprint(data)

        # show data logged in the child run
        filter_child_runs = "tags.mlflow.parentRunId = '{}'".format(run.info.run_id)
        runs = mlflow.search_runs(filter_string=filter_child_runs)
        param_cols = ["params.{}".format(p) for p in params.keys()]
        metric_cols = ["metrics.mean_test_score"]

        print("\n========== child runs ==========\n")
        pd.set_option("display.max_columns", None)  # prevent truncating columns
        print(runs[["run_id", *param_cols, *metric_cols]])

    return model
