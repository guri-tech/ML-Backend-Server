import os
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
from prefect import task
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


@task(log_stdout=True)
def get_data():

    engine = create_engine(os.getenv("DATABASE_URL"))

    sql = "SELECT * FROM bookingtb WHERE is_canceled IS NOT NULL;"
    df = pd.read_sql(sql, con=engine)

    return df


@task(log_stdout=True, nout=2)
def set_model():

    from xgboost import XGBClassifier

    params = {
        "booster": "gbtree",
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 16,
        "learning_rate": 0.1,
        "n_estimators": 400,
        "subsample": 0.9,
        "max_delta_step": 5,
        "gamma": 1,
        "n_jobs": -1,
    }
    model = XGBClassifier(use_label_encoder=False, **params)

    return model, params


@task(log_stdout=True, nout=3)
def preprocessing(df):

    x = df.iloc[:, :-1]
    y = df.iloc[:, [-1]]

    # 수치형 변수(int, float) 컬럼명
    numeric_features = x.select_dtypes(exclude=[object]).columns

    # 범주형 변수(object) 컬럼명
    categorical_features = x.select_dtypes(object).columns

    # 수치형 변수: SimpleImputer 결측치는 중앙값으로 대치 & 정규화
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    # 범주형 변수: 원핫인코딩
    categorical_transformer = OneHotEncoder()

    # 원핫인코딩, 결측치 대치, 정규화
    ct = ColumnTransformer(
        [
            ("numerical", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ]
    )
    x_data = ct.fit_transform(x)
    y_data = y.values.reshape(
        -1,
    )

    return x_data, y_data, ct


@task(log_stdout=True, nout=2)
def train_model(model, x, y):

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y)
    model.fit(x_train, y_train)

    predicted_train = model.predict(x_train)
    predicted_test = model.predict(x_test)

    metrics = {
        "train accuracy": accuracy_score(y_train, predicted_train),
        "train precision": precision_score(y_train, predicted_train),
        "train recall": recall_score(y_train, predicted_train),
        "train f1score": f1_score(y_train, predicted_train),
        "test accuracy": accuracy_score(y_test, predicted_test),
        "test precision": precision_score(y_test, predicted_test),
        "test recall": recall_score(y_test, predicted_test),
        "test f1score": f1_score(y_test, predicted_test),
        "auc": roc_auc_score(y_test, model.predict_proba(x_test)[:, 1]),
    }

    print("train confunsion matrix \n", confusion_matrix(y_train, predicted_train))
    print("\n test confunsion matrix \n", confusion_matrix(y_test, predicted_test))
    return model, metrics


@task(log_stdout=True, nout=2)
def train_model_onepipeline(model, df):

    x = df.iloc[:, :-1]
    y = df.iloc[:, [-1]]

    # 수치형 변수(int, float) 컬럼명
    numeric_features = x.select_dtypes(exclude=["object"]).columns

    # 범주형 변수(object) 컬럼명
    categorical_features = x.select_dtypes(object).columns

    # 수치형 변수: SimpleImputer 결측치는 중앙값으로 대치 & 정규화
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    # 범주형 변수: 원핫인코딩
    categorical_transformer = OneHotEncoder()

    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ]
    )
    clf = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y)
    clf.fit(x_train, y_train)

    predicted_train = clf.predict(x_train)
    predicted_test = clf.predict(x_test)

    metrics = {
        "train accuracy": accuracy_score(y_train, predicted_train),
        "train precision": precision_score(y_train, predicted_train),
        "train recall": recall_score(y_train, predicted_train),
        "train f1score": f1_score(y_train, predicted_train),
        "test accuracy": accuracy_score(y_test, predicted_test),
        "test precision": precision_score(y_test, predicted_test),
        "test recall": recall_score(y_test, predicted_test),
        "test f1score": f1_score(y_test, predicted_test),
        "auc": roc_auc_score(y_test, clf.predict_proba(x_test)[:, 1]),
    }
    print("train confunsion matrix \n", confusion_matrix(y_train, predicted_train))
    print("\n test confunsion matrix \n", confusion_matrix(y_test, predicted_test))
    return clf, metrics


@task(log_stdout=True)
def log_preprocessor(preprocessor):

    mlflow.set_tracking_uri(os.getenv("URI"))
    mlflow.set_experiment("Default")

    with mlflow.start_run() as pre_run:
        model_info = mlflow.sklearn.log_model(preprocessor, "preprocessor")

    # search registered_models with same model_name
    client = MlflowClient()
    models = client.search_model_versions("name='preprocessor'")

    # if there isn't, log_model and register current model to product
    if len(models) == 0:
        client.create_registered_model("preprocessor")

    model_path = RunsArtifactRepository.get_underlying_uri(model_info.model_uri)
    model_version = client.create_model_version(
        "preprocessor",
        model_path,
        model_info.run_id,
        description="filling, encoding, scaling preprocessor",
    )
    # set Production
    client.transition_model_version_stage(
        "preprocessor",
        model_version.version,
        "Production",
        archive_existing_versions=True,
    )

    return "Success"


@task(log_stdout=True)
def log_model(model, model_name, params, metrics, eval_metric):

    mlflow.set_tracking_uri(os.getenv("URI"))
    mlflow.set_experiment("Default")

    with mlflow.start_run() as mlflow_run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        model_info = mlflow.sklearn.log_model(model, model_name)

    # search registered_models with same model_name
    name_filter = f"name='{model_name}'"
    client = MlflowClient()
    models = client.search_model_versions(name_filter)
    # if there isn't, register current model
    if len(models) == 0:
        client.create_registered_model(model_name)

    metric = client.get_run(model_info.run_id).data.metrics[eval_metric]
    model_path = RunsArtifactRepository.get_underlying_uri(model_info.model_uri)
    model_version = client.create_model_version(
        model_name, model_path, model_info.run_id, description=f"{eval_metric}:{metric}"
    )

    return model_version.version


@task(log_stdout=True)
def change_production_model(model_name, current_version, eval_metric):

    production_model = None

    client = MlflowClient()
    current_model = client.get_model_version(model_name, current_version)

    name_filter = f"name='{model_name}'"
    models = client.search_model_versions(name_filter)

    # get current Production model
    for model in models:
        if model.current_stage == "Production":
            production_model = model
    # if there is not Production model
    if production_model is None:
        client.transition_model_version_stage(
            current_model.name, current_model.version, "Production"
        )
        production_model = current_model
    else:
        current_metric = client.get_run(current_model.run_id).data.metrics[eval_metric]
        production_metric = client.get_run(production_model.run_id).data.metrics[
            eval_metric
        ]

        if current_metric > production_metric:
            client.transition_model_version_stage(
                current_model.name,
                current_model.version,
                "Production",
                archive_existing_versions=True,
            )
        production_model = current_model

    return production_model.version
