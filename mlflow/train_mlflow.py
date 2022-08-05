import sys, os
sys.path.append(os.pardir)
import mlflow
from db.select_db import get_db

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score


def compute_model_mlflow(model, params, x, y, numeric_features, categorical_features):

    mlflow.set_tracking_uri('http://127.0.0.1:5000')
    mlflow.set_experiment('HOTEL-EXPERIMENT')
    mlflow.start_run()

    # 수치형 변수: SimpleImputer 결측치는 중앙값으로 대치 & 정규화
    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    # 범주형 변수: 원핫인코딩
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    # 전처리와 모델 학습 파이프라인 분리 -> mlflow에 전처리 과정이 저장 안 됨
    # ct = ColumnTransformer([
    #         ('numerical', numeric_transformer, numeric_features),
    #         ('categorical', categorical_transformer, categorical_features)
    #     ])
    # x_data = ct.fit_transform(x)
    # y_data = y.values.reshape(-1,)
    # x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, stratify=y_data)

    ## 원핫인코딩, 결측치 대치, 정규화
    preprocessor = ColumnTransformer(transformers=[
            ('numerical', numeric_transformer, numeric_features),
            ('categorical', categorical_transformer, categorical_features),
        ])
    clf = Pipeline( steps=[("preprocessor", preprocessor), ("classifier", model)] )
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y)


    clf.fit(x_train, y_train)
    # client = mlflow.tracking.MlflowClient()
    # data = client.get_run(mlflow.active_run().info.run_id).data
    # for i in data:
    #     print(i)

    predicted_train = clf.predict(x_train)
    predicted_test = clf.predict(x_test)
    print(confusion_matrix(y_test, predicted_test))
    print(classification_report(y_test, predicted_test))

    metrics = { "train precision":precision_score(y_train, predicted_train),
                "train recall":recall_score(y_train, predicted_train),
                "train f1score":f1_score(y_train, predicted_train),
                "test precision":precision_score(y_test, predicted_test),
                "test recall":recall_score(y_test, predicted_test) ,
                "test f1score":f1_score(y_test, predicted_test) }

    clf_name = clf.__module__
    mlflow.sklearn.log_model(clf, clf_name, registered_model_name=clf_name)
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)

    mlflow.end_run()

    return clf
