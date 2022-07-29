import sys, os
# p = os.path.abspath('.')
# sys.path.insert(1, p)
sys.path.append(os.pardir)
import mlflow.sklearn
import set_mlflow
from train_code.del_columns import deleteColumns
from db.select_db import get_db

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score

def compute_model_mlflow(model, x, y, numeric_features, categorical_features):
    mlflow.start_run()

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

    mlflow.sklearn.autolog(serialization_format='pickle', registered_model_name=model.__module__)
    model.fit(x_train, y_train)
    # print('best params: ', model.params)
    # print('best score: ', model.best_score_)

    predicted_train = model.predict(x_train)
    predicted_test = model.predict(x_test)
    print(confusion_matrix(y_test, predicted_test))
    print(classification_report(y_test, predicted_test))

    # model_name = model.__module__
    # mlflow.sklearn.log_model(model, model_name)

    metrics = { "train precision":precision_score(y_train, predicted_train),
                "train recall":recall_score(y_train, predicted_train),
                "train f1score":f1_score(y_train, predicted_train),
                "test precision":precision_score(y_test, predicted_test),
                "test recall":recall_score(y_test, predicted_test) ,
                "test f1score":f1_score(y_test, predicted_test) }

    # mlflow.log_params(model.params)
    mlflow.log_metrics(metrics)

    mlflow.end_run()

    return model
