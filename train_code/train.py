import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix

def trainModel(model, x, y, numeric_features, categorical_features):
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
    # print(x_data.shape, y_data.shape)

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, stratify=y_data)
    model.fit(x_train, y_train)

    # 정확도 train, test
    print("########## ", model.__module__, "##########")
    print("train model score: %.3f" % model.score(x_train, y_train))
    print("test model score: %.3f" % model.score(x_test, y_test))
    print("\ntest confusion matrix: \n", confusion_matrix(y_test, model.predict(x_test)))
    # classification report
    print("test Predict: \n", classification_report(y_test, model.predict(x_test)) )

    return model
