import pandas as pd
# import os

# BASE_DIR = os.path.dirname(__file__)
# df = pd.read_csv(f'{BASE_DIR}/../dataset/hotel_3_removeColumns.csv')
# X = df.iloc[:,:-1]
# y = df.iloc[:,[-1]] # 파이프라인이 2차원 행렬 요구

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
from sympy import EX

def trianModel(model, x, y, numeric_features, categorical_features):
    # 수치형 변수: SimpleImputer 결측치는 중앙값으로 대치 & 정규화
    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])

    # 범주형 변수: 원핫인코딩
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ('numerical', numeric_transformer, numeric_features),
            ('categorical', categorical_transformer, categorical_features),
        ])

    # 전처리 후 모델 fit
    clf = Pipeline( steps=[("preprocessor", preprocessor), ("classifier", model)] )

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y)
    try:
        clf.fit(x_train, y_train)
    except Exception as e:
        print(e)
    # 정확도 train, test
    print("train model score: %.3f" % clf.score(x_train, y_train))
    print("test model score: %.3f" % clf.score(x_test, y_test))

    # classification report
    print("test Predict: \n", classification_report(y_test, clf.predict(x_test)) )
    
    return clf
