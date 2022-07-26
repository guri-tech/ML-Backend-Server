import sys, os
sys.path.append(os.pardir)
import pandas as pd
from arange_colums import arangeColumns
from del_columns import deleteColumns
from db.select_db import get_db
from save_model import saveModel
from train import trainModel
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
import joblib

# db에서 데이터 불러오기
df = get_db()

x = df.iloc[:,:-1]
y = df.iloc[:,[-1]]

# 결측치가 많거나 불필요해 보이는 컬럼 삭제
del_col_names = ['agent','company','country','reservation_status_date','reservation_status']
x = deleteColumns(x, del_col_names)

# 수치형 변수: SimpleImputer 결측치는 중앙값으로 대치 & 정규화
numeric_features = x.select_dtypes([int, float]).columns

# 범주형 변수: 원핫인코딩
categorical_features = x.select_dtypes(object).columns

# xgb = XGBClassifier(booster='gbtree', eval_metric='logloss', use_label_encoder=False,
#                    max_depth=6, learning_rate=0.1, n_estimators=100, n_jobs=-1)

xgb = XGBClassifier()
clf_xgb = trainModel(xgb, x, y, numeric_features, categorical_features)

# logi = LogisticRegression(max_iter=1000, n_jobs=-1, solver='sag')
# clf_logi = trainModel(logi, x, y, numeric_features, categorical_features)

# lgbm = LGBMClassifier()
# clf_lgbm = trainModel(lgbm, x, y, numeric_features, categorical_features)

# 모델 저장
saveModel(clf_xgb, 'xgb.pkl')
# saveModel(clf_logi, 'logistic.pkl')
# saveModel(clf_lgbm, 'lgbm.pkl')
