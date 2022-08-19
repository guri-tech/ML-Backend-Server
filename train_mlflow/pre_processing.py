import sys, os

sys.path.append(os.pardir)
from db.select_db import get_db
import pandas as pd
import numpy as np


def pre_processing():

    # 데이터 불러오기
    df = get_db()

    # 결측치가 많거나 불필요해 보이는 컬럼 삭제
    del_col_names = [
        "agent",
        "company",
        "country",
        "reservation_status_date",
        "reservation_status",
    ]
    df_new = df.drop(del_col_names, axis=1)

    x = df_new.iloc[:, :-1]
    y = df_new.iloc[:, [-1]]

    # 수치형 변수(int, float) 컬럼명
    numeric_features = x.select_dtypes([int, float]).columns
    # 범주형 변수(object) 컬럼명
    categorical_features = x.select_dtypes(object).columns

    return x, y, numeric_features, categorical_features
